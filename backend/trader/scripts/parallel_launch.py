#!/usr/bin/env python3
"""
parallel_launch.py — v11: Walk-Forward CV + Anti-Overfit Pipeline.

v11 CHANGES:
  - Default trials reduced: 200 → 100 (smaller search space needs fewer)
  - Passes --n-cv-folds to optimize workers
  - Updated presets with v11 defaults
  - Cleaner progress reporting showing OOS metrics
  - Tighter plateau settings (patience 60, not 140)

Usage:
    python parallel_launch.py                                    # full pipeline
    python parallel_launch.py --validate-only                    # skip optim, just validate
    python parallel_launch.py --trials 100 --preset robust180
    python parallel_launch.py --coins BTC,ETH --skip-validation  # optim only
"""
import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]
PILOT_ROLLOUT_DEFAULT_COINS = ["ETH", "SOL"]

# v11: Updated presets — tighter patience, CV folds
PRESET_CONFIGS = {
    "paper_ready": {
        "plateau_patience": 150,
        "plateau_min_delta": 0.012,
        "plateau_warmup": 80,
        "holdout_days": 240,
        "min_internal_oos_trades": 10,
        "min_total_trades": 28,
        "n_cv_folds": 5,
        "holdout_candidates": 4,
        "holdout_min_trades": 15,
        "holdout_min_sharpe": 0.05,
        "holdout_min_return": 0.0,
        "require_holdout_pass": True,
        "target_trades_per_week": 1.0,
        "plateau_min_completed": 0,
    },
    "robust180": {
        "plateau_patience": 120,
        "plateau_min_delta": 0.015,
        "plateau_warmup": 60,
        "holdout_days": 180,
        "min_internal_oos_trades": 8,
        "min_total_trades": 30,
        "n_cv_folds": 5,
        "holdout_candidates": 3,
        "holdout_min_trades": 15,
        "holdout_min_sharpe": 0.0,
        "holdout_min_return": 0.0,
        "require_holdout_pass": True,
        "target_trades_per_week": 1.0,
        "plateau_min_completed": 0,
    },
    "robust120": {
        "plateau_patience": 90,
        "plateau_min_delta": 0.015,
        "plateau_warmup": 45,
        "holdout_days": 120,
        "min_internal_oos_trades": 6,
        "min_total_trades": 25,
        "n_cv_folds": 5,
        "holdout_candidates": 2,
        "holdout_min_trades": 12,
        "holdout_min_sharpe": 0.0,
        "holdout_min_return": 0.0,
        "require_holdout_pass": True,
        "target_trades_per_week": 1.0,
        "plateau_min_completed": 0,
    },
    "quick": {
        "plateau_patience": 45,
        "plateau_min_delta": 0.03,
        "plateau_warmup": 20,
        "holdout_days": 90,
        "min_internal_oos_trades": 5,
        "min_total_trades": 20,
        "n_cv_folds": 2,
        "holdout_candidates": 1,
        "holdout_min_trades": 10,
        "holdout_min_sharpe": 0.0,
        "holdout_min_return": -0.01,
        "require_holdout_pass": False,
        "target_trades_per_week": 0.8,
        "plateau_min_completed": 0,
    },
    "pilot_rollout": {
        "coins": ",".join(PILOT_ROLLOUT_DEFAULT_COINS),
        "trials": 40,
        "plateau_patience": 45,
        "plateau_min_delta": 0.025,
        "plateau_warmup": 20,
        "holdout_days": 180,
        "min_internal_oos_trades": 10,
        "min_total_trades": 35,
        "n_cv_folds": 5,
        "holdout_candidates": 2,
        "holdout_min_trades": 20,
        "holdout_min_sharpe": 0.10,
        "holdout_min_return": 0.01,
        "require_holdout_pass": True,
        "target_trades_per_week": 1.0,
        "plateau_min_completed": 0,
    },
}


def apply_runtime_preset(args):
    config = PRESET_CONFIGS.get(args.preset)
    if not config:
        return args

    # Respect explicit CLI overrides; only backfill values not provided by user.
    arg_flags = {
        "plateau_patience": "--plateau-patience",
        "plateau_min_delta": "--plateau-min-delta",
        "plateau_warmup": "--plateau-warmup",
        "plateau_min_completed": "--plateau-min-completed",
        "holdout_days": "--holdout-days",
        "min_internal_oos_trades": "--min-internal-oos-trades",
        "min_total_trades": "--min-total-trades",
        "n_cv_folds": "--n-cv-folds",
        "holdout_candidates": "--holdout-candidates",
        "holdout_min_trades": "--holdout-min-trades",
        "holdout_min_sharpe": "--holdout-min-sharpe",
        "holdout_min_return": "--holdout-min-return",
        "require_holdout_pass": "--require-holdout-pass",
        "target_trades_per_week": "--target-trades-per-week",
        "coins": "--coins",
        "trials": "--trials",
    }
    provided = set(sys.argv[1:])
    for key, value in config.items():
        flag = arg_flags.get(key)
        if flag and flag in provided:
            continue
        setattr(args, key, value)
    return args


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _fmt_metric(value, fmt, fallback="?"):
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


def load_optimization_results(results_dir):
    results = {}
    if results_dir.exists():
        for p in results_dir.glob("*_optimization.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                coin = data.get('coin', p.stem.replace('_optimization', ''))
                results[coin] = data
            except Exception:
                pass
    return results


def load_validation_results(results_dir):
    results = {}
    if results_dir.exists():
        for p in results_dir.glob("*_validation.json"):
            try:
                with open(p) as f:
                    data = json.load(f)
                coin = data.get('coin', p.stem.replace('_validation', ''))
                results[coin] = data
            except Exception:
                pass
    return results


def get_coin_trial_progress(optuna_db, target_coins, run_id):
    progress = {
        coin: {"started": 0, "completed": 0, "best_score": None, "best_metrics": {}}
        for coin in target_coins
    }
    if not optuna_db.exists():
        return progress

    # Simple count query as primary method (most reliable under contention)
    count_query = """
        SELECT
            s.study_name,
            COUNT(*) AS started_trials,
            SUM(CASE WHEN CAST(t.state AS TEXT) IN ('1', 'COMPLETE') THEN 1 ELSE 0 END) AS completed_trials
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        GROUP BY s.study_name
    """
    best_trial_query = """
        SELECT t.trial_id, v.value
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        JOIN trial_values v ON v.trial_id = t.trial_id
        WHERE s.study_name = ? AND CAST(t.state AS TEXT) IN ('1', 'COMPLETE') AND v.objective = 0
        ORDER BY v.value DESC LIMIT 1
    """
    attrs_query = """
        SELECT key, value_json
        FROM trial_user_attributes
        WHERE trial_id = ?
          AND key IN ('mean_oos_sharpe', 'oos_sharpe', 'n_trades', 'win_rate', 'max_drawdown', 'std_oos_sharpe')
    """

    for attempt in range(3):
        try:
            with sqlite3.connect(str(optuna_db), timeout=5) as conn:
                conn.execute("PRAGMA busy_timeout = 5000;")
                conn.execute("PRAGMA journal_mode=WAL;")

                rows = conn.execute(count_query).fetchall()
                studies_by_coin = {coin: [] for coin in target_coins}

                # Match current run studies, including multi-seed suffixes like _s42
                for study_name, started_trials, completed_trials in rows:
                    for coin in target_coins:
                        expected_prefix = f"optimize_{coin}_{run_id}"
                        if study_name.startswith(expected_prefix):
                            progress[coin]["started"] += int(started_trials or 0)
                            progress[coin]["completed"] += int(completed_trials or 0)
                            studies_by_coin[coin].append(study_name)
                            break

                # Get best trial details per coin across all matching studies for this run_id
                for coin in target_coins:
                    if progress[coin]["completed"] == 0:
                        continue

                    best_trial_id = None
                    best_score = None
                    for study_name in studies_by_coin.get(coin, []):
                        best_row = conn.execute(best_trial_query, (study_name,)).fetchone()
                        if not best_row:
                            continue
                        candidate_trial_id, candidate_score = best_row
                        if candidate_score is None:
                            continue
                        if best_score is None or float(candidate_score) > float(best_score):
                            best_score = float(candidate_score)
                            best_trial_id = candidate_trial_id

                    if best_trial_id is None:
                        continue

                    progress[coin]["best_score"] = float(best_score)
                    metric_rows = conn.execute(attrs_query, (best_trial_id,)).fetchall()
                    metrics = {}
                    for key, value_json in metric_rows:
                        try:
                            metrics[key] = json.loads(value_json)
                        except (TypeError, json.JSONDecodeError):
                            metrics[key] = value_json
                    progress[coin]["best_metrics"] = metrics
            return progress
        except sqlite3.OperationalError:
            time.sleep(0.5 * (attempt + 1))
        except sqlite3.Error:
            break

    return progress


def print_final_report(script_dir, target_coins, total_time):
    results_dir = script_dir / "optimization_results"
    opt_results = load_optimization_results(results_dir)
    val_results = load_validation_results(results_dir)

    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE PIPELINE REPORT (v11 — Walk-Forward CV)")
    print(f"{'='*80}")
    print(f"   Total runtime: {format_duration(total_time)}")
    print(f"   Coins: {', '.join(target_coins)}")
    print()

    ready_coins = []
    cautious_coins = []
    reject_coins = []

    _f = lambda v, d=3: _fmt_metric(v, f'.{d}f') if v is not None else '?'

    for coin in target_coins:
        opt = opt_results.get(coin, {})
        val = val_results.get(coin, {})

        optim_metrics = opt.get('optim_metrics', {})
        holdout_metrics = opt.get('holdout_metrics', {})
        readiness = val.get('readiness', {})
        version = opt.get('version', 'v10')

        # v11: Show OOS metrics primarily
        mean_oos_sr = optim_metrics.get('mean_oos_sharpe', optim_metrics.get('oos_sharpe', '?'))
        min_oos_sr = optim_metrics.get('min_oos_sharpe', '?')
        std_oos_sr = optim_metrics.get('std_oos_sharpe', '?')
        opt_wr = optim_metrics.get('win_rate', '?')
        opt_dd = optim_metrics.get('max_drawdown', '?')
        opt_trades = optim_metrics.get('n_trades', '?')
        opt_tpy = optim_metrics.get('trades_per_year', '?')
        freq_ratio = optim_metrics.get('frequency_ratio', '?')

        ho_sharpe = holdout_metrics.get('holdout_sharpe', '?')
        ho_return = holdout_metrics.get('holdout_return', '?')
        ho_trades = holdout_metrics.get('holdout_trades', '?')

        val_score = readiness.get('score', '?')
        val_rating = readiness.get('rating', 'N/A')
        deployment_blocked = bool(opt.get('deployment_blocked', False))
        block_reasons = opt.get('deployment_block_reasons', [])
        if deployment_blocked:
            val_rating = 'REJECT'

        emoji = {'READY': '✅', 'CAUTIOUS': '⚠️', 'WEAK': '🟡', 'REJECT': '❌'}.get(val_rating, '⬜')

        print(f"  {emoji} {coin} [{version}]")
        print(f"     ┌─ CV Optimization")
        print(f"     │  Mean OOS SR={_f(mean_oos_sr)} | Min OOS SR={_f(min_oos_sr)} | "
              f"Std={_f(std_oos_sr)} | WR={_fmt_metric(opt_wr, '.1%')} | "
              f"DD={_fmt_metric(opt_dd, '.1%')} | Trades={opt_trades}")
        print(f"     │  Trades/Year={_fmt_metric(opt_tpy, '.1f')} | Frequency ratio={_fmt_metric(freq_ratio, '.2f')}")
        print(f"     ├─ Holdout")
        print(f"     │  Sharpe={_f(ho_sharpe)} | Return={_fmt_metric(ho_return, '.2%')} | Trades={ho_trades}")

        if readiness:
            print(f"     ├─ Validation: {val_rating} ({_fmt_metric(val_score, '.0f')}/100)")

            checks = readiness.get('details', [])
            if checks:
                failed = [c for c in checks if not c['passed']]
                if failed:
                    print(f"     │  Failed: {', '.join(c['name'] for c in failed)}")
                else:
                    print(f"     │  All {len(checks)} checks passed")
        else:
            print(f"     └─ Validation: not run")
        if deployment_blocked and block_reasons:
            print(f"     │  Deployment blocked: {', '.join(block_reasons)}")
        print()

        if val_rating == 'READY':
            ready_coins.append(coin)
        elif val_rating == 'CAUTIOUS':
            cautious_coins.append(coin)
        else:
            reject_coins.append(coin)

    print(f"{'='*80}")
    print(f"📋 RECOMMENDATION")
    print(f"{'='*80}")
    if ready_coins:
        print(f"  ✅ PAPER TRADE NOW:  {', '.join(ready_coins)}")
        print(f"     → Deploy to paper trading with standard position sizing")
    if cautious_coins:
        print(f"  ⚠️  PAPER TRADE (CAUTIOUS): {', '.join(cautious_coins)}")
        print(f"     → Paper trade with 50% reduced position size, monitor closely")
    if reject_coins:
        print(f"  ❌ DO NOT TRADE:     {', '.join(reject_coins)}")
        print(f"     → Needs more data, different features, or parameter re-tuning")
    if not ready_coins and not cautious_coins:
        print(f"  ⚠️  No coins reached paper-trade readiness.")
        print(f"     → Review the validation reports and consider:")
        print(f"       - Collecting more historical data")
        print(f"       - Simplifying the strategy (fewer parameters)")
        print(f"       - Testing on different timeframes")
    print(f"{'='*80}")


def _coin_intent_from_rating(rating: str, coin: str, args) -> str:
    rating_key = (rating or "").upper()
    if args.preset == "pilot_rollout":
        if coin in PILOT_ROLLOUT_DEFAULT_COINS and rating_key in {"READY", "CAUTIOUS", "FULL", "PILOT"}:
            return "PILOT"
        return "SHADOW"
    if rating_key in {"READY", "CAUTIOUS", "FULL", "PILOT"}:
        return "PILOT"
    return "SHADOW"


def write_launch_summary(script_dir: Path, args, target_coins: List[str], run_id: str, total_time: float) -> Path:
    results_dir = script_dir / "optimization_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    opt_results = load_optimization_results(results_dir)
    val_results = load_validation_results(results_dir)

    coins_summary = []
    for coin in target_coins:
        validation = val_results.get(coin, {})
        readiness = validation.get("readiness", {}) if isinstance(validation, dict) else {}
        rating = readiness.get("rating") or "UNKNOWN"
        intent = _coin_intent_from_rating(rating, coin, args)
        coins_summary.append(
            {
                "coin": coin,
                "deployment_intent": intent,
                "readiness_rating": rating,
                "readiness_tier": readiness.get("readiness_tier"),
                "recommended_position_scale": readiness.get("recommended_position_scale"),
                "deployment_blocked": bool(opt_results.get(coin, {}).get("deployment_blocked", False)),
                "deployment_block_reasons": opt_results.get(coin, {}).get("deployment_block_reasons", []),
            }
        )

    summary = {
        "run_id": run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "coins": target_coins,
        "trials": args.trials,
        "runtime_seconds": round(total_time, 2),
        "deployment_tier_map": {row["coin"]: row["deployment_intent"] for row in coins_summary},
        "coin_summaries": coins_summary,
    }

    versioned_path = results_dir / f"launch_summary_{run_id}.json"
    latest_path = results_dir / "launch_summary.json"
    with open(versioned_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    with open(latest_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\n🧾 Launch summary written: {latest_path}")
    return latest_path


def _print_log_tail(log_path, max_lines=30):
    if not log_path.exists():
        return
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return
    tail = lines[-max_lines:]
    if not tail:
        return
    print(f"      └─ Last {len(tail)} log lines from {log_path}:")
    for line in tail:
        print(f"         {line}")


def _estimate_validation_timeout(result_path, timeout_scale=1.0, timeout_cap=7200, mode="full"):
    mode = (mode or "full").lower()
    base = 420 if mode == "paper_screen" else 900
    try:
        with open(result_path) as f:
            data = json.load(f)
        trades = int(data.get('optim_metrics', {}).get('n_trades', 0) or 0)
        folds = int(data.get('n_cv_folds', 1) or 1)
        holdout_days = int(data.get('holdout_days', 180) or 180)
        if mode == "paper_screen":
            estimated = max(base, base + trades * 3 + folds * 45 + holdout_days)
        else:
            estimated = max(base, base + trades * 10 + folds * 120 + holdout_days * 2)
        scaled = int(estimated * max(0.5, timeout_scale))
        return int(min(max(base, scaled), timeout_cap))
    except Exception:
        fallback_seed = 900 if mode == "paper_screen" else 1800
        fallback_min = 420 if mode == "paper_screen" else 900
        fallback = int(fallback_seed * max(0.5, timeout_scale))
        return int(min(max(fallback_min, fallback), timeout_cap))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch parallel optimization + robustness validation (v11 — Walk-Forward CV)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Total trials per coin (default: 100, was 200)")
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1),
                        help="Total worker processes")
    parser.add_argument("--coins", type=str, default=",".join(COINS),
                        help="Comma-separated coin list")
    parser.add_argument("--plateau-patience", type=int, default=120,
                        help="Stop if no improvement for N completed trials after warmup")
    parser.add_argument("--plateau-min-delta", type=float, default=0.015)
    parser.add_argument("--plateau-warmup", type=int, default=60,
                        help="Warmup trials before plateau checks (default: 60)")
    parser.add_argument("--plateau-min-completed", type=int, default=0,
                        help="Never plateau-stop before this many completed trials (0 = auto 40%% of n_trials)")
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="paper_ready",
                        choices=["none", "paper_ready", "robust120", "robust180", "quick", "pilot_rollout"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0)
    parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=3,
                        help="Walk-forward CV folds (default: 3)")
    parser.add_argument("--holdout-candidates", type=int, default=3,
                        help="Top CV candidates to evaluate on holdout per coin")
    parser.add_argument("--require-holdout-pass", action="store_true",
                        help="Block coin deployment if no holdout candidate passes minimum thresholds")
    parser.add_argument("--holdout-min-trades", type=int, default=15)
    parser.add_argument("--holdout-min-sharpe", type=float, default=0.0)
    parser.add_argument("--holdout-min-return", type=float, default=0.0)
    parser.add_argument("--target-trades-per-week", type=float, default=1.0)
    parser.add_argument("--debug-trials", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--log-dir", type=str, default="")
    parser.add_argument("--sampler-seeds", type=str, default="42,1337,2024",
                        help="Comma-separated sampler seeds for consensus runs")
    parser.add_argument("--validation-jobs", type=int, default=3,
                        help="Max parallel validation processes")
    parser.add_argument("--validation-timeout-scale", type=float, default=1.0,
                        help="Multiplier applied to per-coin validation timeout estimates")
    parser.add_argument("--validation-timeout-cap", type=int, default=7200,
                        help="Maximum per-coin validation timeout in seconds")
    parser.add_argument("--validation-fast", action="store_true",
                        help="Run faster robustness checks (lower MC simulation counts)")
    parser.add_argument("--validation-no-timeout", action="store_true",
                        help="Disable per-coin validation timeout (run until completion)")
    parser.add_argument("--screen-threshold", type=float, default=60.0,
                        help="Run full validation only for coins with screen score >= this threshold")
    parser.add_argument("--skip-preflight", action="store_true",
                        help="Skip pre-launch data/feature QA checks")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Run preflight checks and exit without optimization/validation")
    args = parser.parse_args()
    args = apply_runtime_preset(args)

    pipeline_start = time.time()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')

    target_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    if not target_coins:
        raise SystemExit("No coins selected")

    script_dir = Path(__file__).resolve().parent
    trader_root = script_dir.parent

    optimize_path = script_dir / "optimize.py"
    validate_path = script_dir / "validate_robustness.py"

    for name, path in [("optimize.py", optimize_path)]:
        if not path.exists():
            print(f"❌ Cannot find {name} at {path}")
            raise SystemExit(1)

    if not args.skip_validation and not validate_path.exists():
        print(f"⚠️  validate_robustness.py not found at {validate_path}")
        args.skip_validation = True

    data_dir = trader_root / "data"
    db_path = data_dir / "trading.db"
    features_dir = data_dir / "features"

    if not db_path.exists():
        print(f"⚠️  WARNING: SQLite DB not found at {db_path}")
    if not features_dir.exists() or not list(features_dir.glob("*_features.csv")):
        print(f"⚠️  WARNING: No feature files in {features_dir}")

    if not args.skip_preflight or args.preflight_only:
        preflight_output = script_dir / "optimization_results" / f"preflight_{run_id}.json"
        preflight_cmd = [
            sys.executable, "-m", "scripts.preflight_check",
            "--coins", ",".join(target_coins),
            "--db-path", str(db_path),
            "--features-dir", str(features_dir),
            "--output", str(preflight_output),
        ]
        print(f"\n🧪 Running preflight QA: {' '.join(preflight_cmd)}")
        preflight_rc = subprocess.run(preflight_cmd, cwd=str(trader_root), check=False).returncode
        if preflight_rc != 0:
            print(f"⚠️  Preflight returned non-zero ({preflight_rc}). Review report: {preflight_output}")
        else:
            print(f"✅ Preflight complete: {preflight_output}")
        if args.preflight_only:
            raise SystemExit(preflight_rc)

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 1: OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════

    if not args.validate_only:
        n_coins = len(target_coins)
        base_workers = max(1, args.jobs // n_coins)
        remainder_workers = max(0, args.jobs - (base_workers * n_coins))

        worker_counts = {
            coin: base_workers + (1 if i < remainder_workers else 0)
            for i, coin in enumerate(target_coins)
        }
        total_workers = sum(worker_counts.values())

        print(f"{'='*70}")
        print(f"🚀 PHASE 1: OPTIMIZATION v11 ({total_workers} workers)")
        print(f"{'='*70}")
        print(f"   Coins:        {target_coins}")
        print(f"   Target/coin:  {args.trials} trials (was 200-350 in v10)")
        print(f"   Worker split: {worker_counts}")
        print(f"   CV folds:     {args.n_cv_folds} (walk-forward)")
        print(f"   Holdout:      {args.holdout_days} days")
        print(f"   Params:       10 tunable (includes cooldown cadence control)")
        print(f"   Scoring:      Mean OOS Sharpe across CV folds + holdout-guided candidate selection")
        print(f"   Min trades:   total>={args.min_total_trades or 'auto'}, oos>={args.min_internal_oos_trades or 'auto'}")
        print(f"   Holdout cands:{args.holdout_candidates}")
        print(f"   Holdout gate: trades>={args.holdout_min_trades}, SR>={args.holdout_min_sharpe}, Ret>={args.holdout_min_return}")
        print(f"   Enforce gate: {'YES' if args.require_holdout_pass else 'NO'}")
        print(f"   Target cadence: {args.target_trades_per_week:.2f} trades/week")
        print(f"   Preset:       {args.preset}")
        print(f"   Plateau gate: patience={args.plateau_patience}, min_delta={args.plateau_min_delta}, warmup={args.plateau_warmup}, min_completed={args.plateau_min_completed or 'auto(40%)'}")
        print(f"   Seeds:        {args.sampler_seeds}")
        seed_count = max(1, len([s for s in args.sampler_seeds.split(",") if s.strip()]))
        print(f"   Validation:   {'ENABLED' if not args.skip_validation else 'DISABLED'}")
        print(f"{'='*70}")

        print(f"\n   Run ID: {run_id}")

        optuna_db = script_dir / "optuna_trading.db"
        procs = {}
        log_dir = Path(args.log_dir) if args.log_dir else None
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

        for coin in target_coins:
            n_workers = worker_counts[coin]
            cmd = [
                sys.executable, "-m", "scripts.optimize",
                "--coin", coin,
                "--trials", str(args.trials),
                "--jobs", str(n_workers),
                "--plateau-patience", str(args.plateau_patience),
                "--plateau-min-delta", str(args.plateau_min_delta),
                "--plateau-warmup", str(args.plateau_warmup),
                "--plateau-min-completed", str(args.plateau_min_completed),
                "--holdout-days", str(args.holdout_days),
                "--n-cv-folds", str(args.n_cv_folds),
                "--holdout-candidates", str(args.holdout_candidates),
                "--holdout-min-trades", str(args.holdout_min_trades),
                "--holdout-min-sharpe", str(args.holdout_min_sharpe),
                "--holdout-min-return", str(args.holdout_min_return),
                "--target-trades-per-week", str(args.target_trades_per_week),
                "--study-suffix", run_id,
                "--preset", "none",  # already applied
            ]
            if args.min_internal_oos_trades:
                cmd.extend(["--min-internal-oos-trades", str(args.min_internal_oos_trades)])
            if args.require_holdout_pass:
                cmd.append("--require-holdout-pass")
            if args.min_total_trades:
                cmd.extend(["--min-total-trades", str(args.min_total_trades)])
            if args.sampler_seeds:
                cmd.extend(["--sampler-seeds", args.sampler_seeds])
            if args.debug_trials:
                cmd.append("--debug-trials")

            log_file = None
            stdout_target = None
            if log_dir:
                log_file = open(log_dir / f"{coin}_{run_id}.log", 'w')
                stdout_target = log_file

            print(f"   🚀 Launching {coin} ({n_workers} workers)", flush=True)
            proc = subprocess.Popen(
                cmd, cwd=str(trader_root),
                stdout=stdout_target, stderr=subprocess.STDOUT if log_file else None,
            )
            procs[coin] = {'proc': proc, 'log_file': log_file}

        # Monitor progress
        print(f"\n   ⏳ Monitoring progress...\n", flush=True)
        all_done = False
        start_time = time.time()

        while not all_done:
            time.sleep(30)
            elapsed = time.time() - start_time

            still_running = []
            for coin, info in procs.items():
                if info['proc'].poll() is None:
                    still_running.append(coin)

            progress = get_coin_trial_progress(optuna_db, target_coins, run_id)
            status_parts = []
            for coin in target_coins:
                p = progress[coin]
                started = p.get('started', 0)
                completed = p['completed']
                best = p.get('best_score')
                best_str = f" best={best:.3f}" if best is not None else ""
                metrics = p.get('best_metrics', {})
                # v11: show OOS Sharpe in progress
                oos_sr = metrics.get('mean_oos_sharpe', metrics.get('oos_sharpe'))
                oos_str = f" OOS_SR={oos_sr:.3f}" if oos_sr is not None else ""
                running = "🔄" if coin in still_running else "✅"
                progress_target = args.trials * seed_count
                progress_str = f"{started}/{progress_target}"
                if completed and completed != started:
                    progress_str += f" (✓{completed})"
                status_parts.append(f"{running}{coin}:{progress_str}{best_str}{oos_str}")

            print(f"   [{format_duration(elapsed)}] {' | '.join(status_parts)}", flush=True)

            if not still_running:
                all_done = True

        # Cleanup
        for coin, info in procs.items():
            if info['log_file']:
                info['log_file'].close()
            rc = info['proc'].returncode
            if rc != 0:
                print(f"   ⚠️  {coin} exited with code {rc}")
                if log_dir:
                    _print_log_tail(log_dir / f"{coin}_{run_id}.log")

        opt_time = time.time() - start_time
        print(f"\n   ✅ Phase 1 complete in {format_duration(opt_time)}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: VALIDATION
    # ═══════════════════════════════════════════════════════════════════

    if not args.skip_validation:
        print(f"\n{'='*70}")
        print(f"🔬 PHASE 2: ROBUSTNESS VALIDATION")
        print(f"{'='*70}")

        results_dir = script_dir / "optimization_results"
        coins_with_results = [c for c in target_coins
                              if (results_dir / f"{c}_optimization.json").exists()]

        if not coins_with_results:
            print("   No optimization results found. Skipping validation.")
        else:
            print(f"   Screening: {', '.join(coins_with_results)}")

            def _run_validation_mode(coins, mode):
                if not coins:
                    return
                max_parallel = max(1, min(args.validation_jobs, len(coins)))
                pending = list(coins)
                running = {}

                while pending or running:
                    while pending and len(running) < max_parallel:
                        coin = pending.pop(0)
                        print(f"\n   Running {mode} validation for {coin}...")
                        cmd = [
                            sys.executable, "-m", "scripts.validate_robustness",
                            "--coin", coin,
                            "--mode", mode,
                        ]
                        if args.validation_fast:
                            cmd.append("--fast")
                        result_path = results_dir / f"{coin}_optimization.json"
                        timeout_s = _estimate_validation_timeout(
                            result_path,
                            timeout_scale=args.validation_timeout_scale,
                            timeout_cap=args.validation_timeout_cap,
                            mode=mode,
                        )
                        proc = subprocess.Popen(
                            cmd, cwd=str(trader_root),
                            stdout=subprocess.PIPE if not sys.stderr.isatty() else None,
                            stderr=subprocess.PIPE if not sys.stderr.isatty() else None,
                            text=True,
                        )
                        running[coin] = {
                            'proc': proc,
                            'start': time.time(),
                            'timeout': (None if args.validation_no_timeout else timeout_s),
                        }

                    time.sleep(1)
                    finished = []
                    for coin, info in running.items():
                        proc = info['proc']
                        elapsed = time.time() - info['start']
                        if proc.poll() is not None:
                            finished.append(coin)
                            out, err = proc.communicate() if not sys.stderr.isatty() else (None, None)
                            if proc.returncode != 0:
                                print(f"   ❌ Validation failed for {coin} ({mode})")
                                if err:
                                    print(f"      {err[:500]}")
                        elif info['timeout'] is not None and elapsed > info['timeout']:
                            proc.kill()
                            finished.append(coin)
                            print(f"   ⏰ Validation timed out for {coin} ({mode}) after {int(info['timeout'])}s")

                    for coin in finished:
                        running.pop(coin, None)

            _run_validation_mode(coins_with_results, "paper_screen")

            screen_results = load_validation_results(results_dir)
            full_candidates = []
            for coin in coins_with_results:
                score = float(screen_results.get(coin, {}).get('readiness', {}).get('score', 0) or 0)
                if score >= args.screen_threshold:
                    full_candidates.append(coin)

            if full_candidates:
                print(f"\n   Full validation candidates (score >= {args.screen_threshold:.1f}): {', '.join(full_candidates)}")
                _run_validation_mode(full_candidates, "full")
            else:
                ranked_screen = sorted(
                    ((coin, float(screen_results.get(coin, {}).get('readiness', {}).get('score', 0) or 0))
                     for coin in coins_with_results),
                    key=lambda item: item[1],
                    reverse=True,
                )
                fallback_candidates = [coin for coin, score in ranked_screen if score >= max(35.0, args.screen_threshold - 20.0)][:2]
                if fallback_candidates:
                    ranked_str = ", ".join(f"{coin}:{score:.1f}" for coin, score in ranked_screen[:3])
                    print(f"\n   No coins met screen threshold {args.screen_threshold:.1f}; running fallback full validation for: {', '.join(fallback_candidates)}")
                    print(f"   Top screen scores: {ranked_str}")
                    _run_validation_mode(fallback_candidates, "full")
                else:
                    print(f"\n   No coins met screen threshold {args.screen_threshold:.1f}; skipping full validation.")



    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════

    total_time = time.time() - pipeline_start
    print_final_report(script_dir, target_coins, total_time)
    write_launch_summary(script_dir, args, target_coins, run_id, total_time)
