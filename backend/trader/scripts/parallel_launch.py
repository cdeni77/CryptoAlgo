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

# v11: Updated presets — tighter patience, CV folds
PRESET_CONFIGS = {
    "robust180": {
        "plateau_patience": 60,
        "plateau_min_delta": 0.02,
        "plateau_warmup": 30,
        "holdout_days": 180,
        "min_internal_oos_trades": 8,
        "min_total_trades": 30,
        "n_cv_folds": 3,
    },
    "robust120": {
        "plateau_patience": 50,
        "plateau_min_delta": 0.02,
        "plateau_warmup": 25,
        "holdout_days": 120,
        "min_internal_oos_trades": 6,
        "min_total_trades": 25,
        "n_cv_folds": 3,
    },
    "quick": {
        "plateau_patience": 30,
        "plateau_min_delta": 0.03,
        "plateau_warmup": 15,
        "holdout_days": 90,
        "min_internal_oos_trades": 5,
        "min_total_trades": 20,
        "n_cv_folds": 2,
    },
}


def apply_runtime_preset(args):
    config = PRESET_CONFIGS.get(args.preset)
    if not config:
        return args
    for key, value in config.items():
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
        coin: {"completed": 0, "best_score": None, "best_metrics": {}}
        for coin in target_coins
    }
    if not optuna_db.exists():
        return progress

    # Simple count query as primary method (most reliable under contention)
    count_query = """
        SELECT s.study_name, COUNT(*)
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        WHERE t.state = 1
        GROUP BY s.study_name
    """
    best_trial_query = """
        SELECT t.trial_id, v.value
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        JOIN trial_values v ON v.trial_id = t.trial_id
        WHERE s.study_name = ? AND t.state = 1 AND v.objective = 0
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

                # Get all study counts in one query
                rows = conn.execute(count_query).fetchall()
                for study_name, count in rows:
                    for coin in target_coins:
                        expected = f"optimize_{coin}_{run_id}"
                        if study_name == expected:
                            progress[coin]["completed"] = int(count)
                            break

                # Get best trial details per coin
                for coin in target_coins:
                    if progress[coin]["completed"] == 0:
                        continue
                    study_name = f"optimize_{coin}_{run_id}"
                    best_row = conn.execute(best_trial_query, (study_name,)).fetchone()
                    if not best_row:
                        continue
                    best_trial_id, best_score = best_row
                    progress[coin]["best_score"] = float(best_score) if best_score is not None else None

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

        ho_sharpe = holdout_metrics.get('holdout_sharpe', '?')
        ho_return = holdout_metrics.get('holdout_return', '?')
        ho_trades = holdout_metrics.get('holdout_trades', '?')

        val_score = readiness.get('score', '?')
        val_rating = readiness.get('rating', 'N/A')

        emoji = {'READY': '✅', 'CAUTIOUS': '⚠️', 'WEAK': '🟡', 'REJECT': '❌'}.get(val_rating, '⬜')

        print(f"  {emoji} {coin} [{version}]")
        print(f"     ┌─ CV Optimization")
        print(f"     │  Mean OOS SR={_f(mean_oos_sr)} | Min OOS SR={_f(min_oos_sr)} | "
              f"Std={_f(std_oos_sr)} | WR={_fmt_metric(opt_wr, '.1%')} | "
              f"DD={_fmt_metric(opt_dd, '.1%')} | Trades={opt_trades}")
        print(f"     ├─ Holdout")
        print(f"     │  Sharpe={_f(ho_sharpe)} | Return={_fmt_metric(ho_return, '.2%')} | Trades={ho_trades}")

        if readiness:
            sens = val.get('parameter_sensitivity', {})
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch parallel optimization + robustness validation (v11 — Walk-Forward CV)")
    parser.add_argument("--trials", type=int, default=100,
                        help="Total trials per coin (default: 100, was 200)")
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1),
                        help="Total worker processes")
    parser.add_argument("--coins", type=str, default=",".join(COINS),
                        help="Comma-separated coin list")
    parser.add_argument("--plateau-patience", type=int, default=60,
                        help="Stop if no improvement for N trials (default: 60, was 100)")
    parser.add_argument("--plateau-min-delta", type=float, default=0.02)
    parser.add_argument("--plateau-warmup", type=int, default=30,
                        help="Warmup trials before plateau checks (default: 30, was 60)")
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="robust180",
                        choices=["none", "robust120", "robust180", "quick"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0)
    parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=3,
                        help="Walk-forward CV folds (default: 3)")
    parser.add_argument("--debug-trials", action="store_true")
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--log-dir", type=str, default="")
    args = parser.parse_args()
    args = apply_runtime_preset(args)

    pipeline_start = time.time()

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
        print(f"   Params:       9 tunable (reduced from 18)")
        print(f"   Scoring:      Mean OOS Sharpe across CV folds")
        print(f"   Min trades:   total>={args.min_total_trades or 'auto'}, oos>={args.min_internal_oos_trades or 'auto'}")
        print(f"   Preset:       {args.preset}")
        print(f"   Validation:   {'ENABLED' if not args.skip_validation else 'DISABLED'}")
        print(f"{'='*70}")

        run_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
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
                "--holdout-days", str(args.holdout_days),
                "--n-cv-folds", str(args.n_cv_folds),
                "--study-suffix", run_id,
                "--preset", "none",  # already applied
            ]
            if args.min_internal_oos_trades:
                cmd.extend(["--min-internal-oos-trades", str(args.min_internal_oos_trades)])
            if args.min_total_trades:
                cmd.extend(["--min-total-trades", str(args.min_total_trades)])
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
                done = p['completed']
                best = p.get('best_score')
                best_str = f" best={best:.3f}" if best is not None else ""
                metrics = p.get('best_metrics', {})
                # v11: show OOS Sharpe in progress
                oos_sr = metrics.get('mean_oos_sharpe', metrics.get('oos_sharpe'))
                oos_str = f" OOS_SR={oos_sr:.3f}" if oos_sr is not None else ""
                running = "🔄" if coin in still_running else "✅"
                status_parts.append(f"{running}{coin}:{done}/{args.trials}{best_str}{oos_str}")

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
            print(f"   Validating: {', '.join(coins_with_results)}")
            for coin in coins_with_results:
                print(f"\n   Running validation for {coin}...")
                cmd = [sys.executable, "-m", "scripts.validate_robustness", "--coin", coin]
                try:
                    result = subprocess.run(
                        cmd, cwd=str(trader_root),
                        capture_output=not sys.stderr.isatty(),
                        text=True, timeout=1800,
                    )
                    if result.returncode != 0:
                        print(f"   ❌ Validation failed for {coin}")
                        if result.stderr:
                            print(f"      {result.stderr[:500]}")
                except subprocess.TimeoutExpired:
                    print(f"   ⏰ Validation timed out for {coin}")
                except Exception as e:
                    print(f"   ❌ Validation error for {coin}: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════

    total_time = time.time() - pipeline_start
    print_final_report(script_dir, target_coins, total_time)