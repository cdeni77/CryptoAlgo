#!/usr/bin/env python3
"""
parallel_launch.py — v10: Full optimization + robustness validation pipeline.

Launches parallel Optuna workers per coin, then runs post-optimization
robustness validation to produce a paper-trade readiness score.

Changes from v9:
  - Integrated validate_robustness.py as post-optimization step
  - Added --skip-validation, --validate-only flags
  - Real-time progress tracking with ETA
  - Cross-coin summary report with go/no-go recommendations
  - Log capture per worker for debugging
  - New preset: robust180_full (includes validation)

Usage:
    python parallel_launch.py                                    # full pipeline
    python parallel_launch.py --validate-only                    # skip optim, just validate
    python parallel_launch.py --trials 300 --preset robust180
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

PRESET_CONFIGS = {
    "robust180": {
        "plateau_patience": 140,
        "plateau_min_delta": 0.015,
        "plateau_warmup": 80,
        "holdout_days": 180,
    },
    "robust120": {
        "plateau_patience": 120,
        "plateau_min_delta": 0.02,
        "plateau_warmup": 70,
        "holdout_days": 120,
    },
}


def apply_runtime_preset(args: argparse.Namespace) -> argparse.Namespace:
    """Apply launcher-side runtime preset values without mutating trial counts."""
    config = PRESET_CONFIGS.get(args.preset)
    if not config:
        return args

    for key, value in config.items():
        setattr(args, key, value)

    return args


def format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def _fmt_metric(value, fmt: str, fallback: str = "?") -> str:
    """Format a metric value safely."""
    try:
        return format(float(value), fmt)
    except (TypeError, ValueError):
        return fallback


def load_optimization_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all optimization result JSONs."""
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


def load_validation_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all validation result JSONs."""
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


def get_coin_trial_progress(optuna_db: Path, target_coins: List[str], run_id: str) -> Dict[str, Dict[str, object]]:
    """Fetch completed trial counts + current best metrics for each coin in this launcher run."""
    progress = {
        coin: {"completed": 0, "best_score": None, "best_metrics": {}}
        for coin in target_coins
    }
    if not optuna_db.exists():
        return progress

    completed_query = """
        SELECT s.study_id, COUNT(*)
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        WHERE t.state = 1
          AND s.study_name = ?
        GROUP BY s.study_id
    """

    best_trial_query = """
        SELECT t.trial_id, v.value
        FROM studies s
        JOIN trials t ON t.study_id = s.study_id
        JOIN trial_values v ON v.trial_id = t.trial_id
        WHERE s.study_name = ?
          AND t.state = 1
          AND v.objective = 0
        ORDER BY v.value DESC
        LIMIT 1
    """

    attrs_query = """
        SELECT key, value_json
        FROM trial_user_attributes
        WHERE trial_id = ?
          AND key IN ('sharpe', 'profit_factor', 'win_rate', 'max_drawdown', 'n_trades')
    """

    try:
        with sqlite3.connect(optuna_db) as conn:
            conn.execute("PRAGMA busy_timeout = 1000;")
            for coin in target_coins:
                study_name = f"optimize_{coin}_{run_id}"

                completed_row = conn.execute(completed_query, (study_name,)).fetchone()
                progress[coin]["completed"] = int(completed_row[1]) if completed_row else 0

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
    except sqlite3.Error:
        # Best-effort visibility only — keep the launcher running if the DB is busy.
        return progress

    return progress


def print_final_report(script_dir: Path, target_coins: List[str], total_time: float):
    """Print comprehensive summary combining optimization + validation results."""
    results_dir = script_dir / "optimization_results"
    opt_results = load_optimization_results(results_dir)
    val_results = load_validation_results(results_dir)

    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE PIPELINE REPORT")
    print(f"{'='*80}")
    print(f"   Total runtime: {format_duration(total_time)}")
    print(f"   Coins: {', '.join(target_coins)}")
    print()

    ready_coins = []
    cautious_coins = []
    reject_coins = []

    for coin in target_coins:
        opt = opt_results.get(coin, {})
        val = val_results.get(coin, {})

        optim_metrics = opt.get('optim_metrics', {})
        holdout_metrics = opt.get('holdout_metrics', {})
        readiness = val.get('readiness', {})

        # Optimization results
        opt_sharpe = optim_metrics.get('sharpe', '?')
        opt_pf = optim_metrics.get('profit_factor', '?')
        opt_wr = optim_metrics.get('win_rate', '?')
        opt_dd = optim_metrics.get('max_drawdown', '?')
        opt_trades = optim_metrics.get('n_trades', '?')
        n_trials = opt.get('n_trials', '?')

        # Holdout results
        ho_sharpe = holdout_metrics.get('holdout_sharpe', '?')
        ho_return = holdout_metrics.get('holdout_return', '?')
        ho_trades = holdout_metrics.get('holdout_trades', '?')

        # Validation results
        val_score = readiness.get('score', '—')
        val_rating = readiness.get('rating', 'NOT_RUN')

        dsr = val.get('deflated_sharpe', {})
        mc = val.get('mc_shuffle', {})
        sens = val.get('parameter_sensitivity', {})

        emoji = {
            'READY': '✅', 'CAUTIOUS': '⚠️', 'WEAK': '🟡',
            'REJECT': '❌', 'NOT_RUN': '⬜'
        }.get(val_rating, '?')

        print(f"  {emoji} {coin}")
        print(f"     ┌─ Optimization ({n_trials} trials)")

        # Format safely
        def _f(v, fmt='.3f'):
            try:
                return f"{float(v):{fmt}}"
            except (TypeError, ValueError):
                return str(v)

        print(f"     │  Sharpe={_f(opt_sharpe)} | PF={_f(opt_pf)} | "
              f"WR={_f(opt_wr, '.1%')} | DD={_f(opt_dd, '.1%')} | Trades={opt_trades}")
        print(f"     ├─ Holdout")
        print(f"     │  Sharpe={_f(ho_sharpe)} | Return={_f(ho_return, '.2%')} | Trades={ho_trades}")

        if val_rating != 'NOT_RUN':
            print(f"     ├─ Validation Score: {val_score}/100 — {val_rating}")
            if dsr.get('valid'):
                print(f"     │  DSR={_f(dsr.get('dsr', 0))} (p={_f(dsr.get('p_value', 1))})")
            if mc.get('valid'):
                print(f"     │  MC DD95={_f(mc.get('mc_dd_95th', 0), '.1%')} | "
                      f"P(ruin25%)={_f(mc.get('prob_ruin_25pct', 0), '.1%')}")
            if sens.get('valid'):
                print(f"     │  Param fragile={sens.get('fragile', '?')} | "
                      f"Avg SR drop={_f(sens.get('avg_sharpe_drop', 0))}")

            checks = readiness.get('details', [])
            if checks:
                failed = [c for c in checks if not c['passed']]
                if failed:
                    print(f"     └─ Failed checks: {', '.join(c['name'] for c in failed)}")
                else:
                    print(f"     └─ All {len(checks)} checks passed")
        else:
            print(f"     └─ Validation: not run")

        print()

        # Categorize
        if val_rating == 'READY':
            ready_coins.append(coin)
        elif val_rating == 'CAUTIOUS':
            cautious_coins.append(coin)
        else:
            reject_coins.append(coin)

    # Final recommendation
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


def _print_log_tail(log_path: Path, max_lines: int = 30) -> None:
    """Print a short tail from a worker log file for quick debugging."""
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
        description="Launch parallel optimization + robustness validation (v10)")
    parser.add_argument("--trials", type=int, default=200,
                        help="Total trials per coin")
    parser.add_argument("--jobs", type=int, default=(os.cpu_count() or 1),
                        help="Total worker processes (default: all CPU cores)")
    parser.add_argument("--coins", type=str, default=",".join(COINS),
                        help="Comma-separated coin list")
    parser.add_argument("--plateau-patience", type=int, default=100,
                        help="Stop if no best-score improvement for N trials")
    parser.add_argument("--plateau-min-delta", type=float, default=0.02,
                        help="Min score improvement to reset plateau")
    parser.add_argument("--plateau-warmup", type=int, default=60,
                        help="Warmup completed trials before plateau checks")
    parser.add_argument("--holdout-days", type=int, default=180,
                        help="Days reserved as true holdout (never seen by Optuna)")
    parser.add_argument("--preset", type=str, default="robust180",
                        choices=["none", "robust120", "robust180"],
                        help="Optimization preset (default: robust180)")
    parser.add_argument("--debug-trials", action="store_true",
                        help="Enable verbose per-trial output in optimize workers")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip robustness validation (optimization only)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Skip optimization, only run validation on existing results")
    parser.add_argument("--log-dir", type=str, default="",
                        help="Directory to capture per-worker logs (default: no capture)")
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
    # Validate paths
    for name, path in [("optimize.py", optimize_path)]:
        if not path.exists():
            print(f"❌ Cannot find {name} at {path}")
            raise SystemExit(1)

    if not args.skip_validation and not validate_path.exists():
        print(f"⚠️  validate_robustness.py not found at {validate_path}")
        print(f"   Validation will be skipped.")
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
        print(f"🚀 PHASE 1: OPTIMIZATION ({total_workers} workers)")
        print(f"{'='*70}")
        print(f"   Coins:        {target_coins}")
        print(f"   Target/coin:  {args.trials} trials")
        print(f"   Worker split: {worker_counts}")
        print(f"   Holdout:      {args.holdout_days} days (never seen by Optuna)")
        print(f"   Preset:       {args.preset}")
        print(f"   Validation:   {'ENABLED' if not args.skip_validation else 'DISABLED'}")
        print(f"{'='*70}")

        run_id = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')
        print(f"   Study run id: {run_id}")

        log_dir = Path(args.log_dir).expanduser() if args.log_dir else None
        if log_dir is None:
            log_dir = script_dir / "optimization_logs" / run_id
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"   Worker logs:  {log_dir}")

        optuna_db = script_dir / "optuna_trading.db"
        if optuna_db.exists():
            print(f"   ℹ️  Existing optuna DB: {optuna_db.stat().st_size / 1024:.0f} KB")

        processes = []
        failed_starts = []
        optim_start = time.time()

        for coin in target_coins:
            workers_for_coin = max(1, worker_counts[coin])
            base_trials = args.trials // workers_for_coin
            extra = args.trials % workers_for_coin
            trial_splits = [base_trials + (1 if i < extra else 0) for i in range(workers_for_coin)]

            for i, trial_count in enumerate(trial_splits):
                base_cmd = [
                    sys.executable, "-m", "scripts.optimize",
                    "--coin", coin,
                    "--jobs", "1",
                    "--trials", str(max(1, trial_count)),
                    "--plateau-patience", str(args.plateau_patience),
                    "--plateau-min-delta", str(args.plateau_min_delta),
                    "--plateau-warmup", str(args.plateau_warmup),
                    "--holdout-days", str(args.holdout_days),
                    "--preset", "none",
                    "--study-suffix", run_id,
                    "--resume-study",
                ]
                if args.debug_trials:
                    base_cmd.append("--debug-trials")

                print(f"   Starting {coin} worker #{i+1}/{workers_for_coin} ({trial_count} trials)...")

                env = os.environ.copy()
                env.setdefault("PYTHONUNBUFFERED", "1")
                existing_pythonpath = env.get("PYTHONPATH", "")
                pythonpath_parts = [str(trader_root)]
                if existing_pythonpath:
                    pythonpath_parts.append(existing_pythonpath)
                env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

                stdout_target = subprocess.DEVNULL
                stderr_target = subprocess.DEVNULL
                log_file = None
                log_file_path = log_dir / f"{coin}_worker_{i+1}.log"
                log_file = open(log_file_path, 'w')
                stdout_target = log_file
                stderr_target = subprocess.STDOUT

                try:
                    p = subprocess.Popen(
                        base_cmd, env=env, cwd=str(trader_root),
                        stdout=stdout_target, stderr=stderr_target,
                    )
                    processes.append((coin, i, p, log_file, log_file_path))
                except Exception as e:
                    print(f"   ❌ Failed to start {coin} worker #{i+1}: {e}")
                    failed_starts.append((coin, i, str(e)))
                    if log_file:
                        log_file.close()

                time.sleep(0.35)

        if failed_starts:
            print(f"\n⚠️  {len(failed_starts)} workers failed to start!")

        active = len(processes)
        print(f"\n✅ {active} workers started. Waiting for completion...")

        # Wait with progress reporting
        try:
            remaining = list(processes)
            last_report = time.time()
            completed_count = 0
            last_trial_report = {coin: 0 for coin in target_coins}
            trial_report_every = 10

            while remaining:
                still_running = []
                for coin, idx, p, lf, log_path in remaining:
                    ret = p.poll()
                    if ret is not None:
                        completed_count += 1
                        if ret == 0:
                            print(f"   ✅ {coin} worker #{idx+1} done "
                                  f"({completed_count}/{active}, "
                                  f"{format_duration(time.time() - optim_start)} elapsed)")
                        else:
                            print(f"   ❌ {coin} worker #{idx+1} failed (code {ret})")
                            if lf:
                                try:
                                    lf.flush()
                                    with open(lf.name, 'r', encoding='utf-8', errors='replace') as r:
                                        lines = r.readlines()[-12:]
                                    if lines:
                                        print("      ↳ log tail:")
                                        for line in lines:
                                            print(f"         {line.rstrip()}")
                                except OSError:
                                    pass
                        if lf:
                            lf.close()
                    else:
                        still_running.append((coin, idx, p, lf, log_path))
                remaining = still_running

                # Per-coin trial progress updates (every 10 trials by default)
                trial_progress = get_coin_trial_progress(optuna_db, target_coins, run_id)
                for coin in target_coins:
                    coin_progress = trial_progress.get(coin, {})
                    current = min(int(coin_progress.get('completed', 0) or 0), args.trials)
                    next_threshold = ((last_trial_report[coin] // trial_report_every) + 1) * trial_report_every
                    if current >= next_threshold or (current == args.trials and current > last_trial_report[coin]):
                        pct = (current / max(1, args.trials)) * 100
                        best_score = _fmt_metric(coin_progress.get('best_score'), '.3f')

                        best_metrics = coin_progress.get('best_metrics', {}) or {}
                        sharpe = _fmt_metric(best_metrics.get('sharpe'), '.3f')
                        profit_factor = _fmt_metric(best_metrics.get('profit_factor'), '.3f')
                        win_rate = _fmt_metric(best_metrics.get('win_rate'), '.1%')
                        drawdown = _fmt_metric(best_metrics.get('max_drawdown'), '.1%')
                        trades = best_metrics.get('n_trades', '?')

                        print(
                            f"   📈 {coin}: {current}/{args.trials} trials ({pct:.0f}%) | "
                            f"Best score={best_score} | Sharpe={sharpe} | PF={profit_factor} | "
                            f"WR={win_rate} | DD={drawdown} | Trades={trades}"
                        )
                        last_trial_report[coin] = current

                # Periodic progress
                now = time.time()
                if remaining and now - last_report > 60:
                    elapsed = now - optim_start
                    if completed_count > 0:
                        eta = elapsed / completed_count * (active - completed_count)
                        print(f"   ⏳ {completed_count}/{active} done | "
                              f"Elapsed: {format_duration(elapsed)} | "
                              f"ETA: ~{format_duration(eta)}")
                    last_report = now

                if remaining:
                    time.sleep(2.0)

        except KeyboardInterrupt:
            print("\n🛑 Stopping all workers...")
            for coin, idx, p, lf, log_path in processes:
                p.terminate()
            for coin, idx, p, lf, log_path in processes:
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()
                if lf:
                    lf.close()
            print("   Workers terminated.")

        optim_duration = time.time() - optim_start
        print(f"\n🏁 Optimization complete in {format_duration(optim_duration)}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 2: ROBUSTNESS VALIDATION
    # ═══════════════════════════════════════════════════════════════════

    if not args.skip_validation:
        print(f"\n{'='*70}")
        print(f"🔬 PHASE 2: ROBUSTNESS VALIDATION")
        print(f"{'='*70}")

        # Check which coins have optimization results
        results_dir = script_dir / "optimization_results"
        opt_results = load_optimization_results(results_dir)
        coins_with_results = [c for c in target_coins if c in opt_results]

        if not coins_with_results:
            print("   ⚠️  No optimization results found. Skipping validation.")
        else:
            print(f"   Validating: {', '.join(coins_with_results)}")

            validate_cmd = [
                sys.executable, str(validate_path),
                "--all" if len(coins_with_results) == len(target_coins) else "--coin",
            ]
            if len(coins_with_results) != len(target_coins):
                # Validate one at a time
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
            else:
                print(f"\n   Running validation for all coins...")
                cmd = [sys.executable, "-m", "scripts.validate_robustness", "--all"]
                try:
                    subprocess.run(cmd, cwd=str(trader_root), timeout=3600)
                except subprocess.TimeoutExpired:
                    print("   ⏰ Validation timed out")
                except Exception as e:
                    print(f"   ❌ Validation error: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # PHASE 3: FINAL REPORT
    # ═══════════════════════════════════════════════════════════════════

    total_time = time.time() - pipeline_start
    print_final_report(script_dir, target_coins, total_time)
