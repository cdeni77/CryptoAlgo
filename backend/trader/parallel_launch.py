import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


# Usage:
#   python parallel_launch.py --trials 200 --jobs 16
#   python parallel_launch.py --trials 250 --jobs 16 --holdout-days 90
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch parallel Optuna optimization workers")
    parser.add_argument("--trials", type=int, default=200, help="Total trials per coin")
    parser.add_argument("--jobs", type=int, default=16, help="Total worker processes")
    parser.add_argument("--coins", type=str, default=",".join(COINS), help="Comma-separated coin list")
    parser.add_argument("--plateau-patience", type=int, default=120, help="Stop if no best-score improvement for N trials")
    parser.add_argument("--plateau-min-delta", type=float, default=0.01, help="Min score improvement to reset plateau")
    parser.add_argument("--plateau-warmup", type=int, default=80, help="Warmup completed trials before plateau checks")
    parser.add_argument("--holdout-days", type=int, default=90, help="Days reserved as true holdout (never seen by Optuna)")
    parser.add_argument("--debug-trials", action="store_true", help="Enable verbose per-trial output in optimize workers")
    args = parser.parse_args()

    target_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    if not target_coins:
        raise SystemExit("No coins selected")

    # =========================================================================
    # RESOLVE WORKING DIRECTORY
    #
    # Critical for Docker: When the API container spawns this script via
    # ops_service._spawn(cwd=/trader), this script runs with cwd=/trader.
    # Child optimize.py processes MUST also run in the same directory so they
    # can find train_model.py, coin_profiles.py, data/, etc.
    #
    # We resolve the script's own directory as the canonical working dir,
    # which works both locally (script is in backend/trader/) and in Docker
    # (script is at /trader/parallel_launch.py).
    # =========================================================================
    script_dir = Path(__file__).resolve().parent

    # Verify critical files exist in the working directory
    optimize_path = script_dir / "optimize.py"
    train_model_path = script_dir / "train_model.py"

    if not optimize_path.exists():
        print(f"❌ Cannot find optimize.py at {optimize_path}")
        print(f"   Script dir: {script_dir}")
        print(f"   CWD: {os.getcwd()}")
        print(f"   Contents: {list(script_dir.glob('*.py'))[:10]}")
        raise SystemExit(1)

    if not train_model_path.exists():
        print(f"❌ Cannot find train_model.py at {train_model_path}")
        raise SystemExit(1)

    # Check for data
    data_dir = script_dir / "data"
    db_path = data_dir / "trading.db"
    features_dir = data_dir / "features"

    if not db_path.exists():
        print(f"⚠️  WARNING: SQLite DB not found at {db_path}")
        print(f"   Workers will fail to load data. Run the pipeline first.")

    if not features_dir.exists() or not list(features_dir.glob("*_features.csv")):
        print(f"⚠️  WARNING: No feature files in {features_dir}")
        print(f"   Workers will fail to load data. Run compute_features.py first.")

    # Split the requested workers across selected coins.
    n_coins = len(target_coins)
    base_workers = max(1, args.jobs // n_coins)
    remainder_workers = max(0, args.jobs - (base_workers * n_coins))

    worker_counts = {
        coin: base_workers + (1 if i < remainder_workers else 0)
        for i, coin in enumerate(target_coins)
    }

    total_workers = sum(worker_counts.values())

    print(f"🚀 LAUNCHING {total_workers} WORKERS (v9 — True Holdout)")
    print(f"   Coins:        {target_coins}")
    print(f"   Target/coin:  {args.trials} trials")
    print(f"   Worker split: {worker_counts}")
    print(f"   Holdout:      {args.holdout_days} days (never seen by Optuna)")
    print(f"   Script dir:   {script_dir}")
    print(f"   Python:       {sys.executable}")
    print(f"   CWD:          {os.getcwd()}")
    print("   Note: each optimize.py worker runs with --jobs 1 (single process).")
    print("         Parallelism comes from launching many worker processes.")
    print(f"{'='*60}")

    run_id = datetime.utcnow().strftime('%Y%m%d%H%M%S%f')
    print(f"   Study run id: {run_id}")

    # Report optuna DB state
    optuna_db = script_dir / "optuna_trading.db"
    if optuna_db.exists():
        print(f"   ℹ️  Existing optuna DB found at {optuna_db} ({optuna_db.stat().st_size / 1024:.0f} KB)")
        print(f"      Fresh study suffix ensures no collision with old studies.")

    processes = []
    failed_starts = []

    # Launch per-coin workers. Total running processes ~= requested --jobs.
    for coin in target_coins:
        workers_for_coin = max(1, worker_counts[coin])

        # Split trials across workers as evenly as possible per coin.
        base_trials = args.trials // workers_for_coin
        extra = args.trials % workers_for_coin
        trial_splits = [base_trials + (1 if i < extra else 0) for i in range(workers_for_coin)]

        for i, trial_count in enumerate(trial_splits):
            base_cmd = [
                sys.executable,
                str(optimize_path),  # Use absolute path to optimize.py
                "--coin",
                coin,
                "--jobs",
                "1",
                "--trials",
                str(max(1, trial_count)),
                "--plateau-patience",
                str(args.plateau_patience),
                "--plateau-min-delta",
                str(args.plateau_min_delta),
                "--plateau-warmup",
                str(args.plateau_warmup),
                "--holdout-days",
                str(args.holdout_days),
                "--study-suffix",
                run_id,
                "--resume-study",
            ]

            if args.debug_trials:
                base_cmd.append("--debug-trials")

            print(
                f"   Starting {coin} worker #{i + 1}/{workers_for_coin} "
                f"({trial_count} trials, process-level parallelism)..."
            )
            env = os.environ.copy()
            env.setdefault("PYTHONUNBUFFERED", "1")

            try:
                p = subprocess.Popen(
                    base_cmd,
                    env=env,
                    cwd=str(script_dir),  # CRITICAL: ensure children run in trader dir
                )
                processes.append((coin, i, p))
            except Exception as e:
                print(f"   ❌ Failed to start {coin} worker #{i+1}: {e}")
                failed_starts.append((coin, i, str(e)))

            # Stagger starts to reduce initial DB lock contention
            time.sleep(0.35)

    if failed_starts:
        print(f"\n⚠️  {len(failed_starts)} workers failed to start!")
        for coin, idx, err in failed_starts:
            print(f"   {coin} #{idx+1}: {err}")

    active = len(processes)
    print(f"\n✅ {active} workers started. Monitor CPU usage now!")
    print("   Press Ctrl+C to stop all workers.\n")

    # Wait for all workers, reporting exits as they happen
    try:
        remaining = list(processes)
        while remaining:
            still_running = []
            for coin, idx, p in remaining:
                ret = p.poll()
                if ret is not None:
                    if ret == 0:
                        print(f"   ✅ {coin} worker #{idx+1} finished successfully.")
                    else:
                        print(f"   ❌ {coin} worker #{idx+1} exited with code {ret}")
                else:
                    still_running.append((coin, idx, p))
            remaining = still_running
            if remaining:
                time.sleep(2.0)
    except KeyboardInterrupt:
        print("\n🛑 Stopping all workers...")
        for coin, idx, p in processes:
            p.terminate()
        # Give them a moment to clean up
        for coin, idx, p in processes:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

    print(f"\n{'='*60}")
    print(f"🏁 All workers finished.")