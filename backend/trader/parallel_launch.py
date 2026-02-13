import argparse
import subprocess
import sys
import time
from datetime import datetime

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


# Usage:
#   python parallel_launch.py --trials 200 --jobs 16
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="Total trials per coin")
    parser.add_argument("--jobs", type=int, default=16, help="Total worker processes")
    parser.add_argument("--coins", type=str, default=",".join(COINS), help="Comma-separated coin list")
    parser.add_argument("--plateau-patience", type=int, default=120, help="Stop if no best-score improvement for N trials")
    parser.add_argument("--plateau-min-delta", type=float, default=0.01, help="Min score improvement to reset plateau")
    parser.add_argument("--plateau-warmup", type=int, default=80, help="Warmup completed trials before plateau checks")
    args = parser.parse_args()

    target_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    if not target_coins:
        raise SystemExit("No coins selected")

    # Distribute workers as evenly as possible while using all requested cores.
    n_coins = len(target_coins)
    base_workers = max(1, args.jobs // n_coins)
    remainder_workers = max(0, args.jobs - (base_workers * n_coins))

    worker_counts = {
        coin: base_workers + (1 if i < remainder_workers else 0)
        for i, coin in enumerate(target_coins)
    }

    total_workers = sum(worker_counts.values())

    print(f"🚀 LAUNCHING {total_workers} WORKERS")
    print(f"   Coins:        {target_coins}")
    print(f"   Target/coin:  {args.trials} trials")
    print(f"   Worker split: {worker_counts}")
    print(f"{'='*60}")

    run_id = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    print(f"   Study run id: {run_id}")

    processes = []

    # Launch one process per coin and let Optuna handle intra-coin parallelism via
    # --jobs. This keeps logs readable, avoids repeatedly loading the same data,
    # and makes the reported "Cores" value match the configured split.
    for coin in target_coins:
        workers_for_coin = max(1, worker_counts[coin])
        base_cmd = [
            sys.executable,
            "optimize.py",
            "--coin",
            coin,
            "--jobs",
            str(workers_for_coin),
            "--trials",
            str(max(1, args.trials)),
            "--plateau-patience",
            str(args.plateau_patience),
            "--plateau-min-delta",
            str(args.plateau_min_delta),
            "--plateau-warmup",
            str(args.plateau_warmup),
            "--study-suffix",
            run_id,
        ]

        print(f"   Starting {coin} ({args.trials} trials, {workers_for_coin} cores)...")
        p = subprocess.Popen(base_cmd)
        processes.append(p)
        # Stagger starts to reduce initial DB lock contention
        time.sleep(0.35)

    print(f"\n✅ All {len(processes)} workers started. Monitor CPU usage now!")
    print("   Press Ctrl+C to stop all workers.\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all workers...")
        for p in processes:
            p.terminate()
