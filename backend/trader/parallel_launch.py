import argparse
import subprocess
import sys
import time

COINS = ["BTC", "ETH", "SOL", "XRP", "DOGE"]


# Usage:
#   python parallel_launch.py --trials 200 --jobs 10
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="Total trials per coin")
    parser.add_argument("--jobs", type=int, default=10, help="Total worker processes")
    parser.add_argument("--coins", type=str, default=",".join(COINS), help="Comma-separated coin list")
    args = parser.parse_args()

    target_coins = [c.strip().upper() for c in args.coins.split(",") if c.strip()]
    if not target_coins:
        raise SystemExit("No coins selected")

    workers_per_coin = max(1, args.jobs // len(target_coins))
    trials_per_worker = max(1, args.trials // workers_per_coin)

    print(f"🚀 LAUNCHING {workers_per_coin * len(target_coins)} WORKERS")
    print(f"   Coins:        {target_coins}")
    print(f"   Target/coin:  {args.trials} trials")
    print(f"   Workers/coin: {workers_per_coin}")
    print(f"   Trials/worker:{trials_per_worker}")
    print(f"{'='*60}")

    processes = []

    # Launch per-coin workers to reduce study collisions and improve throughput.
    for coin in target_coins:
        base_cmd = [
            sys.executable,
            "optimize.py",
            "--coin",
            coin,
            "--jobs",
            "1",
            "--trials",
            str(trials_per_worker),
        ]

        for i in range(workers_per_coin):
            print(f"   Starting {coin} worker #{i + 1}...")
            p = subprocess.Popen(base_cmd)
            processes.append(p)
            # Stagger starts to reduce initial DB lock contention
            time.sleep(0.4)

    print(f"\n✅ All {len(processes)} workers started. Monitor CPU usage now!")
    print("   Press Ctrl+C to stop all workers.\n")

    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all workers...")
        for p in processes:
            p.terminate()
