import subprocess
import sys
import time
import argparse

# Usage: python parallel_launch.py --trials 200 --jobs 16
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=200, help="Total trials per coin")
    parser.add_argument("--jobs", type=int, default=16, help="Number of parallel workers")
    args = parser.parse_args()

    # Calculate how many trials each worker should do
    trials_per_worker = max(1, args.trials // args.jobs)
    
    print(f"🚀 LAUNCHING {args.jobs} WORKERS")
    print(f"   Total Target: {args.trials} trials")
    print(f"   Per Worker:   {trials_per_worker} trials")
    print(f"{'='*60}")

    processes = []
    
    # We pass '--all' so each worker iterates through all coins
    # We pass '--jobs 1' so the worker itself is single-threaded (pure isolation)
    base_cmd = [sys.executable, "optimize.py", "--all", "--jobs", "1", "--trials", str(trials_per_worker)]

    for i in range(args.jobs):
        print(f"   Starting Worker #{i+1}...")
        p = subprocess.Popen(base_cmd)
        processes.append(p)
        # Stagger starts slightly to prevent initial DB clash
        time.sleep(0.5)

    print(f"\n✅ All {len(processes)} workers started. Monitor CPU usage now!")
    print("   Press Ctrl+C to stop all workers.\n")

    try:
        # Wait for all to finish
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping all workers...")
        for p in processes:
            p.terminate()