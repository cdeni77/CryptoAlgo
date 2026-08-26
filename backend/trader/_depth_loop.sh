#!/usr/bin/env bash
# Keep the unified table fresh while the collectors run, so the morning state is
# current whatever the chain got through. Cheap and idempotent: the store merges
# partitions, so a rebuild only adds.
set -u
cd "$(dirname "$0")"
LOG=/tmp/claude-1000/-home-cdeni-Desktop-Personal-CryptoAlgo-CryptoAlgo-backend-trader/eaf9d71c-b3f5-46b6-aedf-bc3e220c6668/scratchpad
while true; do
  sleep 1800
  (cd ../.. && docker compose run --rm -T trader -m scripts.build_depth) \
    >> "$LOG/depth.log" 2>&1
  echo "[$(date -u +%FT%H:%M:%SZ)] rebuilt" >> "$LOG/depth.log"
done
