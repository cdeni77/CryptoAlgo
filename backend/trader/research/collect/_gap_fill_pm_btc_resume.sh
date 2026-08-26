#!/usr/bin/env bash
# BTC's Polymarket discovery walk has now stopped twice on a bare Predexon 500
# past its retry budget (page 909, then page 844) — 84 and then a further
# chunk short of the 2026-03-02 coverage boundary each time. The walk is
# resumable and skips known pages by slug, so this just resumes it rather
# than treating a Predexon-side outage as done.
#
# Queued behind the ETH/SOL backfill for the same reason every other gap-fill
# script here is queued rather than run inline: editing a script while an
# instance of it is executing can corrupt its in-flight read (observed once
# already tonight — do not repeat it), and Predexon's bucket is 1 req/s
# ORG-WIDE, so nothing else may run against it at the same time regardless.
set -u
cd "$(dirname "$0")/../.."
ROOT=../..
set -a; . $ROOT/.env; set +a
export RESEARCH_STORE=data/research
LOG=/tmp/claude-1000/-home-cdeni-Desktop-Personal-CryptoAlgo-CryptoAlgo-backend-trader/eaf9d71c-b3f5-46b6-aedf-bc3e220c6668/scratchpad
say() { echo "[$(date -u +%FT%H:%M:%SZ)] $*"; }

depth() {
  say "  build_depth"
  (cd $ROOT && docker compose run --rm -T trader -m scripts.build_depth) \
    >> "$LOG/depth.log" 2>&1
  say "  $(tail -4 "$LOG/depth.log" | tr '\n' ' ' | cut -c1-150)"
}

wait_for() { while pgrep -f "$1" > /dev/null; do sleep 30; done; }

say "waiting for the ETH/SOL backfill and its children to clear the Predexon bucket"
wait_for "_gap_fill_pm_eth_sol.sh"
wait_for "_collect_pm.py"

say "clear — resuming BTC discovery toward the 2026-03-02 boundary"
PM_ASSET="btc-" PM_STAGE=discover PM_PAGES=4000 \
  python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_btc_resume_discover.log" 2>&1
say "         $(grep -c '"market_slug": "btc-' data/pm_markets.jsonl 2>/dev/null || echo 0) btc markets known now"

say "pricing whatever this resume found"
PM_ASSET="btc-" PM_STAGE=price PM_PRICED=25000 \
  python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_btc_resume_price.log" 2>&1
say "         done"

depth
say "BTC RESUME DONE"
