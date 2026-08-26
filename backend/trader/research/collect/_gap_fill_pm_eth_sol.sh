#!/usr/bin/env bash
# Polymarket backfill was BTC-only (`_collect_pm.py`'s ASSET_PREFIX defaults
# to 'btc-'), while the live recorder already covers all three assets. The
# whole point of collecting Polymarket is a like-for-like comparison against
# Kalshi's three symbols, so a BTC-only backfill silently answers a narrower
# question than the one being asked. This widens the historical backfill to
# match: same walk, same coverage boundary, just PM_ASSET set to eth-/sol-.
#
# Queued rather than run inline in _gap_fill.sh for the same reason that
# script is separate from _overnight.sh: editing a script file while an
# instance of it is executing can corrupt its in-flight read (observed
# tonight — do not repeat it). Waits for _gap_fill.sh to clear the shared
# 1 req/s Predexon bucket before starting.
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

say "waiting for _gap_fill.sh and its children to clear the Predexon bucket"
wait_for "_gap_fill.sh"
wait_for "_collect_pm.py"
wait_for "_collect_book.py"
wait_for "collect_settlements"

say "clear — starting Polymarket ETH/SOL backfill"

for asset in eth sol; do
  say "ASSET=$asset  discovery to 2026-03-02"
  PM_ASSET="${asset}-" PM_STAGE=discover PM_PAGES=4000 \
    python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_${asset}_discover.log" 2>&1
  say "         $(grep -c "\"market_slug\": \"${asset}-" data/pm_markets.jsonl 2>/dev/null || echo 0) ${asset} markets known now"

  say "ASSET=$asset  pricing"
  PM_ASSET="${asset}-" PM_STAGE=price PM_PRICED=25000 \
    python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_${asset}_price.log" 2>&1
  say "         done with $asset"
done

depth
say "PM ETH/SOL BACKFILL DONE"
