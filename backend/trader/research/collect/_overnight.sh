#!/usr/bin/env bash
# Overnight collection, sequential by necessity.
#
# Predexon enforces 1 req/s on an ORG-WIDE bucket, so these cannot overlap —
# two collectors just make each other retry. Everything is resumable: an
# interrupted job continues from what is already on disk rather than restarting.
#
# Order is by value to the trading system, because the chain may not finish:
#   1. settlements   the venue's own label, ~196 days, cheap. Tests the
#                    Coinbase-vs-BRTI proxy that every target here rests on.
#   2. kalshi book   the training data for every book-derived feature.
#   3. polymarket    the replication set. Most expensive, least urgent.
# `build_depth` runs between jobs so the unified table is fresh by morning even
# if the chain is cut short.
set -u
# This script lives in research/collect/ but everything it touches — data/,
# RESEARCH_STORE, the docker-compose context — is relative to backend/trader/,
# so it cds back there rather than dragging every downstream path two levels
# deeper. `_collect_book.py`/`_collect_pm.py` themselves stayed put in
# behaviour: they resolve their own `data/...` paths against the process cwd,
# which this keeps identical to before the move.
cd "$(dirname "$0")/../.."
ROOT=../..
set -a; . $ROOT/.env; set +a
export RESEARCH_STORE=data/research
LOG=/tmp/claude-1000/-home-cdeni-Desktop-Personal-CryptoAlgo-CryptoAlgo-backend-trader/eaf9d71c-b3f5-46b6-aedf-bc3e220c6668/scratchpad
say() { echo "[$(date -u +%FT%H:%M:%SZ)] $*"; }

# The research store is written by containers and is root-owned, so anything
# touching it runs in one too. The JSONL archives are host-owned and do not.
depth() {
  say "  build_depth"
  (cd $ROOT && docker compose run --rm -T trader -m scripts.build_depth) \
    >> "$LOG/depth.log" 2>&1
  say "  $(tail -4 "$LOG/depth.log" | tr '\n' ' ' | cut -c1-150)"
}

wait_for() {  # let an already-running collector finish before competing for the bucket
  while pgrep -f "$1" > /dev/null; do sleep 30; done
}

say "waiting for any running settlement collection"
wait_for "collect_settlements"

say "JOB 1/3  Kalshi settlements — the venue's own result, all three series"
python -u -m scripts.collect_settlements --venue both >> "$LOG/settle.log" 2>&1
say "         done"

say "JOB 2/3  Kalshi order book — full tick series, every minute indexed"
BOOK_BUDGET=90000 BOOK_WINDOWS=25000 python -u research/collect/_collect_book.py >> "$LOG/book.log" 2>&1
say "         $(wc -l < data/book_full.jsonl 2>/dev/null || echo 0) windows"
depth

say "JOB 3/3  Polymarket — discover to 2026-03-02, then the full book"
PM_STAGE=both PM_PAGES=4000 PM_PRICED=25000 python -u research/collect/_collect_pm.py >> "$LOG/pm.log" 2>&1
say "         $(wc -l < data/pm_markets.jsonl 2>/dev/null || echo 0) markets, $(wc -l < data/pm_prices.jsonl 2>/dev/null || echo 0) books"
depth

say "ALL DONE"
