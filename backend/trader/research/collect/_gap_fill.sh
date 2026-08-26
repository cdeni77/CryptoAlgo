#!/usr/bin/env bash
# Fill in what tonight's runs missed, once they are actually done with the
# Predexon bucket. Queued rather than run inline because two things can leave
# a real gap and neither is worth a special case in _overnight.sh:
#
#   * a transient error truncates a walk before it reaches its real boundary
#     (tonight: a bare 500 stopped Polymarket discovery at page 909, ~84 days
#     short of the 2026-03-02 coverage boundary — get() now retries a 5xx the
#     same way it retries a 429, so this specific failure should not recur,
#     but the walk still needs to be re-run to recover the days already lost)
#   * a page of Kalshi settlements landed after the book backfill's window
#     list was built, so a handful of in-coverage candidates were never
#     attempted at all (measured: 1,024 of them, mostly one bad day)
#
# Both collectors are resumable and skip what is already on disk, so re-running
# them costs only the gap, not the whole history again.
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

# Let tonight's chain and its children finish before touching the shared
# 1 req/s Predexon bucket — competing with them just makes both slower.
say "waiting for the running chain to finish with Predexon"
while pgrep -f "_overnight.sh" > /dev/null \
   || pgrep -f "_collect_book.py" > /dev/null \
   || pgrep -f "_collect_pm.py" > /dev/null \
   || pgrep -f "collect_settlements" > /dev/null; do
  sleep 60
done
say "clear — starting the gap fill"

say "GAP 1/3  Kalshi settlements — incremental catch-up"
python -u -m scripts.collect_settlements --venue both >> "$LOG/gapfill_settle.log" 2>&1
say "         done"

say "GAP 2/3  Kalshi book — the 1,024 in-coverage windows tonight's list missed"
BOOK_BUDGET=5000 BOOK_WINDOWS=25000 \
  python -u research/collect/_collect_book.py >> "$LOG/gapfill_book.log" 2>&1
say "         $(wc -l < data/book_full.jsonl 2>/dev/null || echo 0) windows on disk now"
depth

say "GAP 3/3  Polymarket discovery — resume past page 909 to the 2026-03-02 boundary"
say "         (re-walks known pages first; they are skipped by slug, not re-fetched as new)"
PM_STAGE=discover PM_PAGES=6000 \
  python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_discover.log" 2>&1
say "         $(wc -l < data/pm_markets.jsonl 2>/dev/null || echo 0) markets known now"

say "         pricing whatever GAP 3 found that GAP 3's own stage 2 has not primed yet"
PM_STAGE=price PM_PRICED=30000 \
  python -u research/collect/_collect_pm.py >> "$LOG/gapfill_pm_price.log" 2>&1
say "         $(wc -l < data/pm_prices.jsonl 2>/dev/null || echo 0) books on disk now"
depth

say "GAP FILL DONE"
