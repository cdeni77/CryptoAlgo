#!/usr/bin/env bash
# ARM A2 vs ARM B2 -- the Predexon question, with ONE thing different.
#
# WHY THIS SUPERSEDES ops/retrain_arms.sh
#   `--quote-sources` is matched on the bare source name and applied to BOTH
#   venues. Polymarket's observers are also called `backfill`, `live` and
#   `live_touch`, so naming Kalshi's live sources threw away Polymarket's
#   gamma-API backfill -- 77,759 windows, a different pipeline with none of the
#   deci-cent quantisation the restriction exists to avoid. Under
#   `--complete-cases` every row carrying a `cross_venue` feature went with it:
#   6,465 candidate windows became 956, every fold fell under the 200-window
#   minimum, and the run raised `no fold had enough training windows`.
#
#   The first repair dropped `cross_venue` from both arms. That answered a
#   narrower question and threw away a real feature group to dodge a filter
#   limitation. `--peer-quote-sources` (new) gives the peer its own list, the
#   same shape as the `peer_max_age_seconds` and `peer_null_age` knobs that
#   already exist for the same reason: the peer is a cross-venue FEATURE, never
#   the book we execute against.
#
# SO THE ARMS NOW DIFFER IN EXACTLY ONE THING: whether Predexon may price a
# KALSHI row. Polymarket is identical in both.
#
# NEITHER ARM PROMOTES. This runs `scripts.evaluate`; promotion is a separate,
# deliberate step, and its ledger IS the trial count.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-research/overnight/predexon_arms2}"
cd "$REPO" || exit 2

COMMON=(
  --groups cross_venue,geometry,vol_state,microstructure,cross_asset,offset,market_state,implied_vol,book_flow
  --start 2025-10-01
  --complete-cases
  --entry-offsets 3 6 9 12
  --init-score-source market
  --kelly-fraction 0.10
  --compound
  --min-edge-pp 0.5
  --bankroll 500
  --exclude-price-band 0.35,0.65
  --max-entries-per-symbol-window 4
  --max-positions-per-window 12
  --maker-entry
  --max-stake-dollars 40
  --ensemble-size 5
  --label-source venue
  --edge-lower-bound-z 1.0
  --require-member-agreement
)

run_arm() {
  local name="$1"; shift
  echo "=== ARM ${name} ==="
  docker compose exec -T -w /app trader \
    python -m scripts.evaluate "${COMMON[@]}" "$@" --out "$OUT/${name}"
  echo "exit ${?}"
}

run_arm with_predexon
run_arm kalshi_only \
  --quote-sources live_touch,live_ws,live,live_ws_replay,candles \
  --peer-quote-sources all
