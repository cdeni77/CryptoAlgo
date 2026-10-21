#!/usr/bin/env bash
#
# Two walk-forwards differing in ONE thing: whether the Predexon backfill is
# allowed to price a training row.
#
#   ops/retrain_arms.sh
#
# WHY A COMPARISON AND NOT AN ARGUMENT
#   The repo's own measurements say the backfill quote is unbiased but a far
#   worse forecast: +0.0425 nats against the live mid, which is ~54x the
#   deployed model's entire claimed log_loss_skill (+0.00079). Whole-run P&L
#   splits +$3,493.94 (t=+3.95) on backfilled quotes against -$55.03 (t=-0.12)
#   on live-recorded ones, and the 0-20c band reverses sign.
#   `project-edge-is-vs-backfill-only` records the model beating backfilled
#   quotes at t=4.8 and the live book at t=0.92, negative per fold.
#
#   Measured 2026-10-21, the mechanism is QUANTISATION. Kalshi's tick is
#   `tapered_deci_cent` -- a tenth of a cent outside [0.10, 0.90] -- and
#   Predexon rounds every tail quote to a whole cent: 100% of backfill quotes
#   sit on a whole cent in every band, against ~10% of live ones in the tails.
#   That is 27% of rows, and a 0.5c error on a 4c quote is a 12.5% relative
#   error in the probability, in exactly the confident region log loss is most
#   sensitive to.
#
# THE COST OF ARM B, STATED UP FRONT
#   It drops roughly 57% of training windows (15,138 -> ~6,500), because
#   Kalshi's own candlesticks only begin 2026-08-13 and the live tier begins
#   2026-08-25. A prior live-only attempt measured `log_loss_skill =
#   -0.000118, p = 0.80` -- no skill at all -- on 1,344 windows. This arm has
#   about five times that sample, which is the one variable that has moved.
#
# THE UNCOMFORTABLE OUTCOME IS THE INFORMATIVE ONE
#   If ARM B has no skill, the deployed edge is a backfill artifact.
#
# NEITHER ARM PROMOTES
#   This runs `scripts.evaluate`, which writes scored rows, trades and a
#   report. Promotion stays a separate, deliberate step with its own ledger --
#   and that ledger IS the trial count, so a comparison must not spend it.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${OUT:-research/overnight/predexon_arms}"
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
  # `exec` into the RUNNING trader rather than `run --rm retrain`: the retrain
  # profile's command is `scripts.promote`, which installs. An evaluation must
  # not be able to install.
  docker compose exec -T -w /app trader \
    python -m scripts.evaluate "${COMMON[@]}" "$@" --out "$OUT/${name}"
  echo "exit ${?}"
}

run_arm with_predexon
run_arm kalshi_only \
  --quote-sources live_touch,live_ws,live,live_ws_replay,candles

cat <<'NOTE'

=== how to read it ===
Compare `model_minus_market` and `log_loss_skill` between the two reports, and
read the TAILS (<10c and >90c) separately from the middle band. If the
quantisation story is right the difference should CONCENTRATE in the tails --
that is where Predexon rounds and the live grid does not. A uniform improvement
means something else changed and should be distrusted.
NOTE
