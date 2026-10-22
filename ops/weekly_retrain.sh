#!/usr/bin/env bash
#
# Weekly FORCED retrain for the live trader. Installed via cron, and the time
# here is the one actually installed -- this comment said 0 9 while the crontab
# said 0 5, checked 2026-10-21:
#
#   0 5 * * 0  /home/cdeni/Desktop/Personal/CryptoAlgo/CryptoAlgo/ops/weekly_retrain.sh
#
# WHAT IT DOES
#   1. REFRESHES the fetched datasets -- `venue_strikes`, `venue_candles` and
#      the CF Benchmarks index. None of them is recorded live.
#   2. Runs `scripts.promote` in the one-off `retrain` container: walk-forward,
#      gates, and install **whatever the gates say**.
#   3. Nothing else. The trader is NOT restarted.
#
# **IT ALWAYS INSTALLS, BY STANDING OPERATOR POLICY (2026-10-21.)**
#   The compose `retrain` service passes `--force` with a written reason. The
#   gates still run, still print, and are still recorded in
#   `models/promotions/` -- that ledger IS the trial count and a forced install
#   is marked `forced: true` in it. What changed is that they no longer BLOCK:
#   the operator reads them, the cron does not act on them. The reason is that
#   the account needs the model to keep absorbing new data, and no arm has yet
#   beaten the market anyway (`model_minus_market_p` 0.21-0.23), so a gate that
#   refuses every candidate refuses the retrain rather than the risk.
#
#   The consequence worth naming: `forced` is now constant, so it no longer
#   distinguishes one promotion from another. The gate VALUES in the ledger are
#   what to read week to week, not the pass/fail flag.
#
# WHY THERE IS NO RESTART
#   `scripts/live.py` checks the installed artifact at the first cycle of every
#   window (`reload_if_changed`) and swaps it in when the file changed, keeping
#   the old model if the new one fails to load or is gate-blocked. A restart
#   used to do that job, at an arbitrary instant: it could land on a decision
#   or an order in flight, and it dropped the book sockets and in-memory state
#   (the gap-change history, the bar cache). So exit 0 (installed) is picked up
#   within one window; exit 1 (refused) changes nothing.
#
# WHY THERE IS NO --force
#   The deployed artifact was force-promoted past `calibration_error` 0.0216.
#   That gate has since been traced to a real cost: measured live 2026-09-10,
#   the fitted baseline is the worst calibrated of the three forecasters
#   (over-confidence +0.0362 against the market's +0.0252, all six probability
#   buckets pushed away from 0.5), and `base - mkt` at +12m reads -0.0139 --
#   the baseline losing to the price it trades against. A human may override a
#   gate once, with a written reason, and own it. An automatic weekly job must
#   not.
#
# WHY WEEKLY IS NOT WHAT THE EVIDENCE SUPPORTS
#   Recorded here so the cadence is a decision and not an accident. Recency
#   weighting was swept on 2026-08-25 and rejected monotonically -- half-life
#   None/365/180/90 gave model_minus_market +0.00221/+0.00188/+0.00158/+0.00123
#   with alpha collapsing 0.98 -> 0.68. Old regime is informative, not stale.
#   And the baseline's over-confidence was +0.0202 AT the 2026-08-28 training
#   cut, drifting at only +0.0013/day with r=+0.248 over 14 days -- not
#   significant. So the case for retraining is the ~20% of additional data that
#   accumulates, not decay; weekly is more often than the evidence requires,
#   and is run at the operator's instruction. It is harmless on an EXPANDING
#   window (no --recency-half-life-days, no --train-window-days), which is what
#   the compose service passes.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="$REPO/backend/trader/.tmp/retrain"
mkdir -p "$LOGDIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$LOGDIR/retrain_${STAMP}.log"

cd "$REPO" || exit 2

{
  echo "=== weekly retrain $STAMP ==="
  date -u

  # --- refresh the FETCHED datasets before training on them ---------------
  # `venue_candles` and `venue_strikes` are NOT recorded live; they only
  # advance when their backfill runs. Until 2026-10-22 nothing refreshed them
  # here, so a retrain trained on whatever happened to be in the store --
  # measured that morning, both were 17 hours behind after a manual run the
  # night before, and on a normal Sunday would have been a WEEK behind. The
  # model silently lost the most recent week, which is the week it most needs.
  #
  # Each job skips what it already holds, so a re-run is cheap. A failure is
  # logged and does NOT abort: training on slightly stale data beats not
  # training, and the gap is visible above the gate output.
  echo "--- refreshing fetched datasets ---"
  for job in "backfill_markets --start 2026-10-01" \
             "backfill_candles --start 2026-10-01" \
             "backfill_index --start 2026-10-01 --rate 5.0 --concurrency 4"; do
    echo "+ scripts.$job"
    docker compose run --rm --no-deps -T -w /app --entrypoint python trader \
      -u -m scripts.$job || echo "WARNING: $job exited $? -- continuing"
  done
  echo "--- refresh done ---"

  # `run --rm`, not `up`: one-off, and its exit code is the gate verdict.
  docker compose --profile tools run --rm retrain
  CODE=$?
  echo "promote exit code: $CODE"

  case "$CODE" in
    0)
      echo "artifact INSTALLED; the running trader reloads it at the next "
      echo "window boundary (no restart). Under --force this says nothing "
      echo "about whether the gates passed -- read the gate values in "
      echo "models/promotions/ for that, and the failed_gates list."
      ;;
    1)
      echo "promote ran but installed NOTHING. Under --force that is not a "
      echo "gate refusal -- forcing installs regardless -- so this means the "
      echo "install itself failed: a staging or rename error, or a candidate "
      echo "that could not be built. Investigate; the trader is still running "
      echo "its previous artifact."
      ;;
    *)
      echo "promote did not complete (exit $CODE). Trader untouched."
      ;;
  esac

  # Keep the last 12 weeks of logs; they are the record of what was tried.
  ls -1t "$LOGDIR"/retrain_*.log 2>/dev/null | tail -n +13 | xargs -r rm -f
  echo "=== done $(date -u) ==="
} >> "$LOG" 2>&1

exit 0
