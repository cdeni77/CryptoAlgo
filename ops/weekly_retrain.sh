#!/usr/bin/env bash
#
# Weekly gated retrain for the live trader. Installed via cron:
#
#   0 9 * * 0  /home/cdeni/Desktop/Personal/CryptoAlgo/CryptoAlgo/ops/weekly_retrain.sh
#
# WHAT IT DOES
#   1. Runs `scripts.promote` in the one-off `retrain` container: walk-forward,
#      gates, and install ONLY if every gate passes.
#   2. Restarts the live trader IF AND ONLY IF a new artifact was installed.
#
# WHY THE CONDITIONAL RESTART IS THE POINT
#   `scripts/live.py` calls `load_live` once at startup and never reloads, so
#   promoting an artifact changes nothing about what is being traded until the
#   loop restarts. A weekly promote without a restart is a silent no-op.
#   Equally, restarting when nothing was installed is not free: it resets the
#   live measurement window and triggers a Coinbase 429 burst while the bar
#   history refetches. So: exit 0 (installed) restarts, exit 1 (refused) does
#   not.
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

  # `run --rm`, not `up`: one-off, and its exit code is the gate verdict.
  docker compose --profile tools run --rm retrain
  CODE=$?
  echo "promote exit code: $CODE"

  case "$CODE" in
    0)
      echo "gates PASSED and the artifact was installed; restarting the trader "
      echo "so the loop picks it up (load_live runs only at startup)"
      docker compose restart trader
      echo "trader restarted at $(date -u)"
      ;;
    1)
      echo "gates REFUSED the candidate; nothing installed and the trader is "
      echo "left running its current artifact. The rejection is recorded in "
      echo "models/promotions/ -- that ledger IS the trial count, so a refusal "
      echo "is data, not a failure."
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
