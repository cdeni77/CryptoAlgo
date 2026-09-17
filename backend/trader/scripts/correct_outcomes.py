"""Correct `predictions.outcome` against the venue's own settlement.

**The market gates read this column, and it was filled from Coinbase bars.**
`settle_due` prefers the venue's settlement for the money. `settle_predictions`,
eleven lines away in the same function with the venue data already in scope,
does not — and `PgWriter.scored_against_market()` can return no other label. So
`model_minus_market` and `calibration_vs_market` were computed on our proxy
while the backtest branch of the same gate substitutes `venue_outcome`. Two
label conventions under one gate name, chosen by which branch had data.

Measured 2026-09-16 on 5,471 settled rows at +12m:

    OUR Coinbase label      model_minus_market  +0.000924
    VENUE's own settlement  model_minus_market  +0.000188
    labels disagree on 156 rows                      2.85%

Four fifths of the apparent live edge was self-grading. The disagreement is the
shape a benign proxy should have — it lives in the near-ties, exactly as the
96.98% agreement measured across 56,284 windows predicts — but "benign" is a
statement about the label's bias, not about a statistic computed from it.

**Why a separate step rather than a fix inside `settle_predictions`.** The venue
dict the live loop holds is keyed by market ticker; predictions are keyed by
(symbol, window_open), and deriving one from the other means parsing a ticker,
which is the guess-a-pattern failure `CLAUDE.md` forbids for market resolution.
`venue_settlements` in the research store already carries symbol and
window_open, resolved by the venue itself. So this runs hourly in
`run_live.store_sync_loop`, immediately after `collect_settlements` writes the
rows it reads — out of the trading path entirely.

Idempotent: it updates only rows whose outcome actually differs, so a re-run
corrects nothing and the printed count IS the disagreement.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import pandas as pd

from core.datastore import ResearchStore
from core.pg_writer import PgWriter

logger = logging.getLogger('correct-outcomes')


def venue_labels(store: ResearchStore, *, venue: str = 'kalshi',
                 since: pd.Timestamp | None = None):
    """`(symbol, window_open, settled_up)` from the venue's own settlements."""
    frame = store.read('venue_settlements')
    if frame is None or not len(frame):
        return []
    frame = frame[frame['venue'] == venue]
    frame = frame.dropna(subset=['symbol', 'window_open', 'settled_up'])
    frame = frame.copy()
    frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
    if since is not None:
        frame = frame[frame['window_open'] >= since]
    # One row per window. The store keeps revisions; the newest available_time
    # is the venue's latest word, which is the one to believe.
    if 'available_time' in frame.columns:
        frame = (frame.sort_values('available_time')
                      .drop_duplicates(['symbol', 'window_open'], keep='last'))
    return list(frame[['symbol', 'window_open', 'settled_up']]
                .itertuples(index=False, name=None))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--venue', default='kalshi')
    parser.add_argument('--days', type=float, default=45.0,
                        help='only windows this recent (default 45). The live '
                             'loop has not been running longer than this, and '
                             'a full scan re-reads every settlement ever.')
    parser.add_argument('--dry-run', action='store_true',
                        help='report the disagreement, write nothing')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    url = os.getenv('DATABASE_URL')
    if not url:
        logger.error('DATABASE_URL is unset, so there are no predictions to '
                     'correct. This must run where the serving store is '
                     'reachable — in the compose service, not on the host.')
        return 1

    since = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=float(args.days))
    labels = venue_labels(ResearchStore(), venue=args.venue, since=since)
    if not labels:
        logger.info('no venue settlements in the last %.0f days; nothing to do',
                    args.days)
        return 0

    if args.dry_run:
        logger.info('--dry-run: %d venue label(s) would be checked', len(labels))
        return 0

    examined, corrected = PgWriter(database_url=url).correct_outcomes_from_venue(labels)
    if corrected:
        logger.warning('corrected %d prediction row(s) of %d window(s) checked '
                       '— our Coinbase label disagreed with the venue on these, '
                       'and the market gates read this column',
                       corrected, examined)
    else:
        logger.info('%d window(s) checked, no disagreement', examined)
    return 0


if __name__ == '__main__':
    sys.exit(main())
