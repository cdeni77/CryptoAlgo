"""Correct `predictions.outcome` and settled positions against the venue's settlement.

The market gates read `predictions.outcome`, which the live loop fills from
Coinbase bars. The venue's settlement is the true label.

A separate step, not part of `settle_predictions`: the live loop's venue data
is keyed by ticker, and mapping a ticker to a window would mean parsing it.
`venue_settlements` already carries symbol and window_open. Runs hourly in
`run_live.store_sync_loop`, after `collect_settlements`.

Idempotent: it updates only rows whose outcome differs, so the printed count
is the disagreement.
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
    """Return `(symbol, window_open, settled_up)` from the venue's own settlements."""
    frame = store.read('venue_settlements')
    if frame is None or not len(frame):
        return []
    frame = frame[frame['venue'] == venue]
    frame = frame.dropna(subset=['symbol', 'window_open', 'settled_up'])
    frame = frame.copy()
    frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
    if since is not None:
        frame = frame[frame['window_open'] >= since]
    # One row per window: the store keeps revisions, and the newest
    # available_time wins.
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

    writer = PgWriter(database_url=url)
    examined, corrected = writer.correct_outcomes_from_venue(labels)
    if corrected:
        logger.warning('corrected %d prediction row(s) of %d window(s) checked '
                       '— our Coinbase label disagreed with the venue on these, '
                       'and the market gates read this column',
                       corrected, examined)
    else:
        logger.info('%d window(s) checked, no disagreement', examined)

    # Positions too. A position that matured before the venue's row arrived
    # was settled from our bars. `account.realized_pnl` feeds the circuit
    # breakers, so it must use the venue's label.
    seen, fixed, delta = writer.correct_positions_from_venue(labels)
    if fixed:
        logger.warning('re-settled %d position(s) the venue graded differently: '
                       'realised P&L %+.2f. The breakers read this figure, so it '
                       'was optimistic by that much until now', fixed, delta)
    else:
        logger.info('%d window(s) checked, no position disagreement', seen)
    return 0


if __name__ == '__main__':
    sys.exit(main())
