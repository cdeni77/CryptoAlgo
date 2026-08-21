"""Delete open-interest series that never carried a measurement.

A scrape wrote 720 rows per contract of zero open interest. The cause was
`float(entry.get('openInterestAmount') or entry.get('baseVolume') or 0)` in the
CCXT backfill: an entry with no usable field became 0.0, and a row was created
and stored anyway. Sixteen contracts x 720 hours = 11,520 rows that read as data.

The signature is series-level, not row-level. A venue *reporting* zero open
interest is a measurement and must be kept; a series whose maximum is zero across
its whole history never measured anything. So the unit of deletion is
(venue, symbol), and the test is `max(open_interest_contracts) == 0`.

Usage:
    python -m scripts.purge_dead_open_interest                  # report only
    python -m scripts.purge_dead_open_interest --apply
    python -m scripts.purge_dead_open_interest --apply --venue okx

After applying, re-sync so the research store matches:
    python -m scripts.migrate_to_research_store --venue coinbase
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

TRADER_ROOT = Path(__file__).resolve().parents[1]
if str(TRADER_ROOT) not in sys.path:
    sys.path.insert(0, str(TRADER_ROOT))

from data_collection.storage import SQLiteDatabase          # noqa: E402
from core.datastore import ResearchStore                    # noqa: E402

LOGGER = logging.getLogger('purge_dead_open_interest')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--db-path', default=os.getenv('TRADER_DB_PATH')
                        or str(TRADER_ROOT / 'data' / 'trading.db'))
    parser.add_argument('--store', default=os.getenv('RESEARCH_STORE') or None,
                        help='Also drop the research store open_interest dataset, '
                             'which is rebuilt from SQLite by migrate_to_research_store')
    parser.add_argument('--venue', default=None,
                        help='Restrict to one venue label (e.g. okx)')
    parser.add_argument('--contracts-only', action='store_true',
                        help="Treat a series whose open_interest_contracts is "
                             "identically zero as dead even when open_interest_usd "
                             "carries values. `positioning_features` prefers "
                             "contracts and only falls back to usd, so this is the "
                             "right call when the usd column is itself derived "
                             "junk — check the reported maxima before using it.")
    parser.add_argument('--apply', action='store_true',
                        help='Actually delete. Without this, nothing is written.')
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')
    args = parse_args()

    if not Path(args.db_path).exists():
        LOGGER.error('no database at %s', args.db_path)
        return 1

    database = SQLiteDatabase(args.db_path)
    with database._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT venue, symbol, COUNT(*) AS rows,
                   MAX(open_interest_contracts) AS max_contracts,
                   MAX(COALESCE(open_interest_usd, 0)) AS max_usd,
                   MIN(event_time) AS first, MAX(event_time) AS last
            FROM open_interest
            GROUP BY venue, symbol
            ORDER BY venue, symbol
        """)
        series = [dict(row) for row in cursor.fetchall()]

    if not series:
        print('No open interest rows at all — nothing to purge.')
        return 0

    def verdict(row) -> str:
        """Why this series lives or dies, in the report rather than in my head.

        The first version printed only `max ... contracts` and a keep/DEAD label,
        so a series with zero contracts and a populated usd column read as
        'keep ... max 0 contracts' — a verdict that contradicted the only number
        shown. Both maxima are printed now and the reason is named.
        """
        if row['max_contracts']:
            return 'live'
        if row['max_usd'] and not args.contracts_only:
            return 'usd-only'
        return 'dead'

    scoped = [s for s in series if args.venue is None or s['venue'] == args.venue]
    verdicts = {id(s): verdict(s) for s in scoped}
    dead = [s for s in scoped if verdicts[id(s)] == 'dead']

    print(f'\n{len(series)} open-interest series in {args.db_path}')
    print(f'  {"":6}{"symbol":22} {"venue":<14} {"rows":>6} '
          f'{"max contracts":>15} {"max usd":>15}')
    for s in scoped:
        label = {'live': 'keep', 'usd-only': 'KEEP?', 'dead': 'DEAD'}[verdicts[id(s)]]
        print(f'  {label:6}{s["symbol"]:22} {s["venue"]:<14} {s["rows"]:6} '
              f'{s["max_contracts"]:>15,.0f} {s["max_usd"]:>15,.0f}')

    usd_only = [s for s in scoped if verdicts[id(s)] == 'usd-only']
    if usd_only:
        print(f'\n{len(usd_only)} series have zero contracts but a populated usd '
              f'column. `positioning_features` prefers oi_contracts and only falls '
              f'back to oi_usd, so these still produce features — from the usd '
              f'column alone. If that column is derived junk rather than a '
              f'measurement, re-run with --contracts-only to delete them too.')

    if not dead:
        print('\nNothing unambiguously dead.')
        return 0

    total = sum(s['rows'] for s in dead)
    print(f'\n{len(dead)} dead series, {total:,} rows')

    if not args.apply:
        print('Dry run. Re-run with --apply to delete.')
        return 0

    with database._get_connection() as conn:
        cursor = conn.cursor()
        for s in dead:
            cursor.execute(
                'DELETE FROM open_interest WHERE venue = ? AND symbol = ?',
                (s['venue'], s['symbol']),
            )
        conn.commit()
    print(f'Deleted {total:,} rows from {args.db_path}')

    # The store is derived, so dropping the dataset is safe: the next
    # migrate_to_research_store rebuilds it from what SQLite now holds. Editing
    # the parquet partitions in place would leave the two disagreeing.
    # Derived from the database that was actually purged, not from TRADER_ROOT —
    # pointing --db-path at a scratch copy and having this drop the real store
    # would be a worse bug than the one being cleaned up.
    store_root = args.store
    if store_root is None:
        candidate = Path(args.db_path).resolve().parent / 'research'
        store_root = str(candidate) if candidate.is_dir() else None
    if store_root:
        try:
            ResearchStore(store_root).drop('open_interest')
            print(f'Dropped open_interest from the research store at {store_root}')
            print('Run: python -m scripts.migrate_to_research_store --venue coinbase')
        except Exception as exc:                                  # noqa: BLE001
            LOGGER.warning('could not drop the store dataset: %s', exc)

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
