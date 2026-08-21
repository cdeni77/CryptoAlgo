"""Copy the scraper's SQLite history into the Parquet research store.

The two stores have different jobs. SQLite stays the scraper's landing zone:
row-oriented, mutable, written one bar at a time by a live process. The research
store is columnar, immutable and venue-keyed, and is what feature building and
backtests read.

Migration is idempotent — the store de-duplicates on the full revision key — so
this can be re-run after each backfill.

    python -m scripts.migrate_to_research_store --venue coinbase
    python -m scripts.migrate_to_research_store --coverage
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from core.datastore import ResearchStore, from_sqlite

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
logger = logging.getLogger(__name__)

DEFAULT_DB = os.getenv('TRADER_DB_PATH', './data/trading.db')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db-path', default=DEFAULT_DB, help='Scraper SQLite database')
    parser.add_argument('--store', default=None, help='Research store root (default: data/research)')
    parser.add_argument(
        '--venue', default='unknown',
        help="Venue label for rows written before the venue column existed. "
             "Rows that recorded their own venue keep it.",
    )
    parser.add_argument('--timeframe', default='1h')
    parser.add_argument('--symbols', default=None, help='Comma-separated subset')
    parser.add_argument('--coverage', action='store_true',
                        help='Report what the store already holds, without migrating')
    args = parser.parse_args()

    store = ResearchStore(args.store) if args.store else ResearchStore()

    if args.coverage:
        # Report only. Said out loud because `--coverage --db-path ...` reads as
        # "migrate, then show me", and silently skipping the migration means the
        # numbers below describe whatever was already there.
        print(f'reporting on {store.root} without migrating '
              f'(drop --coverage to migrate first)')
        for dataset in ('bars', 'funding', 'open_interest'):
            frame = store.coverage(dataset)
            print(f"\n{dataset}:")
            print('  (empty)' if frame.empty else frame.to_string(index=False))
        return 0

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error("No database at %s", db_path)
        return 1

    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else None
    counts = from_sqlite(
        store, db_path, venue=args.venue, timeframe=args.timeframe, symbols=symbols
    )

    if not counts:
        logger.warning("Nothing migrated — is the database empty?")
        return 0

    for dataset, rows in counts.items():
        logger.info("%s: %d rows", dataset, rows)

    print("\nCoverage after migration:")
    for dataset in counts:
        frame = store.coverage(dataset)
        if not frame.empty:
            print(f"\n{dataset}:")
            print(frame.to_string(index=False))

    return 0


if __name__ == '__main__':
    sys.exit(main())
