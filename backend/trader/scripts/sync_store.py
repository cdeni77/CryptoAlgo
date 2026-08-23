"""Copy scraped SQLite rows into the Parquet research store.

Two stores, two jobs. SQLite is what the scraper writes: mutable, row-oriented,
one file. The research store is what everything else reads: immutable Parquet
partitioned by dataset, venue, symbol and month, queried through DuckDB, and
fast enough that a five-year feature build is not an afternoon.

**Funding and open interest are migrated even though nothing reads them.** They
are archive — the binary system has no use for either — but no Coinbase endpoint
serves either historically, so unlike bars they cannot be re-fetched at any
price. The SQLite file is the only other copy, and a sync that skipped them
followed by a cleanup of the database would destroy the only irreplaceable data
in this repo. Pass `--no-archive` to skip them deliberately.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.config import DEFAULT_CONFIG
from core.datastore import BARS_DATASET_BY_TIMEFRAME, ResearchStore, from_sqlite


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--db-path', type=str, default='./data/trading.db')
    parser.add_argument('--venue', type=str, default=DEFAULT_CONFIG.venue,
                        help='Label for rows that recorded none. Rows that kept '
                             'their own venue keep it.')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_CONFIG.timeframe)
    parser.add_argument('--symbols', type=str, default=None)
    parser.add_argument('--research-store', type=str, default=None)
    parser.add_argument('--no-archive', action='store_true',
                        help='Skip funding and open interest. They cannot be '
                             're-fetched, so only do this deliberately.')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S', stream=sys.stdout)

    db = Path(args.db_path)
    if not db.exists():
        raise SystemExit(f'no database at {db} — run `python -m scripts.scrape` first')

    store = ResearchStore(args.research_store)
    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else None
    dataset = BARS_DATASET_BY_TIMEFRAME.get(args.timeframe, 'bars')
    print(f'{db} -> {store.root}  (timeframe {args.timeframe} -> dataset {dataset})')

    counts = from_sqlite(store, db, venue=args.venue, timeframe=args.timeframe,
                         symbols=symbols, include_archive=not args.no_archive)
    if not counts:
        print('nothing written — is the timeframe right?')
        return 1
    for name, rows in sorted(counts.items()):
        print(f'  {name:<16} {rows:>12,} rows')

    coverage = store.coverage(dataset)
    if coverage is not None and not coverage.empty:
        print('\ncoverage:')
        print(coverage.to_string(index=False))
    print('\nnext: python -m scripts.baseline')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
