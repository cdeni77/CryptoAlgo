"""Remove instruments outside the traded universe, from the store and the DB.

    python -m scripts.prune_universe --tradeable-five             # dry run
    python -m scripts.prune_universe --tradeable-five --apply
    python -m scripts.prune_universe --keep BTC,ETH,XRP,SOL,ADA --apply

Dry run by default, like `purge_dead_open_interest` before it: it prints every
partition and row count it would remove and changes nothing until `--apply`.

Bars and funding are not the same kind of data
----------------------------------------------
**Bars regenerate.** Coinbase serves candle history by range, so a deleted
instrument's bars come back from one `run_pipeline --backfill-only`.

**Funding and open interest do not.** Both are single-value snapshots on the
product endpoint — no range parameters, no cursor — so they accumulate forward
only, one observation per contract per cycle, and there is no request that
recovers a deleted row. `.gitignore` un-ignores those two datasets specifically
because they are the only irreplaceable thing in the store.

So this refuses to touch them without `--include-irreplaceable`, and even then
prints what will be lost first. A universe decision is reversible; deleting
forward-only data is not.

The other cost, stated because it is easy to forget
---------------------------------------------------
Breadth is the mechanism for the cross-sectional residual formulation. Fourteen
names give residual breadth 10.89 and an IC standard error of 0.031; five give
4.98 and 0.045. Pruning to five raises that error 45% and makes a 3-long/3-short
basket impossible — it discards the only configuration measured in this project
whose IC exceeded its own cost hurdle. Keep a copy if there is any chance of
going back.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path

from core.costs import resolve_base

# Selected by rule in `scripts.instrument_screen`: round trip <= 35bp, fill
# uncertainty <= 0.40x cost, history >= 231d.
TRADEABLE_FIVE = ('BTC', 'ETH', 'XRP', 'SOL', 'ADA')

IRREPLACEABLE = ('funding', 'open_interest')


def _underlying(name: str) -> str | None:
    """The base asset for a partition or DB symbol, either spelling."""
    return resolve_base(name)


def _store_targets(root: Path, keep: set[str]) -> list[tuple[Path, str, str]]:
    """(directory, dataset, symbol) for every partition outside the universe."""
    out: list[tuple[Path, str, str]] = []
    for dataset_dir in sorted(root.glob('*')):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        for symbol_dir in sorted(dataset_dir.glob('venue=*/symbol=*')):
            symbol = symbol_dir.name.split('=', 1)[1]
            base = _underlying(symbol)
            if base is None:
                # Unresolvable: leave it alone and say so rather than guess.
                out.append((symbol_dir, dataset, f'{symbol} (UNRESOLVED — kept)'))
                continue
            if base not in keep:
                out.append((symbol_dir, dataset, symbol))
    return out


def _db_targets(db_path: Path, keep: set[str]) -> dict[str, dict[str, int]]:
    """Row counts per table for symbols outside the universe."""
    tables = {'ohlcv': 'bars', 'funding_rates': 'funding',
              'open_interest': 'open_interest'}
    found: dict[str, dict[str, int]] = {}
    with sqlite3.connect(db_path) as conn:
        for table in tables:
            try:
                rows = conn.execute(
                    f'SELECT symbol, COUNT(*) FROM {table} GROUP BY symbol').fetchall()
            except sqlite3.OperationalError:
                continue
            drop = {}
            for symbol, count in rows:
                base = _underlying(symbol)
                if base is not None and base not in keep:
                    drop[symbol] = int(count)
            if drop:
                found[table] = drop
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--store', default='data/research')
    parser.add_argument('--db-path', default='data/trading.db')
    parser.add_argument('--keep', default=None,
                        help='Comma-separated underlyings to keep (BTC,ETH,...)')
    parser.add_argument('--tradeable-five', action='store_true',
                        help=f'Keep {", ".join(TRADEABLE_FIVE)}')
    parser.add_argument('--include-irreplaceable', action='store_true',
                        help='Also delete funding and open interest, which cannot '
                             'be re-fetched from any endpoint. Off by default.')
    parser.add_argument('--apply', action='store_true',
                        help='Actually delete. Without this, nothing changes.')
    args = parser.parse_args()

    if args.tradeable_five:
        keep = set(TRADEABLE_FIVE)
    elif args.keep:
        keep = {s.strip().upper() for s in args.keep.split(',') if s.strip()}
        resolved = {_underlying(s) or s for s in keep}
        keep = resolved
    else:
        parser.error('pass --tradeable-five or --keep')

    print(f'keeping: {", ".join(sorted(keep))}')
    print(f'mode:    {"APPLY — deletions are permanent" if args.apply else "dry run"}')
    if not args.include_irreplaceable:
        print(f'         funding and open interest are protected '
              f'(--include-irreplaceable to override)')
    print()

    store = Path(args.store)
    targets = _store_targets(store, keep) if store.exists() else []
    by_dataset: dict[str, list[tuple[Path, str]]] = {}
    for path, dataset, symbol in targets:
        by_dataset.setdefault(dataset, []).append((path, symbol))

    removed_dirs = 0
    for dataset in sorted(by_dataset):
        entries = by_dataset[dataset]
        protected = dataset in IRREPLACEABLE and not args.include_irreplaceable
        label = '  [PROTECTED]' if protected else ''
        print(f'{dataset}: {len(entries)} instrument(s){label}')
        for path, symbol in entries:
            size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            print(f'    {symbol:<24} {size / 1e6:7.2f} MB  {path}')
            if args.apply and not protected and 'UNRESOLVED' not in symbol:
                shutil.rmtree(path)
                removed_dirs += 1
        print()

    db_path = Path(args.db_path)
    if db_path.exists():
        db = _db_targets(db_path, keep)
        for table, drop in sorted(db.items()):
            dataset = {'ohlcv': 'bars', 'funding_rates': 'funding',
                       'open_interest': 'open_interest'}[table]
            protected = dataset in IRREPLACEABLE and not args.include_irreplaceable
            label = '  [PROTECTED]' if protected else ''
            print(f'{db_path}:{table}: {sum(drop.values()):,} rows across '
                  f'{len(drop)} symbol(s){label}')
            for symbol, count in sorted(drop.items()):
                print(f'    {symbol:<24} {count:>9,} rows')
            if args.apply and not protected:
                with sqlite3.connect(db_path) as conn:
                    marks = ','.join('?' * len(drop))
                    conn.execute(f'DELETE FROM {table} WHERE symbol IN ({marks})',
                                 list(drop))
                    conn.commit()
                print(f'    deleted. Run VACUUM to reclaim the file size.')
            print()
    else:
        print(f'{db_path}: absent (nothing to prune there)')

    # The feature panel is keyed by a hash of the column names, not the universe,
    # so a panel built on 18 instruments stays on disk and stays loadable after
    # the bars behind it are gone. It regenerates from bars in minutes.
    panels = list((store / 'features').glob('*/*')) if store.exists() else []
    if panels:
        size = sum(f.stat().st_size for f in panels if f.is_file())
        print(f'features: {len(panels)} file(s), {size / 1e6:.1f} MB — stale after '
              f'this prune and NOT removed automatically. `feature_set_hash` hashes '
              f'column names, so a panel built on the old universe is still '
              f'loadable and would silently score against instruments that no '
              f'longer exist. Rebuild with `python -m scripts.build_features`.')

    if not args.apply:
        print('\ndry run: nothing was deleted. Re-run with --apply.')
    else:
        print(f'\nremoved {removed_dirs} store partition(s).')
        if not args.include_irreplaceable:
            print('funding and open interest were left in place.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
