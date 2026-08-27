"""Phase 4: turn the collected JSONL into the two stored layers.

Collection appends JSONL because appending is crash-safe: a `kill -9` costs one
partial line rather than a rewritten partition, and a 47-hour job will be
interrupted. That property is worth having while collecting and worthless
afterwards, so a closed month is converted once, here.

Two outputs, per the design:

    derived   the thirteen packed fields per snapshot, one row per snapshot.
              This is what features read; a training run never touches the
              archive.
    archive   the full ladders, kept because Kalshi destroys books on
              settlement and Predexon is the only source. Questions nobody has
              asked yet — queue position, ladder slope past 5c, replenishment
              shape — are answerable only from this.

Both are Parquet with zstd, partitioned venue/symbol/month to match the
existing research store. Idempotent: converting a month twice produces the
same files, so a re-run after an interrupted conversion is safe.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.collect.fetchers import FIELDS                  # noqa: E402

DATA = Path(os.getenv('COLLECT_DATA', 'data/collection'))
ARCHIVE_IN = DATA / 'archive'
DERIVED_OUT = DATA / 'derived'
LADDERS_OUT = DATA / 'ladders'


def _partitions() -> dict:
    found = defaultdict(list)
    for pattern in ('venue=*/symbol=*/month=*/windows.jsonl.gz',
                    'venue=*/symbol=*/month=*/windows.jsonl'):
        for path in ARCHIVE_IN.glob(pattern):
            parts = {p.split('=')[0]: p.split('=')[1]
                     for p in path.parts if '=' in p}
            found[(parts['venue'], parts['symbol'], parts['month'])].append(path)
    return found


def _open(path):
    """Archives are gzipped; older partitions may not be."""
    return (gzip.open(path, 'rt') if str(path).endswith('.gz')
            else open(path))


def convert(venue: str, symbol: str, month: str, paths, *, keep_ladders=True):
    """One partition, both layers. Returns (n_snapshots, n_windows)."""
    # Keep the FULLEST copy of each window, not the first. A window can appear
    # more than once: an `error` retry re-fetches it, and 3,099 Polymarket
    # windows were archived truncated at the 2,000-snapshot page cap before
    # `fetch_pm` paginated, then reset and re-collected. Keeping whichever copy
    # happened to be written first would silently keep the truncated one and
    # make the re-collection pointless.
    best: dict = {}
    for path in paths:
        with _open(path) as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue                     # a torn last line from a kill
                key = (rec['market_id'], rec['window_open'])
                have = best.get(key)
                if have is None or len(rec.get('series') or []) > len(have.get('series') or []):
                    best[key] = rec

    rows, ladders = [], []
    for rec in best.values():
        opened = pd.Timestamp(rec['window_open'])
        for snap in rec.get('series') or []:
            row = dict(zip(FIELDS, snap))
            row.update(venue=venue, symbol=symbol,
                       market_id=rec['market_id'], window_open=opened)
            rows.append(row)
        if keep_ladders:
            ladders.append({
                'venue': venue, 'symbol': symbol,
                'market_id': rec['market_id'], 'window_open': opened,
                'ladders': json.dumps(rec.get('ladders') or []),
            })
    seen = best
    if not rows:
        return 0, 0

    frame = pd.DataFrame(rows)
    # `ts` is milliseconds since epoch; carry a real timestamp so downstream
    # never has to remember the unit, and an offset so the window grid is
    # queryable without recomputing it per row.
    frame['event_time'] = pd.to_datetime(frame['ts'], unit='ms', utc=True)
    frame['offset_seconds'] = (
        (frame['event_time'] - frame['window_open']).dt.total_seconds())

    out = DERIVED_OUT / f'venue={venue}' / f'symbol={symbol}' / f'month={month}'
    out.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out / 'data.parquet', compression='zstd', index=False)

    if keep_ladders and ladders:
        lout = LADDERS_OUT / f'venue={venue}' / f'symbol={symbol}' / f'month={month}'
        lout.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ladders).to_parquet(lout / 'data.parquet',
                                         compression='zstd', index=False)
    return len(frame), len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--month', default=None, help='only this YYYY-MM')
    parser.add_argument('--no-ladders', action='store_true',
                        help='derived layer only; skip the raw archive')
    args = parser.parse_args()

    total_snaps = total_windows = 0
    for (venue, symbol, month), paths in sorted(_partitions().items()):
        if args.month and month != args.month:
            continue
        snaps, windows = convert(venue, symbol, month, paths,
                                 keep_ladders=not args.no_ladders)
        total_snaps += snaps
        total_windows += windows
        print(f'  {venue:11s} {symbol:8s} {month}  {windows:6,} windows  '
              f'{snaps:9,} snapshots', flush=True)
    print(f'\n{total_windows:,} windows, {total_snaps:,} snapshots')
    if total_windows:
        size = sum(p.stat().st_size for p in DERIVED_OUT.rglob('*.parquet'))
        size += sum(p.stat().st_size for p in LADDERS_OUT.rglob('*.parquet'))
        print(f'{size / 1e9:.2f} GB on disk, '
              f'{size / max(total_windows, 1) / 1024:.0f} KB per window')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
