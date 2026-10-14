"""Convert collected JSONL archives into Parquet layers, one closed month at a time.

`derived` holds the thirteen packed fields per snapshot, which features read.
`ladders` keeps the full ladders, because Kalshi deletes books on settlement.
Both are zstd Parquet partitioned venue/symbol/month. Re-running is idempotent.
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
import pyarrow as pa
import pyarrow.parquet as pq

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


# Explicit schemas, so every batch writes the same types. ParquetWriter rejects
# a type change between row groups.
DERIVED_SCHEMA = pa.schema(
    [('ts', pa.int64())]
    + [(f, pa.float64()) for f in FIELDS[1:]]
    + [('venue', pa.string()), ('symbol', pa.string()),
       ('market_id', pa.string()),
       ('window_open', pa.timestamp('us', tz='UTC')),
       ('event_time', pa.timestamp('us', tz='UTC')),
       ('offset_seconds', pa.float64())])

LADDER_SCHEMA = pa.schema([
    ('venue', pa.string()), ('symbol', pa.string()), ('market_id', pa.string()),
    ('window_open', pa.timestamp('us', tz='UTC')), ('ladders', pa.string())])

# One record can reach tens of MB of JSON, so the ladder layer flushes on a byte
# budget, not a row count.
LADDER_BYTES = int(os.getenv('CONSOLIDATE_LADDER_BYTES', 128 * 1024 * 1024))


def select_copies(paths) -> dict:
    """Pass 1: where the fullest copy of each window lives.

    Returns `{(market_id, window_open): (path, line_number)}`, the location
    only. Records carry their raw ladders and are too large to hold. Each line
    is parsed and dropped, so peak memory is one record.
    """
    best: dict = {}
    for path in paths:
        with _open(path) as handle:
            for lineno, line in enumerate(handle):
                try:
                    rec = json.loads(line)
                    key = (rec['market_id'], rec['window_open'])
                    n = len(rec.get('series') or [])
                except (ValueError, KeyError, TypeError):
                    continue                     # a torn last line from a kill
                del rec                          # before the next allocation
                have = best.get(key)
                if have is None or n > have[0]:
                    best[key] = (n, str(path), lineno)
    return {k: (p, ln) for k, (_, p, ln) in best.items()}


class _Sink:
    """Buffered Parquet writer: bounded memory, one file, stable schema."""

    def __init__(self, path, schema, *, max_rows=None, max_bytes=None):
        self.path, self.schema = path, schema
        self.max_rows, self.max_bytes = max_rows, max_bytes
        self.buf, self.bytes, self.writer, self.rows = [], 0, None, 0

    def add(self, row, nbytes: int = 0) -> None:
        self.buf.append(row)
        self.bytes += nbytes
        if ((self.max_rows and len(self.buf) >= self.max_rows)
                or (self.max_bytes and self.bytes >= self.max_bytes)):
            self.flush()

    def flush(self) -> None:
        if not self.buf:
            return
        frame = pd.DataFrame(self.buf)
        self.buf, self.bytes = [], 0
        table = pa.Table.from_pandas(frame, schema=self.schema,
                                     preserve_index=False)
        del frame
        if self.writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.path, self.schema,
                                           compression='zstd')
        self.writer.write_table(table)
        self.rows += table.num_rows
        del table

    def close(self) -> None:
        self.flush()
        if self.writer is not None:
            self.writer.close()


def convert(venue: str, symbol: str, month: str, paths, *, keep_ladders=True,
            batch_rows: int = 200_000):
    """One partition, both layers, streamed. Returns (n_snapshots, n_windows).

    `select_copies` picks the fullest copy of each window by line number; this
    pass re-reads and emits one record at a time. Peak memory tracks the batch.
    A window can appear more than once after a retry, and an earlier copy can
    be truncated, so the fullest copy wins.
    """
    chosen = select_copies(paths)
    if not chosen:
        return 0, 0
    wanted: dict = {}
    for (path_s, lineno) in chosen.values():
        wanted.setdefault(path_s, set()).add(lineno)

    derived = _Sink(
        DERIVED_OUT / f'venue={venue}' / f'symbol={symbol}' / f'month={month}'
        / 'data.parquet', DERIVED_SCHEMA, max_rows=batch_rows)
    ladders = _Sink(
        LADDERS_OUT / f'venue={venue}' / f'symbol={symbol}' / f'month={month}'
        / 'data.parquet', LADDER_SCHEMA, max_bytes=LADDER_BYTES
    ) if keep_ladders else None

    for path in paths:
        lines = wanted.get(str(path))
        if not lines:
            continue
        with _open(path) as handle:
            for lineno, line in enumerate(handle):
                if lineno not in lines:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                opened = pd.Timestamp(rec['window_open'])
                for snap in rec.get('series') or []:
                    row = dict(zip(FIELDS, snap))
                    ts = row.get('ts')
                    row.update(
                        venue=venue, symbol=symbol,
                        market_id=rec['market_id'], window_open=opened,
                        event_time=pd.Timestamp(ts, unit='ms', tz='UTC'),
                        offset_seconds=(
                            pd.Timestamp(ts, unit='ms', tz='UTC') - opened
                        ).total_seconds())
                    derived.add(row)
                if ladders is not None:
                    blob = json.dumps(rec.get('ladders') or [])
                    ladders.add({'venue': venue, 'symbol': symbol,
                                 'market_id': rec['market_id'],
                                 'window_open': opened, 'ladders': blob},
                                nbytes=len(blob))
                del rec

    derived.close()
    if ladders is not None:
        ladders.close()
    return derived.rows, len(chosen)


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
