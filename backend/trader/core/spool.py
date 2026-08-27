"""An append-only sink for the raw frame stream, and its compaction.

**This deliberately does not use `ResearchStore.write`.** That path reads a
whole `(venue, symbol, month)` partition, concatenates, sorts and rewrites it
under zstd. `run_live.py` names it as the latency threat the entire live process
is arranged around, and it is right for a revisable series — but measured, this
stream runs at 777-862 frames a second across three markets, so a month
partition would reach tens of gigabytes and be rewritten on every flush.
Appending is O(what was appended) and never reads back. Compaction then writes
IMMUTABLE hour-named Parquet files, which `ResearchStore.read` still finds
because it globs `**/*.parquet` and does not depend on the path layout.

**JSONL rather than an in-memory buffer, on purpose.** A crash loses whatever is
buffered, and `record_ladder` already argues this book cannot be rebuilt later at
any cost. Appending a line costs an extra pass over the data and buys
crash-safety on the one dataset with no second chance.

**One row per LEVEL, not per frame.** A delta names one price; a snapshot names
the whole ladder. Emitting a row per level gives both one flat schema, and a
replay regroups by `(market_ticker, seq)` to rebuild the original event. That
keeps the one-applier invariant: replay feeds `BookCache` the same `BookEvent`
the live path folded.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd

from data_collection.stream.base import BookEvent

logger = logging.getLogger('spool')

COLUMNS = ('venue', 'symbol', 'event_time', 'available_time', 'quality',
           'market_ticker', 'seq', 'kind', 'side', 'price', 'size', 'absolute')


def _hour_key(t: float) -> str:
    return datetime.fromtimestamp(t, tz=timezone.utc).strftime('%Y%m%dT%H')


def event_rows(event: BookEvent, symbol: str) -> Iterator[dict]:
    """One record per level. An empty side yields nothing."""
    for side, levels in (('yes', event.yes), ('no', event.no)):
        for price, size in levels:
            yield {'t': event.received, 'venue': event.venue, 'symbol': symbol,
                   'market_ticker': event.market_ticker, 'seq': event.seq,
                   'kind': event.kind, 'side': side, 'price': price,
                   'size': size, 'absolute': event.absolute}


class FrameSpool:
    """Append-only hourly JSONL, one file per venue per UTC hour."""

    def __init__(self, root: Path | str, venue: str) -> None:
        self.root = Path(root) / venue
        self.root.mkdir(parents=True, exist_ok=True)
        self.venue = venue
        self._hour: Optional[str] = None
        self._handle = None

    def _path(self, hour: str) -> Path:
        return self.root / f'{hour}.jsonl'

    def append(self, record: dict) -> None:
        hour = _hour_key(record['t'])
        if hour != self._hour:
            self.close()
            self._hour = hour
            self._handle = self._path(hour).open('a')
        self._handle.write(json.dumps(record) + '\n')

    def extend(self, records: Iterable[dict]) -> int:
        count = 0
        for record in records:
            self.append(record)
            count += 1
        return count

    def flush(self) -> None:
        if self._handle is not None:
            self._handle.flush()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        self._hour = None


def compact(spool_root: Path | str, store_root: Path | str, *,
            keep_days: float = 14.0, now: float | None = None) -> int:
    """Fold closed hourly files into immutable Parquet, then remove them.

    A file whose hour has not yet ended is SKIPPED: compacting it would archive
    a prefix of the stream and then delete the rest.

    `keep_days` prunes compacted Parquet, not the spool. At a measured ~862
    frames a second the raw stream is not affordable indefinitely, so this tier
    is bounded and `venue_ladder` is the one kept forever.
    """
    spool_root, store_root = Path(spool_root), Path(store_root)
    stamp = time.time() if now is None else now
    open_hour = _hour_key(stamp)
    written = 0

    for path in sorted(spool_root.rglob('*.jsonl')):
        if path.stem >= open_hour:
            continue
        rows = []
        for line in path.open():
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            when = pd.Timestamp(rec['t'], unit='s', tz='UTC')
            rows.append({
                'venue': rec['venue'], 'symbol': rec['symbol'],
                'event_time': when, 'available_time': when, 'quality': 'valid',
                'market_ticker': rec['market_ticker'], 'seq': rec.get('seq'),
                'kind': rec['kind'], 'side': rec['side'],
                'price': rec['price'], 'size': rec['size'],
                'absolute': rec['absolute'],
            })
        if rows:
            frame = pd.DataFrame(rows, columns=list(COLUMNS))
            day = path.stem[:8]
            out = (store_root / 'venue_book_events' / f'venue={path.parent.name}'
                   / f'date={day[:4]}-{day[4:6]}-{day[6:]}')
            out.mkdir(parents=True, exist_ok=True)
            target = out / f'{path.stem}.parquet'
            tmp = target.with_suffix('.parquet.tmp')
            frame.to_parquet(tmp, index=False, compression='zstd')
            tmp.replace(target)
            written += len(frame)
            logger.info('compacted %s -> %s (%d rows)', path.name, target.name,
                        len(frame))
        path.unlink()

    # Pruning counts FILES; this returns ROWS. Keeping them separate stops a
    # caller reading "compacted 40,000" as anything but rows.
    prune(store_root, keep_days=keep_days, now=stamp)
    return written


def prune(store_root: Path | str, *, keep_days: float, now: float | None = None) -> int:
    """Delete compacted Parquet older than `keep_days`. Returns files removed."""
    if keep_days <= 0:
        return 0
    root = Path(store_root) / 'venue_book_events'
    if not root.exists():
        return 0
    cutoff = (time.time() if now is None else now) - keep_days * 86400.0
    removed = 0
    for path in root.rglob('*.parquet'):
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    for directory in sorted(root.rglob('*'), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()
    return removed
