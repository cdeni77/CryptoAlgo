"""An append-only sink for the raw frame stream, and its compaction.

Does not use `ResearchStore.write`: that rewrites a whole partition per flush,
which the frame rate cannot afford. Appending JSONL is O(appended) and survives
a crash. Compaction writes immutable hour-named Parquet files, which
`ResearchStore.read` finds by globbing `**/*.parquet`.

One row per level, not per frame. A replay regroups rows into the original
`BookEvent`, so replay and live fold through the same `BookCache`.
Regroup on `(market_ticker, event_time, seq)`: `seq` restarts on every
resubscription, so `(market_ticker, seq)` is not unique.

Three raw tables share this machinery (`RAW_TABLES`). Each has its own
retention, and one free-disk guard bounds all three.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

import pandas as pd

from data_collection.stream.base import BookEvent

logger = logging.getLogger('spool')

# Absolute, not cwd-relative: a cwd-relative spool would land where nothing
# compacts it. `SPOOL_ROOT` overrides.
_TRADER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPOOL_ROOT = Path(os.getenv('SPOOL_ROOT')
                          or _TRADER_ROOT / 'data' / 'spool')

COLUMNS = ('venue', 'symbol', 'event_time', 'available_time', 'quality',
           'market_ticker', 'seq', 'kind', 'side', 'price', 'size', 'absolute')

# Retention and the disk guard. Read from the environment at call time.
DEFAULT_RAW_RETENTION_DAYS = 60.0
DEFAULT_INDEX_RETENTION_DAYS = 365.0
#: **`spot_trades` is a FEATURE SOURCE in waiting, not a replayable buffer.**
#:
#: It inherited `RAW_RETENTION_DAYS` (60) because it sits in the same raw tier
#: as `venue_book_events`, and that quietly capped it at ~5,760 windows -- the
#: floor of any usable `min_train_windows`, so it could never have supported a
#: walk-forward no matter how long it ran. History not kept cannot be
#: recovered, so the retention has to be right BEFORE the data accrues, not
#: after somebody wants it.
#:
#: It is the natural source for trade-level lead-lag, and `cross_asset` is the
#: group with the strongest independent prior support in this project
#: (+0.000183 at t=+3.39, 6/6 folds). Measured 2026-10-21 at ~20 MB/day, so a
#: year costs ~7 GB against 356 GB free.
DEFAULT_SPOT_TRADES_RETENTION_DAYS = 365.0
DEFAULT_MIN_FREE_FRACTION = 0.15
DEFAULT_MIN_FREE_GB = 20.0
# The guard never deletes the newest day of any table: a disk filled by
# something else must not cost the data being written now.
GUARD_FLOOR_DAYS = 1.0


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


def table_spool(root: Path | str, dataset: str, venue: str) -> FrameSpool:
    """The spool for a non-book raw table: `<root>/<dataset>/<venue>/`.

    Book frames keep their original `<root>/<venue>/` layout so files already on
    disk are still compacted after an upgrade.
    """
    return FrameSpool(Path(root) / dataset, venue)


class SafeSpool:
    """A spool that cannot raise into the loop feeding it.

    The trade tape and the index recorder share a process with the trading
    loop; a full disk or a permission error must cost rows, never a reader.
    After a failure the spool drops rows for `retry_after` seconds, then tries
    to reopen. Every drop is counted and the first of each outage is logged.
    """

    def __init__(self, factory: Callable[[], FrameSpool], *,
                 name: str = 'spool', retry_after: float = 60.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._factory = factory
        self._name = name
        self._retry_after = retry_after
        self._clock = clock
        self._spool: Optional[FrameSpool] = None
        self._failed_at: Optional[float] = None
        self.dropped = 0

    def _get(self) -> Optional[FrameSpool]:
        if self._spool is not None:
            return self._spool
        if (self._failed_at is not None
                and self._clock() - self._failed_at < self._retry_after):
            return None
        try:
            self._spool = self._factory()
        except Exception as exc:                          # noqa: BLE001
            self._fail(exc)
            return None
        if self._failed_at is not None:
            logger.warning('%s: writing again after dropping %d rows',
                           self._name, self.dropped)
        self._failed_at = None
        return self._spool

    def _fail(self, exc: BaseException) -> None:
        if self._failed_at is None:
            logger.error('%s: write failed (%s); dropping rows for %.0fs. '
                         'Recording is affected, nothing else is.',
                         self._name, str(exc)[:200], self._retry_after)
        spool, self._spool = self._spool, None
        if spool is not None:
            try:
                spool.close()
            except Exception:                             # noqa: BLE001
                pass
        self._failed_at = self._clock()

    def extend(self, records: Iterable[dict]) -> int:
        records = list(records)
        spool = self._get()
        if spool is None:
            self.dropped += len(records)
            return 0
        try:
            return spool.extend(records)
        except Exception as exc:                          # noqa: BLE001
            self._fail(exc)
            self.dropped += len(records)
            return 0

    def append(self, record: dict) -> None:
        self.extend([record])

    def flush(self) -> None:
        if self._spool is None:
            return
        try:
            self._spool.flush()
        except Exception as exc:                          # noqa: BLE001
            self._fail(exc)

    def close(self) -> None:
        spool, self._spool = self._spool, None
        if spool is not None:
            try:
                spool.close()
            except Exception:                             # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# The raw tables
# ---------------------------------------------------------------------------

def _utc(seconds: float) -> pd.Timestamp:
    return pd.Timestamp(seconds, unit='s', tz='UTC')


def _finite(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _book_row(rec: dict) -> dict:
    when = _utc(rec['t'])
    return {'venue': rec['venue'], 'symbol': rec['symbol'],
            'event_time': when, 'available_time': when, 'quality': 'valid',
            'market_ticker': rec['market_ticker'], 'seq': rec.get('seq'),
            'kind': rec['kind'], 'side': rec['side'],
            'price': rec['price'], 'size': rec['size'],
            'absolute': rec['absolute']}


def _trade_row(rec: dict) -> Optional[dict]:
    # `event_time` is the venue's print time, so a re-delivered trade lands on
    # the same key. `available_time` is when we saw it, never before the print.
    printed = _finite(rec.get('time'))
    if printed is None or rec.get('trade_id') in (None, ''):
        return None
    received = _finite(rec.get('t')) or printed
    return {'venue': rec.get('venue', 'coinbase'), 'symbol': rec['symbol'],
            'event_time': _utc(printed),
            'available_time': _utc(max(printed, received)),
            'quality': 'valid', 'product_id': rec['product_id'],
            'trade_id': str(rec['trade_id']), 'price': _finite(rec.get('price')),
            'size': _finite(rec.get('size')), 'side': rec.get('side'),
            'received_time': _utc(received),
            'sequence_num': rec.get('sequence_num')}


def _index_row(rec: dict) -> dict:
    sampled = float(rec['t'])
    computed = _finite(rec.get('computed')) or sampled
    row = {'venue': rec.get('venue', 'index_proxy'), 'symbol': rec['symbol'],
           'event_time': _utc(sampled),
           'available_time': _utc(max(sampled, computed)),
           'quality': 'valid', 'proxy': _finite(rec.get('proxy')),
           'n_exchanges': int(rec.get('n_exchanges') or 0)}
    for key, value in rec.items():
        if key.startswith('mid_'):
            row[key] = _finite(value)
    return row


@dataclass(frozen=True)
class RawTable:
    dataset: str
    to_row: Callable[[dict], Optional[dict]]
    # Columns that identify a within-file duplicate. A reconnect replays a
    # trade snapshot, so one print can appear twice in an hour's file.
    dedupe: tuple[str, ...] = ()
    retention_env: str = 'RAW_RETENTION_DAYS'
    default_retention: float = DEFAULT_RAW_RETENTION_DAYS
    # The disk guard sheds the lowest rank first: the biggest, most
    # replaceable tier first.
    shed_rank: int = 0


RAW_TABLES: dict[str, RawTable] = {
    'venue_book_events': RawTable('venue_book_events', _book_row, shed_rank=0),
    'spot_trades': RawTable('spot_trades', _trade_row,
                            dedupe=('product_id', 'trade_id'),
                            retention_env='SPOT_TRADES_RETENTION_DAYS',
                            default_retention=DEFAULT_SPOT_TRADES_RETENTION_DAYS,
                            shed_rank=1),
    'index_proxy_1s': RawTable('index_proxy_1s', _index_row,
                               dedupe=('symbol', 'event_time'),
                               retention_env='INDEX_PROXY_RETENTION_DAYS',
                               default_retention=DEFAULT_INDEX_RETENTION_DAYS,
                               shed_rank=2),
}


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.error('%s=%r is not a number; using %s', name, raw, default)
        return default


def retention_days(dataset: str) -> float:
    """Days of `dataset` to keep. 0 or less keeps everything."""
    table = RAW_TABLES[dataset]
    return _env_float(table.retention_env, table.default_retention)


def _spool_files(spool_root: Path, dataset: str) -> list[Path]:
    if dataset == 'venue_book_events':
        # Legacy layout, one level down. Other tables' directories are skipped
        # by name so their files are never read as book frames.
        return sorted(p for p in spool_root.glob('*/*.jsonl')
                      if p.parent.name not in RAW_TABLES)
    return sorted((spool_root / dataset).glob('*/*.jsonl'))


def compact_table(spool_root: Path | str, store_root: Path | str, dataset: str,
                  *, keep_days: float | None = None,
                  now: float | None = None) -> int:
    """Fold one table's closed hourly files into immutable Parquet, then remove them.

    A file whose hour has not ended is skipped: compacting it would archive a
    prefix of the stream and then delete the rest.
    """
    table = RAW_TABLES[dataset]
    spool_root, store_root = Path(spool_root), Path(store_root)
    stamp = time.time() if now is None else now
    open_hour = _hour_key(stamp)
    written = 0

    for path in _spool_files(spool_root, dataset):
        if path.stem >= open_hour:
            continue
        rows = []
        for line in path.open():
            try:
                row = table.to_row(json.loads(line))
            except (ValueError, KeyError, TypeError):
                continue
            if row is not None:
                rows.append(row)
        if rows:
            frame = pd.DataFrame(rows)
            if table.dedupe:
                frame = frame.drop_duplicates(list(table.dedupe), keep='first')
            if dataset == 'venue_book_events':
                frame = frame[list(COLUMNS)]
            day = path.stem[:8]
            out = (store_root / dataset / f'venue={path.parent.name}'
                   / f'date={day[:4]}-{day[4:6]}-{day[6:]}')
            out.mkdir(parents=True, exist_ok=True)
            target = out / f'{path.stem}.parquet'
            tmp = target.with_suffix('.parquet.tmp')
            frame.to_parquet(tmp, index=False, compression='zstd')
            tmp.replace(target)
            written += len(frame)
            logger.info('compacted %s/%s -> %s (%d rows)', dataset, path.name,
                        target.name, len(frame))
        path.unlink()

    keep = retention_days(dataset) if keep_days is None else keep_days
    # `prune` counts files; this returns rows.
    prune(store_root, keep_days=keep, now=stamp, dataset=dataset)
    return written


def compact(spool_root: Path | str, store_root: Path | str, *,
            keep_days: float | None = None, now: float | None = None) -> int:
    """The Kalshi book frames. `keep_days` defaults to `RAW_RETENTION_DAYS`.

    The raw stream is too large to keep forever; `venue_ladder` is the permanent tier.
    """
    return compact_table(spool_root, store_root, 'venue_book_events',
                         keep_days=keep_days, now=now)


def compact_all(spool_root: Path | str, store_root: Path | str, *,
                now: float | None = None,
                disk_usage: Callable = shutil.disk_usage) -> dict[str, int]:
    """Every raw table, then the disk guard. One table failing skips only itself."""
    rows: dict[str, int] = {}
    for dataset in RAW_TABLES:
        try:
            rows[dataset] = compact_table(spool_root, store_root, dataset, now=now)
        except Exception as exc:                          # noqa: BLE001
            logger.warning('compacting %s: %s', dataset, str(exc)[:200])
    try:
        guard_disk(store_root, now=now, disk_usage=disk_usage)
    except Exception as exc:                              # noqa: BLE001
        logger.error('disk guard: %s', str(exc)[:200])
    return rows


def _file_hour(path: Path) -> float:
    """The epoch start of the hour a compacted file holds; mtime as a fallback."""
    try:
        return datetime.strptime(path.stem, '%Y%m%dT%H').replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return path.stat().st_mtime


def _remove_empty_dirs(root: Path) -> None:
    for directory in sorted(root.rglob('*'), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()


def prune(store_root: Path | str, *, keep_days: float, now: float | None = None,
          dataset: str = 'venue_book_events') -> int:
    """Delete compacted Parquet older than `keep_days`. Returns files removed.

    Judged on the hour in the file's name, not its mtime: a restored or copied
    store would otherwise never age out.
    """
    if keep_days <= 0:
        return 0
    root = Path(store_root) / dataset
    if not root.exists():
        return 0
    cutoff = (time.time() if now is None else now) - keep_days * 86400.0
    removed = 0
    for path in root.rglob('*.parquet'):
        if _file_hour(path) + 3600.0 <= cutoff:
            path.unlink()
            removed += 1
    _remove_empty_dirs(root)
    return removed


def free_space_needed(total_bytes: float, *, min_free_fraction: float | None = None,
                      min_free_bytes: float | None = None) -> float:
    """The larger of a fraction of the volume and an absolute floor."""
    fraction = (_env_float('DISK_GUARD_MIN_FREE_FRACTION', DEFAULT_MIN_FREE_FRACTION)
                if min_free_fraction is None else min_free_fraction)
    floor = (_env_float('DISK_GUARD_MIN_FREE_GB', DEFAULT_MIN_FREE_GB) * 1e9
             if min_free_bytes is None else min_free_bytes)
    return max(fraction * float(total_bytes), floor)


def guard_disk(store_root: Path | str, *, min_free_fraction: float | None = None,
               min_free_bytes: float | None = None, now: float | None = None,
               disk_usage: Callable = shutil.disk_usage,
               floor_days: float = GUARD_FLOOR_DAYS) -> int:
    """Shed the oldest raw files until the volume has room again.

    Order: `shed_rank` (book frames, then trades, then the index proxy), oldest
    hour first within a table, never the newest `floor_days`. Returns files
    removed. Every deletion logs at ERROR, because retention is being cut short.
    """
    store_root = Path(store_root)
    usage = disk_usage(store_root)
    need = free_space_needed(usage.total, min_free_fraction=min_free_fraction,
                             min_free_bytes=min_free_bytes)
    if usage.free >= need:
        return 0
    logger.error('DISK GUARD: %.1f GB free of %.1f GB, below the %.1f GB '
                 'floor; pruning the oldest raw partitions',
                 usage.free / 1e9, usage.total / 1e9, need / 1e9)
    protect_from = (time.time() if now is None else now) - floor_days * 86400.0
    candidates = []
    for table in RAW_TABLES.values():
        root = store_root / table.dataset
        if not root.exists():
            continue
        for path in root.rglob('*.parquet'):
            hour = _file_hour(path)
            if hour < protect_from:
                candidates.append((table.shed_rank, hour, str(path)))
    candidates.sort()

    removed, freed = 0, False
    for _rank, hour, name in candidates:
        path = Path(name)
        try:
            path.unlink()
        except OSError as exc:
            logger.error('DISK GUARD: could not remove %s: %s', path, exc)
            continue
        removed += 1
        logger.error('DISK GUARD: removed %s/%s', path.parent.parent.parent.name,
                     path.name)
        if disk_usage(store_root).free >= need:
            freed = True
            break
    for table in RAW_TABLES.values():
        if (store_root / table.dataset).exists():
            _remove_empty_dirs(store_root / table.dataset)
    after = disk_usage(store_root).free
    if freed:
        logger.error('DISK GUARD: %d raw files removed; %.1f GB free',
                     removed, after / 1e9)
    else:
        logger.error('DISK GUARD: removed %d raw files and the volume is STILL '
                     'below its floor (%.1f GB free, %.1f GB wanted). What is '
                     'left is inside the %.0f-day floor or not raw data.',
                     removed, after / 1e9, need / 1e9, floor_days)
    return removed
