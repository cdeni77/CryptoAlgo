"""Research data store: Parquet on disk, DuckDB for queries.

Two stores exist in this system and they have different jobs:

* **This one** is the research store. Immutable Parquet partitioned by dataset,
  venue, symbol and month. It answers "what did the market look like at time t"
  for feature building, backtests and simulation. No server, columnar, and fast
  enough that a full-history feature build is not an afternoon.
* **`core.pg_writer`** is the serving store. PostgreSQL, mutable, holds signals,
  trades and paper state, and is what the API and frontend read. Untouched here.

Two invariants make the research store trustworthy:

**Venue is part of the key.** Bars from Coinbase and bars from Binance are
different series and are stored as such. The pipeline previously stored Binance
perp data under Coinbase contract codes, which silently made every funding and
open-interest feature describe the wrong book.

**Reads are point-in-time.** Every table carries `event_time` (when the thing
happened) and `available_time` (when we could first have known it). `as_of`
filters on `available_time`, so a backtest cannot see a revision that had not
been published yet. Funding and OI in particular are revised.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_ROOT = Path(os.getenv('RESEARCH_STORE', 'data/research'))

# Datasets and the columns they must carry. `event_time` and `available_time`
# are required everywhere — they are what makes a point-in-time read possible.
SCHEMAS: dict[str, tuple[str, ...]] = {
    'bars': (
        'venue', 'symbol', 'event_time', 'available_time',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
    ),
    'funding': (
        'venue', 'symbol', 'event_time', 'available_time',
        'rate', 'mark_price', 'index_price', 'interval_hours', 'is_settlement',
    ),
    'open_interest': (
        'venue', 'symbol', 'event_time', 'available_time',
        'oi_contracts', 'oi_base', 'oi_usd',
    ),
    'book_snapshots': (
        'venue', 'symbol', 'event_time', 'available_time',
        'bid', 'ask', 'bid_size', 'ask_size', 'depth_1pct_bid', 'depth_1pct_ask',
    ),
}

TIME_COLUMNS = ('event_time', 'available_time')

# The store is bitemporal: one event can have several revisions, distinguished by
# when each was published. All revisions are kept, so a read at any `as_of` sees
# exactly the revision that was current then.
EVENT_KEY = ('venue', 'symbol', 'event_time')
REVISION_KEY = ('venue', 'symbol', 'event_time', 'available_time')


class DataStoreError(RuntimeError):
    """Raised when a write would corrupt the store's invariants."""


@dataclass(frozen=True)
class Partition:
    dataset: str
    venue: str
    symbol: str
    month: str          # 'YYYY-MM'

    @property
    def relative(self) -> Path:
        return Path(
            self.dataset,
            f"venue={self.venue}",
            f"symbol={self.symbol}",
            f"month={self.month}",
            "data.parquet",
        )


class ResearchStore:
    """Parquet-backed research store with point-in-time reads."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or DEFAULT_ROOT)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ------------------------------------------------------------

    def write(self, dataset: str, frame: pd.DataFrame, *, overwrite: bool = False) -> int:
        """Write rows into their (venue, symbol, month) partitions.

        Existing partitions are merged and de-duplicated on the full revision
        key, so re-writing identical rows is idempotent while a genuine revision
        is appended rather than replacing its predecessor. Discarding
        superseded revisions would make early point-in-time reads return
        nothing. Pass `overwrite` to replace a partition outright.
        """
        if dataset not in SCHEMAS:
            raise DataStoreError(f"Unknown dataset {dataset!r}; known: {sorted(SCHEMAS)}")
        if frame.empty:
            return 0

        prepared = self._prepare(dataset, frame)
        written = 0

        grouped = prepared.groupby(
            ['venue', 'symbol', prepared['event_time'].dt.strftime('%Y-%m')],
            sort=False,
        )
        for (venue, symbol, month), chunk in grouped:
            part = Partition(dataset, str(venue), str(symbol), str(month))
            path = self.root / part.relative
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists() and not overwrite:
                chunk = self._merge(pd.read_parquet(path), chunk)

            chunk = chunk.sort_values('event_time').reset_index(drop=True)
            tmp = path.with_suffix('.parquet.tmp')
            chunk.to_parquet(tmp, index=False, compression='zstd')
            os.replace(tmp, path)
            written += len(chunk)

        return written

    def _prepare(self, dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        """Validate and normalise a frame against its dataset schema."""
        columns = SCHEMAS[dataset]
        out = frame.copy()

        # `event_time` may arrive as the index.
        if 'event_time' not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.index.name or 'index': 'event_time'})

        missing = {'venue', 'symbol', 'event_time'} - set(out.columns)
        if missing:
            raise DataStoreError(f"{dataset}: missing required columns {sorted(missing)}")

        for col in TIME_COLUMNS:
            if col in out.columns:
                out[col] = pd.to_datetime(out[col], utc=True)

        # Absent an explicit publication time, assume the data was knowable when
        # the event closed. Conservative for bars; callers should set it
        # explicitly for revised series like funding and OI.
        if 'available_time' not in out.columns:
            out['available_time'] = out['event_time']

        for col in columns:
            if col not in out.columns:
                out[col] = pd.NA

        out['venue'] = out['venue'].astype(str).str.lower()
        out['symbol'] = out['symbol'].astype(str).str.upper()

        if (out['available_time'] < out['event_time']).any():
            raise DataStoreError(
                f"{dataset}: available_time precedes event_time — a read at t would "
                "see data published after t"
            )

        return out[list(columns)]

    @staticmethod
    def _merge(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
        """Union two partitions, keeping every distinct revision."""
        for col in TIME_COLUMNS:
            if col in existing.columns:
                existing[col] = pd.to_datetime(existing[col], utc=True)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.sort_values(['event_time', 'available_time'])
        return combined.drop_duplicates(subset=list(REVISION_KEY), keep='last')

    # -- reading ------------------------------------------------------------

    def read(
        self,
        dataset: str,
        *,
        venue: str | None = None,
        symbols: Sequence[str] | None = None,
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        as_of: pd.Timestamp | str | None = None,
        columns: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Read a dataset as of a point in time.

        `as_of` is the whole point: it drops revisions published after the moment
        being simulated, then returns the revision of each event that was current
        at that moment. Omitting it reads the newest revision of everything,
        which is right for feature research and never right for a backtest.

        One row per `(venue, symbol, event_time)` comes back either way.
        """
        if dataset not in SCHEMAS:
            raise DataStoreError(f"Unknown dataset {dataset!r}")

        glob = str(self.root / dataset / '**' / '*.parquet')
        if not any((self.root / dataset).rglob('*.parquet')):
            return pd.DataFrame(columns=list(columns or SCHEMAS[dataset]))

        select = ', '.join(columns) if columns else '*'
        clauses: list[str] = []
        params: list[Any] = []

        if venue:
            clauses.append('venue = ?')
            params.append(venue.lower())
        if symbols:
            placeholders = ', '.join('?' for _ in symbols)
            clauses.append(f'symbol IN ({placeholders})')
            params.extend(s.upper() for s in symbols)
        if start is not None:
            clauses.append('event_time >= ?')
            params.append(pd.Timestamp(start, tz='UTC'))
        if end is not None:
            clauses.append('event_time <= ?')
            params.append(pd.Timestamp(end, tz='UTC'))
        if as_of is not None:
            clauses.append('available_time <= ?')
            params.append(pd.Timestamp(as_of, tz='UTC'))

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
        # Rank revisions per event and keep the newest one still visible at
        # `as_of`, which the WHERE clause has already bounded.
        sql = (
            f"SELECT {select} FROM ("
            f"  SELECT * EXCLUDE (_rn) FROM ("
            f"    SELECT *, ROW_NUMBER() OVER ("
            f"      PARTITION BY {', '.join(EVENT_KEY)} ORDER BY available_time DESC"
            f"    ) AS _rn"
            f"    FROM read_parquet(?, hive_partitioning = false) {where}"
            f"  ) WHERE _rn = 1"
            f") ORDER BY symbol, event_time"
        )
        with duckdb.connect() as con:
            return con.execute(sql, [glob, *params]).df()

    def panel(
        self,
        symbols: Sequence[str],
        *,
        venue: str,
        field: str = 'close',
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        as_of: pd.Timestamp | str | None = None,
    ) -> pd.DataFrame:
        """One field for many symbols as a time x symbol frame.

        This is the shape the cross-sectional features and the pooled model want:
        rank and z-score across the row, not per-symbol loops.
        """
        long = self.read(
            'bars', venue=venue, symbols=symbols, start=start, end=end, as_of=as_of,
            columns=('symbol', 'event_time', field),
        )
        if long.empty:
            return pd.DataFrame()
        wide = long.pivot_table(index='event_time', columns='symbol', values=field, aggfunc='last')
        wide.index = pd.to_datetime(wide.index, utc=True)
        return wide.sort_index()

    def coverage(self, dataset: str) -> pd.DataFrame:
        """Row count and time span per (venue, symbol) — the first thing to check.

        This is what answers "do I actually have Coinbase history for this
        contract, or have I been training on a proxy".
        """
        frame = self.read(dataset, columns=('venue', 'symbol', 'event_time'))
        if frame.empty:
            return pd.DataFrame(columns=['venue', 'symbol', 'rows', 'start', 'end', 'days'])
        grouped = frame.groupby(['venue', 'symbol'])['event_time']
        out = grouped.agg(rows='count', start='min', end='max').reset_index()
        out['days'] = (out['end'] - out['start']).dt.total_seconds() / 86400.0
        return out.sort_values(['venue', 'symbol']).reset_index(drop=True)

    # -- feature matrices ---------------------------------------------------

    def write_features(
        self,
        frame: pd.DataFrame,
        *,
        name: str,
        meta: Optional[dict[str, Any]] = None,
    ) -> tuple[Path, str]:
        """Materialise a feature matrix alongside a content hash.

        The hash covers the column names and the values, so a model artifact can
        record exactly which feature matrix it trained on. When a feature
        definition changes, the hash changes, and a stale model is detectable
        instead of quietly scoring against different inputs.
        """
        directory = self.root / 'features' / name
        directory.mkdir(parents=True, exist_ok=True)

        digest = feature_hash(frame)
        path = directory / f"{digest}.parquet"
        if not path.exists():
            tmp = path.with_suffix('.parquet.tmp')
            frame.to_parquet(tmp, compression='zstd')
            os.replace(tmp, path)

        payload = {
            'name': name,
            'hash': digest,
            'rows': int(len(frame)),
            'columns': list(map(str, frame.columns)),
            **(meta or {}),
        }
        (directory / f"{digest}.json").write_text(json.dumps(payload, indent=2, default=str))
        return path, digest

    def read_features(self, name: str, digest: str) -> pd.DataFrame:
        path = self.root / 'features' / name / f"{digest}.parquet"
        if not path.exists():
            raise DataStoreError(f"No feature matrix {name}/{digest}")
        return pd.read_parquet(path)

    def drop(self, dataset: str) -> None:
        """Remove a dataset entirely. Used by tests and rebuilds."""
        target = self.root / dataset
        if target.exists():
            shutil.rmtree(target)


def feature_hash(frame: pd.DataFrame) -> str:
    """Stable content hash of a feature matrix (columns + values)."""
    hasher = hashlib.sha256()
    hasher.update(json.dumps(list(map(str, frame.columns))).encode())
    hasher.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return hasher.hexdigest()[:16]


def from_sqlite(
    store: ResearchStore,
    sqlite_path: str | Path,
    *,
    venue: str,
    timeframe: str = '1h',
    symbols: Optional[Iterable[str]] = None,
) -> dict[str, int]:
    """Migrate the scraper's SQLite tables into the research store.

    The scraper's schema already carries `event_time`/`available_time`; this
    preserves both rather than collapsing to one, so migrated history supports
    point-in-time reads too.
    """
    import sqlite3

    counts: dict[str, int] = {}
    con = sqlite3.connect(str(sqlite_path))
    try:
        filt = ''
        params: list[Any] = [timeframe]
        if symbols:
            syms = list(symbols)
            filt = f" AND symbol IN ({', '.join('?' for _ in syms)})"
            params.extend(syms)

        bars = pd.read_sql_query(
            "SELECT symbol, event_time, available_time, open, high, low, close, "
            "volume, quote_volume, trade_count FROM ohlcv "
            f"WHERE timeframe = ?{filt}",
            con, params=params,
        )
        if not bars.empty:
            bars['venue'] = venue
            counts['bars'] = store.write('bars', bars)

        funding = pd.read_sql_query(
            "SELECT symbol, event_time, available_time, rate, mark_price, "
            "index_price, is_settlement FROM funding_rates", con,
        )
        if not funding.empty:
            funding['venue'] = venue
            funding['interval_hours'] = 1
            counts['funding'] = store.write('funding', funding)

        oi = pd.read_sql_query(
            "SELECT symbol, event_time, available_time, "
            "open_interest_contracts AS oi_contracts, open_interest_base AS oi_base, "
            "open_interest_usd AS oi_usd FROM open_interest", con,
        )
        if not oi.empty:
            oi['venue'] = venue
            counts['open_interest'] = store.write('open_interest', oi)
    finally:
        con.close()

    return counts
