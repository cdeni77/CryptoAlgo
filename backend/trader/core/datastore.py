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

# Absolute, not cwd-relative. `data/research` resolved against whatever directory
# the process happened to start in, so running a script from the repo root rather
# than from backend/trader created a second, empty store and the first one simply
# went unread — the same hidden-second-copy failure as the compose named volume
# masking the host directory. RESEARCH_STORE still wins.
_TRADER_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.getenv('RESEARCH_STORE') or _TRADER_ROOT / 'data' / 'research')

# Datasets and the columns they must carry. `event_time` and `available_time`
# are required everywhere — they are what makes a point-in-time read possible.
SCHEMAS: dict[str, tuple[str, ...]] = {
    # The dataset this system reads. One-minute bars, and the timeframe is in
    # the dataset name rather than a column because the whole pipeline runs on
    # one timeframe — a `timeframe` column would be constant, and the previous
    # `bars` dataset had no such column at all, which made its implicit hourly
    # granularity a fact you had to know rather than one you could read.
    'minute_bars': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
    ),
    # `bars` is the hourly perp and spot history the previous system read. Kept
    # in the schema because a store on disk may still hold it and an unreadable
    # partition is worse than an unused one; written by nothing.
    'bars': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
    ),
    # The venue's own quote at each decision offset, backfilled from Kalshi's
    # candlesticks. This is the only dataset here that is not derived from
    # Coinbase, and it is the one the market benchmark needs: a backtest built
    # from bars has no order book, so `price_source` substitutes the calibrated
    # baseline for the market and "beat the price" and "beat the baseline" become
    # one question answered twice with the same number.
    #
    # `event_time` is the offset instant (window_open + offset minutes), which is
    # what the quote describes and what the partitioning wants. `window_open` is
    # carried separately because it is the join key against `core/windows.py`.
    'venue_quotes': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'window_open', 'offset_minutes', 'market_ticker',
        'yes_bid', 'yes_ask', 'market_probability', 'spread',
        'volume', 'open_interest', 'usable', 'exclude_reason',
    ),
    'book_snapshots': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'bid', 'ask', 'bid_size', 'ask_size', 'depth_1pct_bid', 'depth_1pct_ask',
    ),
}

# Ordered worst to best. `data_collection.ingest` stamps these from the
# validator; carrying them through means a feature build can exclude flagged
# data instead of silently averaging it in. A 50%-per-hour funding rate is
# stored as SUSPICIOUS rather than dropped, and it has no business reaching the
# carry features.
QUALITY_LEVELS = ('invalid', 'unvalidated', 'suspicious', 'valid')
DEFAULT_MIN_QUALITY = 'valid'

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


def _in_utc(frame: pd.DataFrame) -> pd.DataFrame:
    """Force every timezone-aware column to UTC.

    DuckDB returns `TIMESTAMPTZ` in the *process's local* timezone, so the same
    Parquet file read on two machines produced two different `event_time`
    representations — and the trader container sets `TZ=America/New_York`.
    Nothing downstream re-normalised: `core/features.py` derives minute-of-day
    straight off this index for the intraday seasonality and for
    `us_equity_hours`, whose bounds are written as UTC minutes.

    So the research path indexed seasonality in local time while
    `scripts/live.py:fetch_bars` builds its index with an explicit `tz='UTC'`.
    That is a train/serve skew in a fitted object, and it moves with DST twice a
    year. Measured elsewhere in this audit: the same bars under an
    `America/New_York` index shift the fitted seasonal peak by 353 minutes.

    Pinned here rather than at each call site, because the point is that no
    consumer should have to remember.
    """
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].dt.tz_convert('UTC')
    return frame


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

        # An unstamped row is unvalidated by definition, never valid by default.
        if 'quality' not in out.columns:
            out['quality'] = 'unvalidated'
        out['quality'] = out['quality'].fillna('unvalidated').astype(str).str.lower()

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
        min_quality: str | None = DEFAULT_MIN_QUALITY,
    ) -> pd.DataFrame:
        """Read a dataset as of a point in time.

        `as_of` is the whole point: it drops revisions published after the moment
        being simulated, then returns the revision of each event that was current
        at that moment. Omitting it reads the newest revision of everything,
        which is right for feature research and never right for a backtest.

        One row per `(venue, symbol, event_time)` comes back either way.

        `min_quality` defaults to 'valid', so flagged data is excluded unless
        asked for. Pass a lower level to include it, or None for everything.
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
        # Bound as plain `datetime`, not `pd.Timestamp`. Under pandas 3.0 the
        # Timestamp is no longer implicitly convertible by the driver's binder and
        # every `--start`/`--end` run raised, which meant no run could be
        # date-limited at all — including the small ones you would use to check a
        # change quickly.
        if start is not None:
            clauses.append('event_time >= ?')
            params.append(pd.Timestamp(start, tz='UTC').to_pydatetime())
        if end is not None:
            clauses.append('event_time <= ?')
            params.append(pd.Timestamp(end, tz='UTC').to_pydatetime())
        if as_of is not None:
            clauses.append('available_time <= ?')
            params.append(pd.Timestamp(as_of, tz='UTC').to_pydatetime())
        if min_quality is not None:
            if min_quality not in QUALITY_LEVELS:
                raise DataStoreError(
                    f"Unknown quality {min_quality!r}; expected one of {QUALITY_LEVELS}"
                )
            acceptable = QUALITY_LEVELS[QUALITY_LEVELS.index(min_quality):]
            clauses.append(f"quality IN ({', '.join('?' for _ in acceptable)})")
            params.extend(acceptable)

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
            return _in_utc(con.execute(sql, [glob, *params]).df())

    def panel(
        self,
        symbols: Sequence[str],
        *,
        venue: str,
        field: str = 'close',
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        as_of: pd.Timestamp | str | None = None,
        min_quality: str | None = DEFAULT_MIN_QUALITY,
    ) -> pd.DataFrame:
        """One field for many symbols as a time x symbol frame.

        This is the shape the cross-sectional features and the pooled model want:
        rank and z-score across the row, not per-symbol loops.
        """
        long = self.read(
            'bars', venue=venue, symbols=symbols, start=start, end=end, as_of=as_of,
            columns=('symbol', 'event_time', field), min_quality=min_quality,
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
        frame = self.read(
            dataset, columns=('venue', 'symbol', 'event_time'), min_quality=None
        )
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


# Which research dataset a scraped timeframe belongs in. The 15-minute binary
# system reads `minute_bars` and nothing else; anything coarser is archive.
BARS_DATASET_BY_TIMEFRAME = {'1m': 'minute_bars'}


def from_sqlite(
    store: ResearchStore,
    sqlite_path: str | Path,
    *,
    venue: str,
    timeframe: str = '1m',
    symbols: Optional[Iterable[str]] = None,
    include_archive: bool = True,
) -> dict[str, int]:
    """Migrate the scraper's SQLite tables into the research store.

    The scraper's schema already carries `event_time`/`available_time`; this
    preserves both rather than collapsing to one, so migrated history supports
    point-in-time reads too.

    `venue` is the caller's label for rows that recorded none. Rows that did keep
    theirs — which is what the venue column is for, now that it is part of the
    scraper's unique key.
    """
    import sqlite3

    def venue_expression(table: str, fallbacks: tuple[str, ...]) -> str:
        """`venue` where the table has it, else the best legacy proxy.

        Funding and open interest gained a venue column late. Before that,
        funding's origin was inferred from `funding_source` and open interest's
        from `source` — which held the *client library* ('ccxt'), not an exchange.
        Older databases still read from those, so the fallback chain stays.
        """
        cursor = con.execute(f'PRAGMA table_info({table})')
        present = {row[1] for row in cursor.fetchall()}
        candidates = [c for c in ('venue', *fallbacks) if c in present]
        if not candidates:
            return "'unknown' AS row_venue"
        return f"COALESCE({', '.join(candidates)}, 'unknown') AS row_venue"

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
            "SELECT symbol, event_time, available_time, quality, open, high, low, "
            "close, volume, quote_volume, trade_count, "
            + venue_expression('ohlcv', ())
            + f" FROM ohlcv WHERE timeframe = ?{filt}",
            con, params=params,
        )
        if not bars.empty:
            # Prefer the venue recorded per row; fall back to the caller's label
            # for rows written before the column existed.
            bars['venue'] = bars['row_venue'].where(
                bars['row_venue'] != 'unknown', venue
            )
            bars = bars.drop(columns=['row_venue'])
            dataset = BARS_DATASET_BY_TIMEFRAME.get(timeframe, 'bars')
            counts[dataset] = store.write(dataset, bars)

        # Funding and open interest are archive: nothing in the binary system
        # reads them, and no endpoint serves either historically, so they cannot
        # be re-fetched at any price. They are migrated anyway, because the
        # SQLite database is the only other copy and deleting it after a sync
        # that skipped them would destroy the only irreplaceable data here.
        def has_table(name: str) -> bool:
            row = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
            ).fetchone()
            return row is not None

        if not include_archive:
            return counts

        funding = pd.read_sql_query(
            "SELECT symbol, event_time, available_time, quality, rate, mark_price, "
            "index_price, is_settlement, "
            + venue_expression('funding_rates', ('funding_source',))
            + " FROM funding_rates",
            con,
        ) if has_table('funding_rates') else pd.DataFrame()
        if not funding.empty:
            funding['venue'] = funding['row_venue'].where(
                funding['row_venue'] != 'unknown', venue
            )
            funding = funding.drop(columns=['row_venue'])
            funding['interval_hours'] = 1
            counts['funding'] = store.write('funding', funding)

        oi = pd.read_sql_query(
            "SELECT symbol, event_time, available_time, quality, "
            "open_interest_contracts AS oi_contracts, open_interest_base AS oi_base, "
            "open_interest_usd AS oi_usd, "
            + venue_expression('open_interest', ('source',))
            + " FROM open_interest",
            con,
        ) if has_table('open_interest') else pd.DataFrame()
        if not oi.empty:
            oi['venue'] = oi['row_venue'].where(oi['row_venue'] != 'unknown', venue)
            oi = oi.drop(columns=['row_venue'])
            counts['open_interest'] = store.write('open_interest', oi)
    finally:
        con.close()

    return counts
