"""Research data store: Parquet on disk, DuckDB for queries.

This is the research store: immutable Parquet partitioned by dataset, venue,
symbol and month. `core.pg_writer` is the separate serving store (PostgreSQL).

Invariants:

* Venue is part of the key. Bars from different venues are different series.
* Reads are point-in-time. Every table carries `event_time` (when it happened)
  and `available_time` (when we could first know it). `as_of` filters on
  `available_time`, so a backtest never sees an unpublished revision.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# Absolute, not cwd-relative, so a script started elsewhere cannot create a
# second, empty store. RESEARCH_STORE overrides.
_TRADER_ROOT = Path(__file__).resolve().parents[1]


def default_root() -> Path:
    """The store root, resolved at call time, not at import.

    Scripts set `RESEARCH_STORE` after importing this module, so an import-time
    value would ignore their `--research-store` flag.
    """
    return Path(os.getenv('RESEARCH_STORE') or _TRADER_ROOT / 'data' / 'research')


# Kept because several modules import it. It is the import-time value; anything
# that must honour a later override calls `default_root()`.
DEFAULT_ROOT = default_root()

# Datasets and the columns they must carry. `event_time` and `available_time`
# are required everywhere: they make point-in-time reads possible.
SCHEMAS: dict[str, tuple[str, ...]] = {
    # One-minute bars: the dataset this system reads. The timeframe is in the
    # dataset name, not a column.
    'minute_bars': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
    ),
    # Hourly history from a retired system. Kept so stored partitions stay
    # readable; nothing writes it.
    'bars': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'open', 'high', 'low', 'close', 'volume', 'quote_volume', 'trade_count',
    ),
    # The venue's quote at each decision offset, backfilled from Kalshi
    # candlesticks. `event_time` is the offset instant (window_open + offset);
    # `window_open` is the join key against `core/windows.py`.
    'venue_quotes': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'window_open', 'offset_minutes', 'market_ticker',
        'yes_bid', 'yes_ask', 'market_probability', 'spread',
        'volume', 'open_interest', 'usable', 'exclude_reason',
    ),
    # The raw ladder, both sides, every minute a market is open.
    # It cannot be backfilled: the orderbook endpoint returns empty once a market
    # settles, and no historical endpoint carries size. Store every level so any
    # later question about book shape stays answerable.
    'venue_ladder': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'window_open', 'minute_into_window',
        'yes_levels', 'no_levels', 'yes_total', 'no_total',
        # How this ladder reached us, and the cache age when sampled.
        # `transport` is part of the event key: REST and WS rows for one minute
        # are independent observations, not a revision of each other.
        'transport', 'book_age_ms',
    ),
    # The Polymarket ladder in the same columns, so the venues join on
    # (symbol, window_open, minute). `outcomes` proves which token is "Up";
    # `token_id_up` is the token the book was read from.
    'pm_ladder': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'window_open', 'minute_into_window',
        'yes_levels', 'no_levels', 'yes_total', 'no_total',
        'outcomes', 'token_id_up',
        # Keep in step with venue_ladder.
        'transport', 'book_age_ms',
    ),
    # The raw stream, one row per book level per frame.
    # Not written through `write()`, which rewrites a whole partition per call
    # and cannot keep up with the frame rate. `core/spool.compact` writes
    # immutable hour-named Parquet files, which `read()` finds by glob.
    # Typed columns, not JSON text: they are all of a delta's fields and compress far better.
    'venue_book_events': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'seq', 'kind', 'side', 'price', 'size', 'absolute',
    ),
    # The venue's own settlement: the authority on the label. Our target is
    # built from Coinbase bars, a proxy for CF Benchmarks BRTI.
    # One schema for both venues: Kalshi's `result` ('yes'/'no') and
    # Polymarket's `winning_side` ('A'/'B') both reduce to `settled_up`.
    'venue_settlements': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'window_open', 'close_time', 'settlement_time',
        'result', 'settled_up', 'volume', 'open_interest', 'last_price',
    ),
    # The market's forward volatility, from inverting Kalshi's KXBTCD threshold
    # ladder. `core/vol.py` estimates sigma only from past returns.
    # `r2` says whether the ladder was consistent enough for the fit to mean anything.
    'venue_implied_vol': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'event_ticker', 'close_time', 'minutes_to_close',
        'implied_sigma_per_min', 'implied_spot', 'atm_strike', 'n_strikes',
        'r2',
    ),
    # The one summarised book, for both venues and both provenances, at every
    # minute of the window. `scripts/build_depth.py` is the only writer.
    # `source` separates a recorded book from a reconstructed one.
    # `quote_age_seconds` is required: Predexon serves book changes, so a quiet
    # book is carried forward, and a stale price compared with a fresh forecast
    # manufactures edge.
    'venue_depth': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'market_ticker', 'window_open', 'offset_minutes',
        'yes_bid', 'yes_ask', 'yes_bid_size', 'yes_ask_size',
        'depth_bid_1c', 'depth_bid_5c', 'depth_ask_1c', 'depth_ask_5c',
        'depth_bid_total', 'depth_ask_total',
        'levels_bid', 'levels_ask', 'seq', 'gaps', 'source',
        'quote_age_seconds',
        # Which sampler produced the ladder half of the row, as against the
        # `source` of the row itself. Null on older rows.
        'transport',
    ),
    # Every Coinbase trade the spot stream saw, written by `core/spool`, never
    # through `write()`. `event_time` is the print, `available_time` our receipt.
    # Bounded by RAW_RETENTION_DAYS.
    'spot_trades': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'product_id', 'trade_id', 'price', 'size', 'side', 'received_time',
        'sequence_num',
    ),
    # A per-second median of constituent-exchange mids, standing in for the CF
    # Benchmarks index Kalshi settles on (`core/index_proxy.py`). Written by
    # `core/spool`; `event_time` is the sampling instant.
    'index_proxy_1s': (
        'venue', 'symbol', 'event_time', 'available_time', 'quality',
        'proxy', 'n_exchanges',
        'mid_coinbase', 'mid_kraken', 'mid_bitstamp', 'mid_gemini',
    ),
}

# Ordered worst to best. `data_collection.ingest` stamps these from the
# validator, so a feature build can exclude flagged data.
QUALITY_LEVELS = ('invalid', 'unvalidated', 'suspicious', 'valid')
DEFAULT_MIN_QUALITY = 'valid'

TIME_COLUMNS = ('event_time', 'available_time')

# The store is bitemporal: one event can have several revisions, keyed by
# publication time. All revisions are kept, so a read at any `as_of` sees the
# revision current then.
EVENT_KEY = ('venue', 'symbol', 'event_time')

# Datasets where one event is observed by more than one instrument.
# `read` keeps one row per event key (the latest `available_time`), so a
# revision supersedes its predecessor. Independent observers of the same event
# (live versus backfill, REST versus WS) need their own key column, or one
# silently hides the other. Revisions still collapse within an observer.
EVENT_KEY_EXTRA: dict[str, tuple[str, ...]] = {
    # `offset_minutes` as well as `source`: `event_time` is
    # `window_open + offset_minutes`, so minute 15 of one window and minute 0 of
    # the next share an instant but are different markets.
    # `transport` is not in this key: older rows lack the column, and a missing
    # key column makes `read` fail. Adding it needs a backfill first.
    'venue_depth': ('source', 'offset_minutes'),
    # REST and WS samplers both write the same minute; comparing them needs both rows.
    'venue_ladder': ('transport',),
    'pm_ladder': ('transport',),
    # Every level of a snapshot shares one `seq`, so side and price complete the key.
    'venue_book_events': ('market_ticker', 'seq', 'side', 'price'),
    # The venue's trade id makes a print unique, so a re-delivered trade
    # collapses to one row.
    'spot_trades': ('product_id', 'trade_id'),
}


def event_key(dataset: str) -> tuple[str, ...]:
    """The columns that identify one observation of one event."""
    return EVENT_KEY + EVENT_KEY_EXTRA.get(dataset, ())
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

    DuckDB returns `TIMESTAMPTZ` in the process's local timezone, and the
    container sets `TZ=America/New_York`. Minute-of-day features and the live
    path assume UTC, so every read is normalised here.
    """
    for column in frame.columns:
        dtype = frame[column].dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            frame[column] = frame[column].dt.tz_convert('UTC')
    return frame


def _utc_bound(value) -> datetime:
    """A query bound as a plain UTC `datetime`, from naive or aware input.

    The DuckDB binder needs a plain `datetime`, not `pd.Timestamp`. An aware
    bound is converted, never relabelled.
    """
    stamp = pd.Timestamp(value)
    stamp = stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')
    return stamp.to_pydatetime()


class ResearchStore:
    """Parquet-backed research store with point-in-time reads."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or default_root())
        self.root.mkdir(parents=True, exist_ok=True)

    # -- writing ------------------------------------------------------------

    def write(self, dataset: str, frame: pd.DataFrame, *, overwrite: bool = False) -> int:
        """Write rows into their (venue, symbol, month) partitions.

        Existing partitions are merged and de-duplicated on the full revision
        key: identical rows are idempotent, and a real revision is appended,
        not replaced, so early point-in-time reads still work.
        Pass `overwrite` to replace a partition outright.
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
                chunk = self._merge(pd.read_parquet(path), chunk,
                                    event_key(dataset) + ('available_time',))

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

        # Without an explicit publication time, assume the data was knowable
        # when the event closed. Revised series must set it explicitly.
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
    def _merge(existing: pd.DataFrame, incoming: pd.DataFrame,
               revision_key: Sequence[str] = REVISION_KEY) -> pd.DataFrame:
        """Union two partitions, keeping every distinct revision."""
        for col in TIME_COLUMNS:
            if col in existing.columns:
                existing[col] = pd.to_datetime(existing[col], utc=True)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.sort_values(['event_time', 'available_time'])
        return combined.drop_duplicates(subset=list(revision_key), keep='last')

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

        `as_of` drops revisions published after that moment, then returns the
        revision of each event current then. Without it, the newest revision is
        returned: fine for research, never for a backtest.

        One row per event key comes back either way. `min_quality` defaults to
        'valid'; pass a lower level to include flagged data, or None for everything.
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
        # Bound as plain `datetime`, not `pd.Timestamp`: see `_utc_bound`.
        if start is not None:
            clauses.append('event_time >= ?')
            params.append(_utc_bound(start))
        if end is not None:
            clauses.append('event_time <= ?')
            params.append(_utc_bound(end))
        if as_of is not None:
            clauses.append('available_time <= ?')
            params.append(_utc_bound(as_of))
        if min_quality is not None:
            if min_quality not in QUALITY_LEVELS:
                raise DataStoreError(
                    f"Unknown quality {min_quality!r}; expected one of {QUALITY_LEVELS}"
                )
            acceptable = QUALITY_LEVELS[QUALITY_LEVELS.index(min_quality):]
            clauses.append(f"quality IN ({', '.join('?' for _ in acceptable)})")
            params.extend(acceptable)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ''
        # `union_by_name`, because older partitions lack newer columns, and a
        # schema mismatch in the glob would make the whole dataset unreadable.
        # Rank revisions per event and keep the newest one visible at `as_of`
        # (the WHERE clause already bounds it).
        sql = (
            f"SELECT {select} FROM ("
            f"  SELECT * EXCLUDE (_rn) FROM ("
            f"    SELECT *, ROW_NUMBER() OVER ("
            f"      PARTITION BY {', '.join(event_key(dataset))} "
            f"ORDER BY available_time DESC"
            f"    ) AS _rn"
            f"    FROM read_parquet(?, hive_partitioning = false, "
            f"union_by_name = true) {where}"
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
        """One field for many symbols as a time x symbol frame."""
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
        """Row count and time span per (venue, symbol)."""
        frame = self.read(
            dataset, columns=('venue', 'symbol', 'event_time', 'quality'),
            min_quality=None
        )
        if frame.empty:
            return pd.DataFrame(columns=['venue', 'symbol', 'rows', 'usable',
                                         'demoted', 'start', 'end', 'days'])
        grouped = frame.groupby(['venue', 'symbol'])['event_time']
        out = grouped.agg(rows='count', start='min', end='max').reset_index()
        out['days'] = (out['end'] - out['start']).dt.total_seconds() / 86400.0
        # `rows` is every row; `usable` is what a default (`min_quality='valid'`)
        # read returns. `demoted` shows bars the validator has flagged.
        if 'quality' in frame.columns:
            usable = (frame[frame['quality'].astype(str) == 'valid']
                      .groupby(['venue', 'symbol'])['event_time'].count()
                      .rename('usable').reset_index())
            out = out.merge(usable, on=['venue', 'symbol'], how='left')
            out['usable'] = out['usable'].fillna(0).astype(int)
        else:
            out['usable'] = out['rows']
        out['demoted'] = out['rows'] - out['usable']
        cols = ['venue', 'symbol', 'rows', 'usable', 'demoted',
                'start', 'end', 'days']
        return (out[cols].sort_values(['venue', 'symbol'])
                .reset_index(drop=True))

    def drop(self, dataset: str) -> None:
        """Remove a dataset entirely. Used by tests and rebuilds."""
        target = self.root / dataset
        if target.exists():
            shutil.rmtree(target)


# Which research dataset a scraped timeframe belongs in. Only `minute_bars` is read.
BARS_DATASET_BY_TIMEFRAME = {'1m': 'minute_bars'}


def from_sqlite(
    store: ResearchStore,
    sqlite_path: str | Path,
    *,
    venue: str,
    timeframe: str = '1m',
    symbols: Optional[Iterable[str]] = None,
) -> dict[str, int]:
    """Migrate the scraper's SQLite tables into the research store.

    Keeps both `event_time` and `available_time`, so migrated history supports
    point-in-time reads. `venue` labels only rows that recorded none.
    """
    import sqlite3

    def venue_expression(table: str, fallbacks: tuple[str, ...]) -> str:
        """`venue` where the table has it, else 'unknown'."""
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
            # Prefer the venue recorded per row; use the caller's label for
            # rows written before the column existed.
            bars['venue'] = bars['row_venue'].where(
                bars['row_venue'] != 'unknown', venue
            )
            bars = bars.drop(columns=['row_venue'])
            dataset = BARS_DATASET_BY_TIMEFRAME.get(timeframe, 'bars')
            counts[dataset] = store.write(dataset, bars)
    finally:
        con.close()

    return counts
