"""SQLite storage for OHLCV bars, with bi-temporal timestamps."""

import logging
import re
import sqlite3
import pandas as pd

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .models import OHLCVBar

logger = logging.getLogger(__name__)


class StorageWriteError(RuntimeError):
    """A write failed for a reason that is not about the data.

    Raised, not counted: a returned 0 cannot tell "the venue served nothing"
    from "this database is read-only".
    """


class DatabaseBase(ABC):
    """Abstract base class for database implementations."""
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize database schema."""
        pass
    
    @abstractmethod
    def insert_ohlcv(self, bar: OHLCVBar) -> bool:
        """Insert OHLCV bar."""
        pass
    
    @abstractmethod
    def insert_ohlcv_batch(self, bars: List[OHLCVBar]) -> int:
        """Insert batch of OHLCV bars. Returns count inserted."""
        pass
    
    @abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get OHLCV data."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


class SQLiteDatabase(DatabaseBase):
    """SQLite database with a bi-temporal schema for point-in-time queries."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._ensure_directory()
        self._conn: Optional[sqlite3.Connection] = None
    
    def _ensure_directory(self):
        """Ensure database directory exists."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
    
    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection with proper handling."""
        conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def initialize(self) -> None:
        """Create database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # OHLCV table with bi-temporal timestamps
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlcv (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    venue TEXT DEFAULT 'unknown',
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL,
                    trade_count INTEGER,
                    quality TEXT DEFAULT 'valid',
                    quality_notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    -- Venue is part of the key. Without it, INSERT OR REPLACE
                    -- makes each venue overwrite the other's bar for the same
                    -- hour, so the two-venue design collapses to whichever
                    -- provider wrote last -- and the cross-venue features
                    -- (basis, lead-lag) get no rows at all, silently.
                    UNIQUE(symbol, timeframe, venue, event_time)
                )
            """)
            
            # Indexes for efficient queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe_event 
                ON ohlcv(symbol, timeframe, event_time)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlcv_available_time 
                ON ohlcv(available_time)
            """)
            
            # Migrate older databases. Their bars keep venue 'unknown', because
            # their true origin is not recoverable.
            self._add_missing_columns(cursor, "ohlcv", {
                "venue": "TEXT DEFAULT 'unknown'",
            })
            # Older databases lack venue in the key, and SQLite cannot alter a
            # constraint in place.
            self._rebuild_for_venue_key(cursor, "ohlcv",
                                        "UNIQUE(symbol, timeframe, venue, event_time)")

            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")

    @staticmethod
    def _add_missing_columns(cursor, table: str, columns: Dict[str, str]) -> None:
        """Add any of `columns` the table does not have. Idempotent."""
        cursor.execute(f"PRAGMA table_info({table})")
        present = {row[1] for row in cursor.fetchall()}
        for name, definition in columns.items():
            if name not in present:
                logger.info("adding %s.%s", table, name)
                cursor.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")

    @staticmethod
    def _rebuild_for_venue_key(cursor, table: str, wanted_unique: str) -> None:
        """Rebuild `table` if its unique key does not include venue.

        SQLite cannot alter a UNIQUE constraint, so this recreates the table and
        copies the rows. Without venue in the key, `INSERT OR REPLACE` lets one
        venue overwrite the other. Overwritten rows need a new backfill.
        """
        cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
        )
        row = cursor.fetchone()
        if not row or not row[0]:
            return
        create_sql = row[0]
        normalised = ' '.join(create_sql.split())
        if wanted_unique.replace(' ', '') in normalised.replace(' ', ''):
            return  # already keyed on venue

        logger.warning(
            "%s is keyed without venue, so one venue's rows have been overwriting "
            "the other's. Rebuilding the table; re-run the backfill to recover the "
            "missing venue.", table,
        )

        cursor.execute(f"PRAGMA table_info({table})")
        columns = [r[1] for r in cursor.fetchall()]
        column_list = ', '.join(columns)

        # Swap the old unique clause for the wanted one, keeping every column
        # definition exactly as it was.
        rebuilt = re.sub(
            r'UNIQUE\s*\([^)]*\)', wanted_unique, create_sql, count=1
        ).replace(f'TABLE IF NOT EXISTS {table}', f'TABLE {table}__new', 1)
        if f'{table}__new' not in rebuilt:
            rebuilt = rebuilt.replace(f'TABLE {table}', f'TABLE {table}__new', 1)

        cursor.execute(f"DROP TABLE IF EXISTS {table}__new")
        cursor.execute(rebuilt)
        # OR IGNORE keeps the first row, which is deterministic.
        cursor.execute(
            f"INSERT OR IGNORE INTO {table}__new ({column_list}) "
            f"SELECT {column_list} FROM {table}"
        )
        cursor.execute(f"DROP TABLE {table}")
        cursor.execute(f"ALTER TABLE {table}__new RENAME TO {table}")
    
    def insert_ohlcv(self, bar: OHLCVBar) -> bool:
        """Insert single OHLCV bar."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (symbol, timeframe, venue, event_time, available_time,
                     open, high, low, close, volume, quote_volume,
                     trade_count, quality, quality_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bar.symbol,
                    bar.timeframe,
                    bar.venue,
                    bar.event_time,
                    bar.available_time,
                    bar.open,
                    bar.high,
                    bar.low,
                    bar.close,
                    bar.volume,
                    bar.quote_volume,
                    bar.trade_count,
                    bar.quality.value,
                    bar.quality_notes,
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert OHLCV: {e}")
                return False
    
    def insert_ohlcv_batch(self, bars: List[OHLCVBar]) -> int:
        """Insert batch of OHLCV bars."""
        if not bars:
            return 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
            failures: List[str] = []
            
            for bar in bars:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv 
                        (symbol, timeframe, venue, event_time, available_time,
                         open, high, low, close, volume, quote_volume,
                         trade_count, quality, quality_notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        bar.symbol,
                        bar.timeframe,
                        bar.venue,
                        bar.event_time,
                        bar.available_time,
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume,
                        bar.quote_volume,
                        bar.trade_count,
                        bar.quality.value,
                        bar.quality_notes,
                    ))
                    inserted += 1
                except sqlite3.OperationalError as e:
                    # A connection-level failure (read-only file, full disk,
                    # lock) affects every row. Fail once, with the cause.
                    raise StorageWriteError(
                        f"cannot write to {self.db_path}: {e}. "
                        f"{inserted} of {len(bars)} bars were written before this. "
                        f"Check that the file and its directory are writable by "
                        f"this process — SQLite needs the *directory* too, for its "
                        f"-wal and -journal files — and that the disk is not full."
                    ) from e
                except Exception as e:
                    failures.append(f"{bar.bar_id}: {e}")
            
            conn.commit()
            if failures and not inserted:
                # Every row failed. Returning 0 would read as "no data" upstream.
                raise StorageWriteError(
                    f"all {len(bars)} bars rejected, first: {failures[0]}"
                )
            if failures:
                logger.error(
                    "%d of %d bars rejected (first: %s)",
                    len(failures), len(bars), failures[0],
                )
            return inserted
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get OHLCV data with optional point-in-time constraint."""
        with self._get_connection() as conn:
            query = """
                SELECT event_time, open, high, low, close, volume, 
                       quote_volume, trade_count, quality
                FROM ohlcv
                WHERE symbol = ? 
                  AND timeframe = ?
                  AND event_time >= ?
                  AND event_time < ?
            """
            params: List[Any] = [symbol, timeframe, start, end]
            
            if as_of is not None:
                query += " AND available_time <= ?"
                params.append(as_of)
            
            query += " ORDER BY event_time ASC"
            
            df = pd.read_sql_query(
                query,
                conn,
                params=params,
                parse_dates=["event_time"]
            )
            
            if not df.empty:
                df.set_index("event_time", inplace=True)
            
            return df
    
    def find_gaps(
        self,
        table: str,
        symbol: str,
        start: datetime,
        end: datetime,
        *,
        step_seconds: int,
        venue: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> list[tuple[datetime, datetime]]:
        """Windows inside [start, end) with no rows, including interior holes."""
        allowed = {'ohlcv'}
        if table not in allowed:
            raise ValueError(f'unknown table {table!r}')

        sql = f"SELECT event_time FROM {table} WHERE symbol = ?"
        params: list = [symbol]
        if timeframe is not None and table == 'ohlcv':
            sql += " AND timeframe = ?"
            params.append(timeframe)
        if venue is not None:
            sql += " AND venue = ?"
            params.append(venue)
        sql += " AND event_time >= ? AND event_time < ? ORDER BY event_time"
        # Bind the datetime, not an isoformat string. Rows are stored as
        # 'YYYY-MM-DD HH:MM:SS', and a 'T' separator compares wrong as text.
        params += [start, end]

        with self._get_connection() as conn:
            rows = conn.execute(sql, params).fetchall()

        step = timedelta(seconds=step_seconds)
        # A gap opens when consecutive rows are more than one step apart.
        tolerance = step * 1.5
        gaps: list[tuple[datetime, datetime]] = []
        cursor = start
        for row in rows:
            raw = row['event_time']
            # SQLite hands back either a str or a datetime depending on the
            # detect_types configuration; accept both.
            moment = raw if isinstance(raw, datetime) else datetime.fromisoformat(str(raw))
            if moment - cursor > tolerance:
                gaps.append((cursor, moment))
            cursor = max(cursor, moment + step)
        if end - cursor > tolerance:
            gaps.append((cursor, end))
        return gaps

    def get_latest_ohlcv_time(
        self, symbol: str, timeframe: str, venue: Optional[str] = None
    ) -> Optional[datetime]:
        """MAX event_time for this instrument, optionally for one venue.

        Pass `venue` for a watermark: without it, one venue's bars hide another
        venue's gap.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = (
                "SELECT MAX(event_time) as max_time FROM ohlcv "
                "WHERE symbol = ? AND timeframe = ?"
            )
            params: list = [symbol, timeframe]
            if venue is not None:
                sql += " AND venue = ?"
                params.append(venue)
            cursor.execute(sql, params)
            row = cursor.fetchone()
            value = row["max_time"] if row else None
            return datetime.fromisoformat(value) if value else None

    def get_earliest_ohlcv_time(
        self, symbol: str, timeframe: str, venue: Optional[str] = None
    ) -> Optional[datetime]:
        """MIN event_time for this instrument, optionally for one venue.

        Pass `venue` for a watermark: without it, one venue's bars hide another
        venue's gap.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            sql = (
                "SELECT MIN(event_time) as min_time FROM ohlcv "
                "WHERE symbol = ? AND timeframe = ?"
            )
            params: list = [symbol, timeframe]
            if venue is not None:
                sql += " AND venue = ?"
                params.append(venue)
            cursor.execute(sql, params)
            row = cursor.fetchone()
            value = row["min_time"] if row else None
            return datetime.fromisoformat(value) if value else None

    def close(self) -> None:
        """Close database (no-op for SQLite with context manager pattern)."""
        logger.info("SQLite database closed")


def create_database(db_type: str = "sqlite", **kwargs) -> DatabaseBase:
    """Factory function to create appropriate database implementation."""
    if db_type == "sqlite":
        return SQLiteDatabase(kwargs.get("db_path", "data/trading.db"))
    else:
        raise ValueError(f"Unknown database type: {db_type}")
