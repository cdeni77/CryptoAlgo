"""
Database storage layer for the Crypto Trading System.

Supports SQLite (development) and TimescaleDB (production).
All data is stored with bi-temporal timestamps for backtesting integrity.
"""

import logging
import sqlite3
from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

from .models import (
    OHLCVBar,
    Trade,
    FundingRate,
    OpenInterest,
    OrderBookSnapshot,
    OrderBookLevel,
    DataQuality,
)

logger = logging.getLogger(__name__)


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
        """
        Get OHLCV data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (1m, 1h, etc.)
            start: Start time (inclusive)
            end: End time (exclusive)
            as_of: Point-in-time query - only return data that was available
                   at this time. Critical for backtesting.
        """
        pass
    
    @abstractmethod
    def insert_funding_rate(self, funding: FundingRate) -> bool:
        """Insert funding rate."""
        pass
    
    @abstractmethod
    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get funding rate history."""
        pass
    
    @abstractmethod
    def insert_open_interest(self, oi: OpenInterest) -> bool:
        """Insert open interest snapshot."""
        pass
    
    @abstractmethod
    def get_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get open interest history."""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close database connection."""
        pass


class SQLiteDatabase(DatabaseBase):
    """
    SQLite database for development and testing.
    
    Uses bi-temporal schema for point-in-time queries.
    """
    
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
                    UNIQUE(symbol, timeframe, event_time)
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
            
            # Trades table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    trade_id TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    price REAL NOT NULL,
                    size REAL NOT NULL,
                    side TEXT NOT NULL,
                    quality TEXT DEFAULT 'valid',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, trade_id)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_trades_symbol_event 
                ON trades(symbol, event_time)
            """)
            
            # Funding rates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    rate REAL NOT NULL,
                    mark_price REAL NOT NULL,
                    index_price REAL NOT NULL,
                    is_settlement INTEGER DEFAULT 0,
                    quality TEXT DEFAULT 'valid',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, event_time)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_funding_symbol_event 
                ON funding_rates(symbol, event_time)
            """)
            
            # Open interest table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS open_interest (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    open_interest_contracts REAL NOT NULL,
                    open_interest_base REAL,
                    open_interest_usd REAL,
                    quality TEXT DEFAULT 'valid',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, event_time)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_oi_symbol_event 
                ON open_interest(symbol, event_time)
            """)
            
            # Order book snapshots (compressed JSON for efficiency)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS orderbook_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    best_bid REAL,
                    best_ask REAL,
                    mid_price REAL,
                    spread_bps REAL,
                    bids_json TEXT NOT NULL,
                    asks_json TEXT NOT NULL,
                    quality TEXT DEFAULT 'valid',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_orderbook_symbol_event 
                ON orderbook_snapshots(symbol, event_time)
            """)
            
            # Data quality tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data_type TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    quality TEXT NOT NULL,
                    issues TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
    
    def insert_ohlcv(self, bar: OHLCVBar) -> bool:
        """Insert single OHLCV bar."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO ohlcv 
                    (symbol, timeframe, event_time, available_time, 
                     open, high, low, close, volume, quote_volume, 
                     trade_count, quality, quality_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    bar.symbol,
                    bar.timeframe,
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
            
            for bar in bars:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv 
                        (symbol, timeframe, event_time, available_time, 
                         open, high, low, close, volume, quote_volume, 
                         trade_count, quality, quality_notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        bar.symbol,
                        bar.timeframe,
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
                except Exception as e:
                    logger.error(f"Failed to insert bar {bar.bar_id}: {e}")
            
            conn.commit()
            return inserted
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get OHLCV data with optional point-in-time constraint.
        
        The as_of parameter is CRITICAL for backtesting:
        - If None: returns all data (for analysis only)
        - If set: returns only data that was available at that time
        """
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
            
            # Point-in-time constraint
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
    
    def insert_funding_rate(self, funding: FundingRate) -> bool:
        """Insert funding rate."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO funding_rates
                    (symbol, event_time, available_time, rate, 
                     mark_price, index_price, is_settlement, quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    funding.symbol,
                    funding.event_time,
                    funding.available_time,
                    funding.rate,
                    funding.mark_price,
                    funding.index_price,
                    1 if funding.is_settlement else 0,
                    funding.quality.value,
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert funding rate: {e}")
                return False
    
    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get funding rate history."""
        with self._get_connection() as conn:
            query = """
                SELECT event_time, rate, mark_price, index_price, 
                       is_settlement, quality
                FROM funding_rates
                WHERE symbol = ?
                  AND event_time >= ?
                  AND event_time < ?
            """
            params: List[Any] = [symbol, start, end]
            
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
                df["is_settlement"] = df["is_settlement"].astype(bool)
            
            return df
    
    def insert_open_interest(self, oi: OpenInterest) -> bool:
        """Insert open interest snapshot."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO open_interest
                    (symbol, event_time, available_time, open_interest_contracts,
                     open_interest_base, open_interest_usd, quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    oi.symbol,
                    oi.event_time,
                    oi.available_time,
                    oi.open_interest_contracts,
                    oi.open_interest_base,
                    oi.open_interest_usd,
                    oi.quality.value,
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert open interest: {e}")
                return False
    
    def get_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get open interest history."""
        with self._get_connection() as conn:
            query = """
                SELECT event_time, open_interest_contracts, 
                       open_interest_base, open_interest_usd, quality
                FROM open_interest
                WHERE symbol = ?
                  AND event_time >= ?
                  AND event_time < ?
            """
            params: List[Any] = [symbol, start, end]
            
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
    
    def insert_orderbook_snapshot(self, book: OrderBookSnapshot) -> bool:
        """Insert order book snapshot."""
        import json
        
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO orderbook_snapshots
                    (symbol, event_time, available_time, best_bid, best_ask,
                     mid_price, spread_bps, bids_json, asks_json, quality)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    book.symbol,
                    book.event_time,
                    book.available_time,
                    book.best_bid,
                    book.best_ask,
                    book.mid_price,
                    book.spread_bps,
                    json.dumps([l.to_tuple() for l in book.bids]),
                    json.dumps([l.to_tuple() for l in book.asks]),
                    book.quality.value,
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert orderbook: {e}")
                return False
    
    def get_latest_ohlcv_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the most recent OHLCV event time for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(event_time) as max_time
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            if result and result["max_time"]:
                return datetime.fromisoformat(result["max_time"])
            return None
    
    def get_data_quality_summary(
        self,
        data_type: str,
        start: datetime,
        end: datetime
    ) -> Dict[str, Any]:
        """Get data quality summary for a time period."""
        table_map = {
            "ohlcv": "ohlcv",
            "funding": "funding_rates",
            "oi": "open_interest",
            "orderbook": "orderbook_snapshots",
        }
        
        table = table_map.get(data_type)
        if not table:
            return {}
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute(f"""
                SELECT quality, COUNT(*) as count
                FROM {table}
                WHERE event_time >= ? AND event_time < ?
                GROUP BY quality
            """, (start, end))
            
            results = cursor.fetchall()
            
            summary = {
                "total": 0,
                "valid": 0,
                "suspicious": 0,
                "invalid": 0,
            }
            
            for row in results:
                quality = row["quality"]
                count = row["count"]
                summary["total"] += count
                summary[quality] = count
            
            if summary["total"] > 0:
                summary["validity_rate"] = summary["valid"] / summary["total"] * 100
            else:
                summary["validity_rate"] = 0.0
            
            return summary
    
    def close(self) -> None:
        """Close database (no-op for SQLite with context manager pattern)."""
        logger.info("SQLite database closed")


class TimescaleDatabase(DatabaseBase):
    """
    TimescaleDB database for production.
    
    Uses hypertables for efficient time-series storage and querying.
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        database: str,
        user: str,
        password: str,
        pool_size: int = 5
    ):
        self.host = host
        self.port = port
        self.database = database
        self.user = user
        self.password = password
        self.pool_size = pool_size
        self._pool = None
    
    async def connect(self):
        """Create connection pool."""
        try:
            import asyncpg
            
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=2,
                max_size=self.pool_size,
            )
            logger.info(f"Connected to TimescaleDB at {self.host}:{self.port}")
            
        except ImportError:
            raise ImportError(
                "asyncpg required for TimescaleDB. "
                "Install with: pip install asyncpg"
            )
    
    def initialize(self) -> None:
        """
        Initialize TimescaleDB schema.
        
        Note: This is a simplified version. Production would need
        proper async handling and migration system.
        """
        # TimescaleDB schema would use hypertables
        # This is a placeholder - actual implementation would be async
        logger.info("TimescaleDB initialization - implement with proper async")
        raise NotImplementedError("Use SQLite for now, TimescaleDB coming soon")
    
    def insert_ohlcv(self, bar: OHLCVBar) -> bool:
        raise NotImplementedError()
    
    def insert_ohlcv_batch(self, bars: List[OHLCVBar]) -> int:
        raise NotImplementedError()
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        raise NotImplementedError()
    
    def insert_funding_rate(self, funding: FundingRate) -> bool:
        raise NotImplementedError()
    
    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        raise NotImplementedError()
    
    def insert_open_interest(self, oi: OpenInterest) -> bool:
        raise NotImplementedError()
    
    def get_open_interest(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        raise NotImplementedError()
    
    def close(self) -> None:
        if self._pool:
            # Would need async close
            pass


def create_database(
    db_type: str = "sqlite",
    **kwargs
) -> DatabaseBase:
    """
    Factory function to create appropriate database implementation.
    
    Args:
        db_type: "sqlite" or "timescaledb"
        **kwargs: Database-specific configuration
    
    Returns:
        DatabaseBase instance
    """
    if db_type == "sqlite":
        return SQLiteDatabase(kwargs.get("db_path", "data/trading.db"))
    elif db_type == "timescaledb":
        return TimescaleDatabase(**kwargs)
    else:
        raise ValueError(f"Unknown database type: {db_type}")
