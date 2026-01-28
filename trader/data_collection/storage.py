"""
Database storage layer for the Crypto Trading System.

Supports SQLite (development) and TimescaleDB (production).
All data is stored with bi-temporal timestamps for backtesting integrity.
"""

import logging
import sqlite3
import pandas as pd

from abc import ABC, abstractmethod
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from .models import (
    OHLCVBar,
    FundingRate,
    OpenInterest,
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
        """Get OHLCV data."""
        pass
    
    @abstractmethod
    def insert_funding_rate(self, funding: FundingRate) -> bool:
        """Insert funding rate."""
        pass
    
    @abstractmethod
    def insert_funding_rate_batch(self, rates: List[FundingRate]) -> int:
        """Insert batch of funding rates."""
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
            
            # Funding rates table - ENHANCED
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS funding_rates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    event_time TIMESTAMP NOT NULL,
                    available_time TIMESTAMP NOT NULL,
                    rate REAL NOT NULL,
                    mark_price REAL,
                    index_price REAL,
                    is_settlement INTEGER DEFAULT 0,
                    quality TEXT DEFAULT 'valid',
                    source TEXT DEFAULT 'unknown',
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
                    source TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, event_time)
                )
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_oi_symbol_event 
                ON open_interest(symbol, event_time)
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
    
    def insert_funding_rate(self, funding: FundingRate) -> bool:
        """Insert funding rate."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO funding_rates
                    (symbol, event_time, available_time, rate, 
                     mark_price, index_price, is_settlement, quality, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    funding.symbol,
                    funding.event_time,
                    funding.available_time,
                    funding.rate,
                    funding.mark_price,
                    funding.index_price,
                    1 if funding.is_settlement else 0,
                    funding.quality.value,
                    "coinbase",
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert funding rate: {e}")
                return False
    
    def insert_funding_rate_batch(self, rates: List[FundingRate]) -> int:
        """Insert batch of funding rates."""
        if not rates:
            return 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
            
            for funding in rates:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO funding_rates
                        (symbol, event_time, available_time, rate, 
                         mark_price, index_price, is_settlement, quality, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        funding.symbol,
                        funding.event_time,
                        funding.available_time,
                        funding.rate,
                        funding.mark_price,
                        funding.index_price,
                        1 if funding.is_settlement else 0,
                        funding.quality.value,
                        getattr(funding, 'source', 'ccxt'),
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert funding rate: {e}")
            
            conn.commit()
            return inserted
    
    def get_funding_rates(
        self,
        symbol: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        as_of: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Get funding rate history for a symbol."""
        query = """
            SELECT event_time, rate
            FROM funding_rates
            WHERE symbol = ?
        """
        params: List[Any] = [symbol]
        
        if start is not None:
            query += " AND event_time >= ?"
            params.append(start)
        
        if end is not None:
            query += " AND event_time < ?"
            params.append(end)
        
        if as_of is not None:
            query += " AND available_time <= ?"
            params.append(as_of)
        
        query += " ORDER BY event_time ASC"
        
        with self._get_connection() as conn:
            df = pd.read_sql_query(
                query,
                conn,
                params=params,
                parse_dates=["event_time"]
            )
        
        if not df.empty:
            df.set_index("event_time", inplace=True)
        
        return df
    
    def insert_open_interest(self, oi: OpenInterest) -> bool:
        """Insert open interest snapshot."""
        with self._get_connection() as conn:
            try:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO open_interest
                    (symbol, event_time, available_time, open_interest_contracts,
                     open_interest_base, open_interest_usd, quality, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    oi.symbol,
                    oi.event_time,
                    oi.available_time,
                    oi.open_interest_contracts,
                    oi.open_interest_base,
                    oi.open_interest_usd,
                    oi.quality.value,
                    "ccxt",
                ))
                conn.commit()
                return True
            except Exception as e:
                logger.error(f"Failed to insert open interest: {e}")
                return False
    
    def insert_open_interest_batch(self, oi_list: List[OpenInterest]) -> int:
        """Insert batch of open interest records."""
        if not oi_list:
            return 0
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            inserted = 0
            
            for oi in oi_list:
                try:
                    cursor.execute("""
                        INSERT OR REPLACE INTO open_interest
                        (symbol, event_time, available_time, open_interest_contracts,
                         open_interest_base, open_interest_usd, quality, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        oi.symbol,
                        oi.event_time,
                        oi.available_time,
                        oi.open_interest_contracts,
                        oi.open_interest_base,
                        oi.open_interest_usd,
                        oi.quality.value,
                        "ccxt",
                    ))
                    inserted += 1
                except Exception as e:
                    logger.error(f"Failed to insert OI: {e}")
            
            conn.commit()
            return inserted
    
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
    
    def get_earliest_ohlcv_time(self, symbol: str, timeframe: str) -> Optional[datetime]:
        """Get the earliest OHLCV event time for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(event_time) as min_time
                FROM ohlcv
                WHERE symbol = ? AND timeframe = ?
            """, (symbol, timeframe))
            
            result = cursor.fetchone()
            if result and result["min_time"]:
                return datetime.fromisoformat(result["min_time"])
            return None
    
    def get_latest_funding_time(self, symbol: str) -> Optional[datetime]:
        """Get the most recent funding rate time for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MAX(event_time) as max_time
                FROM funding_rates
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            if result and result["max_time"]:
                return datetime.fromisoformat(result["max_time"])
            return None
    
    def get_earliest_funding_time(self, symbol: str) -> Optional[datetime]:
        """Get the earliest funding rate time for a symbol."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(event_time) as min_time
                FROM funding_rates
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            if result and result["min_time"]:
                return datetime.fromisoformat(result["min_time"])
            return None
    
    def get_funding_stats(self) -> Dict[str, Any]:
        """Get statistics about stored funding data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT symbol, 
                       COUNT(*) as count,
                       MIN(event_time) as earliest,
                       MAX(event_time) as latest,
                       AVG(rate) as avg_rate,
                       MIN(rate) as min_rate,
                       MAX(rate) as max_rate
                FROM funding_rates
                GROUP BY symbol
            """)
            
            results = {}
            for row in cursor.fetchall():
                results[row["symbol"]] = {
                    "count": row["count"],
                    "earliest": row["earliest"],
                    "latest": row["latest"],
                    "avg_rate_bps": row["avg_rate"] * 10000 if row["avg_rate"] else 0,
                    "min_rate_bps": row["min_rate"] * 10000 if row["min_rate"] else 0,
                    "max_rate_bps": row["max_rate"] * 10000 if row["max_rate"] else 0,
                }
            
            return results
    
    def close(self) -> None:
        """Close database (no-op for SQLite with context manager pattern)."""
        logger.info("SQLite database closed")


def create_database(db_type: str = "sqlite", **kwargs) -> DatabaseBase:
    """Factory function to create appropriate database implementation."""
    if db_type == "sqlite":
        return SQLiteDatabase(kwargs.get("db_path", "data/trading.db"))
    else:
        raise ValueError(f"Unknown database type: {db_type}")