"""
Data Collection Module for Crypto Perpetual Futures Trading System.
"""

from .models import (
    OHLCVBar,
    Trade,
    FundingRate,
    OpenInterest,
    OrderBookSnapshot,
    OrderBookLevel,
    TickerUpdate,
    Side,
    DataQuality,
    BiTemporalMixin,
)

from .storage import (
    DatabaseBase,
    SQLiteDatabase,
    create_database,
)

from .ccxt_connector import CCXTConnector

__all__ = [
    # Models
    "OHLCVBar",
    "Trade", 
    "FundingRate",
    "OpenInterest",
    "OrderBookSnapshot",
    "OrderBookLevel",
    "TickerUpdate",
    "Side",
    "DataQuality",
    "BiTemporalMixin",
    
    # Storage
    "DatabaseBase",
    "SQLiteDatabase",
    "create_database",
    
    # Connectors
    "CCXTConnector",
]