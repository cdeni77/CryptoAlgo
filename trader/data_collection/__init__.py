"""
Data Collection Module for Crypto Perpetual Futures Trading System.

This module provides:
- Real-time data collection via WebSocket
- Historical data backfill via REST APIs
- Data validation and quality tracking
- Bi-temporal storage for backtesting integrity
- Message queue for real-time data flow
"""

# Core modules (no external dependencies beyond stdlib)
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

from .validator import (
    DataValidator,
    ValidationConfig,
    DataQualityTracker,
    DataValidationResult,
)

from .queue import (
    MessageQueueBase,
    InMemoryQueue,
    QueueMessage,
    Channels,
)

from .storage import (
    DatabaseBase,
    SQLiteDatabase,
    create_database,
)

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
    
    # Validation
    "DataValidator",
    "ValidationConfig",
    "DataQualityTracker",
    "DataValidationResult",
    
    # Queue
    "MessageQueueBase",
    "InMemoryQueue",
    "QueueMessage",
    "Channels",
    
    # Storage
    "DatabaseBase",
    "SQLiteDatabase",
    "create_database",
]

# Optional imports - require external packages
try:
    from .queue import RedisQueue, create_queue
    __all__.extend(["RedisQueue", "create_queue"])
except ImportError:
    pass

try:
    from .storage import TimescaleDatabase
    __all__.append("TimescaleDatabase")
except ImportError:
    pass

try:
    from .coinbase_connector import (
        CoinbaseRESTClient,
        CoinbaseWebSocketClient,
        CoinbaseAuth,
        RateLimiter,
    )
    __all__.extend([
        "CoinbaseRESTClient",
        "CoinbaseWebSocketClient",
        "CoinbaseAuth",
        "RateLimiter",
    ])
except ImportError:
    pass

try:
    from .ccxt_connector import CCXTConnector
    __all__.append("CCXTConnector")
except ImportError:
    pass

try:
    from .pipeline import (
        DataPipeline,
        PipelineConfig,
        create_pipeline,
    )
    __all__.extend([
        "DataPipeline",
        "PipelineConfig",
        "create_pipeline",
    ])
except ImportError:
    pass
