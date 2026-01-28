"""
Configuration settings for the Crypto Perpetual Futures Trading System.

All configuration is centralized here for easy modification and environment-based overrides.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class Environment(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class CoinbaseAPIConfig:
    """Coinbase Advanced Trade API configuration."""
    
    # API credentials (loaded from environment)
    api_key: str = field(default_factory=lambda: os.environ.get("CDP_API_KEY", ""))
    api_secret: str = field(default_factory=lambda: os.environ.get("CDP_API_SECRET", ""))
    
    # Base URLs
    rest_base_url: str = "https://api.coinbase.com"
    websocket_url: str = "wss://advanced-trade-ws.coinbase.com"
    
    # Rate limits (requests per second)
    public_rate_limit: int = 10
    private_rate_limit: int = 15
    websocket_connections_per_sec: int = 750
    websocket_msg_per_sec_unauth: int = 8
    
    # Timeouts (seconds)
    request_timeout: int = 30
    websocket_ping_interval: int = 30
    websocket_reconnect_delay: int = 5
    
    # Retry configuration
    max_retries: int = 3
    retry_backoff_factor: float = 2.0


@dataclass
class CCXTConfig:
    """CCXT library configuration for historical data."""
    
    # Supported exchanges for cross-reference
    exchanges: List[str] = field(default_factory=lambda: [
        "binance",
        "bybit", 
        "okx"
    ])
    
    # Rate limiting
    enable_rate_limit: bool = True
    rate_limit_requests_per_second: int = 5
    
    # Data fetch settings
    default_limit: int = 1000
    max_retries: int = 3


@dataclass
class DatabaseConfig:
    """Time-series database configuration."""
    
    # Database type: "timescaledb", "questdb", or "sqlite" (for development)
    db_type: str = field(default_factory=lambda: os.environ.get("DB_TYPE", "sqlite"))
    
    # Connection settings
    host: str = field(default_factory=lambda: os.environ.get("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.environ.get("DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.environ.get("DB_NAME", "crypto_trading"))
    user: str = field(default_factory=lambda: os.environ.get("DB_USER", ""))
    password: str = field(default_factory=lambda: os.environ.get("DB_PASSWORD", ""))
    
    # SQLite path (for development)
    sqlite_path: str = field(default_factory=lambda: os.environ.get(
        "SQLITE_PATH", "/home/claude/crypto_trading_system/data/trading.db"
    ))
    
    # Connection pool settings
    pool_size: int = 5
    max_overflow: int = 10
    
    # Data retention (days)
    ohlcv_retention_days: int = 365 * 3  # 3 years
    trades_retention_days: int = 90
    orderbook_retention_days: int = 30


@dataclass
class MessageQueueConfig:
    """Message queue configuration for real-time data flow."""
    
    # Queue type: "redis", "memory" (for development)
    queue_type: str = field(default_factory=lambda: os.environ.get("QUEUE_TYPE", "memory"))
    
    # Redis settings
    redis_host: str = field(default_factory=lambda: os.environ.get("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.environ.get("REDIS_PORT", "6379")))
    redis_db: int = 0
    
    # Queue settings
    max_queue_size: int = 10000
    batch_size: int = 100
    flush_interval_ms: int = 100


@dataclass
class InstrumentConfig:
    """Trading instrument configuration."""
    
    # Target perpetual futures contracts
    # Note: As of Jan 2026, Coinbase US offers BTC-PERP and ETH-PERP
    # Expand this list as more contracts become available
    perpetual_symbols: List[str] = field(default_factory=lambda: [
        "BTC-PERP",
        "ETH-PERP",
        # Future additions when available:
        # "SOL-PERP",
        # "XRP-PERP", 
        # "AVAX-PERP",
    ])
    
    # Corresponding spot symbols for basis calculation
    spot_symbols: List[str] = field(default_factory=lambda: [
        "BTC-USD",
        "ETH-USD",
    ])
    
    # Contract specifications (Coinbase nano contracts)
    contract_sizes: Dict[str, float] = field(default_factory=lambda: {
        "BTC-PERP": 0.01,   # 1/100 BTC
        "ETH-PERP": 0.1,    # 1/10 ETH
    })
    
    # Minimum notional (USD)
    min_notional: float = 10.0


@dataclass
class DataCollectionConfig:
    """Data collection specific settings."""
    
    # OHLCV settings
    ohlcv_timeframes: List[str] = field(default_factory=lambda: [
        "1m", "5m", "15m", "1h", "4h", "1d"
    ])
    primary_timeframe: str = "1h"  # Primary timeframe for modeling
    
    # Historical backfill
    backfill_days: int = 365  # Days of history to fetch on initialization
    
    # Data validation
    max_price_change_pct: float = 10.0  # Flag moves > 10% in 1 minute as suspicious
    max_gap_minutes: int = 5  # Flag gaps > 5 minutes
    
    # Funding rate settings
    funding_interval_hours: int = 1  # Coinbase accrues hourly
    funding_settlement_times: List[str] = field(default_factory=lambda: [
        "00:00", "12:00"  # Twice daily settlement
    ])
    
    # Order book settings
    orderbook_depth: int = 50  # Levels to store
    orderbook_snapshot_interval_sec: int = 60  # How often to snapshot full book


@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = field(default_factory=lambda: os.environ.get("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File logging
    log_to_file: bool = True
    log_file_path: str = "/home/claude/crypto_trading_system/logs/data_collection.log"
    max_log_size_mb: int = 100
    backup_count: int = 5


@dataclass
class Config:
    """Main configuration class aggregating all settings."""
    
    environment: Environment = Environment.DEVELOPMENT
    
    coinbase: CoinbaseAPIConfig = field(default_factory=CoinbaseAPIConfig)
    ccxt: CCXTConfig = field(default_factory=CCXTConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    queue: MessageQueueConfig = field(default_factory=MessageQueueConfig)
    instruments: InstrumentConfig = field(default_factory=InstrumentConfig)
    data_collection: DataCollectionConfig = field(default_factory=DataCollectionConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    @classmethod
    def from_environment(cls) -> "Config":
        """Create configuration based on environment variables."""
        env_str = os.environ.get("ENVIRONMENT", "development").lower()
        try:
            env = Environment(env_str)
        except ValueError:
            env = Environment.DEVELOPMENT
        
        config = cls(environment=env)
        
        # Override settings based on environment
        if env == Environment.PRODUCTION:
            config.database.db_type = "timescaledb"
            config.queue.queue_type = "redis"
            config.logging.level = "WARNING"
        
        return config


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.from_environment()
    return _config


def reset_config():
    """Reset configuration (useful for testing)."""
    global _config
    _config = None
