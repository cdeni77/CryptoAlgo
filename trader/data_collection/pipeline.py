"""
Data Collection Pipeline Orchestrator.

Coordinates all components of the data collection system:
- REST API polling
- WebSocket streaming
- Data validation
- Database storage
- Message queue for real-time data flow
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field

from .models import (
    OHLCVBar,
    Trade,
    FundingRate,
    OpenInterest,
    OrderBookSnapshot,
    TickerUpdate,
    DataQuality,
)
from .validator import DataValidator, ValidationConfig, DataQualityTracker
from .queue import MessageQueueBase, InMemoryQueue, Channels, QueueMessage
from .storage import DatabaseBase, SQLiteDatabase, create_database
from .coinbase_connector import CoinbaseRESTClient, CoinbaseWebSocketClient
from .ccxt_connector import CCXTConnector

logger = logging.getLogger(__name__)


def ensure_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """
    Convert datetime to naive UTC for consistent comparison.
    
    All internal datetime handling uses naive UTC to avoid timezone issues.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        # Convert to UTC then strip timezone
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def ensure_aware_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert datetime to timezone-aware UTC."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass
class PipelineConfig:
    """Configuration for the data pipeline."""
    
    # Symbols to track
    symbols: List[str] = field(default_factory=lambda: ["BTC-PERP", "ETH-PERP"])
    
    # Timeframes to collect
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    
    # API credentials
    coinbase_api_key: Optional[str] = None
    coinbase_api_secret: Optional[str] = None
    
    # Proxy settings (for regions where exchanges are blocked)
    proxy: Optional[str] = None  # e.g., "http://127.0.0.1:7890"
    
    # CCXT exchange settings
    ccxt_exchanges: List[str] = field(default_factory=lambda: ["binance", "bybit"])
    ccxt_use_fallbacks: bool = True  # Try fallback exchanges if primary ones fail
    
    # Database settings
    db_type: str = "sqlite"
    db_path: str = "/home/claude/crypto_trading_system/data/trading.db"
    
    # Queue settings
    queue_type: str = "memory"
    
    # Collection intervals (seconds)
    ohlcv_poll_interval: int = 60  # Poll every minute for 1m candles
    funding_poll_interval: int = 300  # Poll every 5 minutes
    oi_poll_interval: int = 300  # Poll every 5 minutes
    
    # Historical backfill
    backfill_days: int = 30  # Days of history to fetch on startup
    
    # Validation
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Feature flags
    enable_funding_polling: bool = True  # Set to False if no Coinbase credentials


class DataPipeline:
    """
    Main data collection pipeline.
    
    Architecture:
    
    ┌─────────────────────────────────────────────────────────────┐
    │                    DATA COLLECTION LAYER                     │
    ├─────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
    │  │  WebSocket   │  │   REST API   │  │   CCXT      │       │
    │  │  Connector   │  │   Poller     │  │   Backfill  │       │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
    │         │                 │                 │               │
    │         └─────────────────┼─────────────────┘               │
    │                           ▼                                 │
    │                   ┌───────────────┐                         │
    │                   │ Message Queue │                         │
    │                   └───────┬───────┘                         │
    │                           ▼                                 │
    │                   ┌───────────────┐                         │
    │                   │   Validator   │                         │
    │                   └───────┬───────┘                         │
    │                           ▼                                 │
    │                   ┌───────────────┐                         │
    │                   │   Database    │                         │
    │                   └───────────────┘                         │
    └─────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Components
        self._rest_client: Optional[CoinbaseRESTClient] = None
        self._ws_client: Optional[CoinbaseWebSocketClient] = None
        self._ccxt_connector: Optional[CCXTConnector] = None
        self._queue: Optional[MessageQueueBase] = None
        self._database: Optional[DatabaseBase] = None
        self._validator: DataValidator = DataValidator(config.validation_config)
        self._quality_tracker: DataQualityTracker = DataQualityTracker()
        
        # State
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._last_ohlcv_times: Dict[str, Dict[str, datetime]] = {}  # symbol -> timeframe -> last time
        
        # Callbacks for external consumers
        self._on_ohlcv_callbacks: List[Callable[[OHLCVBar], None]] = []
        self._on_ticker_callbacks: List[Callable[[TickerUpdate], None]] = []
        self._on_funding_callbacks: List[Callable[[FundingRate], None]] = []
    
    async def initialize(self):
        """Initialize all pipeline components."""
        logger.info("Initializing data pipeline...")
        
        # Initialize database
        if self.config.db_type == "sqlite":
            self._database = SQLiteDatabase(self.config.db_path)
        else:
            self._database = create_database(
                self.config.db_type,
                db_path=self.config.db_path
            )
        self._database.initialize()
        logger.info(f"Database initialized: {self.config.db_type}")
        
        # Initialize message queue
        self._queue = InMemoryQueue()
        await self._queue.start()
        logger.info(f"Message queue initialized: {self.config.queue_type}")
        
        # Initialize REST client
        self._rest_client = CoinbaseRESTClient(
            api_key=self.config.coinbase_api_key,
            api_secret=self.config.coinbase_api_secret,
        )
        logger.info("REST client initialized")
        
        # Initialize WebSocket client
        self._ws_client = CoinbaseWebSocketClient(
            api_key=self.config.coinbase_api_key,
            api_secret=self.config.coinbase_api_secret,
            on_ticker=self._handle_ws_ticker,
            on_orderbook=self._handle_ws_orderbook,
            on_error=self._handle_ws_error,
        )
        logger.info("WebSocket client initialized")
        
        # Initialize CCXT connector for backfill
        self._ccxt_connector = CCXTConnector(
            exchanges=self.config.ccxt_exchanges,
            proxy=self.config.proxy,
            use_fallbacks=self.config.ccxt_use_fallbacks,
        )
        await self._ccxt_connector.initialize()
        logger.info("CCXT connector initialized")
        
        # Load last known times from database
        await self._load_last_times()
        
        logger.info("Data pipeline initialization complete")
    
    async def _load_last_times(self):
        """Load last OHLCV times from database for each symbol/timeframe."""
        for symbol in self.config.symbols:
            self._last_ohlcv_times[symbol] = {}
            for timeframe in self.config.timeframes:
                last_time = self._database.get_latest_ohlcv_time(symbol, timeframe)
                if last_time:
                    # Ensure naive UTC
                    last_time = ensure_naive_utc(last_time)
                    self._last_ohlcv_times[symbol][timeframe] = last_time
                    logger.debug(f"Last {symbol} {timeframe} bar: {last_time}")
    
    async def start(self):
        """Start the data pipeline."""
        if self._running:
            logger.warning("Pipeline already running")
            return
        
        self._running = True
        logger.info("Starting data pipeline...")
        
        # Start WebSocket
        await self._ws_client.connect()
        await self._ws_client.subscribe(
            channels=["ticker"],
            product_ids=self.config.symbols,
        )
        await self._ws_client.start()
        
        # Subscribe to validated data channels
        await self._queue.subscribe(
            Channels.VALIDATED_OHLCV,
            self._handle_validated_ohlcv
        )
        
        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._ohlcv_poll_loop()),
            asyncio.create_task(self._validation_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]
        
        # Only poll funding if enabled
        if self.config.enable_funding_polling and self.config.coinbase_api_key:
            self._tasks.append(asyncio.create_task(self._funding_poll_loop()))
        
        logger.info("Data pipeline started")
    
    async def stop(self):
        """Stop the data pipeline gracefully."""
        logger.info("Stopping data pipeline...")
        self._running = False
        
        # Cancel tasks
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        
        # Close components
        if self._ws_client:
            await self._ws_client.stop()
        
        if self._rest_client:
            await self._rest_client.close()
        
        if self._ccxt_connector:
            await self._ccxt_connector.close()
        
        if self._queue:
            await self._queue.close()
        
        if self._database:
            self._database.close()
        
        logger.info("Data pipeline stopped")
    
    async def backfill(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        use_ccxt: bool = True,
    ):
        """
        Backfill historical data.
        
        Args:
            start: Start datetime (default: config.backfill_days ago)
            end: End datetime (default: now)
            symbols: Symbols to backfill (default: config.symbols)
            timeframes: Timeframes to backfill (default: config.timeframes)
            use_ccxt: Use CCXT for backfill (Coinbase may have limited history)
        """
        # Normalize to naive UTC for consistent handling
        end = ensure_naive_utc(end) if end else datetime.utcnow()
        start = ensure_naive_utc(start) if start else (end - timedelta(days=self.config.backfill_days))
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        
        logger.info(f"Starting backfill from {start} to {end}")
        logger.info(f"Symbols: {symbols}, Timeframes: {timeframes}")
        
        for symbol in symbols:
            for timeframe in timeframes:
                # Check if we already have data
                last_time = self._last_ohlcv_times.get(symbol, {}).get(timeframe)
                
                # Ensure last_time is also naive UTC for comparison
                if last_time:
                    last_time = ensure_naive_utc(last_time)
                
                actual_start = last_time if last_time and last_time > start else start
                
                if actual_start >= end:
                    logger.info(f"Skipping {symbol} {timeframe} - already up to date")
                    continue
                
                logger.info(f"Backfilling {symbol} {timeframe} from {actual_start}")
                
                try:
                    if use_ccxt:
                        # Use CCXT for backfill (better historical coverage)
                        bars = await self._ccxt_connector.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            start=actual_start,
                            end=end,
                        )
                    else:
                        # Use Coinbase REST API
                        bars = await self._rest_client.get_candles_range(
                            product_id=symbol,
                            granularity=timeframe,
                            start=actual_start,
                            end=end,
                        )
                    
                    if bars:
                        # Validate and store
                        valid_bars = []
                        prev_bar = None
                        
                        for bar in bars:
                            result = self._validator.validate_ohlcv(bar, prev_bar)
                            self._quality_tracker.record_validation(result)
                            
                            if result.is_valid or result.quality == DataQuality.SUSPICIOUS:
                                valid_bars.append(bar)
                            prev_bar = bar
                        
                        # Batch insert
                        inserted = self._database.insert_ohlcv_batch(valid_bars)
                        logger.info(
                            f"Backfilled {inserted}/{len(bars)} bars for {symbol} {timeframe}"
                        )
                        
                        # Update last time
                        if valid_bars:
                            self._last_ohlcv_times.setdefault(symbol, {})[timeframe] = ensure_naive_utc(valid_bars[-1].event_time)
                    
                except Exception as e:
                    logger.error(f"Error backfilling {symbol} {timeframe}: {e}")
                    import traceback
                    traceback.print_exc()
        
        logger.info("Backfill complete")
        logger.info(f"Quality summary: {self._quality_tracker.get_summary()}")
    
    # =========================================================================
    # Poll Loops
    # =========================================================================
    
    async def _ohlcv_poll_loop(self):
        """Periodically poll for new OHLCV data."""
        while self._running:
            try:
                for symbol in self.config.symbols:
                    for timeframe in self.config.timeframes:
                        await self._poll_ohlcv(symbol, timeframe)
                
                await asyncio.sleep(self.config.ohlcv_poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OHLCV poll error: {e}")
                await asyncio.sleep(5)
    
    async def _poll_ohlcv(self, symbol: str, timeframe: str):
        """Poll OHLCV for a single symbol/timeframe."""
        try:
            # Get recent bars
            end = datetime.utcnow()
            start = end - timedelta(hours=2)  # Fetch last 2 hours to catch any missed
            
            bars = await self._rest_client.get_candles(
                product_id=symbol,
                granularity=timeframe,
                start=start,
                end=end,
            )
            
            # Publish to queue for validation
            for bar in bars:
                await self._queue.publish(
                    Channels.RAW_OHLCV,
                    bar.to_dict()
                )
                
        except Exception as e:
            logger.error(f"Error polling OHLCV {symbol} {timeframe}: {e}")
    
    async def _funding_poll_loop(self):
        """Periodically poll for funding rate data."""
        while self._running:
            try:
                for symbol in self.config.symbols:
                    await self._poll_funding(symbol)
                
                await asyncio.sleep(self.config.funding_poll_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Funding poll error: {e}")
                await asyncio.sleep(5)
    
    async def _poll_funding(self, symbol: str):
        """Poll funding rate for a single symbol."""
        try:
            funding = await self._rest_client.get_funding_rate(symbol)
            
            if funding:
                await self._queue.publish(
                    Channels.RAW_FUNDING,
                    funding.to_dict()
                )
        except Exception as e:
            logger.error(f"Error polling funding {symbol}: {e}")
    
    # =========================================================================
    # Validation Loop
    # =========================================================================
    
    async def _validation_loop(self):
        """Process and validate data from queues."""
        while self._running:
            try:
                # Process raw OHLCV
                ohlcv_messages = await self._queue.get_batch(
                    Channels.RAW_OHLCV,
                    max_messages=100,
                    timeout_ms=100,
                )
                
                for msg in ohlcv_messages:
                    bar = OHLCVBar.from_dict(msg.data)
                    
                    # Get previous bar for continuity check
                    prev_time = self._last_ohlcv_times.get(bar.symbol, {}).get(bar.timeframe)
                    # Note: In production, fetch actual previous bar from DB
                    
                    result = self._validator.validate_ohlcv(bar, None)
                    self._quality_tracker.record_validation(result)
                    
                    if result.is_valid or result.quality == DataQuality.SUSPICIOUS:
                        # Store valid/suspicious data
                        self._database.insert_ohlcv(bar)
                        
                        # Publish to validated channel
                        await self._queue.publish(
                            Channels.VALIDATED_OHLCV,
                            bar.to_dict()
                        )
                        
                        # Update tracking
                        self._last_ohlcv_times.setdefault(bar.symbol, {})[bar.timeframe] = ensure_naive_utc(bar.event_time)
                
                # Process raw funding
                funding_messages = await self._queue.get_batch(
                    Channels.RAW_FUNDING,
                    max_messages=100,
                    timeout_ms=100,
                )
                
                for msg in funding_messages:
                    funding = FundingRate(
                        symbol=msg.data["symbol"],
                        event_time=datetime.fromisoformat(msg.data["event_time"]),
                        available_time=datetime.fromisoformat(msg.data["available_time"]),
                        rate=float(msg.data["rate"]),
                        mark_price=float(msg.data["mark_price"]),
                        index_price=float(msg.data["index_price"]),
                    )
                    
                    result = self._validator.validate_funding_rate(funding)
                    
                    if result.is_valid or result.quality == DataQuality.SUSPICIOUS:
                        self._database.insert_funding_rate(funding)
                        
                        await self._queue.publish(
                            Channels.VALIDATED_FUNDING,
                            funding.to_dict()
                        )
                
                await asyncio.sleep(0.01)  # Small sleep to prevent tight loop
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Validation loop error: {e}")
                await asyncio.sleep(1)
    
    # =========================================================================
    # WebSocket Handlers
    # =========================================================================
    
    def _handle_ws_ticker(self, ticker: TickerUpdate):
        """Handle ticker update from WebSocket."""
        # Validate
        result = self._validator.validate_ticker(ticker)
        
        if result.is_valid:
            # Notify callbacks
            for callback in self._on_ticker_callbacks:
                try:
                    callback(ticker)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")
    
    def _handle_ws_orderbook(self, book: OrderBookSnapshot):
        """Handle orderbook update from WebSocket."""
        result = self._validator.validate_orderbook(book)
        
        if result.is_valid:
            # Store snapshot periodically (not every update)
            # In production, this would be rate-limited
            pass
    
    def _handle_ws_error(self, error: Exception):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
        # Could implement alerting here
    
    def _handle_validated_ohlcv(self, msg: QueueMessage):
        """Handle validated OHLCV from queue."""
        bar = OHLCVBar.from_dict(msg.data)
        
        # Notify callbacks
        for callback in self._on_ohlcv_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"OHLCV callback error: {e}")
    
    # =========================================================================
    # Health Check
    # =========================================================================
    
    async def _health_check_loop(self):
        """Periodic health check and logging."""
        while self._running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Log queue sizes
                if hasattr(self._queue, 'get_queue_sizes'):
                    sizes = self._queue.get_queue_sizes()
                    logger.info(f"Queue sizes: {sizes}")
                
                # Log quality stats
                summary = self._quality_tracker.get_summary()
                logger.info(f"Data quality: {summary['validity_rate']}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
    
    # =========================================================================
    # Public Interface
    # =========================================================================
    
    def on_ohlcv(self, callback: Callable[[OHLCVBar], None]):
        """Register callback for new OHLCV data."""
        self._on_ohlcv_callbacks.append(callback)
    
    def on_ticker(self, callback: Callable[[TickerUpdate], None]):
        """Register callback for ticker updates."""
        self._on_ticker_callbacks.append(callback)
    
    def on_funding(self, callback: Callable[[FundingRate], None]):
        """Register callback for funding rate updates."""
        self._on_funding_callbacks.append(callback)
    
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
    ):
        """
        Get OHLCV data from database.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe
            start: Start time
            end: End time
            as_of: Point-in-time query (for backtesting)
        """
        # Ensure naive UTC for database queries
        start = ensure_naive_utc(start)
        end = ensure_naive_utc(end)
        as_of = ensure_naive_utc(as_of)
        
        return self._database.get_ohlcv(symbol, timeframe, start, end, as_of)
    
    def get_funding_rates(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        as_of: Optional[datetime] = None,
    ):
        """Get funding rate history from database."""
        # Ensure naive UTC for database queries
        start = ensure_naive_utc(start)
        end = ensure_naive_utc(end)
        as_of = ensure_naive_utc(as_of)
        
        return self._database.get_funding_rates(symbol, start, end, as_of)
    
    def get_quality_summary(self) -> Dict[str, Any]:
        """Get current data quality summary."""
        return self._quality_tracker.get_summary()


async def create_pipeline(config: Optional[PipelineConfig] = None) -> DataPipeline:
    """Factory function to create and initialize a data pipeline."""
    config = config or PipelineConfig()
    pipeline = DataPipeline(config)
    await pipeline.initialize()
    return pipeline