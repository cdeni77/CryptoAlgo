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

from .models import OHLCVBar, FundingRate, TickerUpdate, DataQuality, OrderBookSnapshot
from .validator import DataValidator, ValidationConfig, DataQualityTracker
from .queue import MessageQueueBase, InMemoryQueue, Channels, QueueMessage
from .storage import DatabaseBase, SQLiteDatabase, create_database
from .coinbase_connector import CoinbaseRESTClient, CoinbaseWebSocketClient
from .ccxt_connector import CCXTConnector

logger = logging.getLogger(__name__)


def ensure_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


@dataclass
class PipelineConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTC-PERP", "ETH-PERP"])
    timeframes: List[str] = field(default_factory=lambda: ["1m", "5m", "15m", "1h"])
    coinbase_api_key: Optional[str] = None
    coinbase_api_secret: Optional[str] = None
    proxy: Optional[str] = None
    ccxt_exchanges: List[str] = field(default_factory=lambda: ["binance", "bybit"])
    ccxt_use_fallbacks: bool = True
    db_type: str = "sqlite"
    db_path: str = "./data/trading.db"
    queue_type: str = "memory"
    ohlcv_poll_interval: int = 60
    funding_poll_interval: int = 300
    backfill_days: int = 30
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    enable_funding_polling: bool = True


class DataPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._rest_client: Optional[CoinbaseRESTClient] = None
        self._ws_client: Optional[CoinbaseWebSocketClient] = None
        self._ccxt_connector: Optional[CCXTConnector] = None
        self._queue: Optional[MessageQueueBase] = None
        self._database: Optional[DatabaseBase] = None
        self._validator: DataValidator = DataValidator(config.validation_config)
        self._quality_tracker: DataQualityTracker = DataQualityTracker()
        self._running = False
        self._tasks: List[asyncio.Task] = []
        self._last_ohlcv_times: Dict[str, Dict[str, datetime]] = {}
        self._first_ohlcv_times: Dict[str, Dict[str, datetime]] = {}
        self._on_ohlcv_callbacks: List[Callable[[OHLCVBar], None]] = []
        self._on_ticker_callbacks: List[Callable[[TickerUpdate], None]] = []
        self._on_funding_callbacks: List[Callable[[FundingRate], None]] = []
        self._funding_failed_symbols = set()

    async def initialize(self):
        logger.info("Initializing data pipeline...")
        if self.config.db_type == "sqlite":
            self._database = SQLiteDatabase(self.config.db_path)
        else:
            self._database = create_database(self.config.db_type, db_path=self.config.db_path)
        self._database.initialize()
        logger.info(f"Database initialized: {self.config.db_type}")

        self._queue = InMemoryQueue()
        await self._queue.start()
        logger.info(f"Message queue initialized: {self.config.queue_type}")

        self._rest_client = CoinbaseRESTClient(
            api_key=self.config.coinbase_api_key,
            api_secret=self.config.coinbase_api_secret,
        )
        logger.info("REST client initialized")

        self._ws_client = CoinbaseWebSocketClient(
            api_key=self.config.coinbase_api_key,
            api_secret=self.config.coinbase_api_secret,
            on_ticker=self._handle_ws_ticker,
            on_orderbook=self._handle_ws_orderbook,
            on_error=self._handle_ws_error,
        )
        logger.info("WebSocket client initialized")

        self._ccxt_connector = CCXTConnector(
            exchanges=self.config.ccxt_exchanges,
            proxy=self.config.proxy,
            use_fallbacks=self.config.ccxt_use_fallbacks,
        )
        await self._ccxt_connector.initialize()
        logger.info("CCXT connector initialized")

        await self._load_last_times()
        await self._load_first_times()
        logger.info("Data pipeline initialization complete")

    async def _load_last_times(self):
        for symbol in self.config.symbols:
            self._last_ohlcv_times[symbol] = {}
            for timeframe in self.config.timeframes:
                last_time = self._database.get_latest_ohlcv_time(symbol, timeframe)
                if last_time:
                    last_time = ensure_naive_utc(last_time)
                    self._last_ohlcv_times[symbol][timeframe] = last_time

    async def _load_first_times(self):
        for symbol in self.config.symbols:
            self._first_ohlcv_times[symbol] = {}
            for timeframe in self.config.timeframes:
                first_time = self._database.get_earliest_ohlcv_time(symbol, timeframe)
                if first_time:
                    first_time = ensure_naive_utc(first_time)
                    self._first_ohlcv_times[symbol][timeframe] = first_time

    async def start(self):
        if self._running:
            logger.warning("Pipeline already running")
            return
        self._running = True
        logger.info("Starting data pipeline...")

        try:
            await self._ws_client.connect()
            await self._ws_client.subscribe(channels=["ticker"], product_ids=self.config.symbols)
            await self._ws_client.start()
            logger.info("WebSocket connected and subscribed")
        except Exception as e:
            logger.warning(f"WebSocket failed ({e}), falling back to REST polling only")

        await self._queue.subscribe(Channels.VALIDATED_OHLCV, self._handle_validated_ohlcv)

        self._tasks = [
            asyncio.create_task(self._ohlcv_poll_loop()),
            asyncio.create_task(self._validation_loop()),
            asyncio.create_task(self._health_check_loop()),
        ]

        if self.config.enable_funding_polling and self.config.coinbase_api_key:
            self._tasks.append(asyncio.create_task(self._funding_poll_loop()))

        logger.info("Data pipeline started")

    async def stop(self):
        logger.info("Stopping data pipeline...")
        self._running = False
        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
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

    def _granularity_to_seconds(self, granularity: str) -> int:
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
            "1h": 3600, "2h": 7200, "6h": 21600, "1d": 86400,
        }
        return mapping.get(granularity.lower(), 3600)

    async def _fetch_bars(self, symbol, timeframe, start_dt, end_dt, use_ccxt):
        if use_ccxt:
            return await self._ccxt_connector.fetch_ohlcv(symbol=symbol, timeframe=timeframe, start=start_dt, end=end_dt)
        else:
            return await self._rest_client.get_candles_range(product_id=symbol, granularity=timeframe, start=start_dt, end=end_dt)

    def _process_and_insert_bars(self, bars, symbol, timeframe):
        if not bars:
            return 0
        valid_bars = []
        prev_bar = None
        for bar in bars:
            result = self._validator.validate_ohlcv(bar, prev_bar)
            self._quality_tracker.record_validation(result)
            if result.is_valid or result.quality == DataQuality.SUSPICIOUS:
                valid_bars.append(bar)
            prev_bar = bar
        if not valid_bars:
            return 0
        inserted = self._database.insert_ohlcv_batch(valid_bars)
        logger.info(f"Backfilled {inserted}/{len(bars)} bars for {symbol} {timeframe}")
        first_event = ensure_naive_utc(valid_bars[0].event_time)
        last_event = ensure_naive_utc(valid_bars[-1].event_time)
        current_last = self._last_ohlcv_times.get(symbol, {}).get(timeframe)
        if current_last is None or last_event > current_last:
            self._last_ohlcv_times.setdefault(symbol, {})[timeframe] = last_event
        current_first = self._first_ohlcv_times.get(symbol, {}).get(timeframe)
        if current_first is None or first_event < current_first:
            self._first_ohlcv_times.setdefault(symbol, {})[timeframe] = first_event
        return inserted

    async def backfill(self, start: Optional[datetime] = None, end: Optional[datetime] = None,
                       symbols: Optional[List[str]] = None, timeframes: Optional[List[str]] = None, use_ccxt: bool = True):
        end = ensure_naive_utc(end) if end else datetime.utcnow()
        start = ensure_naive_utc(start) if start else (end - timedelta(days=self.config.backfill_days))
        symbols = symbols or self.config.symbols
        timeframes = timeframes or self.config.timeframes
        logger.info(f"Starting backfill from {start} to {end}")
        logger.info(f"Symbols: {symbols}, Timeframes: {timeframes}")
        for symbol in symbols:
            for timeframe in timeframes:
                tf_seconds = self._granularity_to_seconds(timeframe)
                tf_delta = timedelta(seconds=tf_seconds)
                first_time = self._first_ohlcv_times.get(symbol, {}).get(timeframe)
                last_time = self._last_ohlcv_times.get(symbol, {}).get(timeframe)
                if first_time:
                    first_time = ensure_naive_utc(first_time)
                if last_time:
                    last_time = ensure_naive_utc(last_time)
                fetched_any = False
                if use_ccxt and (not first_time or first_time > start + tf_delta):
                    prepend_end = first_time if first_time else end
                    prepend_start = start
                    if prepend_start < prepend_end:
                        logger.info(f"Prepending {symbol} {timeframe} from {prepend_start} to {prepend_end}")
                        try:
                            bars = await self._fetch_bars(symbol, timeframe, prepend_start, prepend_end, use_ccxt)
                            inserted = self._process_and_insert_bars(bars, symbol, timeframe)
                            if inserted:
                                fetched_any = True
                        except Exception as e:
                            logger.error(f"Error prepending {symbol} {timeframe}: {e}")
                            import traceback
                            traceback.print_exc()
                append_start = start
                if last_time:
                    append_start = last_time + tf_delta
                if append_start < end:
                    logger.info(f"Appending {symbol} {timeframe} from {append_start} to {end}")
                    try:
                        bars = await self._fetch_bars(symbol, timeframe, append_start, end, use_ccxt)
                        inserted = self._process_and_insert_bars(bars, symbol, timeframe)
                        if inserted:
                            fetched_any = True
                    except Exception as e:
                        logger.error(f"Error appending {symbol} {timeframe}: {e}")
                        import traceback
                        traceback.print_exc()
                if not fetched_any:
                    logger.info(f"Skipping {symbol} {timeframe} - already up to date in requested range")
        logger.info("Backfill complete")
        logger.info(f"Quality summary: {self._quality_tracker.get_summary()}")

    async def _ohlcv_poll_loop(self):
        while self._running:
            try:
                now = datetime.utcnow()
                for symbol in self.config.symbols:
                    for timeframe in self.config.timeframes:
                        tf_seconds = self._granularity_to_seconds(timeframe)
                        tf_delta = timedelta(seconds=tf_seconds)
                        last_time = self._last_ohlcv_times.get(symbol, {}).get(timeframe)
                        if last_time:
                            last_time = ensure_naive_utc(last_time)
                        if last_time:
                            next_expected = last_time + tf_delta
                        else:
                            aligned = int(now.timestamp()) // tf_seconds * tf_seconds
                            next_expected = datetime.utcfromtimestamp(aligned)
                        if now >= next_expected - timedelta(seconds=30):
                            start_fetch = next_expected - tf_delta * 2
                            start_fetch = max(start_fetch, now - timedelta(days=1))
                            end_fetch = now
                            if start_fetch < end_fetch:
                                logger.debug(f"Polling {symbol} {timeframe} from {start_fetch}")
                                await self._poll_ohlcv(symbol, timeframe, start_fetch, end_fetch)
                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"OHLCV poll loop error: {e}")
                await asyncio.sleep(10)

    async def _poll_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime):
        try:
            start = ensure_naive_utc(start)
            end = ensure_naive_utc(end)
            if start >= end:
                return
            bars = await self._rest_client.get_candles_range(product_id=symbol, granularity=timeframe, start=start, end=end)
            if not bars:
                return
            last_known = self._last_ohlcv_times.get(symbol, {}).get(timeframe)
            new_bars = [b for b in bars if not last_known or b.event_time > last_known]
            if new_bars:
                logger.info(f"🕯  NEW {len(new_bars)} {timeframe} candles for {symbol}")
                for bar in new_bars:
                    await self._queue.publish(Channels.RAW_OHLCV, bar.to_dict())
        except Exception as e:
            logger.error(f"Error polling OHLCV {symbol} {timeframe}: {e}")

    async def _funding_poll_loop(self):
        while self._running:
            try:
                for symbol in self.config.symbols:
                    await self._poll_funding(symbol)
                await asyncio.sleep(self.config.funding_poll_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Funding poll loop error: {e}")
                await asyncio.sleep(10)

    async def _poll_funding(self, symbol: str):
        try:
            funding = await self._rest_client.get_funding_rate(symbol)
            if funding:
                await self._queue.publish(Channels.RAW_FUNDING, funding.to_dict())
                if symbol in self._funding_failed_symbols:
                    logger.info(f"Funding rate now available for {symbol}")
                    self._funding_failed_symbols.remove(symbol)
            else:
                if symbol not in self._funding_failed_symbols:
                    logger.info(f"No funding rate available for {symbol} – likely no open position")
                    self._funding_failed_symbols.add(symbol)
        except Exception as e:
            if symbol not in self._funding_failed_symbols:
                logger.info(f"Funding rate polling failed for {symbol} (will retry silently): {e}")
                self._funding_failed_symbols.add(symbol)

    async def _validation_loop(self):
        while self._running:
            try:
                ohlcv_messages = await self._queue.get_batch(Channels.RAW_OHLCV, max_messages=100, timeout_ms=100)
                for msg in ohlcv_messages:
                    bar = OHLCVBar.from_dict(msg.data)
                    result = self._validator.validate_ohlcv(bar, None)
                    self._quality_tracker.record_validation(result)
                    if result.is_valid or result.quality == DataQuality.SUSPICIOUS:
                        self._database.insert_ohlcv(bar)
                        await self._queue.publish(Channels.VALIDATED_OHLCV, bar.to_dict())
                        self._last_ohlcv_times.setdefault(bar.symbol, {})[bar.timeframe] = ensure_naive_utc(bar.event_time)
                funding_messages = await self._queue.get_batch(Channels.RAW_FUNDING, max_messages=100, timeout_ms=100)
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
                        await self._queue.publish(Channels.VALIDATED_FUNDING, funding.to_dict())
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Validation loop error: {e}")
                await asyncio.sleep(1)

    def _handle_ws_ticker(self, ticker: TickerUpdate):
        result = self._validator.validate_ticker(ticker)
        if result.is_valid:
            for callback in self._on_ticker_callbacks:
                try:
                    callback(ticker)
                except Exception as e:
                    logger.error(f"Ticker callback error: {e}")

    def _handle_ws_orderbook(self, book: OrderBookSnapshot):
        result = self._validator.validate_orderbook(book)
        if result.is_valid:
            pass

    def _handle_ws_error(self, error: Exception):
        logger.error(f"WebSocket error: {error}")

    def _handle_validated_ohlcv(self, msg: QueueMessage):
        bar = OHLCVBar.from_dict(msg.data)
        for callback in self._on_ohlcv_callbacks:
            try:
                callback(bar)
            except Exception as e:
                logger.error(f"OHLCV callback error: {e}")

    async def _health_check_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)
                if hasattr(self._queue, 'get_queue_sizes'):
                    sizes = self._queue.get_queue_sizes()
                    logger.info(f"Queue sizes: {sizes}")
                summary = self._quality_tracker.get_summary()
                logger.info(f"Data quality: {summary['validity_rate']}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    def on_ohlcv(self, callback: Callable[[OHLCVBar], None]):
        self._on_ohlcv_callbacks.append(callback)

    def on_ticker(self, callback: Callable[[TickerUpdate], None]):
        self._on_ticker_callbacks.append(callback)

    def on_funding(self, callback: Callable[[FundingRate], None]):
        self._on_funding_callbacks.append(callback)

    def get_ohlcv(self, symbol: str, timeframe: str, start: datetime, end: datetime, as_of: Optional[datetime] = None):
        start = ensure_naive_utc(start)
        end = ensure_naive_utc(end)
        as_of = ensure_naive_utc(as_of)
        return self._database.get_ohlcv(symbol, timeframe, start, end, as_of)

    def get_funding_rates(self, symbol: str, start: datetime, end: datetime, as_of: Optional[datetime] = None):
        start = ensure_naive_utc(start)
        end = ensure_naive_utc(end)
        as_of = ensure_naive_utc(as_of)
        return self._database.get_funding_rates(symbol, start, end, as_of)

    def get_quality_summary(self) -> Dict[str, Any]:
        return self._quality_tracker.get_summary()


async def create_pipeline(config: Optional[PipelineConfig] = None) -> DataPipeline:
    config = config or PipelineConfig()
    pipeline = DataPipeline(config)
    await pipeline.initialize()
    return pipeline