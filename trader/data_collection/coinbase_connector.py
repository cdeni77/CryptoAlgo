"""
Coinbase Advanced Trade API Connector.

Handles both REST API and WebSocket connections for real-time and historical data.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import aiohttp
from coinbase import jwt_generator  # Official SDK import

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
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    """Token bucket rate limiter for API requests."""
    requests_per_second: float
    burst_size: int = 10
    
    def __post_init__(self):
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, waiting if necessary."""
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(
                self.burst_size,
                self._tokens + elapsed * self.requests_per_second
            )
            self._last_update = now
            
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class CoinbaseAuth:
    """Authentication handler for Coinbase Advanced Trade API (using official SDK)."""
    
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key  # full "organizations/{org_id}/apiKeys/{key_id}"
        self.api_secret = api_secret.replace('\\n', '\n')  # Ensure real newlines
    
    def generate_jwt(self, method: str, path: str) -> str:
        """Generate JWT using official coinbase-advanced-py SDK."""
        jwt_uri = jwt_generator.format_jwt_uri(method.upper(), path)
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, self.api_key, self.api_secret)
        return jwt_token


class CoinbaseRESTClient:
    """
    REST API client for Coinbase Advanced Trade.
    
    Handles:
    - Historical OHLCV data
    - Product information
    - Funding rates
    - Account data (authenticated)
    """
    
    BASE_URL = "https://api.coinbase.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        rate_limit_public: int = 10,
        rate_limit_private: int = 15,
    ):
        self.auth = CoinbaseAuth(api_key, api_secret) if api_key and api_secret else None
        self._session: Optional[aiohttp.ClientSession] = None
        self._public_limiter = RateLimiter(rate_limit_public)
        self._private_limiter = RateLimiter(rate_limit_private)
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None,
        authenticated: bool = False,
    ) -> Tuple[int, Any]:
        """Make API request with rate limiting and error handling."""
        session = await self._ensure_session()
        
        limiter = self._private_limiter if authenticated else self._public_limiter
        await limiter.acquire()
        
        url = f"{self.BASE_URL}{path}"
        if params:
            url += "?" + urlencode(params)
        
        headers = {"Content-Type": "application/json"}
        
        if authenticated:
            if not self.auth:
                raise ValueError("Authentication required but no credentials provided")
            jwt_token = self.auth.generate_jwt(method, path)
            headers["Authorization"] = f"Bearer {jwt_token}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with session.request(
                    method,
                    url,
                    headers=headers,
                    json=body if body else None,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return response.status, data
                    else:
                        error_text = await response.text()
                        logger.error(f"Error {response.status}: {error_text}")
                        
                    if response.status == 429:
                        retry_after = int(response.headers.get("Retry-After", 1))
                        logger.warning(f"Rate limited, waiting {retry_after}s")
                        await asyncio.sleep(retry_after)
                        continue
                    
                    return response.status, {"error": error_text}
                    
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
            
            except aiohttp.ClientError as e:
                logger.error(f"Request error: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        raise RuntimeError(f"Failed after {max_retries} retries")
    
    async def get_products(self) -> List[Dict]:
        """Get all available products."""
        status, data = await self._request("GET", "/api/v3/brokerage/products", authenticated=True)
        if status != 200:
            logger.error(f"Failed to get products: {data}")
            return []
        return data.get("products", [])
    
    async def get_perpetual_products(self) -> List[Dict]:
        """Get perpetual futures products only."""
        products = await self.get_products()
        return [p for p in products if p.get("product_type") == "FUTURE" and "PERP" in p.get("product_id", "")]
    
    async def get_candles(
        self,
        product_id: str,
        granularity: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 300,
    ) -> List[OHLCVBar]:
        """
        Fetch OHLCV candles.
        
        Args:
            product_id: e.g., "BTC-PERP"
            granularity: "1m", "5m", "15m", "1h", "6h", "1d"
            start: Start time
            end: End time
            limit: Max candles to return (max 300)
        """
        granularity_map = {
            "1m": "ONE_MINUTE",
            "5m": "FIVE_MINUTE",
            "15m": "FIFTEEN_MINUTE",
            "1h": "ONE_HOUR",
            "6h": "SIX_HOUR",
            "1d": "ONE_DAY",
        }
        
        api_granularity = granularity_map.get(granularity, granularity)
        
        params = {
            "granularity": api_granularity,
            "limit": min(limit, 300),
        }
        
        if start:
            params["start"] = int(start.timestamp())
        if end:
            params["end"] = int(end.timestamp())
        
        path = f"/api/v3/brokerage/products/{product_id}/candles"
        status, data = await self._request("GET", path, params=params)
        
        if status != 200:
            logger.error(f"Failed to get candles for {product_id}: {data}")
            return []
        
        candles = data.get("candles", [])
        bars = []
        
        for c in candles:
            try:
                event_time = datetime.utcfromtimestamp(int(c["start"]))
                tf_seconds = self._granularity_to_seconds(granularity)
                available_time = event_time + timedelta(seconds=tf_seconds + 5)
                
                bar = OHLCVBar(
                    symbol=product_id,
                    timeframe=granularity,
                    event_time=event_time,
                    available_time=available_time,
                    open=float(c["open"]),
                    high=float(c["high"]),
                    low=float(c["low"]),
                    close=float(c["close"]),
                    volume=float(c["volume"]),
                )
                bars.append(bar)
            except (KeyError, ValueError) as e:
                logger.warning(f"Failed to parse candle: {e}")
                continue
        
        bars.sort(key=lambda x: x.event_time)
        return bars
    
    async def get_candles_range(
        self,
        product_id: str,
        granularity: str,
        start: datetime,
        end: datetime,
    ) -> List[OHLCVBar]:
        """Fetch candles for a date range, handling pagination."""
        all_bars = []
        current_start = start
        tf_seconds = self._granularity_to_seconds(granularity)
        expected_bars = int((end - start).total_seconds() / tf_seconds)
        
        logger.info(
            f"Fetching {product_id} {granularity} candles from {start} to {end} "
            f"(~{expected_bars} bars)"
        )
        
        while current_start < end:
            batch_end = min(
                current_start + timedelta(seconds=tf_seconds * 300),
                end
            )
            
            bars = await self.get_candles(
                product_id=product_id,
                granularity=granularity,
                start=current_start,
                end=batch_end,
            )
            
            if not bars:
                logger.warning(f"No candles returned for {current_start} to {batch_end}")
                break
            
            all_bars.extend(bars)
            current_start = bars[-1].event_time + timedelta(seconds=tf_seconds)
            
            if len(all_bars) % 1000 == 0:
                logger.info(f"Fetched {len(all_bars)} bars...")
        
        logger.info(f"Fetched {len(all_bars)} total bars for {product_id}")
        return all_bars
    
    async def get_ticker(self, product_id: str) -> Optional[TickerUpdate]:
        """Get current ticker for a product."""
        path = f"/api/v3/brokerage/products/{product_id}/ticker"
        status, data = await self._request("GET", path, params={"limit": 1})
        
        if status != 200:
            logger.error(f"Failed to get ticker for {product_id}: {data}")
            return None
        
        try:
            trades = data.get("trades", [])
            best_bid = data.get("best_bid")
            best_ask = data.get("best_ask")
            
            if not trades:
                return None
            
            latest = trades[0]
            now = datetime.utcnow()
            
            return TickerUpdate(
                symbol=product_id,
                event_time=now,
                available_time=now,
                price=float(latest.get("price", 0)),
                best_bid=float(best_bid) if best_bid else 0.0,
                best_ask=float(best_ask) if best_ask else 0.0,
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse ticker: {e}")
            return None
    
    async def get_perpetuals_portfolio_summary(self) -> Optional[Dict]:
        """Get perpetuals portfolio summary (authenticated)."""
        if not self.auth:
            logger.error("Authentication required for portfolio summary")
            return None
        
        status, data = await self._request(
            "GET",
            "/api/v3/brokerage/intx/portfolio",
            authenticated=True
        )
        
        if status != 200:
            logger.error(f"Failed to get portfolio summary: {data}")
            return None
        
        return data
    
    def _granularity_to_seconds(self, granularity: str) -> int:
        """Convert granularity string to seconds."""
        mapping = {
            "1m": 60, "5m": 300, "15m": 900, "1h": 3600,
            "4h": 14400, "6h": 21600, "1d": 86400,
        }
        return mapping.get(granularity, 3600)
    
    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Coinbase REST client closed")


class CoinbaseWebSocketClient:
    """WebSocket client for real-time Coinbase data."""
    
    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        on_ticker: Optional[Callable[[TickerUpdate], None]] = None,
        on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
    ):
        self.auth = CoinbaseAuth(api_key, api_secret) if api_key and api_secret else None
        self.on_ticker = on_ticker
        self.on_orderbook = on_orderbook
        self.on_error = on_error
        
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._subscriptions: Dict[str, List[str]] = {}
        self._orderbook_state: Dict[str, OrderBookSnapshot] = {}
        self._reconnect_delay = 5
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(
                self.WS_URL,
                heartbeat=30,
                receive_timeout=60,
            )
            self._running = True
            logger.info("WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe(
        self,
        channels: List[str],
        product_ids: List[str],
    ) -> bool:
        """Subscribe to channels for products."""
        if not self._ws or self._ws.closed:
            if not await self.connect():
                return False
        
        message = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channel": channels[0] if len(channels) == 1 else channels,
        }
        
        if self.auth:
            timestamp = str(int(time.time()))
            channel_str = ",".join(channels)
            products_str = ",".join(product_ids)
            sign_message = f"{timestamp}{channel_str}{products_str}"
            signature = hmac.new(
                self.auth.api_secret.encode(),
                sign_message.encode(),
                hashlib.sha256
            ).hexdigest()
            message.update({
                "api_key": self.auth.api_key,
                "timestamp": timestamp,
                "signature": signature,
            })
        
        try:
            await self._ws.send_json(message)
            for channel in channels:
                if channel not in self._subscriptions:
                    self._subscriptions[channel] = []
                self._subscriptions[channel].extend(product_ids)
            logger.info(f"Subscribed to {channels} for {product_ids}")
            return True
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    async def start(self):
        """Start receiving messages."""
        if not self._ws:
            await self.connect()
        
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket client started")
    
    async def _receive_loop(self):
        """Main message receiving loop."""
        while self._running:
            try:
                if not self._ws or self._ws.closed:
                    logger.warning("WebSocket closed, reconnecting...")
                    await asyncio.sleep(self._reconnect_delay)
                    await self._reconnect()
                    continue
                
                msg = await self._ws.receive(timeout=30)
                
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(json.loads(msg.data))
                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.warning("WebSocket closed by server")
                    await self._reconnect()
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    await self._reconnect()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Receive error: {e}")
                if self.on_error:
                    self.on_error(e)
                await asyncio.sleep(1)
    
    async def _handle_message(self, data: Dict):
        """Process incoming WebSocket message."""
        msg_type = data.get("type") or data.get("channel")
        
        if msg_type == "subscriptions":
            logger.debug(f"Subscription confirmed: {data}")
            return
        if msg_type == "error":
            logger.error(f"WebSocket error: {data.get('message')}")
            return
        if msg_type == "ticker":
            await self._handle_ticker(data)
        elif msg_type in ("l2update", "level2"):
            await self._handle_orderbook_update(data)
        elif msg_type == "snapshot":
            await self._handle_orderbook_snapshot(data)
    
    async def _handle_ticker(self, data: Dict):
        """Handle ticker update."""
        try:
            events = data.get("events", [{}])
            for event in events:
                tickers = event.get("tickers", [])
                for t in tickers:
                    now = datetime.utcnow()
                    ticker = TickerUpdate(
                        symbol=t.get("product_id", ""),
                        event_time=now,
                        available_time=now,
                        price=float(t.get("price", 0)),
                        best_bid=float(t.get("best_bid", 0)),
                        best_ask=float(t.get("best_ask", 0)),
                        volume_24h=float(t.get("volume_24_h", 0)) if t.get("volume_24_h") else None,
                    )
                    if self.on_ticker:
                        self.on_ticker(ticker)
        except Exception as e:
            logger.error(f"Failed to handle ticker: {e}")
    
    async def _handle_orderbook_snapshot(self, data: Dict):
        """Handle full orderbook snapshot."""
        try:
            product_id = data.get("product_id")
            if not product_id:
                return
            
            now = datetime.utcnow()
            bids = [
                OrderBookLevel(price=float(b[0]), size=float(b[1]))
                for b in data.get("bids", [])
            ]
            asks = [
                OrderBookLevel(price=float(a[0]), size=float(a[1]))
                for a in data.get("asks", [])
            ]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            
            snapshot = OrderBookSnapshot(
                symbol=product_id,
                event_time=now,
                available_time=now,
                bids=bids,
                asks=asks,
            )
            self._orderbook_state[product_id] = snapshot
            
            if self.on_orderbook:
                self.on_orderbook(snapshot)
        except Exception as e:
            logger.error(f"Failed to handle orderbook snapshot: {e}")
    
    async def _handle_orderbook_update(self, data: Dict):
        """Handle incremental orderbook update."""
        try:
            events = data.get("events", [])
            for event in events:
                product_id = event.get("product_id")
                if not product_id or product_id not in self._orderbook_state:
                    continue
                
                current = self._orderbook_state[product_id]
                for update in event.get("updates", []):
                    side = update.get("side")
                    price = float(update.get("price_level", 0))
                    size = float(update.get("new_quantity", 0))
                    
                    if side == "bid":
                        self._update_book_side(current.bids, price, size, descending=True)
                    else:
                        self._update_book_side(current.asks, price, size, descending=False)
                
                now = datetime.utcnow()
                current.event_time = now
                current.available_time = now
                
                if self.on_orderbook:
                    self.on_orderbook(current)
        except Exception as e:
            logger.error(f"Failed to handle orderbook update: {e}")
    
    def _update_book_side(
        self,
        levels: List[OrderBookLevel],
        price: float,
        size: float,
        descending: bool = True
    ):
        """Update one side of the order book."""
        for i, level in enumerate(levels):
            if abs(level.price - price) < 1e-10:
                if size == 0:
                    levels.pop(i)
                else:
                    levels[i].size = size
                return
        
        if size > 0:
            levels.append(OrderBookLevel(price=price, size=size))
            levels.sort(key=lambda x: x.price, reverse=descending)
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        while self._running:
            try:
                await asyncio.sleep(25)
                if self._ws and not self._ws.closed:
                    await self._ws.ping()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Heartbeat error: {e}")
    
    async def _reconnect(self):
        """Reconnect and resubscribe."""
        logger.info("Attempting reconnection...")
        if self._ws and not self._ws.closed:
            await self._ws.close()
        
        if not await self.connect():
            return
        
        for channel, products in self._subscriptions.items():
            if products:
                await self.subscribe([channel], products)
        logger.info("Reconnection complete")
    
    async def stop(self):
        """Stop the WebSocket client."""
        self._running = False
        
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
        
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        
        logger.info("WebSocket client stopped")
    
    def get_orderbook(self, product_id: str) -> Optional[OrderBookSnapshot]:
        """Get current orderbook state for a product."""
        return self._orderbook_state.get(product_id)