"""
Coinbase Advanced Trade API Connector.
"""

import asyncio
import hashlib
import hmac
import json
import logging
import time
import aiohttp

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlencode
from coinbase import jwt_generator  

from .timeutil import (
    epoch_seconds_to_naive_utc,
    naive_utc_to_epoch_seconds,
    utc_now,
)
from .models import (
    OHLCVBar, OrderBookSnapshot, OrderBookLevel, TickerUpdate,
)

logger = logging.getLogger(__name__)


@dataclass
class RateLimiter:
    requests_per_second: float
    burst_size: int = 10
    
    def __post_init__(self):
        self._tokens = float(self.burst_size)
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            self._tokens = min(self.burst_size, self._tokens + elapsed * self.requests_per_second)
            self._last_update = now
            if self._tokens < 1:
                wait_time = (1 - self._tokens) / self.requests_per_second
                await asyncio.sleep(wait_time)
                self._tokens = 0
            else:
                self._tokens -= 1


class CoinbaseAuth:
    def __init__(self, api_key: str, api_secret:  str):
        self.api_key = api_key
        self.api_secret = api_secret.replace('\\n', '\n')
    
    def generate_jwt(self, method: str, path: str) -> str:
        jwt_uri = jwt_generator.format_jwt_uri(method.upper(), path)
        jwt_token = jwt_generator.build_rest_jwt(jwt_uri, self.api_key, self.api_secret)
        return jwt_token


# Waits between attempts on an empty batch. Geometric, because the failure being
# recovered from is a rate limit — and a rate limit answers an immediate retry
# exactly as it answered the first request.
EMPTY_BATCH_BACKOFF_SECONDS = (1.0, 4.0, 15.0)


# The venue's hard cap on candles per request. Both `start` and `end` are
# inclusive, so a request may span at most this many candle STARTS — see
# `get_candles_range`, where using the full span instead of `n - 1` silently
# destroyed one minute in every 301.
MAX_CANDLES_PER_REQUEST = 300


class CoinbaseRESTClient:
    BASE_URL = "https://api.coinbase.com"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 rate_limit_public: int = 10, rate_limit_private: int = 15):
        self.auth = CoinbaseAuth(api_key, api_secret) if api_key and api_secret else None
        self._session: Optional[aiohttp.ClientSession] = None
        self._public_limiter = RateLimiter(rate_limit_public)
        self._private_limiter = RateLimiter(rate_limit_private)
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _request(self, method: str, path: str, params: Optional[Dict] = None, body: Optional[Dict] = None,
                       authenticated: bool = False) -> Tuple[int, Any]:
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
                async with session.request(method, url, headers=headers, json=body if body else None,
                                           timeout=aiohttp.ClientTimeout(total=30)) as response:
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
    
    async def get_candles(self, product_id: str, granularity: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None, limit: int = 300) -> List[OHLCVBar]:
        granularity_map = {
            "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR", "2h": "TWO_HOUR", "6h": "SIX_HOUR", "1d": "ONE_DAY",
        }
        api_granularity = granularity_map.get(granularity, "ONE_HOUR")
        params = {"granularity": api_granularity, "limit": min(limit, MAX_CANDLES_PER_REQUEST)}
        if start:
            params["start"] = naive_utc_to_epoch_seconds(start)
        if end:
            params["end"] = naive_utc_to_epoch_seconds(end)
        path = f"/api/v3/brokerage/market/products/{product_id}/candles"
        status, data = await self._request("GET", path, params=params, authenticated=False)
        if status != 200:
            logger.warning(f"Coinbase candle fetch failed {status}: {data}")
            return []
        candles = data.get("candles", [])
        bars = []
        tf_seconds = self._granularity_to_seconds(granularity)
        for c in candles:
            try:
                start_ts = int(c["start"])
                event_time_utc = datetime.fromtimestamp(start_ts, tz=timezone.utc)
                event_time = epoch_seconds_to_naive_utc(start_ts)
                available_time = (event_time_utc + timedelta(seconds=tf_seconds)).replace(tzinfo=None)
                bar = OHLCVBar(
                    symbol=product_id, timeframe=granularity, event_time=event_time, available_time=available_time,
                    open=float(c["open"]), high=float(c["high"]), low=float(c["low"]),
                    close=float(c["close"]), volume=float(c["volume"]),
                )
                bars.append(bar)
            except (KeyError, ValueError, TypeError):
                continue
        bars.sort(key=lambda x: x.event_time)
        # Drop the candle that has not closed yet. `end` is almost always
        # `utc_now()` at the call sites, so the newest candle returned was the
        # one for the minute in progress: a partial bar whose high, low, close
        # and volume are all wrong, stored permanently because
        # `DataPipeline._poll` only accepts `event_time > last_known` and so never
        # re-requests it. Observed directly — a scrape started at 04:17:29Z stored
        # `04:17:00` for both symbols with 51.9% of a minute's volume.
        #
        # `available_time` is already `event_time + timeframe`, i.e. the instant
        # the bar closes, so this is the same "knowable at close" rule the rest of
        # the system uses rather than a second convention.
        now = utc_now()
        closed = [b for b in bars if b.available_time <= now]
        if len(closed) != len(bars):
            logger.debug('%s: dropped %d unclosed candle(s)', product_id,
                         len(bars) - len(closed))
        return closed
    
    async def get_candles_range(self, product_id: str, granularity: str, start: datetime, end: datetime) -> List[OHLCVBar]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        all_bars = []
        current_start = start
        attempt = 0
        skipped_windows: List[tuple] = []
        tf_seconds = self._granularity_to_seconds(granularity)
        # `limit - 1`, not `limit`. Coinbase treats both `start` and `end` as
        # INCLUSIVE, so a span of `limit * tf` names `limit + 1` candle starts.
        # With `limit=300` the API returned the newest 300 and dropped the oldest
        # — `current_start` itself — and the loop below then advanced past it.
        # One minute destroyed in every 301, forever.
        #
        # It was not a rounding detail. In the five-year store this produced
        # 8,721 of BTC's 10,124 missing minutes: 98.9% of the gaps between holes
        # were exactly 301 minutes, the rate was 0.003318 against 1/301 =
        # 0.003322, and 5,121 of BTC's holes fell on the identical minute in ETH
        # against 28.9 expected by chance. Two independent order books do not go
        # untraded in the same 8,700 minutes.
        batch_span = timedelta(seconds=tf_seconds * (MAX_CANDLES_PER_REQUEST - 1))
        while current_start < end:
            batch_end = min(current_start + batch_span, end)
            if (batch_end - current_start).total_seconds() < tf_seconds:
                break
            bars = await self.get_candles(product_id, granularity, current_start, batch_end)
            if bars:
                all_bars.extend(bars)
                last_time = bars[-1].event_time.replace(tzinfo=timezone.utc) if bars[-1].event_time.tzinfo is None else bars[-1].event_time
                next_start = last_time + timedelta(seconds=tf_seconds)
                attempt = 0
                current_start = max(next_start, current_start + timedelta(seconds=tf_seconds))
            else:
                # `get_candles` returns [] for both "no data in this window" and
                # "the request failed", and advancing past a failure silently
                # drops 300 bars.
                #
                # Retried with BACKOFF, not immediately. The previous version
                # retried once with no delay, which is the one thing that cannot
                # recover the most likely cause: a rate limit answers the second
                # request exactly as it answered the first. On a five-year
                # backfill of ~8,700 batches per symbol, two gave up this way and
                # left two five-hour holes in the history.
                if attempt < len(EMPTY_BATCH_BACKOFF_SECONDS):
                    delay = EMPTY_BATCH_BACKOFF_SECONDS[attempt]
                    attempt += 1
                    logger.warning(
                        'empty candle batch for %s %s..%s: attempt %d/%d, '
                        'waiting %.1fs',
                        product_id, current_start, batch_end, attempt,
                        len(EMPTY_BATCH_BACKOFF_SECONDS), delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                skipped_windows.append((current_start, batch_end))
                attempt = 0
                current_start = batch_end
            await asyncio.sleep(0.1)
        if skipped_windows:
            logger.error(
                '%s %s: %d window(s) still empty after %d attempts with backoff '
                'and missing from the history: %s%s',
                product_id, granularity, len(skipped_windows),
                len(EMPTY_BATCH_BACKOFF_SECONDS) + 1,
                ', '.join(f'{a:%Y-%m-%d %H:%M}..{b:%Y-%m-%d %H:%M}'
                          for a, b in skipped_windows[:5]),
                '' if len(skipped_windows) <= 5 else f' (+{len(skipped_windows) - 5} more)',
            )
            logger.error(
                'recover these with `python -m scripts.scrape --fill-gaps`, which '
                're-requests only the minutes the store is missing'
            )
        # Recorded on the client so a caller can act on them rather than parse a
        # log line. Overwritten per call, which is what a caller wants.
        self.last_skipped_windows = list(skipped_windows)
        unique = {b.event_time: b for b in all_bars}
        return sorted(unique.values(), key=lambda x: x.event_time)
    
    async def get_ticker(self, product_id: str) -> Optional[TickerUpdate]:
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
            now = utc_now()
            return TickerUpdate(
                symbol=product_id, event_time=now, available_time=now,
                price=float(latest.get("price", 0)),
                best_bid=float(best_bid) if best_bid else 0.0,
                best_ask=float(best_ask) if best_ask else 0.0,
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to parse ticker: {e}")
            return None
    
    def _granularity_to_seconds(self, granularity: str) -> int:
        mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "6h": 21600, "1d": 86400}
        return mapping.get(granularity, 3600)
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        logger.debug("Coinbase REST client closed")


class CoinbaseWebSocketClient:
    WS_URL = "wss://advanced-trade-ws.coinbase.com"
    
    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 on_ticker: Optional[Callable[[TickerUpdate], None]] = None,
                 on_orderbook: Optional[Callable[[OrderBookSnapshot], None]] = None,
                 on_error: Optional[Callable[[Exception], None]] = None):
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
        try:
            self._session = aiohttp.ClientSession()
            self._ws = await self._session.ws_connect(self.WS_URL, heartbeat=30, receive_timeout=60)
            self._running = True
            logger.info("WebSocket connected")
            return True
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            if self.on_error:
                self.on_error(e)
            return False
    
    async def subscribe(self, channels: List[str], product_ids: List[str]) -> bool:
        if not self._ws or self._ws.closed:
            if not await self.connect():
                return False
        message = {
            "type": "subscribe",
            "product_ids": product_ids,
            "channel": channels[0] if len(channels) == 1 else channels,
        }
        try:
            await self._ws.send_json(message)
            for channel in channels:
                self._subscriptions.setdefault(channel, []).extend(product_ids)
            logger.info(f"Subscribed to {channels} for {product_ids}")
            return True
        except Exception as e:
            logger.error(f"Subscription failed: {e}")
            return False
    
    async def start(self):
        if not self._ws:
            await self.connect()
        self._running = True
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info("WebSocket client started")
    
    async def _receive_loop(self):
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
        try:
            events = data.get("events", [{}])
            for event in events:
                tickers = event.get("tickers", [])
                for t in tickers:
                    now = utc_now()
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
        try:
            product_id = data.get("product_id")
            if not product_id:
                return
            now = utc_now()
            bids = [OrderBookLevel(price=float(b[0]), size=float(b[1])) for b in data.get("bids", [])]
            asks = [OrderBookLevel(price=float(a[0]), size=float(a[1])) for a in data.get("asks", [])]
            bids.sort(key=lambda x: x.price, reverse=True)
            asks.sort(key=lambda x: x.price)
            snapshot = OrderBookSnapshot(symbol=product_id, event_time=now, available_time=now, bids=bids, asks=asks)
            self._orderbook_state[product_id] = snapshot
            if self.on_orderbook:
                self.on_orderbook(snapshot)
        except Exception as e:
            logger.error(f"Failed to handle orderbook snapshot: {e}")
    
    async def _handle_orderbook_update(self, data: Dict):
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
                now = utc_now()
                current.event_time = now
                current.available_time = now
                if self.on_orderbook:
                    self.on_orderbook(current)
        except Exception as e:
            logger.error(f"Failed to handle orderbook update: {e}")
    
    def _update_book_side(self, levels: List[OrderBookLevel], price: float, size: float, descending: bool = True):
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
        return self._orderbook_state.get(product_id)