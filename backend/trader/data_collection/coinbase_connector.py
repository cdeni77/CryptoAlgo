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
    ensure_naive_utc,
    epoch_seconds_to_naive_utc,
    naive_utc_to_epoch_seconds,
    utc_now,
)
from .models import (
    OHLCVBar, FundingRate, OpenInterest, OrderBookSnapshot, OrderBookLevel,
    TickerUpdate, DataQuality,
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


def _funding_interval_hours(raw: Any) -> float:
    """Parse `funding_interval` ("3600s") into hours. Defaults to 1.

    Read rather than assumed: every rate is divided by this to reach
    decimal/hour, so a venue that switches to 8h settlement would otherwise
    inflate every carry feature eightfold with nothing to notice it.
    """
    if raw in (None, ""):
        return 1.0
    text = str(raw).strip().lower()
    try:
        if text.endswith("s"):
            return max(float(text[:-1]) / 3600.0, 1e-9)
        if text.endswith("h"):
            return max(float(text[:-1]), 1e-9)
        return max(float(text) / 3600.0, 1e-9)
    except ValueError:
        logger.warning("unparseable funding_interval %r, assuming hourly", raw)
        return 1.0


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
    
    async def get_products(self, product_type: str = "SPOT") -> List[Dict]:
        params = {"product_type": product_type}
        status, data = await self._request("GET", "/api/v3/brokerage/products", params=params, authenticated=True)
        if status != 200:
            logger.error(f"Failed to get products: {data}")
            return []
        return data.get("products", [])
    
    async def get_perpetual_products(self, target_codes: Optional[List[str]] = None) -> List[Dict]:
        products = await self.get_products(product_type="FUTURE")
        if not target_codes:
            return products
        targets = {t.upper() for t in target_codes}
        return [p for p in products if any(t in p.get("product_id", "").upper() for t in targets)]
    
    async def get_candles(self, product_id: str, granularity: str, start: Optional[datetime] = None,
                          end: Optional[datetime] = None, limit: int = 300) -> List[OHLCVBar]:
        granularity_map = {
            "1m": "ONE_MINUTE", "5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE",
            "1h": "ONE_HOUR", "2h": "TWO_HOUR", "6h": "SIX_HOUR", "1d": "ONE_DAY",
        }
        api_granularity = granularity_map.get(granularity, "ONE_HOUR")
        params = {"granularity": api_granularity, "limit": min(limit, 300)}
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
        return bars
    
    async def get_candles_range(self, product_id: str, granularity: str, start: datetime, end: datetime) -> List[OHLCVBar]:
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        all_bars = []
        current_start = start
        retried_window = False
        skipped_windows: List[tuple] = []
        tf_seconds = self._granularity_to_seconds(granularity)
        batch_duration = tf_seconds * 300
        while current_start < end:
            batch_end = min(current_start + timedelta(seconds=batch_duration), end)
            if (batch_end - current_start).total_seconds() < tf_seconds:
                break
            bars = await self.get_candles(product_id, granularity, current_start, batch_end)
            if bars:
                all_bars.extend(bars)
                last_time = bars[-1].event_time.replace(tzinfo=timezone.utc) if bars[-1].event_time.tzinfo is None else bars[-1].event_time
                next_start = last_time + timedelta(seconds=tf_seconds)
                retried_window = False
                current_start = max(next_start, current_start + timedelta(seconds=tf_seconds))
            else:
                # `get_candles` returns [] for both "no data in this window" and
                # "the request failed", and advancing past a failure silently
                # drops 300 bars. Retry once, then record the gap rather than
                # stepping over it as though the market had been quiet.
                if not retried_window:
                    retried_window = True
                    logger.warning(
                        'empty candle batch for %s %s..%s: retrying once',
                        product_id, current_start, batch_end,
                    )
                    continue
                skipped_windows.append((current_start, batch_end))
                retried_window = False
                current_start = batch_end
            await asyncio.sleep(0.1)
        if skipped_windows:
            logger.error(
                '%s %s: %d window(s) still empty after a retry and missing from '
                'the history: %s',
                product_id, granularity, len(skipped_windows),
                ', '.join(f'{a:%Y-%m-%d %H:%M}..{b:%Y-%m-%d %H:%M}'
                          for a, b in skipped_windows[:5]),
            )
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
    
    async def get_funding_rate_history(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        limit: int = 200,
    ) -> List[FundingRate]:
        """Historical funding for a CDE product. There is no such endpoint.

        Established by probing a live US account (`scripts/probe_funding.py`):

        - `/api/v3/brokerage/products/{id}` returns the **current** rate only,
          under `future_product_details.funding_rate`, with `funding_time` being
          the *next* settlement. No range parameters, no cursor.
        - `/api/v3/brokerage/intx/funding-rates`, which is what this method used
          to call, is Coinbase *International* Exchange. `GET
          /api/v3/brokerage/portfolios` on a US account returns a single
          `DEFAULT` portfolio and no INTX one, so every `intx/` path is a 404 by
          design, not by misconfiguration.

        So CDE funding cannot be backfilled: it has to be accumulated forward,
        one observation per hourly settlement, by `get_funding_rate`. The
        consequence is strategic rather than technical — the carry head can only
        ever be validated on history collected since collection began, so start
        collecting before you need it.

        This returns empty rather than raising: a caller asking for history is
        not making a mistake, the data simply does not exist. The scrape counts
        the zero and now exits non-zero, which is where the loudness belongs.
        """
        logger.info(
            "%s: CDE publishes no historical funding, only the current rate. "
            "Funding must be accumulated forward — see the docstring",
            product_id,
        )
        return []

    async def _unreachable_intx_funding_history(
        self,
        product_id: str,
        start: datetime,
        end: datetime,
        limit: int = 200,
    ) -> List[FundingRate]:
        """The former INTX implementation, kept for a venue that has the endpoint.

        Unreferenced on the CDE path by design. Coinbase International does
        expose paginated funding history at this path, so if this system is ever
        pointed at an INTX account the parsing below is correct and only the
        caller needs changing.
        """
        if not self.auth:
            logger.warning("Authentication required for funding rate history")
            return []

        start = ensure_naive_utc(start)
        end = ensure_naive_utc(end)

        path = "/api/v3/brokerage/intx/funding-rates"
        cursor = None
        all_rates: List[FundingRate] = []

        while True:
            params = {
                "product_id": product_id,
                "start": naive_utc_to_epoch_seconds(start),
                "end": naive_utc_to_epoch_seconds(end),
                "limit": limit,
            }
            if cursor:
                params["cursor"] = cursor

            status, data = await self._request("GET", path, params=params, authenticated=True)
            if status != 200:
                logger.debug(f"Funding history request failed for {product_id}: {data}")
                break

            raw_rows = (
                data.get("funding_rates")
                or data.get("results")
                or data.get("data")
                or []
            )
            if not raw_rows:
                break

            for row in raw_rows:
                ts = (
                    row.get("timestamp")
                    or row.get("event_time")
                    or row.get("time")
                    or row.get("funding_time")
                )
                if ts is None:
                    continue
                if isinstance(ts, str):
                    try:
                        event_time = datetime.fromisoformat(ts.replace("Z", "+00:00")).replace(tzinfo=None)
                    except ValueError:
                        continue
                else:
                    ts_num = float(ts)
                    if ts_num > 10_000_000_000:
                        ts_num /= 1000.0
                    event_time = epoch_seconds_to_naive_utc(ts_num)

                if event_time < start or event_time >= end:
                    continue

                rate_value = row.get("funding_rate")
                if rate_value is None:
                    rate_value = row.get("rate")
                if rate_value is None:
                    continue

                interval_hours = float(row.get("interval_hours") or row.get("intervalHours") or 1.0)
                interval_hours = max(interval_hours, 1.0)

                all_rates.append(FundingRate(
                    symbol=product_id,
                    event_time=event_time,
                    available_time=event_time + timedelta(seconds=5),
                    rate=float(rate_value) / interval_hours,
                    mark_price=float(row.get("mark_price") or row.get("markPrice") or 0.0),
                    index_price=float(row.get("index_price") or row.get("indexPrice") or 0.0),
                    is_settlement=bool(row.get("is_settlement", False)),
                    funding_source="coinbase",
                ))

            cursor = data.get("cursor") or data.get("next_cursor")
            if not cursor:
                break

        all_rates.sort(key=lambda x: x.event_time)
        return all_rates

    async def _product(self, product_id: str) -> Optional[Dict[str, Any]]:
        """The product payload, or None. One place that knows the path."""
        if not self.auth:
            logger.warning("Authentication required for product details")
            return None
        try:
            status, data = await self._request(
                "GET", f"/api/v3/brokerage/products/{product_id}", authenticated=True
            )
        except Exception as e:                                   # noqa: BLE001
            logger.debug("product lookup for %s failed: %s", product_id, e)
            return None
        if status != 200:
            logger.debug("product lookup for %s: HTTP %s", product_id, status)
            return None
        return data

    async def get_contract_snapshot(
        self, product_id: str
    ) -> Tuple[Optional[FundingRate], Optional[OpenInterest]]:
        """Funding rate and open interest from a single request.

        Both live on the same payload, so fetching them separately doubled the
        request count and let the two observations straddle a settlement — the
        funding row keyed to 22:00 could be paired with open interest read after
        the 22:00 print. One request, one instant, two records.
        """
        data = await self._product(product_id)
        if data is None:
            return None, None
        return self._parse_funding(product_id, data), self._parse_open_interest(product_id, data)

    async def get_funding_rate(self, product_id: str) -> Optional[FundingRate]:
        """The current funding rate for a CDE contract.

        Read from `/api/v3/brokerage/products/{id}` under
        `future_product_details`, which is where CDE actually publishes it:

            "funding_interval": "3600s",
            "funding_rate": "0.000009",
            "funding_time": "2026-08-21T16:00:00Z",
            "index_price": "77451.711238",
            "settlement_price": "72675",

        This used to ask `/api/v3/brokerage/intx/products/{id}`. INTX is
        Coinbase *International* Exchange — a different venue from CDE — so a US
        account, whose only portfolio is `DEFAULT`, got 404 on every call and the
        scrape stored zero funding rates. Note also that the sibling
        `perpetual_details` block exists but its `funding_rate` is an empty
        string; the populated values are one level up.

        `funding_time` is the *next* settlement, so it is the timestamp this rate
        applies to, and it is what the bar is keyed on.
        """
        data = await self._product(product_id)
        if data is None:
            return None
        return self._parse_funding(product_id, data)

    def _parse_funding(self, product_id: str, data: Dict[str, Any]) -> Optional[FundingRate]:
        """The funding half of a product payload. Pure, so it can be tested."""
        try:
            details = data.get("future_product_details") or {}
            raw_rate = details.get("funding_rate")
            if raw_rate in (None, ""):
                logger.debug("%s reports no funding_rate", product_id)
                return None

            # Normalise to decimal/hour, matching the CCXT path. CDE settles
            # hourly (`funding_interval: 3600s`), so the divisor is 1 there — but
            # reading it rather than assuming it means a venue that changes the
            # interval cannot silently rescale every carry feature.
            interval_hours = _funding_interval_hours(details.get("funding_interval"))
            rate = float(raw_rate) / max(interval_hours, 1e-9)

            event_time = (
                ensure_naive_utc(
                    datetime.fromisoformat(
                        str(details["funding_time"]).replace("Z", "+00:00")
                    )
                )
                if details.get("funding_time")
                else utc_now()
            )
            return FundingRate(
                symbol=product_id,
                event_time=event_time,
                available_time=utc_now(),
                rate=rate,
                mark_price=float(data.get("price") or 0.0),
                index_price=float(details.get("index_price") or 0.0),
                funding_source="coinbase",
                venue="coinbase",
            )
        except Exception as e:
            logger.debug(f"Could not get funding rate for {product_id}: {e}")
            return None
    
    # Where the venue puts open interest. Checked in order, and a miss is a miss:
    # the previous implementation went to CCXT because this client had no
    # open-interest method, and a comment recorded that absence as "Coinbase
    # exposes no open-interest endpoint" — a statement about our code, promoted
    # to a fact about the venue. It is on the product payload, on the contract we
    # actually trade.
    OPEN_INTEREST_PATHS: Tuple[Tuple[str, ...], ...] = (
        ('future_product_details', 'open_interest'),
        ('future_product_details', 'perpetual_details', 'open_interest'),
        ('open_interest',),
    )

    async def get_open_interest(self, product_id: str) -> Optional[OpenInterest]:
        """Current open interest for one contract, in contracts.

        `/api/v3/brokerage/products/{id}` reports it under
        `future_product_details.open_interest` — 268,164 for BIP-20DEC30-CDE
        against 21,579,279 for gate's BTC/USDT:USDT, which is how far apart the
        two books are. The units are contracts, matching the sibling `volume_24h`
        (~1.46M/day for BIP).

        Like funding, this is a snapshot: a current value with no range
        parameters and no cursor, so open interest accumulates forward one
        observation per run and cannot be backfilled.

        Returns None when the venue reports nothing. It never substitutes a zero
        — writing 0.0 for an absent field is what put 11,520 rows of fabricated
        open interest into the store, where five features went all-NaN and
        `liquidation_cascade_24h` quietly carried on with its OI term disabled.
        """
        data = await self._product(product_id)
        if data is None:
            return None
        return self._parse_open_interest(product_id, data)

    def _parse_open_interest(self, product_id: str, data: Dict[str, Any]) -> Optional[OpenInterest]:
        """The open-interest half of a product payload. Pure, so it can be tested."""
        try:
            raw = None
            for keys in self.OPEN_INTEREST_PATHS:
                node: Any = data
                for key in keys:
                    node = (node or {}).get(key) if isinstance(node, dict) else None
                if node not in (None, ""):
                    raw = node
                    break

            if raw is None:
                logger.debug("%s reports no open_interest", product_id)
                return None

            try:
                contracts = float(raw)
            except (TypeError, ValueError):
                logger.warning("%s: uninterpretable open_interest %r", product_id, raw)
                return None

            now = utc_now()
            return OpenInterest(
                symbol=product_id,
                event_time=now,
                available_time=now,
                open_interest_contracts=contracts,
                venue="coinbase",
            )
        except Exception as e:                                   # noqa: BLE001
            logger.debug(f"Could not get open interest for {product_id}: {e}")
            return None

    async def _get_funding_rate_from_portfolio(self, product_id: str) -> Optional[FundingRate]:
        try:
            summary = await self.get_perpetuals_portfolio_summary()
            if not summary:
                return None
            positions = summary.get("positions", [])
            for pos in positions:
                if pos.get("product_id") == product_id:
                    now = utc_now()
                    return FundingRate(
                        symbol=product_id, event_time=now, available_time=now,
                        rate=float(pos.get("funding_rate", 0)),
                        mark_price=float(pos.get("mark_price", 0)),
                        index_price=float(pos.get("index_price", 0)),
                    )
            return None
        except Exception as e:
            logger.debug(f"Could not get funding rate from portfolio: {e}")
            return None
    
    async def get_perpetuals_portfolio_summary(self) -> Optional[Dict]:
        if not self.auth:
            logger.error("Authentication required for portfolio summary")
            return None
        status, data = await self._request("GET", "/api/v3/brokerage/intx/portfolio", authenticated=True)
        if status != 200:
            logger.error(f"Failed to get portfolio summary: {data}")
            return None
        return data
    
    def _granularity_to_seconds(self, granularity: str) -> int:
        mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "6h": 21600, "1d": 86400}
        return mapping.get(granularity, 3600)
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Coinbase REST client closed")


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