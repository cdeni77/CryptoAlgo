"""The Kalshi trading client: quotes, balance, orders, fills.

Markets are resolved by asking the venue, never by building a ticker.
`resolve_window_market` lists the venue's markets and picks the one whose
`close_time` is the window's settlement. A built ticker can silently match the
wrong market after a venue rename.

Compare `close_time` (UTC), not the ticker. The 15-minute tickers are named in
Eastern time, so `KXBTC15M-26AUG230045` closes at 04:45Z. `KXBTCD` is an hourly
strike ladder, not an up/down market.

The venue lists one open market per series: the window now running.

Live decisions are priced against the real book. `Prediction.price_source`
records whether a decision used the book or the baseline.

Safety: `place_order` refuses unless the client was constructed with
`live=True`, and every caller defaults to dry run.

Authentication is RSA, not HMAC. The client signs `timestamp + method + path`
with RSA-PSS over SHA-256. Headers: the base64 signature in
`KALSHI-ACCESS-SIGNATURE`, the key id in `KALSHI-ACCESS-KEY`, and epoch
milliseconds in `KALSHI-ACCESS-TIMESTAMP`.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import aiohttp

from core.costs import tick_at

from .timeutil import naive_utc_to_epoch_seconds

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
DEMO_BASE_URL = 'https://demo-api.kalshi.co/trade-api/v2'

# The historical tier is on a different host. The live endpoints serve about
# the last three months; older data answers on `/historical/...` at this host.
# A complete history queries both and merges them (see `all_fills`).
HISTORICAL_BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'

# Kalshi quotes in whole cents, 1..99, through the middle of the range.
CENT = 0.01
# ...and in tenths of a cent below 10c and above 90c (`tapered_deci_cent`).
DECI_CENT = 0.001

# Sort key for a row with no timestamp. It sorts first instead of raising.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class KalshiError(RuntimeError):
    """The venue refused, or the response was not what the caller assumed.

    `status` is the HTTP status when the venue answered at all. A 5xx on an
    order POST is ambiguous — the order may have been accepted — so callers
    look the order up rather than treat it as a refusal.
    """

    def __init__(self, *args, status: Optional[int] = None) -> None:
        super().__init__(*args)
        self.status = status

    @property
    def ambiguous(self) -> bool:
        return self.status is not None and self.status >= 500


class NotLiveError(KalshiError):
    """An order was attempted on a client that was not constructed for it."""


def _load_private_key(pem: str | bytes):
    from cryptography.hazmat.primitives import serialization

    data = pem.encode() if isinstance(pem, str) else pem
    return serialization.load_pem_private_key(data, password=None)


@dataclass(frozen=True)
class Quote:
    """One market's book, as probabilities in [0, 1], not cents.

    The venue serves prices as dollar strings (`yes_bid_dollars: "0.1900"`).
    The integer-cent fields (`yes_bid: 19`) come back null. Both are accepted.
    """

    ticker: str
    yes_bid: Optional[float]
    yes_ask: Optional[float]
    no_bid: Optional[float]
    no_ask: Optional[float]
    last_price: Optional[float]
    volume: int
    open_interest: int
    close_time: Optional[datetime]
    status: str
    # Contracts resting at the touch.
    yes_bid_size: Optional[float] = None
    yes_ask_size: Optional[float] = None
    # The exchange the market lives on. An order must carry the market's own
    # `exchange_index`; the venue answers `404 market_not_found` otherwise.
    exchange_index: int = 0
    # The strike the market settles against, published once the window opens.
    # Prefer it over a strike computed from bars.
    floor_strike: Optional[float] = None
    strike_type: Optional[str] = None
    open_time: Optional[datetime] = None
    # The touch's source ('stream' or 'rest'), its read time (epoch seconds),
    # and the full ladder, so depth features describe the priced snapshot.
    book_source: Optional[str] = None
    book_time: Optional[float] = None
    yes_levels: tuple = ()
    no_levels: tuple = ()

    @property
    def mid(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return (self.yes_bid + self.yes_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid

    def ask_for(self, side: str) -> Optional[float]:
        """What it costs to buy `side` right now, crossing the spread."""
        return self.yes_ask if side == 'up' else self.no_ask

    def size_for(self, side: str) -> Optional[float]:
        """Contracts available at the touch on the side being bought."""
        return self.yes_ask_size if side == 'up' else self.yes_bid_size

    def depth_dollars(self, side: str) -> Optional[float]:
        """What the touch is worth in dollars — the real cap on a stake."""
        price = self.ask_for(side)
        size = self.size_for(side)
        if price is None or size is None:
            return None
        return price * size

    def tradeable(self) -> bool:
        return self.status == 'active' and self.yes_bid is not None and self.yes_ask is not None


def _price(raw: dict, name: str) -> Optional[float]:
    """A price in dollars, or None for an absent or zero level.

    Reads `{name}_dollars` (a dollar string) first, then `{name}` (integer cents).
    """
    dollars = raw.get(f'{name}_dollars')
    if dollars is not None:
        try:
            value = float(dollars)
        except (TypeError, ValueError):
            logger.warning('%s_dollars was %r, which is not a number; treating the '
                           'level as absent', name, dollars)
            value = 0.0
        return value if value > 0 else None
    cents = raw.get(name)
    if cents is None:
        return None
    try:
        value = float(cents)
    except (TypeError, ValueError):
        return None
    return value * CENT if value > 0 else None


def _quantity(raw: dict, *names: str) -> float:
    """A count, from a `_fp` fixed-point string or a plain number."""
    for name in names:
        for key in (f'{name}_fp', name):
            value = raw.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return 0.0


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


def _money(raw: dict, name: str) -> Optional[float]:
    """A dollar amount. Zero is a value; None means absent or unparseable.

    Reads `{name}_dollars` first, then `{name}` in integer cents. Unlike `_price`,
    zero stays zero: a losing position settles at revenue exactly 0.
    """
    dollars = raw.get(f'{name}_dollars')
    if dollars is not None:
        try:
            return float(dollars)
        except (TypeError, ValueError):
            logger.warning('%s_dollars was %r, which is not a number', name, dollars)
            return None
    cents = raw.get(name)
    if cents is None:
        return None
    try:
        return float(cents) * CENT
    except (TypeError, ValueError):
        logger.warning('%s was %r, which is not a number', name, cents)
        return None


def _fee(raw: dict, name: str = 'fee_cost') -> Optional[float]:
    """A fee in dollars. Zero is a value; None means absent or unparseable.

    Unsuffixed `fee_cost` is already in dollars, unlike `revenue`, which is in
    cents. `{name}_dollars` wins when the venue sends it.
    """
    dollars = raw.get(f'{name}_dollars')
    if dollars is not None:
        try:
            return float(dollars)
        except (TypeError, ValueError):
            logger.warning('%s_dollars was %r, which is not a number', name, dollars)
            return None
    value = raw.get(name)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        logger.warning('%s was %r, which is not a number', name, value)
        return None


@dataclass(frozen=True)
class Fill:
    """One execution, as the venue recorded it. The account of record for entry.

    Where it differs from our `Position` row (partial fill, price improvement,
    fee), this one is right.

    `price` is what was paid for `side`, in dollars per contract. A NO fill
    carries its price in `no_price`, not `yes_price`.
    """

    trade_id: str
    order_id: Optional[str]
    ticker: str
    side: str                      # 'up' (YES) or 'down' (NO), our vocabulary
    action: Optional[str]          # 'buy' or 'sell', as the venue said it
    contracts: float
    price: Optional[float]         # dollars paid per contract for `side`
    is_taker: Optional[bool]
    created_time: Optional[datetime]
    raw: dict


@dataclass(frozen=True)
class Settlement:
    """One market resolved, as the venue paid it. The account of record for PnL.

    It overrides our own settlement, which uses Coinbase as a proxy for CF
    Benchmarks BRTI, and our predicted fee.
    """

    ticker: str
    event_ticker: Optional[str]
    market_result: Optional[str]   # 'yes' or 'no', per the venue
    yes_contracts: float
    no_contracts: float
    yes_cost: Optional[float]
    no_cost: Optional[float]
    revenue: Optional[float]
    fee_cost: Optional[float]
    settled_time: Optional[datetime]
    raw: dict

    @property
    def cost(self) -> Optional[float]:
        """What the position cost, both sides together, fees excluded."""
        if self.yes_cost is None and self.no_cost is None:
            return None
        return (self.yes_cost or 0.0) + (self.no_cost or 0.0)

    @property
    def contracts(self) -> float:
        return self.yes_contracts + self.no_contracts

    @property
    def pnl(self) -> Optional[float]:
        """`revenue - cost - fee_cost`, or None when revenue or cost is absent.

        A missing field is not a zero. `revenue` is the payout; `fee_cost` is a
        separate debit charged at order time.
        """
        cost = self.cost
        if self.revenue is None or cost is None:
            return None
        return self.revenue - cost - (self.fee_cost or 0.0)


@dataclass(frozen=True)
class Trade:
    """One print on the public tape, by any account. Never use it for P&L.

    `/historical/trades` and `/markets/trades` carry no account id. The `taker_*`
    fields name the aggressor's side, not ours.

    Uses: a market price to mark an open position, and a check that a fill
    printed where the venue said (join on `trade_id`). For a taker fill,
    `taker_side` matches `Fill.side`.
    """

    trade_id: str
    ticker: str
    contracts: float
    yes_price: Optional[float]
    no_price: Optional[float]
    created_time: Optional[datetime]
    is_block_trade: Optional[bool]
    # The aggressor's side ('up'/'down'), and the book side hit ('bid'/'ask',
    # verbatim from the venue).
    taker_side: Optional[str] = None
    taker_book_side: Optional[str] = None


def _bool(value: Any) -> Optional[bool]:
    """A tri-state read: True, False, or "the venue did not say"."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ('true', 't', '1', 'yes'):
        return True
    if text in ('false', 'f', '0', 'no'):
        return False
    return None


def parse_fill(raw: dict) -> Fill:
    """A fills row, with side 'yes'/'no' mapped to 'up'/'down'.

    An unknown side passes through verbatim. A guessed side would invert PnL.
    """
    side_raw = str(raw.get('side', '')).strip().lower()
    side = {'yes': 'up', 'no': 'down'}.get(side_raw, side_raw)
    price = _money(raw, 'yes_price') if side == 'up' else _money(raw, 'no_price')
    return Fill(
        trade_id=str(raw.get('trade_id') or ''),
        order_id=str(raw['order_id']) if raw.get('order_id') else None,
        ticker=str(raw.get('ticker') or ''),
        side=side,
        action=str(raw['action']).strip().lower() if raw.get('action') else None,
        contracts=_quantity(raw, 'count'),
        price=price,
        is_taker=_bool(raw.get('is_taker')),
        created_time=_parse_time(raw.get('created_time')),
        raw=dict(raw),
    )


def parse_settlement(raw: dict) -> Settlement:
    """A settlements row, in dollars, with zeros preserved.

    `fee_cost` goes through `_fee`, not `_money`: unsuffixed, it is already in
    dollars, while unsuffixed `revenue` is in cents.
    """
    return Settlement(
        ticker=str(raw.get('ticker') or ''),
        event_ticker=str(raw['event_ticker']) if raw.get('event_ticker') else None,
        market_result=(str(raw['market_result']).strip().lower()
                       if raw.get('market_result') else None),
        yes_contracts=_quantity(raw, 'yes_count'),
        no_contracts=_quantity(raw, 'no_count'),
        yes_cost=_money(raw, 'yes_total_cost'),
        no_cost=_money(raw, 'no_total_cost'),
        revenue=_money(raw, 'revenue'),
        fee_cost=_fee(raw),
        settled_time=_parse_time(raw.get('settled_time')),
        raw=dict(raw),
    )


def parse_trade(raw: dict) -> Trade:
    """A tape print. The taker side is translated; the book side is not.

    Prefers `taker_outcome_side`, the newer name, over its alias `taker_side`.
    """
    taker = raw.get('taker_outcome_side') or raw.get('taker_side')
    taker_side = None
    if taker:
        text = str(taker).strip().lower()
        taker_side = {'yes': 'up', 'no': 'down'}.get(text, text)
    book_side = raw.get('taker_book_side')
    return Trade(
        trade_id=str(raw.get('trade_id') or ''),
        ticker=str(raw.get('ticker') or ''),
        contracts=_quantity(raw, 'count'),
        yes_price=_money(raw, 'yes_price'),
        no_price=_money(raw, 'no_price'),
        created_time=_parse_time(raw.get('created_time')),
        is_block_trade=_bool(raw.get('is_block_trade')),
        taker_side=taker_side,
        taker_book_side=str(book_side).strip().lower() if book_side else None,
    )


class KalshiClient:
    """Async client. One session, signed requests, and an explicit live flag."""

    def __init__(
        self,
        *,
        key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
        historical_base_url: Optional[str] = None,
        live: bool = False,
        timeout_seconds: float = 15.0,
    ):
        self.key_id = key_id or os.getenv('KALSHI_KEY_ID', '')
        pem = private_key_pem or os.getenv('KALSHI_PRIVATE_KEY', '')
        path = private_key_path or os.getenv('KALSHI_PRIVATE_KEY_PATH', '')
        if not pem and path:
            pem = Path(path).read_text()
        self._pem = pem
        self._key = _load_private_key(pem) if pem else None
        self.base_url = (base_url or os.getenv('KALSHI_BASE_URL') or DEFAULT_BASE_URL).rstrip('/')
        # The host for `/historical/...` only. See HISTORICAL_BASE_URL.
        self.historical_base_url = (
            historical_base_url or os.getenv('KALSHI_HISTORICAL_BASE_URL')
            or HISTORICAL_BASE_URL
        ).rstrip('/')
        self.live = live
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._session: Optional[aiohttp.ClientSession] = None

    # ---- lifecycle ------------------------------------------------------
    async def __aenter__(self) -> 'KalshiClient':
        self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    @property
    def configured(self) -> bool:
        return bool(self.key_id and self._key)

    # ---- signing --------------------------------------------------------
    def _headers(self, method: str, path: str) -> dict[str, str]:
        """Sign `timestamp + method + path`. The path excludes the query string."""
        if not self.configured:
            raise KalshiError(
                'Kalshi credentials are not configured. Set KALSHI_KEY_ID and '
                'either KALSHI_PRIVATE_KEY (the PEM itself) or '
                'KALSHI_PRIVATE_KEY_PATH (a file holding it).'
            )
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        timestamp = str(int(time.time() * 1000))
        message = f'{timestamp}{method.upper()}{path}'.encode()
        signature = self._key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                        salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            'KALSHI-ACCESS-KEY': self.key_id,
            'KALSHI-ACCESS-SIGNATURE': base64.b64encode(signature).decode(),
            'KALSHI-ACCESS-TIMESTAMP': timestamp,
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        }

    async def _request(self, method: str, path: str, *,
                       params: Optional[dict] = None,
                       body: Optional[dict] = None,
                       base: Optional[str] = None) -> dict:
        if self._session is None:
            raise KalshiError('client is not open; use `async with KalshiClient(...)`')
        # The signature covers the path without the query or host. `base`
        # changes only the host; both hosts mount `/trade-api/v2`.
        signed_path = f'/trade-api/v2{path}'
        headers = self._headers(method, signed_path)
        url = f'{(base or self.base_url).rstrip("/")}{path}'
        async with self._session.request(
            method, url, params=params,
            data=json.dumps(body) if body is not None else None,
            headers=headers,
        ) as response:
            text = await response.text()
            if response.status >= 400:
                raise KalshiError(
                    f'{method} {path} -> {response.status}: {text[:400]}',
                    status=response.status)
            return json.loads(text) if text else {}

    # ---- reading --------------------------------------------------------
    async def balance(self, *, exchange_index: Optional[int] = None) -> float:
        """Available balance in dollars, or NaN when the venue did not say.

        Pass `exchange_index` for the balance spendable on that shard. Omit it
        for the whole-account total. Prefers `balance_dollars` over `balance`
        (cents). Returns NaN, never 0, for a missing field, because callers
        write the balance onto the account.
        """
        payload = await self._request('GET', '/portfolio/balance')

        # Balances are local to an exchange shard. `balance_breakdown` has one
        # entry per `exchange_index`, and an order can spend only its market's
        # shard.
        if exchange_index is not None:
            breakdown = payload.get('balance_breakdown')
            if isinstance(breakdown, list) and breakdown:
                for row in breakdown:
                    try:
                        if int(row.get('exchange_index', -1)) == int(exchange_index):
                            return float(row.get('balance'))
                    except (TypeError, ValueError):
                        logger.error('balance_breakdown row was %r', row)
                        return float('nan')
                # A shard the venue does not list holds nothing.
                return 0.0
            # No breakdown: the venue does not shard, so use the total.

        dollars = payload.get('balance_dollars')
        if dollars is not None:
            try:
                return float(dollars)
            except (TypeError, ValueError):
                logger.error('balance_dollars was %r, which is not a number', dollars)
                return float('nan')
        cents = payload.get('balance')
        if cents is None:
            logger.error('the venue returned no balance field; keys were %s',
                         sorted(payload)[:12])
            return float('nan')
        try:
            return float(cents) * CENT
        except (TypeError, ValueError):
            logger.error('balance was %r, which is not a number', cents)
            return float('nan')

    async def markets(self, **params) -> list[dict]:
        """One page of markets. `series_ticker`, `status`, `limit` all apply."""
        payload = await self._request('GET', '/markets', params=_clean(params))
        return list(payload.get('markets', []))

    async def market(self, ticker: str) -> dict:
        payload = await self._request('GET', f'/markets/{ticker}')
        return dict(payload.get('market', {}))

    async def quote(self, ticker: str) -> Quote:
        raw = await self.market(ticker)
        return _to_quote(raw)

    async def resolve_window_market(
        self,
        series_ticker: str,
        settle_time: datetime,
        *,
        tolerance_seconds: int = 90,
    ) -> Optional[dict]:
        """Find the market for one 15-minute window by its close time.

        Returns the open market in the series that closes when this window
        settles, or None. A caller that gets None abstains.
        """
        candidates = await self.markets(
            series_ticker=series_ticker, status='open', limit=200)
        best, best_gap = None, None
        for market in candidates:
            close = _parse_time(market.get('close_time'))
            if close is None:
                continue
            gap = abs((close - settle_time).total_seconds())
            if best_gap is None or gap < best_gap:
                best, best_gap = market, gap
        if best is None or best_gap is None or best_gap > tolerance_seconds:
            logger.warning(
                'no %s market closes within %ds of %s (%d open markets, closest '
                '%s away) — abstaining rather than trading a neighbour',
                series_ticker, tolerance_seconds, settle_time.isoformat(),
                len(candidates),
                f'{best_gap:.0f}s' if best_gap is not None else 'n/a',
            )
            return None
        return best

    async def orderbook(self, ticker: str) -> dict:
        """The full ladder, parsed to the touch. See `parse_orderbook`."""
        return parse_orderbook(
            await self._request('GET', f'/markets/{ticker}/orderbook'))

    async def book_quote(self, meta: 'Quote') -> 'Quote':
        """`meta` with its touch and ladder replaced by one `/orderbook` read.

        Raises `KalshiError` when the book cannot be read; the caller abstains.
        """
        try:
            payload = await self._request('GET', f'/markets/{meta.ticker}/orderbook')
        except (asyncio.TimeoutError, OSError, aiohttp.ClientError) as exc:
            raise KalshiError(f'{meta.ticker}: orderbook read failed ({exc!r})') from exc
        yes, no = book_levels(payload)
        return quote_from_levels(meta, yes, no, source='rest', at=time.time())

    async def positions(self) -> list[dict]:
        payload = await self._request('GET', '/portfolio/positions')
        return list(payload.get('market_positions', []))

    @staticmethod
    def position_size(row: dict) -> float:
        """Signed contracts held in one market, from a positions row.

        The venue sends `position_fp`, a fixed-point string such as "-5.00";
        `position` is absent. Negative is a NO position, so test `!= 0` for
        "anything open".
        """
        return _quantity(row, 'position')

    async def fills(self, **params) -> list[dict]:
        payload = await self._request('GET', '/portfolio/fills', params=_clean(params))
        return list(payload.get('fills', []))

    async def settlements(self, **params) -> list[dict]:
        """Settled positions, as the venue resolved them. The venue is authoritative."""
        payload = await self._request('GET', '/portfolio/settlements',
                                      params=_clean(params))
        return list(payload.get('settlements', []))

    # ---- the venue's own ledger ----------------------------------------
    #
    # The venue's ledger is the account of record. Where it disagrees with our
    # own arithmetic, the venue is right.

    async def historical_cutoff(self) -> dict[str, Optional[datetime]]:
        """Where the live tier stops and the historical tier begins.

        The cutoff moves, so it is read, not assumed. Returns the venue's
        timestamps parsed to UTC, or `{}` when unreachable; callers then query
        both tiers.
        """
        try:
            payload = await self._request('GET', '/historical/cutoff',
                                         base=self.historical_base_url)
        except KalshiError as exc:
            logger.warning('the historical cutoff was unreadable (%s); querying '
                           'both tiers rather than guessing a retention window', exc)
            return {}
        out: dict[str, Optional[datetime]] = {}
        for key, value in payload.items():
            when = _parse_time(value)
            if when is not None:
                out[str(key)] = when
        return out

    async def _pages(self, path: str, key: str, *, base: Optional[str] = None,
                     max_pages: int = 40, **params) -> list[dict]:
        """Follow the cursor to the end, up to `max_pages`.

        An empty or repeated cursor ends the read. `limit` is at most 200 on
        these routes. Hitting the page cap logs an error, because the ledger is
        then truncated.
        """
        rows: list[dict] = []
        cursor: Optional[str] = None
        seen: set[str] = set()
        for page in range(max_pages):
            query = dict(params)
            if cursor:
                query['cursor'] = cursor
            payload = await self._request('GET', path, params=_clean(query), base=base)
            batch = payload.get(key) or []
            rows.extend(batch)
            cursor = payload.get('cursor') or None
            # A repeated cursor is a loop, not a next page.
            if not cursor or cursor in seen or not batch:
                return rows
            seen.add(cursor)
        logger.error(
            'stopped paginating %s after %d pages (%d rows) with a cursor still '
            'outstanding. The ledger below this point is MISSING, not empty.',
            path, max_pages, len(rows))
        return rows

    async def all_fills(self, *, since: Optional[datetime] = None,
                        limit: int = 200) -> list[Fill]:
        """Every fill, live tier and historical tier merged, newest first.

        Deduplicated on `trade_id`, because the tiers overlap around the cutoff.
        """
        params: dict[str, Any] = {'limit': limit}
        if since is not None:
            params['min_ts'] = naive_utc_to_epoch_seconds(since)

        rows = await self._pages('/portfolio/fills', 'fills', **params)
        try:
            rows += await self._pages('/portfolio/fills/historical', 'fills',
                                      base=self.historical_base_url, **params)
        except KalshiError as exc:
            # Not fatal. The live tier alone is still right for a recent window.
            logger.warning(
                'the historical fills tier was unavailable (%s); this ledger '
                'covers the live tier only and is short of anything older than '
                'the cutoff', exc)

        fills: dict[str, Fill] = {}
        for raw in rows:
            fill = parse_fill(raw)
            if not fill.trade_id:
                logger.warning('a fill arrived with no trade_id (%s); skipping it '
                               'rather than inventing a key', sorted(raw)[:8])
                continue
            fills[fill.trade_id] = fill
        return sorted(fills.values(),
                      key=lambda f: (f.created_time or _EPOCH), reverse=True)

    async def all_settlements(self, *, since: Optional[datetime] = None,
                              limit: int = 200) -> list[Settlement]:
        """Every settled market, both tiers merged, newest first. This is the P&L.

        A held binary pays one fee at entry and settles once, so a settlement row
        holds its cost, revenue and fee. Keyed by ticker.
        """
        params: dict[str, Any] = {'limit': limit}
        if since is not None:
            params['min_ts'] = naive_utc_to_epoch_seconds(since)

        rows = await self._pages('/portfolio/settlements', 'settlements', **params)
        try:
            rows += await self._pages('/portfolio/settlements/historical',
                                      'settlements',
                                      base=self.historical_base_url, **params)
        except KalshiError as exc:
            logger.warning(
                'the historical settlements tier was unavailable (%s); this P&L '
                'covers the live tier only', exc)

        settled: dict[str, Settlement] = {}
        for raw in rows:
            row = parse_settlement(raw)
            if not row.ticker:
                continue
            settled[row.ticker] = row
        return sorted(settled.values(),
                      key=lambda s: (s.settled_time or _EPOCH), reverse=True)

    async def market_trades(self, *, ticker: Optional[str] = None,
                            since: Optional[datetime] = None,
                            limit: int = 100,
                            historical: bool = False) -> list[Trade]:
        """The public tape: every account's trades, not ours. See `Trade`.

        `historical=True` reads `/historical/trades` for prints older than the
        cutoff; the live route is `/markets/trades`.
        """
        path = '/historical/trades' if historical else '/markets/trades'
        base = self.historical_base_url if historical else None
        params: dict[str, Any] = {'limit': limit, 'ticker': ticker}
        if since is not None:
            params['min_ts'] = naive_utc_to_epoch_seconds(since)
        rows = await self._pages(path, 'trades', base=base, **params)
        return [parse_trade(raw) for raw in rows]

    async def reconcile(self, *, exchange_index: Optional[int] = None) -> dict:
        """Balance, open positions, recent fills and settlements, from the venue.

        `exchange_index` narrows the balance to one shard. See `balance`.
        """
        balance = await self.balance(exchange_index=exchange_index)
        positions = await self.positions()
        fills = await self.fills(limit=200)
        settled = []
        try:
            settled = await self.settlements(limit=200)
        except KalshiError as exc:
            logger.warning('settlements unavailable (%s); falling back to fills', exc)
        return {
            'balance': balance,
            'positions': positions,
            'fills': fills,
            'settlements': settled,
        }

    async def orders(self, **params) -> list[dict]:
        payload = await self._request('GET', '/portfolio/orders', params=_clean(params))
        return list(payload.get('orders', []))

    # ---- writing --------------------------------------------------------
    async def place_order(
        self,
        *,
        ticker: str,
        side: str,
        exchange_index: int = 0,
        contracts: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
        time_in_force: str = 'immediate_or_cancel',
    ) -> dict:
        """Buy `contracts` of `side` at no worse than `limit_price` (dollars).

        Refuses unless the client was constructed with `live=True`.

        Defaults to `immediate_or_cancel`: the order never rests on a wasting
        market, and a partial fill beats the nothing `fill_or_kill` returns.
        """
        if not self.live:
            raise NotLiveError(
                'this client was not constructed with live=True, so it will not '
                'place orders. That is deliberate: a script meant to observe '
                'that instead trades cannot be undone.'
            )
        if side not in ('up', 'down'):
            raise ValueError(f"side must be 'up' or 'down', got {side!r}")
        if contracts < 1:
            raise ValueError(f'contracts must be at least 1, got {contracts}')

        # V2 orders (V1 returns 410). V2 quotes one book from the YES side:
        # `bid` buys YES, `ask` sells YES, and selling YES buys NO at
        # `1 - price`. There is no `action`, `type` or `yes_price`/`no_price`,
        # and `self_trade_prevention_type` is required.
        #
        # `limit_price` is what we pay for `side`. Convert it to a YES limit:
        # paying 0.30 for NO is selling YES at 0.70.
        book_side = 'bid' if side == 'up' else 'ask'
        yes_limit = limit_price if side == 'up' else 1.0 - limit_price

        # Round so the limit can still fill: a bid rounds up, an ask rounds down.
        # Round onto the venue's tick (a tenth of a cent below 10c and above
        # 90c), so the order never reaches past the chosen limit by a whole cent.
        #
        # Work in integer tenths of a cent. The epsilon absorbs float error:
        # 0.60 is 599.999... tenths.
        step = int(round(float(tick_at(yes_limit)) / DECI_CENT))
        if book_side == 'bid':
            ticks = int(math.ceil(yes_limit / DECI_CENT / step - 1e-9)) * step
        else:
            ticks = int(math.floor(yes_limit / DECI_CENT / step + 1e-9)) * step
        price = ticks * DECI_CENT
        if not CENT - 1e-9 <= price <= 1.0 - CENT + 1e-9:
            raise ValueError(
                f'limit {limit_price} on the {side} side is a YES limit of '
                f'{yes_limit:.4f} ({price:.4f}), outside 1c..99c'
            )

        body = {
            'ticker': ticker,
            'side': book_side,
            # Decimal strings, not integers. `count` accepts up to 2 decimals and
            # `price` 2-4; a bare int for count is rejected.
            'count': f'{int(contracts)}.00',
            'price': f'{price:.4f}',
            'time_in_force': time_in_force,
            # Required in V2. `taker_at_cross` crosses the spread; `maker` would
            # rest.
            'self_trade_prevention_type': 'taker_at_cross',
            # The market's exchange shard. The default 0 gives a 404 for a
            # market on another shard.
            'exchange_index': int(exchange_index),
        }
        if client_order_id:
            body['client_order_id'] = client_order_id
        logger.info('placing %s (%s) %d @ %.4f YES on %s',
                    side, book_side, contracts, price, ticker)
        payload = await self._request('POST', '/portfolio/events/orders', body=body)
        # Log the request and the full reply. The POST reply's shape differs
        # from the GET order record, so the reason for a kill is only here.
        logger.info('order request  %s', json.dumps(body, sort_keys=True))
        logger.info('order reply    %s', json.dumps(payload, sort_keys=True, default=str))
        return dict(payload.get('order', payload))

    async def order_by_client_id(self, client_order_id: str, *,
                                 ticker: Optional[str] = None,
                                 max_pages: int = 3) -> Optional[dict]:
        """Find our order by the `client_order_id` we sent, or None.

        `GET /portfolio/orders` cannot filter on it, so this lists the market's
        orders and matches. None does not prove the order was refused.
        """
        rows = await self._pages('/portfolio/orders', 'orders',
                                 max_pages=max_pages, ticker=ticker, limit=100)
        for row in rows:
            if str(row.get('client_order_id') or '') == str(client_order_id):
                return dict(row)
        return None


NAN = float('nan')


def parse_orderbook(payload: dict) -> dict:
    """The touch of a REST orderbook, in YES-denominated terms.

    A live response is `{'orderbook': {'orderbook_fp': {'yes_dollars':
    [[price, size], ...], 'no_dollars': [...]}}}`, with decimal strings. The
    stream uses other keys, so every known shape is accepted.

    The book is two bid stacks, so the YES ask is `1 - best_no_bid`.
    `no_bid_size` is what a YES buy crosses. An empty stack gives NaN, never
    zero.
    """
    raw = payload.get('orderbook', payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        raw = {}
    stacks = raw.get('orderbook_fp') if isinstance(raw.get('orderbook_fp'), dict) else raw

    def touch(*names):
        for name in names:
            levels = stacks.get(name)
            if not levels:
                continue
            best = None
            for entry in levels:
                try:
                    price, size = float(entry[0]), float(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if size > 0 and (best is None or price > best[0]):
                    best = (price, size)
            if best is not None:
                return best
        return None

    yes = touch('yes_dollars', 'yes', 'yes_dollars_fp')
    no = touch('no_dollars', 'no', 'no_dollars_fp')
    return {
        'yes_bid': yes[0] if yes else NAN,
        'yes_bid_size': yes[1] if yes else NAN,
        'no_bid': no[0] if no else NAN,
        'no_bid_size': no[1] if no else NAN,
        # The conversions. `1 - best_no_bid` is what buying YES costs.
        'yes_ask': (1.0 - no[0]) if no else NAN,
        'no_ask': (1.0 - yes[0]) if yes else NAN,
    }


def _stacks(payload: dict) -> dict:
    raw = payload.get('orderbook', payload) if isinstance(payload, dict) else {}
    if not isinstance(raw, dict):
        return {}
    return raw.get('orderbook_fp') if isinstance(raw.get('orderbook_fp'), dict) else raw


def book_levels(payload: dict) -> tuple[list, list]:
    """(YES bids, NO bids) as sorted `(price, size)` from a REST orderbook.

    Same nesting and spellings `parse_orderbook` accepts; empty or zero-size
    levels are dropped.
    """
    stacks = _stacks(payload)

    def levels(*names):
        for name in names:
            out = []
            for entry in stacks.get(name) or []:
                try:
                    price, size = float(entry[0]), float(entry[1])
                except (TypeError, ValueError, IndexError):
                    continue
                if size > 0:
                    out.append((price, size))
            if out:
                return sorted(out)
        return []

    return (levels('yes_dollars', 'yes', 'yes_dollars_fp'),
            levels('no_dollars', 'no', 'no_dollars_fp'))


def quote_from_levels(meta: Quote, yes_levels, no_levels, *,
                      source: str, at: float) -> Quote:
    """`meta` (status, shard, strike) with the touch taken from a ladder.

    The book is two bid stacks: the YES ask is `1 - best_no_bid`, and its size
    is the NO bid's size. An empty stack gives None, never zero.
    """
    book = parse_orderbook({'yes': list(yes_levels or []),
                            'no': list(no_levels or [])})

    def finite(value):
        value = float(value) if value is not None else float('nan')
        return value if value == value else None

    yes_bid, no_bid = finite(book['yes_bid']), finite(book['no_bid'])
    return replace(
        meta,
        yes_bid=yes_bid,
        no_bid=no_bid,
        yes_ask=(1.0 - no_bid) if no_bid is not None else None,
        no_ask=(1.0 - yes_bid) if yes_bid is not None else None,
        yes_bid_size=finite(book['yes_bid_size']),
        yes_ask_size=finite(book['no_bid_size']),
        book_source=source,
        book_time=float(at),
        yes_levels=tuple(tuple(level) for level in (yes_levels or [])),
        no_levels=tuple(tuple(level) for level in (no_levels or [])),
    )


def market_meta(raw: dict) -> Quote:
    """A market's metadata with the touch blanked.

    `/markets` serves cached prices, so only a book read may price a decision.
    """
    return replace(_to_quote(raw), yes_bid=None, yes_ask=None, no_bid=None,
                   no_ask=None, yes_bid_size=None, yes_ask_size=None)


def _clean(params: dict) -> dict:
    return {k: v for k, v in params.items() if v is not None}


def _to_quote(raw: dict) -> Quote:
    strike = raw.get('floor_strike')
    return Quote(
        ticker=str(raw.get('ticker', '')),
        yes_bid=_price(raw, 'yes_bid'),
        yes_ask=_price(raw, 'yes_ask'),
        no_bid=_price(raw, 'no_bid'),
        no_ask=_price(raw, 'no_ask'),
        last_price=_price(raw, 'last_price'),
        volume=int(_quantity(raw, 'volume')),
        open_interest=int(_quantity(raw, 'open_interest')),
        exchange_index=int(raw.get('exchange_index') or 0),
        close_time=_parse_time(raw.get('close_time')),
        status=str(raw.get('status', 'unknown')),
        yes_bid_size=_quantity(raw, 'yes_bid_size') or None,
        yes_ask_size=_quantity(raw, 'yes_ask_size') or None,
        floor_strike=float(strike) if strike is not None else None,
        strike_type=raw.get('strike_type'),
        open_time=_parse_time(raw.get('open_time')),
    )
