"""The Kalshi trading client: quotes, balance, orders, fills.

**Markets are resolved by asking the venue, never by building a ticker.** A
15-minute up/down market has a ticker, and it would be easy to construct one
from a series prefix, a date and a time. That is a guess that keeps working
until the venue renames a series or changes a format, and then it fails by
finding *no* market — or, far worse, the wrong one. So `resolve_window_market`
lists the venue's own markets and picks the one whose close time is the window's
settlement. It re-derives its answer from the venue every run and fails loudly
when the venue moves.

Two measured reasons this is not over-caution. `KXBTCD` looks like the daily
Bitcoin series and is in fact hourly with an explicit strike in the ticker
(`KXBTCD-26AUG2317-T86749.99`) — a threshold ladder, not an up/down market;
pointed at it, every window abstained with the distance printed rather than
trading a contract 30 minutes away. And the 15-minute tickers are named in
**Eastern** time while `close_time` is UTC: `KXBTC15M-26AUG230045` closes at
04:45Z, because 00:45 EDT is 04:45 UTC. Parsing the ticker for its settlement
would mean hardcoding the venue's timezone and its daylight-saving rule, and
would be wrong twice a year. `close_time` is unambiguous, so that is what is
compared.

The venue lists **one open market per series** — the window currently running.
The next is created when this one settles, so a lookahead resolution correctly
finds nothing, and the live loop only ever needs the window it is inside.

**Live decisions must be priced against the real book.** The backtest has no
quote history, so it stands the calibrated baseline in for the market — a
deliberate, conservative null. A live decision has no such excuse: it reads the
bid and ask, and `Prediction.price_source` records which of the two priced it.
A row that cannot tell those apart makes a backtest look like a fill.

**Nothing places an order unless it is asked twice.** `place_order` refuses
unless the client was constructed with `live=True`, and every caller defaults to
dry run. The failure mode being designed against is a script that was meant to
observe and instead traded, which is unrecoverable in a way a missed trade is
not.

Authentication is an RSA key, not an HMAC secret: Kalshi signs
`timestamp + method + path` with RSA-PSS over SHA-256 and sends the signature
base64 in `KALSHI-ACCESS-SIGNATURE`, the key id in `KALSHI-ACCESS-KEY`, and
milliseconds since the epoch in `KALSHI-ACCESS-TIMESTAMP`.
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import aiohttp

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
DEMO_BASE_URL = 'https://demo-api.kalshi.co/trade-api/v2'

# Kalshi quotes in whole cents, 1..99.
CENT = 0.01


class KalshiError(RuntimeError):
    """The venue refused, or the response was not what the caller assumed."""


class NotLiveError(KalshiError):
    """An order was attempted on a client that was not constructed for it."""


def _load_private_key(pem: str | bytes):
    from cryptography.hazmat.primitives import serialization

    data = pem.encode() if isinstance(pem, str) else pem
    return serialization.load_pem_private_key(data, password=None)


@dataclass(frozen=True)
class Quote:
    """One side of the book, as probabilities rather than cents.

    Converted at the boundary on purpose: everything above this module reasons on
    the probability scale, and a stray factor of 100 between cents and dollars is
    the classic bug in a binary system.

    **The venue serves prices as dollar-denominated strings**, in fields suffixed
    `_dollars`: `yes_bid_dollars: "0.1900"`. This module originally read integer
    cents from `yes_bid`, which is absent — so every quote parsed as null, every
    book looked empty, and the first live sampling run reported "no two-sided
    book on any symbol" against a market that was quoting 0.19/0.20 with 1,594
    contracts on the bid. Both encodings are accepted now, because a venue that
    changed once can change back.
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
    # Contracts resting at the touch. The depth assumption
    # (`Config.max_stake_dollars`) has been an unmeasured guess; these make it
    # measurable, and the first observation was 59 contracts on the ask at 20c —
    # about $12, well under the $25 the sizing rules were willing to stake.
    yes_bid_size: Optional[float] = None
    yes_ask_size: Optional[float] = None
    # The number the market settles against, published once the window opens.
    # The live path should prefer this over anything computed from bars.
    floor_strike: Optional[float] = None
    strike_type: Optional[str] = None
    open_time: Optional[datetime] = None

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
    """A price in dollars, from whichever encoding the venue used.

    Kalshi serves `yes_bid_dollars: "0.1900"` — a string, already in dollars.
    Older documentation describes `yes_bid: 19`, an integer in cents. Reading only
    the second gets None from a live market and an empty book from a market that
    is quoting, so both are tried and the dollar form wins.
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


def _cents(value: Any) -> Optional[float]:
    """Integer cents to dollars. Retained for the legacy encoding and tests."""
    if value is None:
        return None
    try:
        cents = float(value)
    except (TypeError, ValueError):
        return None
    return cents * CENT if cents > 0 else None


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value).replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(text).astimezone(timezone.utc)
    except ValueError:
        return None


class KalshiClient:
    """Async client. One session, signed requests, and an explicit live flag."""

    def __init__(
        self,
        *,
        key_id: Optional[str] = None,
        private_key_pem: Optional[str] = None,
        private_key_path: Optional[str] = None,
        base_url: Optional[str] = None,
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
                       body: Optional[dict] = None) -> dict:
        if self._session is None:
            raise KalshiError('client is not open; use `async with KalshiClient(...)`')
        # The signature covers the path without the query, so build it from the
        # same string that is signed rather than from the final URL.
        signed_path = f'/trade-api/v2{path}'
        headers = self._headers(method, signed_path)
        url = f'{self.base_url}{path}'
        async with self._session.request(
            method, url, params=params,
            data=json.dumps(body) if body is not None else None,
            headers=headers,
        ) as response:
            text = await response.text()
            if response.status >= 400:
                raise KalshiError(
                    f'{method} {path} -> {response.status}: {text[:400]}')
            return json.loads(text) if text else {}

    # ---- reading --------------------------------------------------------
    async def balance(self) -> float:
        """Available balance in dollars, or NaN when the venue did not say.

        Two corrections. It read only the integer-cent `balance` field, while
        every other number in this file prefers the `_dollars` form — this module
        already learned that lesson for quotes, where reading only the cents
        field reported "no two-sided book" against a market quoting 0.19/0.20.
        A venue that serves `balance_dollars` here would have been read 100x low.

        And a missing or unparseable field defaulted to `0`, silently, producing a
        plausible-looking $0.00. That is the worst possible failure for this
        particular number: `reconcile_with_venue` writes it straight onto the
        account, so a parse change would have overwritten a correct bankroll with
        zero — and `check_venue.py` treats a zero balance as a legitimate
        outcome ("a zero balance still proves auth"), so the one script whose job
        is to catch this would have printed OK. NaN instead, which the callers
        already treat as unsafe.
        """
        payload = await self._request('GET', '/portfolio/balance')
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

        Asks the venue for open markets in the series and returns the one closing
        when this window settles. Returns None rather than guessing — a caller
        that cannot find its market must abstain, not trade a neighbouring one.
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

    async def positions(self) -> list[dict]:
        payload = await self._request('GET', '/portfolio/positions')
        return list(payload.get('market_positions', []))

    async def fills(self, **params) -> list[dict]:
        payload = await self._request('GET', '/portfolio/fills', params=_clean(params))
        return list(payload.get('fills', []))

    async def settlements(self, **params) -> list[dict]:
        """Settled positions, as the venue resolved them.

        The authority on what a position was worth. Settling from our own bars
        means settling against an OHLC mean of Coinbase standing in for sixty
        seconds of CF Benchmarks BRTI — a close proxy that will sometimes
        disagree, and when it does our books are wrong and the venue's are right.
        """
        payload = await self._request('GET', '/portfolio/settlements',
                                      params=_clean(params))
        return list(payload.get('settlements', []))

    async def reconcile(self) -> dict:
        """Balance, open positions and recent fills, as the venue sees them.

        One call site for everything authoritative, so the live loop can compare
        its own arithmetic against the account of record rather than assume the
        two agree.
        """
        balance = await self.balance()
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
        contracts: int,
        limit_price: float,
        client_order_id: Optional[str] = None,
        time_in_force: str = 'fill_or_kill',
    ) -> dict:
        """Buy `contracts` of `side` at no worse than `limit_price` (dollars).

        Refuses unless the client was constructed with `live=True`. The default
        is `fill_or_kill`: a 15-minute market is a wasting asset, and a resting
        order that fills eight minutes later is filling against a barrier
        probability that no longer holds — the decision it came from has expired.
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

        # V2. The V1 endpoint returns 410 `deprecated_v1_order_endpoint`, found on
        # the first real cycle — no amount of testing against our own mocks would
        # have caught it.
        #
        # This is not a renamed path. V2 quotes a SINGLE book from the YES side:
        # `bid` means buy YES, `ask` means sell YES, and selling YES is
        # economically buying NO at `1 - price`. There is no `action`, no `type`,
        # no `yes_price`/`no_price`, and `self_trade_prevention_type` is required.
        #
        # So `limit_price` arrives as what we would PAY for `side` (that is what
        # `decide` computes) and has to be converted to a YES-denominated limit:
        # paying 0.30 for NO is selling YES at 0.70. Sending 0.30 as an `ask`
        # would offer to sell YES at thirty cents — the same class of error as
        # inverting the side outright, and the mutation that survived all 230
        # tests before this was written.
        book_side = 'bid' if side == 'up' else 'ask'
        yes_limit = limit_price if side == 'up' else 1.0 - limit_price

        # Round so the limit never becomes one that CANNOT fill. A bid fills
        # against asks at or below it, so round up; an ask fills against bids at
        # or above it, so round down. Rounding the wrong way under `fill_or_kill`
        # is a guaranteed kill, which the old code did in the tails by using
        # banker's rounding on a deci-cent ladder.
        #
        # The epsilon absorbs binary representation: 0.60 is 59.999...c in float.
        if book_side == 'bid':
            cents = int(math.ceil(yes_limit / CENT - 1e-9))
        else:
            cents = int(math.floor(yes_limit / CENT + 1e-9))
        if not 1 <= cents <= 99:
            raise ValueError(
                f'limit {limit_price} on the {side} side is a YES limit of '
                f'{yes_limit:.4f} ({cents}c), outside 1c..99c'
            )

        body = {
            'ticker': ticker,
            'side': book_side,
            # Decimal strings, not integers. `count` accepts up to 2 decimals and
            # `price` 2-4; a bare int for count is rejected.
            'count': f'{int(contracts)}.00',
            'price': f'{cents / 100.0:.4f}',
            'time_in_force': time_in_force,
            # Required in V2. `taker_at_cross` crosses the spread, which is what a
            # fill_or_kill on a wasting 15-minute market is for; `maker` would rest
            # and is the opposite of the intent.
            'self_trade_prevention_type': 'taker_at_cross',
        }
        if client_order_id:
            body['client_order_id'] = client_order_id
        logger.info('placing %s (%s) %d @ %dc YES on %s',
                    side, book_side, contracts, cents, ticker)
        payload = await self._request('POST', '/portfolio/events/orders', body=body)
        return dict(payload.get('order', payload))

    async def cancel(self, order_id: str) -> dict:
        if not self.live:
            raise NotLiveError('client is not live')
        return await self._request('DELETE', f'/portfolio/orders/{order_id}')


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
        close_time=_parse_time(raw.get('close_time')),
        status=str(raw.get('status', 'unknown')),
        yes_bid_size=_quantity(raw, 'yes_bid_size') or None,
        yes_ask_size=_quantity(raw, 'yes_ask_size') or None,
        floor_strike=float(strike) if strike is not None else None,
        strike_type=raw.get('strike_type'),
        open_time=_parse_time(raw.get('open_time')),
    )
