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

from .timeutil import naive_utc_to_epoch_seconds

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = 'https://api.elections.kalshi.com/trade-api/v2'
DEMO_BASE_URL = 'https://demo-api.kalshi.co/trade-api/v2'

# **The historical tier is a different host, and that is not a typo.** Since
# 2026-02-19 the venue partitions its data: the live endpoints serve roughly the
# last three months and refuse to look further back, and everything older moved
# to `/historical/...`. Those routes answer on `external-api.kalshi.com` while
# this account's trading routes answer on the host above, so a single base URL
# cannot reach both. Building a complete fill history means querying the live
# endpoint and the historical one and merging — which is what `all_fills` does.
#
# Overridable, because a venue that split its hosts once can merge them again,
# and the failure mode is a 404 that names a route rather than a wrong number.
HISTORICAL_BASE_URL = 'https://external-api.kalshi.com/trade-api/v2'

# Kalshi quotes in whole cents, 1..99.
CENT = 0.01

# Sorts a row the venue gave no timestamp to, without dropping it. `None`
# is not orderable against a datetime, and a ledger that raises on one
# missing field is worse than one that puts that row last.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


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
    # **Which exchange the market lives on.** The order body's `exchange_index`
    # defaults to 0, and on 2026-08-25 every KX*15M market reported 2 — orders
    # defaulting to 0 came back `404 market_not_found` for a market every read
    # endpoint confirmed was active. 274 orders had filled earlier the same night,
    # so the series moved rather than the code breaking. Carrying the market's own
    # value means a future migration costs nothing.
    exchange_index: int = 0
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


def _money(raw: dict, name: str) -> Optional[float]:
    """A dollar amount, from whichever encoding the venue used. **Zero is a value.**

    Same trap as `_price` and a different resolution. The venue serves
    `revenue_dollars: "21.0000"` alongside a legacy `revenue: 2100` in integer
    cents, so the suffixed form has to win or the number is 100x wrong.

    But `_price` maps a zero to None deliberately — a zero *quote* means there is
    no level there, not that a contract is free. On a settlement that reasoning
    inverts: **a losing position settles at revenue exactly 0**, and the loss is
    the whole point of recording it. Reusing `_price` here would have turned every
    loser into a missing measurement, which is the one direction of error an
    equity curve must never make. So zero is returned as zero, and only a genuinely
    absent or unparseable field is None.
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
    """A fee, which is **already in dollars** even without the `_dollars` suffix.

    `_money` falls back to multiplying an unsuffixed field by a cent, which is
    right for `revenue` (the venue serves `revenue: 500` for a $5.00 payout) and
    wrong here. The venue does not serve `fee_cost_dollars` on a settlement at
    all, so that fallback fired every time and stored the fee a hundred times too
    small — and `pnl = revenue - cost - fee_cost` was inflated by the difference.
    Measured on the live account: 365 settlements reported $0.28 of fees against
    roughly $28 actually charged, and realised P&L read $40.34 instead of about
    $12.62.

    The proof is the published schedule. A live row with five NO contracts at
    $4.515 carried `fee_cost: 0.030700`, and
    `ceil(0.07 * 5 * p * (1-p) * 10000) / 10000` with `p = 4.515/5` is 0.0307 —
    the fee in dollars, to the hundredth of a cent, exactly as `core/costs.py`
    computes it. A fee is never a round number of cents on this venue, so an
    unsuffixed `fee_cost` that is already fractional cannot be cents.

    The suffixed form still wins where the venue sends one, and zero stays zero:
    a settlement really can be charged nothing.
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

    Our own `Position` row says what we *believed* we bought: the price `decide()`
    sized at and the fee it predicted from the published schedule. This says what
    happened. They differ whenever an order partially fills, fills better than the
    limit, or is charged a fee we mispriced — 16 of the first 323 live fills were
    partial — and where they differ this one is right.

    `price` is what was paid for `side`, on the probability scale, which is the
    scale everything above this module reasons on. The venue quotes a single
    YES-denominated book, so a NO fill arrives as `no_price` and the two are not
    interchangeable: reading `yes_price` on a NO fill books a 30c purchase as 70c.
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

    This is the row that replaces our arithmetic. We settle from an OHLC mean of
    Coinbase standing in for sixty seconds of CF Benchmarks BRTI — a close proxy
    that will sometimes disagree — and we predict the fee from the published
    schedule. The venue does neither: it knows what it paid and what it charged.

    `pnl` is `revenue - cost - fee_cost` and is `None` when any of the three was
    absent, rather than treating a missing field as zero. A settlement whose
    revenue did not parse is not a break-even trade.
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
        """Realised profit on this market. `None` when the venue left a gap.

        The fee is subtracted rather than assumed netted out. Kalshi charges at
        order time and settlement is free, so `revenue` is the payout and
        `fee_cost` is a separate debit that already left the balance. Getting that
        assumption backwards double-counts the fee, which is why
        `venue_ledger.balance_check` compares the ledger's own cash flows against
        the venue's balance instead of trusting this arithmetic unattended.
        """
        cost = self.cost
        if self.revenue is None or cost is None:
            return None
        return self.revenue - cost - (self.fee_cost or 0.0)


@dataclass(frozen=True)
class Trade:
    """One print on the public tape. **Anonymous, and not ours.**

    `/historical/trades` and `/markets/trades` serve every trade the exchange
    printed in a market, by anyone. There is no account field, no side that is
    "ours", and nothing that distinguishes our 5 contracts from a stranger's 500 —
    so this cannot compute a portfolio, and reading a tape total as a position is
    how a P&L page comes to show someone else's money.

    **The `taker_*` fields do not make it ours.** The payload carries
    `taker_outcome_side`, `taker_side` and `taker_book_side`, which is the closest
    this endpoint comes to naming a participant — and what they name is which side
    the *aggressor of that print* crossed on. Any account could be the aggressor,
    including someone else's. There is still no account id, so a tape filtered to
    `taker_outcome_side == 'yes'` is not our buys; it is everybody's.

    What it is good for is the two things a portfolio page actually lacks: a
    market-observed last price to mark an open position at (our own forecast must
    never do that job — marking a binary at the probability we believe books
    conviction as profit), and an independent check that a fill printed at the
    price the venue told us it did. `Fill.trade_id` joins to `trade_id` here, and
    for a fill where `Fill.is_taker` is true `taker_side` should agree with
    `Fill.side` — which is what makes the check a check rather than a restatement.
    """

    trade_id: str
    ticker: str
    contracts: float
    yes_price: Optional[float]
    no_price: Optional[float]
    created_time: Optional[datetime]
    is_block_trade: Optional[bool]
    # Which side the aggressor took, in this project's vocabulary ('up'/'down'),
    # and which side of the book they hit ('bid'/'ask', kept verbatim because it
    # is the venue's own single-book language and not a direction).
    taker_side: Optional[str] = None
    taker_book_side: Optional[str] = None

    def price_for(self, side: str) -> Optional[float]:
        """What this print paid for `side`.

        Both `yes_price` and `no_price` are read from the payload rather than one
        being derived as `1 - other`. They are usually complementary and nothing
        here needs them to be: a venue that quotes them independently, or a
        documented example where they do not sum to a dollar, must not silently
        become a wrong number.
        """
        return self.yes_price if side == 'up' else self.no_price


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
    """A fills row, in this project's vocabulary.

    The side mapping is the load-bearing part. The venue names the YES book:
    `side: "yes"` is a YES contract and `side: "no"` is a NO contract, which this
    project calls 'up' and 'down'. Anything else is carried through verbatim
    rather than guessed at, because a mislabelled side inverts a trade's PnL and
    an unfamiliar string is not evidence for either direction.
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

    Money fields go through `_money`, so a `revenue` of 0 on a lost market stays 0
    rather than becoming a missing measurement.

    **`fee_cost` does NOT go through `_money`** — see `_fee`. The venue serves
    `revenue` in integer cents and `fee_cost` in dollars, both unsuffixed, so the
    one cents fallback cannot serve both. Reading the fee through it divided every
    fee by a hundred and inflated every settled trade's P&L.
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

    `taker_outcome_side` is preferred over `taker_side`: the payload serves both,
    which is the venue's usual pattern for a field it has renamed, and reading the
    older name first would go stale silently the day the alias is dropped.
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
        # The host for `/historical/...` only. Defaults to a different host from
        # `base_url` on purpose — see HISTORICAL_BASE_URL.
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
        # The signature covers the path without the query, so build it from the
        # same string that is signed rather than from the final URL.
        #
        # `base` names a different HOST, never a different path prefix: the
        # historical tier answers on `external-api.kalshi.com` under the same
        # `/trade-api/v2` mount, so the signed string is identical and only the
        # host changes. Deriving the signature from the URL instead would sign a
        # different string per host and every historical read would 401.
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
                    f'{method} {path} -> {response.status}: {text[:400]}')
            return json.loads(text) if text else {}

    # ---- reading --------------------------------------------------------
    async def balance(self, *, exchange_index: Optional[int] = None) -> float:
        """Available balance in dollars, or NaN when the venue did not say.

        Pass `exchange_index` to get the balance **spendable on that shard**,
        which is the only figure an order against a market on it can draw on.
        Omit it for the whole-account total.

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

        # **Balances are local to an exchange shard.** Kalshi shards by category
        # and `balance_breakdown` carries one entry per `exchange_index`. The
        # KX*15M crypto series report shard 2, and on 2026-08-25 this account had
        # $106.61 on shard 0 and $1.35 on shard 2 — so `balance_dollars` of
        # $107.96 was 80x the capital an order could actually reach, and every
        # order came back `insufficient_balance` against an apparently healthy
        # balance. A caller that names the shard gets what it can spend there; a
        # caller that does not keeps the whole-account meaning.
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
                # A shard the venue does not list holds nothing. Reporting the
                # total here would be the original bug wearing a new name.
                return 0.0
            # No breakdown at all: an older response shape, or a venue that does
            # not shard. Falling through to the total is right, and refusing
            # would halt trading on a venue behaving correctly.

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

    async def orderbook(self, ticker: str) -> dict:
        """The full ladder, parsed to the touch. See `parse_orderbook`."""
        return parse_orderbook(
            await self._request('GET', f'/markets/{ticker}/orderbook'))

    async def quote_with_live_book(self, ticker: str) -> Quote:
        """`quote()`, but with the touch taken from the LIVE orderbook.

        **`/markets` serves a CACHED touch, and pricing a decision off it is
        what stopped the live loop filling.** Measured 2026-09-04 by sampling
        both endpoints for the same market two seconds apart: `/markets`
        returned an identical `0.9120/0.9160` on ETH for five consecutive
        samples across ten seconds while `/orderbook` moved 0.934 -> 0.960 ->
        0.951 -> 0.950. Across 28 paired samples the two disagreed on 96.4%,
        mean 0.66c with the ask understated by up to 6.3c.

        That is not a rounding detail, for two reasons:

        * **The order cannot fill.** The captured kill priced a 0.74 ask against
          a real 0.76 and sent a 74c bid — a cent below the actual BID. Ten of
          ten orders after the 2026-09-04 restart died this way.
        * **The MODEL is fed the stale number too.** The promoted artifact is
          `init_score_source=market`, so its prediction is a correction to the
          price; `market_probability` is the midpoint of this touch. A quote
          several cents stale means the model corrects the wrong number, which
          is a forecast error and not merely a fill error.

        It also fixes a live/training mismatch: the backtest reads
        `venue_depth`, built by `scripts/build_depth.py` from RECORDED LADDERS —
        the real book — so `/markets` was a source live used and training never
        saw. "One `decide()`" guarantees the decision is shared; nothing
        guaranteed the price reaching it came from the same place.

        An empty book is trusted, because an empty book is a real state. Only a
        FAILED request falls back to the cached touch, and says so.
        """
        quote = await self.quote(ticker)
        try:
            book = await self.orderbook(ticker)
        except (KalshiError, asyncio.TimeoutError, OSError) as exc:
            logger.warning(
                '%s: the live orderbook did not answer (%s), so this decision '
                'is priced off the CACHED /markets touch and may be several '
                'cents stale. Expect it not to fill.', ticker, exc)
            return quote

        def finite(value):
            value = float(value) if value is not None else float('nan')
            return value if value == value else None

        yes_bid, no_bid = finite(book['yes_bid']), finite(book['no_bid'])
        return replace(
            quote,
            yes_bid=yes_bid,
            no_bid=no_bid,
            # Two BID stacks, so both asks are conversions. Buying YES crosses
            # the NO stack at `1 - best_no_bid`.
            yes_ask=(1.0 - no_bid) if no_bid is not None else None,
            no_ask=(1.0 - yes_bid) if yes_bid is not None else None,
            yes_bid_size=finite(book['yes_bid_size']),
            # The size behind the YES ask is what rests on the NO bid.
            yes_ask_size=finite(book['no_bid_size']),
        )

    async def positions(self) -> list[dict]:
        payload = await self._request('GET', '/portfolio/positions')
        return list(payload.get('market_positions', []))

    @staticmethod
    def position_size(row: dict) -> float:
        """Signed contracts held in one market, from a positions row.

        **The field is `position_fp`, and it is a fixed-point string.** The live
        loop read `row['position']`, which V2 does not send at all, so
        `int(row.get('position') or 0)` was `int(0)` for every row — every open
        position looked closed. Measured against a position we had just watched
        fill:

            {"ticker": "KXBTC15M-26AUG241000-00", "position_fp": "-5.00",
             "market_exposure_dollars": "2.150000", ...}

        This is the same trap as the quote fields, which arrive as
        `yes_bid_dollars` strings while the integer-cent fields documented
        elsewhere come back null. `_quantity` already existed for it; the
        positions path simply never used it.

        Negative is a short YES, which is how a NO position is held — so callers
        asking "is anything open here" want `!= 0`, not `> 0`.
        """
        return _quantity(row, 'position')

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

    # ---- the venue's own ledger ----------------------------------------
    #
    # Everything below reads what the venue recorded, and it exists because our
    # own arithmetic is not the account of record. The paper engine debits a
    # bankroll at the price `decide()` sized at and the fee it predicted from the
    # published schedule; live, both are guesses about someone else's ledger.
    # Where they disagree the venue is right, and the first live night produced
    # exactly that disagreement — see `adopt_venue_balance`.

    async def historical_cutoff(self) -> dict[str, Optional[datetime]]:
        """Where the live tier stops and the historical tier begins.

        Since 2026-02-19 the live endpoints refuse to look past a moving cutoff
        (roughly three months). Reading it rather than assuming a retention
        window is the same discipline as resolving a market by asking: a constant
        keeps working until the venue moves it, and then the ledger is silently
        short of its oldest rows.

        Returns a dict of whatever timestamps the venue named, parsed to UTC. An
        unreachable cutoff is not fatal — `all_fills` then queries both tiers
        unconditionally, which is merely wasteful — so this returns `{}` rather
        than raising.
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
        """Follow the cursor to the end. Bounded, because an unbounded loop is not a read.

        The venue pages at `limit` (200 max on these routes) and returns a
        `cursor` for the next page; an empty or repeated cursor is the end. The
        page cap is a circuit breaker rather than a limit anyone should hit — 40
        pages is 8,000 rows — and it *logs* when it trips instead of silently
        returning a truncated ledger, because a P&L short of its oldest fills
        looks exactly like a P&L.
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
            # A cursor the venue repeats is a loop, not a next page. Observed on
            # other venues at the last page; cheap to defend against.
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

        Two tiers and one ledger. The live route serves the last ~3 months and the
        historical route everything before it, so a complete history means asking
        both and merging — the venue's own documentation says so, and a caller that
        asks only the live route gets a P&L that begins three months ago and looks
        complete.

        Deduplicated on `trade_id` because the tiers overlap around the cutoff:
        the same fill can appear in both, and counting one twice doubles a
        position's cost basis.
        """
        params: dict[str, Any] = {'limit': limit}
        if since is not None:
            params['min_ts'] = naive_utc_to_epoch_seconds(since)

        rows = await self._pages('/portfolio/fills', 'fills', **params)
        try:
            rows += await self._pages('/portfolio/fills/historical', 'fills',
                                      base=self.historical_base_url, **params)
        except KalshiError as exc:
            # Not fatal, and not hidden. The live tier alone is a correct answer
            # for a recent window and a wrong one for a lifetime P&L, so the
            # caller is told which it got.
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
        """Every settled market, both tiers merged, newest first.

        **This is the P&L.** A binary bought once and held pays exactly one fee at
        entry and settles once at $1 or $0, so a settlement row is the complete
        economic history of a position: what it cost, what it returned, what it
        was charged. Our own books recompute all three from Coinbase bars and the
        published fee schedule, and both are approximations of this.

        Keyed by ticker, which is unique per settled market.
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
        """The public tape. **Everyone's trades, not ours — see `Trade`.**

        Useful for exactly two jobs, and dangerous for a third. It marks an open
        position at a price the market printed rather than at the probability we
        believe, and it verifies that a fill printed where the venue said it did
        (`Fill.trade_id` joins to `trade_id`). It cannot compute a portfolio: no
        row here says who traded, so summing the tape sums the exchange.

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
        """Balance, open positions and recent fills, as the venue sees them.

        One call site for everything authoritative, so the live loop can compare
        its own arithmetic against the account of record rather than assume the
        two agree.

        `exchange_index` narrows the balance to the shard the traded markets live
        on. Without it the cross-shard total is reported, which is not money any
        single order can draw on — see `balance`.
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

        The default is `immediate_or_cancel`, not `fill_or_kill`. Both refuse to
        rest — correct on a wasting asset, where an order filling eight minutes
        later is filling against a barrier probability that no longer holds — but
        `fill_or_kill` is all-or-nothing. Nine contracts wanted against five
        resting returns nothing at all; `immediate_or_cancel` takes the five.
        With a positive edge a partial fill strictly beats a kill, and the
        accounting already handles partials (16 of the first 323 live fills were
        partial).
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
            # immediate_or_cancel on a wasting 15-minute market is for; `maker`
            # would rest and is the opposite of the intent.
            'self_trade_prevention_type': 'taker_at_cross',
            # Which exchange holds the market. Omitting it defaults to 0, and a
            # market on another exchange is then simply not found — a 404 that
            # names the market rather than the mismatch.
            'exchange_index': int(exchange_index),
        }
        if client_order_id:
            body['client_order_id'] = client_order_id
        logger.info('placing %s (%s) %d @ %dc YES on %s',
                    side, book_side, contracts, cents, ticker)
        payload = await self._request('POST', '/portfolio/events/orders', body=body)
        # **The venue's own answer, in full.** The caller reads `status` and a
        # count off this reply and reports "did not fill" when neither is
        # there — and the POST reply's shape is NOT the GET order record's, so
        # `status` came back None on every attempt and the reason, whatever it
        # is, was discarded unread. Ten consecutive kills on 2026-09-04 were
        # diagnosed as a price miss on exactly this missing evidence, against a
        # book the stream shows our limit was 2c through. Log both sides once
        # per order: three lines a window is nothing, and the alternative is
        # inferring a refusal from its absence.
        logger.info('order request  %s', json.dumps(body, sort_keys=True))
        logger.info('order reply    %s', json.dumps(payload, sort_keys=True, default=str))
        return dict(payload.get('order', payload))

    async def cancel(self, order_id: str) -> dict:
        if not self.live:
            raise NotLiveError('client is not live')
        return await self._request('DELETE', f'/portfolio/orders/{order_id}')


NAN = float('nan')


def parse_orderbook(payload: dict) -> dict:
    """The touch of a REST orderbook, in YES-denominated terms.

    **The shape is nested and its own thing.** A live response is
    `{'orderbook': {'orderbook_fp': {'yes_dollars': [[price, size], ...],
    'no_dollars': [...]}}}`, with prices AND sizes as decimal strings and the
    deci-cent grid present in the tails. The stream sends different keys again
    (`yes_dollars_fp`, `price_dollars`/`delta_fp`), and reading one shape
    against the other yields an empty book and no exception — which is how a
    book read can report "nothing there" about a market quoting thousands of
    contracts. Every known shape is accepted here for exactly that reason.

    Kalshi's book is two BID stacks, so an ask is a conversion and not a field:
    the YES ask is `1 - best_no_bid`. Sizes stay on the stack they came from —
    `no_bid_size` is what a YES *buy* crosses.

    NaN, never zero, on an empty stack: a settled market returns empty ladders,
    and zero would read as a real price at the bottom of the grid.
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
