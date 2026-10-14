"""A live estimate of the index Kalshi settles on, from its constituent exchanges.

Kalshi settles BTC on CF Benchmarks' BRTI (ETH and SOL on CF's equivalent
real-time indices), not on Coinbase. Our label is a Coinbase one-minute OHLC
mean, and measured against Kalshi's own results it disagrees on ~35% of windows
within 1bp of the strike and ~14% within 1-2bp — exactly the near-ties this
system trades disproportionately. A better estimate of the index itself is the
most promising information nobody here has yet.

**This is an approximation, and the gap is known.** CF's methodology builds a
consolidated order book from its constituent exchanges, derives a volume-weighted
"utilised depth" mid-price curve from it and takes a weighted median across
exchanges, with outlier and staleness rules on each venue. This takes the plain
MEDIAN of each fresh exchange's top-of-book mid: no depth, no volume weighting,
no outlier filter beyond the median itself, and only four venues (Coinbase,
Kraken, Bitstamp, Gemini — the US-dollar constituents that matter most for
volume). It should track the index far better than one venue's trades do; it
will not reproduce it, and `research/validate/_validate_index_proxy.py` is
what says by how much.

**Freshness is per connection, not per quote.** A quiet book sends nothing, so
"no update for five seconds" means "unchanged", not "stale", as long as the
connection is demonstrably delivering — and on an in-order socket any later
message proves that. An exchange counts when its connection delivered something
within `max_staleness` seconds AND that symbol's quote changed within
`MAX_QUIET_SECONDS`, the second bound catching a feed that keeps heartbeating
while one book silently stops.

Pure: no sockets and no clock. Every receive time is passed in.
"""
from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Optional

import pandas as pd

EXCHANGES = ('coinbase', 'kraken', 'bitstamp', 'gemini')
MID_COLUMNS = tuple(f'mid_{e}' for e in EXCHANGES)

# Each venue's name for our three symbols.
PRODUCTS: dict[str, dict[str, str]] = {
    'coinbase': {'BTC-USD': 'BTC-USD', 'ETH-USD': 'ETH-USD', 'SOL-USD': 'SOL-USD'},
    'kraken': {'BTC-USD': 'BTC/USD', 'ETH-USD': 'ETH/USD', 'SOL-USD': 'SOL/USD'},
    'bitstamp': {'BTC-USD': 'btcusd', 'ETH-USD': 'ethusd', 'SOL-USD': 'solusd'},
    'gemini': {'BTC-USD': 'BTCUSD', 'ETH-USD': 'ETHUSD', 'SOL-USD': 'SOLUSD'},
}

URLS = {
    'coinbase': 'wss://advanced-trade-ws.coinbase.com',
    'kraken': 'wss://ws.kraken.com/v2',
    'bitstamp': 'wss://ws.bitstamp.net',
    'gemini': 'wss://api.gemini.com/v2/marketdata',
}

# A symbol whose quote has not changed for this long is dropped even on a live
# connection. Generous: a top-of-book on any of these venues moves within
# seconds, so a minute of nothing is a stuck book, not a calm market.
MAX_QUIET_SECONDS = 60.0


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class SideBook:
    """One side of an L2 book, with the best level cached.

    Recomputing the best only when the best itself is removed keeps a
    30,000-level Coinbase book at O(1) per update in the common case.
    """

    def __init__(self, bids: bool) -> None:
        self.bids = bids
        self.levels: dict[float, float] = {}
        self._best: Optional[float] = None
        self._dirty = False

    def clear(self) -> None:
        self.levels.clear()
        self._best, self._dirty = None, False

    def set(self, price: float, size: float) -> None:
        if size <= 0.0:
            if self.levels.pop(price, None) is not None and price == self._best:
                self._dirty = True
            return
        self.levels[price] = size
        if self._dirty:
            return
        if (self._best is None or (price > self._best if self.bids
                                   else price < self._best)):
            self._best = price

    def best(self) -> Optional[float]:
        if self._dirty:
            self._best = ((max if self.bids else min)(self.levels)
                          if self.levels else None)
            self._dirty = False
        return self._best


@dataclass
class Quote:
    bid: Optional[float] = None
    ask: Optional[float] = None
    updated: Optional[float] = None
    last_trade: Optional[float] = None
    last_trade_at: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        # A crossed or locked book on ONE venue is a broken book, not a price.
        if self.bid is None or self.ask is None or not (0 < self.bid < self.ask):
            return None
        return 0.5 * (self.bid + self.ask)


@dataclass
class ExchangeState:
    """Everything known about one venue, for every symbol, on one connection."""

    name: str
    symbols: tuple[str, ...] = ('BTC-USD', 'ETH-USD', 'SOL-USD')
    quotes: dict[str, Quote] = field(default_factory=dict)
    books: dict[str, tuple[SideBook, SideBook]] = field(default_factory=dict)
    alive_at: Optional[float] = None
    messages: int = 0

    def __post_init__(self) -> None:
        venue = PRODUCTS.get(self.name, {})
        self.by_product = {venue.get(s, s): s for s in self.symbols}
        for symbol in self.symbols:
            self.quotes.setdefault(symbol, Quote())

    def symbol_for(self, product) -> Optional[str]:
        return self.by_product.get(str(product)) if product is not None else None

    def alive(self, at: float) -> None:
        self.alive_at = at
        self.messages += 1

    def disconnect(self) -> None:
        """Nothing on a dead connection is current, and a new one resends it all."""
        self.alive_at = None
        for quote in self.quotes.values():
            quote.bid = quote.ask = quote.updated = None
        for bid, ask in self.books.values():
            bid.clear()
            ask.clear()

    def set_bbo(self, symbol: str, bid, ask, at: float) -> None:
        quote = self.quotes.setdefault(symbol, Quote())
        quote.bid, quote.ask, quote.updated = _number(bid), _number(ask), at

    def book(self, symbol: str) -> tuple[SideBook, SideBook]:
        if symbol not in self.books:
            self.books[symbol] = (SideBook(bids=True), SideBook(bids=False))
        return self.books[symbol]

    def reset_book(self, symbol: str) -> None:
        bid, ask = self.book(symbol)
        bid.clear()
        ask.clear()

    def apply_level(self, symbol: str, is_bid: bool, price, size) -> None:
        price, size = _number(price), _number(size)
        if price is None or size is None:
            return
        bid, ask = self.book(symbol)
        (bid if is_bid else ask).set(price, size)

    def book_touched(self, symbol: str, at: float) -> None:
        bid, ask = self.book(symbol)
        self.set_bbo(symbol, bid.best(), ask.best(), at)

    def trade(self, symbol: str, price, at: float) -> None:
        value = _number(price)
        if value is None:
            return
        quote = self.quotes.setdefault(symbol, Quote())
        quote.last_trade, quote.last_trade_at = value, at

    def mid(self, symbol: str, now: float, *, max_staleness: float,
            max_quiet: float = MAX_QUIET_SECONDS) -> Optional[float]:
        quote = self.quotes.get(symbol)
        if (quote is None or quote.updated is None or self.alive_at is None
                or now - self.alive_at > max_staleness
                or now - quote.updated > max_quiet):
            return None
        return quote.mid


def proxy_row(states: Iterable[ExchangeState], symbol: str, now: float, *,
              max_staleness: float = 5.0, min_exchanges: int = 2) -> dict:
    """One per-second row: the median fresh mid, how many fed it, and each mid.

    Fewer than `min_exchanges` fresh venues is NaN rather than one venue's
    price: a single exchange is exactly the proxy this replaces.
    """
    row: dict = {'symbol': symbol, 'proxy': float('nan'), 'n_exchanges': 0}
    mids = []
    for state in states:
        mid = state.mid(symbol, now, max_staleness=max_staleness)
        row[f'mid_{state.name}'] = float('nan') if mid is None else mid
        if mid is not None:
            mids.append(mid)
    for column in MID_COLUMNS:
        row.setdefault(column, float('nan'))
    row['n_exchanges'] = len(mids)
    if len(mids) >= max(1, int(min_exchanges)):
        row['proxy'] = float(statistics.median(mids))
    return row


# ---------------------------------------------------------------------------
# Parsers. Each folds one decoded message into an ExchangeState and returns a
# reason when the connection must be rebuilt, else None.
# ---------------------------------------------------------------------------

def subscribe_messages(exchange: str, symbols: Iterable[str]) -> list[dict]:
    products = [PRODUCTS[exchange][s] for s in symbols]
    if exchange == 'coinbase':
        # `heartbeats` keeps a quiet product's connection open and gives the
        # connection-level liveness every second.
        return [{'type': 'subscribe', 'product_ids': products, 'channel': channel}
                for channel in ('heartbeats', 'level2', 'market_trades')]
    if exchange == 'kraken':
        # `event_trigger: bbo` fires the ticker on every best-bid/offer change,
        # so no book has to be maintained for a top-of-book mid.
        return [{'method': 'subscribe',
                 'params': {'channel': 'ticker', 'symbol': products,
                            'event_trigger': 'bbo', 'snapshot': True}},
                {'method': 'subscribe',
                 'params': {'channel': 'trade', 'symbol': products,
                            'snapshot': False}}]
    if exchange == 'bitstamp':
        # `order_book_<pair>` pushes the whole top-100 on every change, so each
        # message is a snapshot and there is no delta to lose.
        return [{'event': 'bts:subscribe', 'data': {'channel': f'{kind}_{p}'}}
                for p in products for kind in ('order_book', 'live_trades')]
    if exchange == 'gemini':
        return [{'type': 'subscribe',
                 'subscriptions': [{'name': 'l2', 'symbols': products}]}]
    raise ValueError(f'unknown exchange {exchange!r}')


def handle_coinbase(payload: dict, state: ExchangeState, received: float) -> Optional[str]:
    """Advanced Trade `level2` (served on channel `l2_data`), `market_trades`."""
    if not isinstance(payload, dict):
        return None
    if payload.get('type') == 'error':
        return f"venue error: {str(payload.get('message') or payload)[:160]}"
    state.alive(received)
    channel = payload.get('channel')
    for event in payload.get('events') or []:
        if channel == 'l2_data':
            symbol = state.symbol_for(event.get('product_id'))
            if symbol is None:
                continue
            if event.get('type') == 'snapshot':
                state.reset_book(symbol)
            for update in event.get('updates') or []:
                side = str(update.get('side') or '').lower()
                if side not in ('bid', 'offer', 'ask'):
                    continue
                state.apply_level(symbol, side == 'bid', update.get('price_level'),
                                  update.get('new_quantity'))
            state.book_touched(symbol, received)
        elif channel == 'market_trades':
            for trade in event.get('trades') or []:
                symbol = state.symbol_for(trade.get('product_id'))
                if symbol is not None:
                    state.trade(symbol, trade.get('price'), received)
    return None


def handle_kraken(payload: dict, state: ExchangeState, received: float) -> Optional[str]:
    """Spot WebSocket v2: `ticker` (bbo-triggered), `trade`, `heartbeat`, `status`."""
    if not isinstance(payload, dict):
        return None
    if payload.get('method') in ('subscribe', 'unsubscribe') and \
            payload.get('success') is False:
        return f"subscribe refused: {str(payload.get('error') or payload)[:160]}"
    state.alive(received)
    channel = payload.get('channel')
    data = payload.get('data')
    if not isinstance(data, list):
        return None
    for entry in data:
        if not isinstance(entry, dict):
            continue
        symbol = state.symbol_for(entry.get('symbol'))
        if symbol is None:
            continue
        if channel == 'ticker':
            state.set_bbo(symbol, entry.get('bid'), entry.get('ask'), received)
        elif channel == 'trade':
            state.trade(symbol, entry.get('price'), received)
    return None


def handle_bitstamp(payload: dict, state: ExchangeState, received: float) -> Optional[str]:
    """`order_book_<pair>` (top-100 snapshots) and `live_trades_<pair>`."""
    if not isinstance(payload, dict):
        return None
    event = payload.get('event')
    if event == 'bts:request_reconnect':
        return 'venue requested a reconnect'
    if event == 'bts:error':
        return f"venue error: {str(payload.get('data') or payload)[:160]}"
    state.alive(received)
    channel = str(payload.get('channel') or '')
    data = payload.get('data') or {}
    if event == 'data' and channel.startswith('order_book_'):
        symbol = state.symbol_for(channel[len('order_book_'):])
        if symbol is None:
            return None

        def top(levels):
            try:
                return levels[0][0]
            except (IndexError, KeyError, TypeError):
                return None
        state.set_bbo(symbol, top(data.get('bids')), top(data.get('asks')), received)
    elif event == 'trade' and channel.startswith('live_trades_'):
        symbol = state.symbol_for(channel[len('live_trades_'):])
        if symbol is not None:
            state.trade(symbol, data.get('price_str', data.get('price')), received)
    return None


def handle_gemini(payload: dict, state: ExchangeState, received: float) -> Optional[str]:
    """Market data v2: `l2_updates` (first one per symbol is the full book), `trade`.

    The book is cleared on every (re)connect, so applying the first, complete
    `l2_updates` onto an empty book IS the snapshot.
    """
    if not isinstance(payload, dict):
        return None
    if payload.get('result') == 'error':
        return f"venue error: {str(payload.get('reason') or payload)[:160]}"
    state.alive(received)
    kind = payload.get('type')
    if kind == 'l2_updates':
        symbol = state.symbol_for(payload.get('symbol'))
        if symbol is None:
            return None
        for change in payload.get('changes') or []:
            try:
                side, price, size = change[0], change[1], change[2]
            except (IndexError, KeyError, TypeError):
                continue
            side = str(side).lower()
            if side not in ('buy', 'sell'):
                continue
            state.apply_level(symbol, side == 'buy', price, size)
        state.book_touched(symbol, received)
        for trade in payload.get('trades') or []:
            state.trade(symbol, trade.get('price'), received)
    elif kind == 'trade':
        symbol = state.symbol_for(payload.get('symbol'))
        if symbol is not None:
            state.trade(symbol, payload.get('price'), received)
    return None


HANDLERS = {'coinbase': handle_coinbase, 'kraken': handle_kraken,
            'bitstamp': handle_bitstamp, 'gemini': handle_gemini}


# ---------------------------------------------------------------------------
# The in-process accessor
# ---------------------------------------------------------------------------

class IndexProxyCache:
    """The latest proxy row per symbol and a rolling one-second history.

    Like `record_stream.CACHE`, process-wide so the trading loop can read what
    the recorder maintains. The history covers a whole window plus its strike
    minute with room to spare, which is all `index_features_row` needs.
    """

    def __init__(self, history_seconds: int = 1800) -> None:
        self.history_seconds = int(history_seconds)
        self._history: dict[str, deque] = {}
        self._latest: dict[str, dict] = {}

    def record(self, row: dict) -> None:
        symbol = row['symbol']
        history = self._history.setdefault(
            symbol, deque(maxlen=self.history_seconds))
        history.append((float(row['t']), row.get('proxy'),
                        int(row.get('n_exchanges') or 0)))
        self._latest[symbol] = dict(row)

    def latest(self, symbol: str) -> Optional[dict]:
        row = self._latest.get(symbol)
        return dict(row) if row is not None else None

    def snapshot(self, symbol: str) -> pd.DataFrame:
        """The history in the shape `index_proxy_1s` is stored in."""
        rows = list(self._history.get(symbol, ()))
        frame = pd.DataFrame(rows, columns=['t', 'proxy', 'n_exchanges'])
        frame['event_time'] = pd.to_datetime(frame['t'], unit='s', utc=True)
        frame['proxy'] = pd.to_numeric(frame['proxy'], errors='coerce')
        frame['symbol'] = symbol
        return frame[['symbol', 'event_time', 'proxy', 'n_exchanges']]

    def symbols(self) -> list[str]:
        return sorted(self._history)
