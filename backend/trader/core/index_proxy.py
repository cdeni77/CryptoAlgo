"""A live estimate of the index Kalshi settles on, from its constituent exchanges.

Kalshi settles BTC on CF Benchmarks' BRTI (ETH and SOL on CF's equivalents),
not on Coinbase. This is an approximation: the plain median of each fresh
exchange's top-of-book mid across four venues. It has no depth weighting and no
outlier filter beyond the median. `research/validate/_validate_index_proxy.py`
measures the gap.

Freshness is per connection, not per quote: a quiet book sends nothing.
An exchange counts when its connection delivered something within
`max_staleness` seconds and that symbol's quote changed within `MAX_QUIET_SECONDS`.

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
# connection: a minute without a top-of-book change is a stuck book.
MAX_QUIET_SECONDS = 60.0


def _number(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


class SideBook:
    """One side of an L2 book, with the best level cached.

    The best is recomputed only when the best level is removed, so an update
    is O(1) in the common case.
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
        # A crossed or locked book on one venue is broken, not a price.
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

    Fewer than `min_exchanges` fresh venues gives NaN, not one venue's price.
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
        # `heartbeats` gives connection liveness every second. `ticker` carries
        # the best bid and ask on each message, so no book is folded.
        return [{'type': 'subscribe', 'product_ids': products, 'channel': channel}
                for channel in ('heartbeats', 'ticker')]
    if exchange == 'kraken':
        # `event_trigger: bbo` fires the ticker on every best-bid/offer change.
        return [{'method': 'subscribe',
                 'params': {'channel': 'ticker', 'symbol': products,
                            'event_trigger': 'bbo', 'snapshot': True}},
                {'method': 'subscribe',
                 'params': {'channel': 'trade', 'symbol': products,
                            'snapshot': False}}]
    if exchange == 'bitstamp':
        # `order_book_<pair>` pushes the whole top-100 on every change: each
        # message is a snapshot.
        return [{'event': 'bts:subscribe', 'data': {'channel': f'{kind}_{p}'}}
                for p in products for kind in ('order_book', 'live_trades')]
    if exchange == 'gemini':
        return [{'type': 'subscribe',
                 'subscriptions': [{'name': 'l2', 'symbols': products}]}]
    raise ValueError(f'unknown exchange {exchange!r}')


def handle_coinbase(payload: dict, state: ExchangeState, received: float) -> Optional[str]:
    """Advanced Trade `ticker`: `events[].tickers[]`, each with `product_id`,
    `best_bid`, `best_ask` and the last trade `price`."""
    if not isinstance(payload, dict):
        return None
    if payload.get('type') == 'error':
        return f"venue error: {str(payload.get('message') or payload)[:160]}"
    state.alive(received)
    if payload.get('channel') != 'ticker':
        return None
    for event in payload.get('events') or []:
        for ticker in event.get('tickers') or []:
            symbol = state.symbol_for(ticker.get('product_id'))
            if symbol is None:
                continue
            state.set_bbo(symbol, ticker.get('best_bid'), ticker.get('best_ask'),
                          received)
            state.trade(symbol, ticker.get('price'), received)
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

    The book is cleared on every (re)connect, so the first complete
    `l2_updates` applied onto an empty book is the snapshot.
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

    Process-wide, so the trading loop reads what the recorder maintains.
    The history covers a window plus its strike minute, as `index_features_row` needs.
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
