"""One-minute OHLCV bars folded from a trade stream, on REST candle semantics.

The bar for minute `m` is stamped at `m`'s start, opens at its first trade,
closes at its last, and its volume is the summed trade size — what Coinbase's
`/candles` serves. A minute with no trades has no bar, as in REST.

**A bar is only as good as the coverage under it.** A minute is served only
when the stream was connected, gap-free and untainted for the whole of it, and
the clock has passed its close by `grace_seconds` (trades arrive tens of ms
after they print). Anything else is `None`, and the caller falls back to REST.
An empty minute inside coverage is a real "no trades"; outside it, it is
nothing at all — conflating the two would invent quiet minutes.

Pure and synchronous: no sockets, no clock unless one is passed in.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional, Union

logger = logging.getLogger(__name__)

MINUTE = 60


def minute_of(epoch: float) -> int:
    """The start of the minute holding `epoch`, in whole epoch seconds."""
    return int(epoch // MINUTE) * MINUTE


def parse_time(value: Union[str, float, int, datetime]) -> float:
    """Epoch seconds from an ISO string (nanoseconds allowed), number or datetime."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, datetime):
        return value.timestamp()
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace('Z', '+00:00')).timestamp()
    except ValueError:
        # Coinbase can send nine fractional digits, which `fromisoformat`
        # refuses below Python 3.11's tolerance; pandas reads them exactly.
        import pandas as pd
        return pd.Timestamp(text).value / 1e9


def _order_key(when: float, trade_id: str) -> tuple:
    # Ties on the timestamp break on the venue's id, numerically when it is one.
    return (when, 0, int(trade_id), '') if trade_id.isdigit() else (when, 1, 0, trade_id)


@dataclass
class _Minute:
    open_key: tuple
    open: float
    close_key: tuple
    close: float
    high: float
    low: float
    volume: float
    ids: set = field(default_factory=set)

    def add(self, key: tuple, price: float, size: float, trade_id: str) -> None:
        if key < self.open_key:
            self.open_key, self.open = key, price
        if key > self.close_key:
            self.close_key, self.close = key, price
        self.high = max(self.high, price)
        self.low = min(self.low, price)
        self.volume += size
        self.ids.add(trade_id)


class TradeTape:
    """Bars for every product on ONE connection, plus that connection's coverage.

    Coverage is per connection because a sequence gap or a reconnect cannot be
    attributed to a product: the missing message could have carried any of them.
    """

    def __init__(self, *, grace_seconds: float = 0.75,
                 retain_minutes: int = 180,
                 clock_margin_seconds: float = 1.0,
                 now: Callable[[], float] = time.time) -> None:
        self.grace_seconds = float(grace_seconds)
        self.retain_minutes = int(retain_minutes)
        # A trade printed just before we subscribed can carry a timestamp a
        # little after our clock's `connected` instant; the first covered minute
        # must start clear of that.
        self.clock_margin_seconds = float(clock_margin_seconds)
        self._now = now
        self._minutes: dict[str, dict[int, _Minute]] = {}
        # Minutes a trade reached AFTER they were declared final.
        self._tainted: dict[str, set[int]] = {}
        # Continuous coverage as [start, end) in epoch seconds; end None = open.
        self._coverage: list[list[Optional[float]]] = []
        self.late_trades = 0

    # -- coverage ----------------------------------------------------------

    def connected(self, at: Optional[float] = None) -> None:
        at = self._now() if at is None else float(at)
        if self._coverage and self._coverage[-1][1] is None:
            return
        self._coverage.append([at, None])

    def disconnected(self, at: Optional[float] = None) -> None:
        at = self._now() if at is None else float(at)
        if self._coverage and self._coverage[-1][1] is None:
            self._coverage[-1][1] = at

    def gap(self, since: float, at: Optional[float] = None) -> None:
        """A message was lost somewhere in (since, at]: coverage breaks there."""
        at = self._now() if at is None else float(at)
        if self._coverage and self._coverage[-1][1] is None:
            self._coverage[-1][1] = min(float(since), at)
        self._coverage.append([at, None])

    @property
    def is_connected(self) -> bool:
        return bool(self._coverage) and self._coverage[-1][1] is None

    # -- trades ------------------------------------------------------------

    def add(self, product: str, trade_id, price, size, when,
            *, received: Optional[float] = None, snapshot: bool = False) -> bool:
        """Fold one trade. False when it was a duplicate or unparseable.

        `snapshot` trades are the recent history a subscription opens with.
        Arriving late is their nature, so they taint a minute only when that
        minute was already being served — a new trade there means one was lost.
        """
        try:
            trade_id = str(trade_id)
            price, size = float(price), float(size)
            at = parse_time(when)
        except (TypeError, ValueError):
            return False
        if not trade_id or price <= 0 or size < 0:
            return False
        minute = minute_of(at)
        book = self._minutes.setdefault(product, {})
        acc = book.get(minute)
        if acc is not None and trade_id in acc.ids:
            return False
        received = self._now() if received is None else float(received)
        late = received >= minute + MINUTE + self.grace_seconds
        if late and snapshot:
            late = self.reliable(product, minute, now=received)
        if late:
            # Late: whatever was served for this minute may have missed it.
            self.late_trades += 1
            self._tainted.setdefault(product, set()).add(minute)
            logger.warning('%s: trade %s for %s arrived %.2fs after the minute '
                           'closed; the minute is no longer served from the '
                           'stream', product, trade_id,
                           time.strftime('%H:%M', time.gmtime(minute)),
                           received - minute - MINUTE)
        key = _order_key(at, trade_id)
        if acc is None:
            book[minute] = _Minute(open_key=key, open=price, close_key=key,
                                   close=price, high=price, low=price,
                                   volume=size, ids={trade_id})
            self._prune(received)
        else:
            acc.add(key, price, size, trade_id)
        return True

    def _prune(self, now: float) -> None:
        floor = minute_of(now) - self.retain_minutes * MINUTE
        for book in self._minutes.values():
            for minute in [m for m in book if m < floor]:
                del book[minute]
        for marks in self._tainted.values():
            for minute in [m for m in marks if m < floor]:
                marks.discard(minute)
        self._coverage = [c for c in self._coverage
                          if c[1] is None or c[1] >= floor]

    # -- reading -----------------------------------------------------------

    def final(self, minute: int, *, now: Optional[float] = None) -> bool:
        now = self._now() if now is None else float(now)
        return now >= minute + MINUTE + self.grace_seconds

    def reliable(self, product: str, minute: int, *,
                 now: Optional[float] = None) -> bool:
        """Final, fully inside one coverage interval, and untainted."""
        now = self._now() if now is None else float(now)
        minute = int(minute)
        if not self.final(minute, now=now):
            return False
        if minute in self._tainted.get(product, ()):
            return False
        # Older than what is retained: its bar may have been pruned, and an
        # absent bar must never read as a quiet minute.
        if minute < minute_of(now) - (self.retain_minutes - 1) * MINUTE:
            return False
        end = minute + MINUTE + self.grace_seconds
        for start, stop in self._coverage:
            if start + self.clock_margin_seconds > minute:
                continue
            if stop is None or end <= stop:
                return True
        return False

    def bar(self, product: str, minute: int) -> Optional[dict]:
        acc = self._minutes.get(product, {}).get(int(minute))
        if acc is None:
            return None
        return {'minute': int(minute), 'open': acc.open, 'high': acc.high,
                'low': acc.low, 'close': acc.close, 'volume': acc.volume,
                'trades': len(acc.ids)}

    def span(self, product: str, first: int, last: int, *,
             now: Optional[float] = None) -> Optional[list[dict]]:
        """Bars for every minute in [first, last], or None unless ALL are reliable.

        Minutes without trades are absent from the list, exactly as REST omits
        them — which is only a claim because every minute was covered.
        """
        now = self._now() if now is None else float(now)
        out: list[dict] = []
        for minute in range(int(first), int(last) + 1, MINUTE):
            if not self.reliable(product, minute, now=now):
                return None
            bar = self.bar(product, minute)
            if bar is not None:
                out.append(bar)
        return out

    def products(self) -> list[str]:
        return sorted(self._minutes)
