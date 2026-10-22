"""One-minute index bars folded from the live `cfbenchmarks_value` stream.

The bar is keyed on its START minute, the convention `core/windows.py` reads:
the strike is the mean over `[t0 - 1min, t0)`, so a bar stamped at its end
would shift every window by a minute and nothing would raise.

**`mean_price` is the venue's own number, not ours.** A frame carries
`avg_60s_data` covering `[end - 60s, end)`, so the frame whose window ends at
`m + 60` describes exactly minute `m`. That is the quantity Kalshi settles on
-- better than the `(O+H+L+C)/4` proxy `core/windows.bar_mean` falls back to,
and better than a mean we compute ourselves from the same prints, because it
is the number rather than an estimate of it.

A partial average is refused, not rounded down to a worse estimate: it
averages a different interval. `IndexValue.settles` is the guard, and it
matters on every reconnect. Where no full average arrives for a minute,
`mean_price` falls back to the mean of the prints we saw, and `mean_source`
says which happened -- a bar that cannot say where its mean came from is the
same defect as a quote that cannot say whether it was a fill or a backtest.

An index has no volume. `volume` is left for the caller to carry across from
spot, as `scripts/backfill_index` does, because a null column drops every row
under `--complete-cases`.
"""

from __future__ import annotations

import math
from typing import Iterable, Optional

MINUTE = 60.0

#: A minute of one-second prints. Fewer means the stream was not connected for
#: all of it, and the bar's own OHLC is then built from a partial interval.
FULL_MINUTE_PRINTS = 55


def minute_of(at: float) -> int:
    """The START second of the minute containing `at`."""
    return int(at) - int(at) % int(MINUTE)


class IndexBars:
    """Fold index prints into minute bars. Not thread-safe; one event loop."""

    def __init__(self, retain_minutes: int = 240) -> None:
        self._prints: dict[tuple[str, int], list[float]] = {}
        #: The venue's full-window average for a minute, where one arrived.
        self._means: dict[tuple[str, int], float] = {}
        self._retain = int(retain_minutes)

    def add(self, value) -> None:
        """Record one `IndexValue`."""
        key = (value.symbol, minute_of(value.at))
        self._prints.setdefault(key, []).append(float(value.value))
        # The average ending at `m + 60` is exactly minute `m`. Refused while
        # the window is partial -- see the module docstring.
        if value.settles and value.avg_window_end is not None:
            end = float(value.avg_window_end)
            start = int(end) - int(MINUTE)
            if start % int(MINUTE) == 0:
                self._means[(value.symbol, start)] = float(value.avg_60s)
        self._evict(value.symbol, minute_of(value.at))

    def _evict(self, symbol: str, newest: int) -> None:
        floor = newest - self._retain * int(MINUTE)
        for store in (self._prints, self._means):
            for key in [k for k in store
                        if k[0] == symbol and k[1] < floor]:
                del store[key]

    def complete(self, symbol: str, *, now: float) -> list[int]:
        """Minutes that have ENDED, oldest first.

        A minute still in progress has no bar: its close is not yet its close.
        """
        cutoff = minute_of(now)
        return sorted(m for (s, m) in self._prints if s == symbol and m < cutoff)

    def bar(self, symbol: str, minute: int) -> Optional[dict]:
        """The bar for one ended minute, or None if nothing was seen."""
        prints = self._prints.get((symbol, minute))
        if not prints:
            return None
        mean = self._means.get((symbol, minute))
        return {
            'symbol': symbol,
            'minute': minute,
            'open': prints[0], 'close': prints[-1],
            'high': max(prints), 'low': min(prints),
            # The venue's own average where we have it, ours otherwise, and
            # the row says which.
            'mean_price': mean if mean is not None else sum(prints) / len(prints),
            'mean_source': 'venue_avg_60s' if mean is not None else 'prints',
            'n_prints': len(prints),
            'complete': len(prints) >= FULL_MINUTE_PRINTS,
        }

    def since(self, symbol: str, *, after: int = -1,
              now: float) -> list[dict]:
        """Ended minutes after `after`, oldest first. **Non-destructive.**

        The writer and the decision path read the same fold: one persists
        minutes it has not yet written, the other splices the newest minutes
        onto the stored history. A `drain` that forgot what it yielded would
        delete the bar the next decision needs -- the fold is sized by
        `retain_minutes`, and eviction is the only thing that should remove
        from it.
        """
        out = []
        for minute in self.complete(symbol, now=now):
            if minute <= after:
                continue
            bar = self.bar(symbol, minute)
            if bar is not None:
                out.append(bar)
        return out

    def drain(self, symbol: str, *, now: float) -> Iterable[dict]:
        """Yield and forget every ended minute's bar.

        Only for a standalone run that nothing else reads from. Inside
        `run_live` use `since`, or the decision path loses the minute.
        """
        for minute in self.complete(symbol, now=now):
            bar = self.bar(symbol, minute)
            self._prints.pop((symbol, minute), None)
            self._means.pop((symbol, minute), None)
            if bar is not None:
                yield bar

    def depth(self) -> int:
        """How many minutes are held, across every symbol."""
        return len(self._prints)


#: What the index does not have and spot does. Verified on both transports
#: 2026-10-22: the WS frame carries `{type, time, id, value}` and the REST
#: point `{time, value}` -- no volume field exists on either.
ACTIVITY_COLUMNS = ('volume', 'quote_volume', 'trade_count')


def carry_activity(index, spot):
    """Put spot's activity columns onto index bars, matched on `event_time`.

    **One definition, because training and live must do this identically.**
    An index has no volume, so after a migration the bars are a hybrid: BRTI
    price with Coinbase activity. `signed_volume_15` then takes its sign from
    the index's returns and its magnitude from spot, and `vwap_gap_15`
    multiplies one source's price by the other's volume. Those features are
    defensible only while both paths build them the same way -- two
    implementations would give them different meanings in training and
    serving, and nothing would raise.

    A left join: an index minute with no spot bar keeps the index price and a
    null volume, rather than being dropped. Deciding what to do about the null
    belongs to `--complete-cases`, not here.
    """
    import pandas as pd

    if index is None or not len(index):
        return index
    if spot is None or not len(spot):
        return index
    have = [c for c in ACTIVITY_COLUMNS if c in spot.columns]
    if not have:
        return index
    right = spot[['event_time', *have]].copy()
    right['event_time'] = pd.to_datetime(right['event_time'], utc=True)
    out = index.copy()
    out['event_time'] = pd.to_datetime(out['event_time'], utc=True)
    out = out.drop(columns=[c for c in have if c in out.columns])
    return out.merge(right, on='event_time', how='left')


def rows_for_store(bars: Iterable[dict], *, available_time,
                   venue: str = 'cfbenchmarks') -> list[dict]:
    """Bars as `minute_bars` rows.

    `available_time` is the capture instant. The bar for minute `m` is only
    knowable once `m` has ended, so a row claiming to be knowable at `m`
    itself would be a lookahead the store is built to refuse.
    """
    import pandas as pd

    out = []
    for bar in bars:
        event_time = pd.Timestamp(bar['minute'], unit='s', tz='UTC')
        out.append({
            'venue': venue, 'symbol': bar['symbol'],
            'event_time': event_time,
            'available_time': max(pd.Timestamp(available_time), event_time),
            'quality': 'valid' if bar['complete'] else 'suspicious',
            'open': bar['open'], 'high': bar['high'],
            'low': bar['low'], 'close': bar['close'],
            'mean_price': bar['mean_price'],
            'volume': math.nan, 'quote_volume': math.nan,
            'trade_count': math.nan,
        })
    return out
