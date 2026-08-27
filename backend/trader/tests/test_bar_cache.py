"""The 1,500-minute window is kept between cycles; only the tail is re-asked.

Re-downloading twenty-five hours every sixty seconds to learn one new minute was
measured at 3.2s — the largest single cost in the live cycle, and the reason
fetching the three symbols concurrently barely helped: `get_candles_range` pages
at 300 candles, so a full window is ~6 sequential round trips per symbol.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from scripts import live


class _Bar:
    def __init__(self, when, close):
        self.event_time = when
        self.open = self.high = self.low = self.close = close
        self.volume = 1.0


class _Client:
    """Serves one bar a minute, and records the span it was asked for."""

    def __init__(self, close_price=1.0):
        self.spans: list[tuple] = []
        self.close_price = close_price

    async def get_candles_range(self, symbol, timeframe, start, end):
        self.spans.append((symbol, start, end))
        out, when = [], start
        while when <= end:
            out.append(_Bar(when, self.close_price))
            when = when + timedelta(minutes=1)
        return out

    async def close(self):
        pass


@pytest.fixture(autouse=True)
def _clear_cache():
    live._BAR_CACHE.clear()
    yield
    live._BAR_CACHE.clear()


@pytest.fixture
def client(monkeypatch):
    c = _Client()
    monkeypatch.setattr(live, 'CoinbaseRESTClient', lambda **kw: c)
    return c


@pytest.mark.asyncio
async def test_the_first_cycle_fetches_the_whole_window(client):
    out = await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    symbol = DEFAULT_CONFIG.symbols[0]
    _, start, end = next(s for s in client.spans if s[0] == symbol)
    assert (end - start) >= timedelta(minutes=99)
    assert len(out[symbol]) >= 100


@pytest.mark.asyncio
async def test_the_second_cycle_asks_only_for_the_tail(client):
    await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    client.spans.clear()
    await live.fetch_bars(DEFAULT_CONFIG, minutes=100)

    symbol = DEFAULT_CONFIG.symbols[0]
    _, start, end = next(s for s in client.spans if s[0] == symbol)
    assert (end - start) <= timedelta(minutes=live.BAR_REFETCH_MINUTES + 2), (
        'the second cycle re-downloaded the whole window'
    )


@pytest.mark.asyncio
async def test_the_window_still_spans_the_full_period_after_an_incremental(client):
    first = await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    symbol = DEFAULT_CONFIG.symbols[0]
    n_first = len(first[symbol])
    second = await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    assert len(second[symbol]) >= n_first - 2, 'the window shrank'
    times = second[symbol]['event_time']
    assert times.is_monotonic_increasing and not times.duplicated().any()


@pytest.mark.asyncio
async def test_a_refetched_minute_takes_the_newer_value(client):
    """The newest cached bar is the minute still forming; the later read is the
    finished one. Keeping the cached copy would freeze a partial candle into the
    window, and every log_rv_* feature is built off these closes."""
    await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    symbol = DEFAULT_CONFIG.symbols[0]
    stale_last = live._BAR_CACHE[symbol]['event_time'].iloc[-1]

    client.close_price = 999.0
    out = await live.fetch_bars(DEFAULT_CONFIG, minutes=100)
    row = out[symbol].loc[out[symbol]['event_time'] == stale_last]
    assert float(row['close'].iloc[0]) == 999.0, 'the partial candle was frozen in'


@pytest.mark.asyncio
async def test_a_cache_too_short_for_the_window_falls_back_to_a_full_fetch(client):
    await live.fetch_bars(DEFAULT_CONFIG, minutes=50)
    client.spans.clear()
    # A wider window than the cache covers: an incremental tail would leave a
    # hole in the middle that no feature would notice.
    await live.fetch_bars(DEFAULT_CONFIG, minutes=400)
    symbol = DEFAULT_CONFIG.symbols[0]
    _, start, end = next(s for s in client.spans if s[0] == symbol)
    assert (end - start) >= timedelta(minutes=399), 'left a hole in the window'


def test_merge_prefers_the_fresh_copy_and_trims_to_the_floor():
    base = pd.Timestamp('2026-08-27 12:00', tz='UTC')
    cached = pd.DataFrame({'event_time': [base, base + pd.Timedelta(minutes=1)],
                           'close': [1.0, 2.0]})
    fresh = pd.DataFrame({'event_time': [base + pd.Timedelta(minutes=1),
                                         base + pd.Timedelta(minutes=2)],
                          'close': [99.0, 3.0]})
    merged = live._merge_bars(cached, fresh, floor=base)
    assert list(merged['close']) == [1.0, 99.0, 3.0]

    trimmed = live._merge_bars(cached, fresh, floor=base + pd.Timedelta(minutes=1))
    assert list(trimmed['close']) == [99.0, 3.0]
