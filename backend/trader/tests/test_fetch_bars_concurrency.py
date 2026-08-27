"""The three symbol fetches are independent, so they run concurrently.

Measured at 3.17s in series — three round trips to one host, run one after
another for no reason. It was the largest single cost in the live cycle.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from core.config import DEFAULT_CONFIG
from scripts import live


class _Client:
    """Each call sleeps; in series three take 3x as long as in parallel."""

    def __init__(self, delay=0.15, fail=()):
        self.delay, self.fail, self.calls = delay, set(fail), []

    async def get_candles_range(self, symbol, timeframe, start, end):
        self.calls.append(symbol)
        await asyncio.sleep(self.delay)
        if symbol in self.fail:
            raise RuntimeError('venue said no')
        return [type('B', (), {'event_time': start, 'open': 1.0, 'high': 1.0,
                               'low': 1.0, 'close': 1.0, 'volume': 1.0})()]

    async def close(self):
        pass


@pytest.fixture
def patched(monkeypatch):
    def _install(client):
        monkeypatch.setattr(live, 'CoinbaseRESTClient', lambda **kw: client)
        return client
    return _install


@pytest.mark.asyncio
async def test_the_symbols_are_fetched_concurrently(patched):
    client = patched(_Client(delay=0.15))
    started = time.perf_counter()
    out = await live.fetch_bars(DEFAULT_CONFIG)
    elapsed = time.perf_counter() - started

    n = len(DEFAULT_CONFIG.symbols)
    assert len(out) == n and len(client.calls) == n
    assert elapsed < 0.15 * n * 0.7, (
        f'{elapsed:.2f}s for {n} symbols at 0.15s each — still serial')


@pytest.mark.asyncio
async def test_one_symbol_failing_does_not_lose_the_others(patched):
    """A partial outage must not become a total one."""
    failed = DEFAULT_CONFIG.symbols[0]
    patched(_Client(delay=0.01, fail=[failed]))
    out = await live.fetch_bars(DEFAULT_CONFIG)
    assert failed not in out
    assert len(out) == len(DEFAULT_CONFIG.symbols) - 1


@pytest.mark.asyncio
async def test_every_symbol_failing_returns_empty_rather_than_raising():
    """The cycle checks `if not bars` and abstains; it must not see an exception."""
    class AllFail(_Client):
        async def get_candles_range(self, symbol, timeframe, start, end):
            raise RuntimeError('venue down')

    import scripts.live as mod
    original = mod.CoinbaseRESTClient
    mod.CoinbaseRESTClient = lambda **kw: AllFail()
    try:
        assert await live.fetch_bars(DEFAULT_CONFIG) == {}
    finally:
        mod.CoinbaseRESTClient = original
