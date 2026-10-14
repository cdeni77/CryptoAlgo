"""Tests that a decision is priced from the stream book when it is fresh.

A REST `/orderbook` read is the fallback, run concurrently for every symbol that needs
one. A failed read gives no quote; the `/markets` touch is never used.
"""
from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest

from core.stream_book import BookCache
from data_collection.kalshi_client import KalshiError, market_meta
from data_collection.stream.base import BookEvent
from livetrade import quotes as live_quotes
from scripts import live

SETTLE = pd.Timestamp('2026-10-14 12:15', tz='UTC')
TICKERS = {'BTC-USD': 'KXBTC15M-26OCT140815-15', 'ETH-USD': 'KXETH15M-26OCT140815-15',
           'SOL-USD': 'KXSOL15M-26OCT140815-15'}


def _market(ticker):
    return {'ticker': ticker, 'status': 'active', 'exchange_index': 2,
            'close_time': '2026-10-14T12:15:00Z', 'floor_strike': 100.0,
            # The cached touch, wrong on purpose: it must never price.
            'yes_bid_dollars': '0.1000', 'yes_ask_dollars': '0.1100'}


class FakeKalshi:
    def __init__(self, *, fail_book=(), delay=0.0):
        self.resolved, self.books, self.fail_book, self.delay = [], [], set(fail_book), delay

    async def resolve_window_market(self, series, settle_time):
        self.resolved.append(series)
        symbol = next(s for s, t in TICKERS.items() if t.startswith(series))
        return _market(TICKERS[symbol])

    async def market(self, ticker):
        return _market(ticker)

    async def book_quote(self, meta):
        from data_collection.kalshi_client import quote_from_levels
        self.books.append(meta.ticker)
        await asyncio.sleep(self.delay)
        if meta.ticker in self.fail_book:
            raise KalshiError('orderbook -> 503', status=503)
        return quote_from_levels(meta, [(0.55, 10.0)], [(0.43, 20.0)],
                                 source='rest', at=time.time())


def _cache(clock, tickers):
    cache = BookCache(now=lambda: clock['now'])
    for i, ticker in enumerate(tickers):
        cache.apply(BookEvent(venue='kalshi', market_ticker=ticker, kind='snapshot',
                              received=clock['now'], seq=i + 1,
                              yes=[(0.60, 5.0), (0.58, 9.0)], no=[(0.38, 7.0)]))
    return cache


@pytest.fixture(autouse=True)
def _fresh_markets():
    live_quotes._MARKETS.clear()
    yield
    live_quotes._MARKETS.clear()


class TestBookCacheFreshness:
    def test_a_live_connection_vouches_for_a_quiet_book(self):
        clock = {'now': 1000.0}
        cache = _cache(clock, ['A', 'B'])
        clock['now'] = 1004.0
        # Another market's frame keeps the connection live.
        cache.apply(BookEvent(venue='kalshi', market_ticker='B', kind='delta',
                              received=1004.0, seq=3, yes=[(0.60, 1.0)], no=[],
                              absolute=False))
        assert cache.current('A', 2.0) is not None

    def test_a_silent_connection_does_not(self):
        clock = {'now': 1000.0}
        cache = _cache(clock, ['A'])
        clock['now'] = 1002.5
        assert cache.current('A', 2.0) is None

    def test_a_gapped_book_does_not(self):
        clock = {'now': 1000.0}
        cache = _cache(clock, ['A', 'B'])
        cache.apply(BookEvent(venue='kalshi', market_ticker='B', kind='delta',
                              received=1000.0, seq=9, yes=[(0.6, 1.0)], no=[],
                              absolute=False))
        assert cache.current('A', 2.0) is None

    def test_a_dropped_connection_needs_a_new_snapshot(self):
        clock = {'now': 1000.0}
        cache = _cache(clock, ['A', 'B'])
        cache.invalidate('kalshi')
        cache.reset_sequence('kalshi')
        cache.apply(BookEvent(venue='kalshi', market_ticker='B', kind='snapshot',
                              received=1000.5, seq=1, yes=[(0.6, 1.0)], no=[(0.3, 1.0)]))
        clock['now'] = 1000.6
        assert cache.current('A', 2.0) is None, 'A missed every delta since the drop'
        assert cache.current('B', 2.0) is not None
        assert not cache.any_gapped(), 'awaiting a snapshot must not force a rebuild'


class TestFetchQuotes:
    def test_the_stream_prices_when_fresh_and_rest_is_not_called(self):
        clock = {'now': time.time()}
        cache = _cache(clock, list(TICKERS.values()))
        kalshi = FakeKalshi()
        quotes = asyncio.run(live.fetch_quotes(kalshi, list(TICKERS), SETTLE,
                                               cache=cache, max_age_seconds=2.0))
        assert set(quotes) == set(TICKERS)
        assert kalshi.books == []
        quote, ticker = quotes['BTC-USD']
        assert quote.book_source == 'stream' and ticker == TICKERS['BTC-USD']
        assert quote.yes_bid == pytest.approx(0.60)
        assert quote.yes_ask == pytest.approx(0.62)  # 1 - best NO bid
        assert quote.exchange_index == 2 and quote.floor_strike == 100.0

    def test_a_stale_stream_falls_back_to_rest_concurrently(self):
        clock = {'now': time.time() - 30}
        cache = _cache(clock, list(TICKERS.values()))
        clock['now'] = time.time()
        kalshi = FakeKalshi(delay=0.2)
        started = time.perf_counter()
        quotes = asyncio.run(live.fetch_quotes(kalshi, list(TICKERS), SETTLE,
                                               cache=cache, max_age_seconds=2.0))
        assert time.perf_counter() - started < 0.45, 'the REST reads ran in series'
        assert sorted(kalshi.books) == sorted(TICKERS.values())
        assert {q.book_source for q, _ in quotes.values()} == {'rest'}

    def test_a_failed_book_read_is_no_quote_not_the_cached_touch(self):
        kalshi = FakeKalshi(fail_book=[TICKERS['ETH-USD']])
        quotes = asyncio.run(live.fetch_quotes(kalshi, list(TICKERS), SETTLE,
                                               cache=None))
        assert 'ETH-USD' not in quotes
        assert set(quotes) == {'BTC-USD', 'SOL-USD'}

    def test_the_market_is_resolved_once_per_window(self):
        kalshi = FakeKalshi()
        for _ in range(3):
            asyncio.run(live.fetch_quotes(kalshi, list(TICKERS), SETTLE, cache=None))
        assert len(kalshi.resolved) == 3, 'one resolution per symbol, not per read'
        asyncio.run(live.fetch_quotes(kalshi, list(TICKERS),
                                      SETTLE + pd.Timedelta(minutes=15), cache=None))
        assert len(kalshi.resolved) == 6, 'a new window resolves again'

    def test_market_metadata_never_carries_the_cached_touch(self):
        meta = market_meta(_market('X'))
        assert meta.yes_bid is None and meta.yes_ask is None
        assert not meta.tradeable()
