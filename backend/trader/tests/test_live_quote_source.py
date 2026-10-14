"""Tests that a decision is priced from `/orderbook`, never the cached `/markets` touch.

The cached touch can be cents stale, so a limit set from it may never cross.
"""
from __future__ import annotations

import asyncio

import pytest

from data_collection.kalshi_client import (
    KalshiClient, KalshiError, market_meta, parse_orderbook,
)


# The venue's real shapes. `/markets` is stale here on purpose, two cents under
# the book; that is the condition under test.
STALE_MARKET = {'market': {
    'ticker': 'KXSOL15M-26SEP041145-45',
    'yes_bid_dollars': '0.7300', 'yes_ask_dollars': '0.7400',
    'no_bid_dollars': '0.2600', 'no_ask_dollars': '0.2700',
    'yes_bid_size_fp': '11.00', 'yes_ask_size_fp': '12.00',
    'status': 'active', 'exchange_index': 2, 'volume': 900,
    'open_interest': 100, 'last_price_dollars': '0.7400',
}}
LIVE_BOOK = {'orderbook': {'orderbook_fp': {
    'yes_dollars': [['0.0010', '99999.00'], ['0.7400', '10.00'],
                    ['0.7500', '64.00']],
    'no_dollars': [['0.0010', '99999.00'], ['0.2300', '9.00'],
                   ['0.2400', '32.82']],
}}}


class FakeClient(KalshiClient):
    """Real parsing and merging, fake transport."""

    def __init__(self, *, book=LIVE_BOOK, raise_on_book=None):
        self._book, self._raise = book, raise_on_book
        self.calls: list[str] = []
        self.live = False

    async def _request(self, method, path, **kw):
        self.calls.append(path)
        if path.endswith('/orderbook'):
            if self._raise is not None:
                raise self._raise
            return self._book
        return STALE_MARKET


def _live_quote(client):
    """The live quote path for one symbol: market metadata with the touch blanked,
    priced by one `/orderbook` read.
    """
    return client.book_quote(market_meta(STALE_MARKET['market']))

def test_the_touch_comes_from_the_book_not_the_cached_quote():
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    # /markets said 0.73/0.74. The book has a 0.75 bid, and the YES ask is
    # 1 - best_no_bid = 1 - 0.24 = 0.76.
    assert q.yes_bid == pytest.approx(0.75)
    assert q.yes_ask == pytest.approx(0.76)
    assert q.ask_for('up') == pytest.approx(0.76)


def test_the_no_side_is_converted_and_not_relabelled():
    """Buying NO costs `1 - best_yes_bid`."""
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    assert q.no_bid == pytest.approx(0.24)
    assert q.no_ask == pytest.approx(0.25)  # 1 - 0.75
    assert q.ask_for('down') == pytest.approx(0.25)


def test_the_captured_kill_would_now_cross():
    """With the live touch the decision prices at 0.76, not the stale 0.74, so the limit
    can fill.
    """
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    assert q.ask_for('up') > 0.74, 'the old cached ask was 0.74 and never filled'


def test_the_sizes_come_from_the_stack_that_gets_crossed():
    """`yes_ask_size` is the size resting on the NO bid, which a YES buy crosses."""
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    assert q.yes_bid_size == pytest.approx(64.0)
    assert q.yes_ask_size == pytest.approx(32.82)


def test_metadata_still_comes_from_markets():
    """Status, shard and close time still come from `/markets`; only the touch is replaced."""
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    assert q.exchange_index == 2
    assert q.status == 'active'
    assert q.volume == 900


def test_an_empty_book_is_trusted_rather_than_papered_over():
    """An empty book is reported as empty; the cached touch is not substituted."""
    c = FakeClient(book={'orderbook': {'orderbook_fp': {}}})
    q = asyncio.run(_live_quote(c))
    assert q.yes_bid is None and q.yes_ask is None
    assert q.ask_for('up') is None


def test_a_FAILED_book_request_is_no_quote_not_the_cached_touch():
    """A failed book request raises; the cached touch is never used as a fallback."""
    c = FakeClient(raise_on_book=KalshiError('503'))
    with pytest.raises(KalshiError):
        asyncio.run(_live_quote(c))


def test_a_timed_out_book_request_is_no_quote_either():
    c = FakeClient(raise_on_book=asyncio.TimeoutError())
    with pytest.raises(KalshiError):
        asyncio.run(_live_quote(c))


def test_the_rest_quote_says_where_it_came_from_and_carries_its_ladder():
    c = FakeClient()
    q = asyncio.run(_live_quote(c))
    assert q.book_source == 'rest' and q.book_time is not None
    assert (0.75, 64.0) in q.yes_levels and (0.24, 32.82) in q.no_levels


def test_it_reads_the_book_exactly_once_per_quote():
    c = FakeClient()
    asyncio.run(_live_quote(c))
    assert sum(1 for p in c.calls if p.endswith('/orderbook')) == 1


def test_parse_orderbook_ignores_the_dust_level():
    """The 0.0010 dust level with large size is not read as the touch."""
    book = parse_orderbook(LIVE_BOOK)
    assert book['yes_bid'] == pytest.approx(0.75)
    assert book['no_bid'] == pytest.approx(0.24)
