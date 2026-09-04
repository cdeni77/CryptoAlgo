"""A decision is priced off the LIVE book, never `/markets`' cached touch.

**This is what stopped the live loop filling on 2026-09-04.** `fetch_quotes`
priced every decision off `kalshi.quote(ticker)`, which reads `/markets` and its
`yes_bid_dollars`/`yes_ask_dollars` fields. Those are cached. Measured by
sampling both endpoints for the same market two seconds apart, `/markets`
returned an identical `0.9120/0.9160` on ETH for five consecutive samples across
ten seconds while `/orderbook` moved 0.934 -> 0.960 -> 0.951 -> 0.950. Across 28
paired samples the two disagreed on 96.4%, mean 0.66c, ask understated by up to
6.3c.

The captured kill: the loop believed the ask was 0.74 and sent a 74c bid; the
real book was 0.75/0.76, so our limit was a cent below the actual BID and could
never cross. Ten of ten orders after the restart died this way.

Two consequences, and the second is the worse one:

  * the order cannot fill, and
  * `market_probability` — the midpoint of this touch — is the `init_score` of a
    market-initialised artifact, so the model spends its capacity correcting a
    price that is several cents stale. That is a forecast error, not a fill one.

It also closes a live/training mismatch: the backtest reads `venue_depth`, built
from RECORDED LADDERS, so `/markets` was a price source live used and training
never saw.
"""
from __future__ import annotations

import asyncio

import pytest

from data_collection.kalshi_client import KalshiClient, KalshiError, parse_orderbook


# The venue's real shapes, recorded from the live account 2026-09-04. `/markets`
# is deliberately STALE here — two cents under the book — which is the whole
# condition under test.
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


def test_the_touch_comes_from_the_book_not_the_cached_quote():
    c = FakeClient()
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    # /markets said 0.73/0.74. The book says 0.75 bid, and the YES ask is
    # 1 - best_no_bid = 1 - 0.24 = 0.76.
    assert q.yes_bid == pytest.approx(0.75)
    assert q.yes_ask == pytest.approx(0.76)
    assert q.ask_for('up') == pytest.approx(0.76)


def test_the_no_side_is_converted_and_not_relabelled():
    """Buying NO costs `1 - best_yes_bid`. Getting this backwards prices the
    wrong side of the book, which is the same class of error as inverting the
    trade outright."""
    c = FakeClient()
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.no_bid == pytest.approx(0.24)
    assert q.no_ask == pytest.approx(0.25)     # 1 - 0.75
    assert q.ask_for('down') == pytest.approx(0.25)


def test_the_captured_kill_would_now_cross():
    """The regression test for the actual failure.

    Our limit was 0.74 against a real 0.76 ask. With the live touch the decision
    prices at 0.76, so `order_limit_price` starts from a number that can fill.
    """
    c = FakeClient()
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.ask_for('up') > 0.74, 'the old cached ask was 0.74 and never filled'


def test_the_sizes_come_from_the_stack_that_gets_crossed():
    """`yes_ask_size` is what rests on the NO bid — a YES buy crosses NO bids.
    `decide()` caps the stake on this, so taking it from the wrong stack sizes
    against liquidity that is not there."""
    c = FakeClient()
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.yes_bid_size == pytest.approx(64.0)
    assert q.yes_ask_size == pytest.approx(32.82)


def test_metadata_still_comes_from_markets():
    """Only the touch is replaced. Status, shard and close time have no
    orderbook equivalent, and `exchange_index` in particular must survive — a
    wrong shard is a 404 that names the market rather than the mismatch."""
    c = FakeClient()
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.exchange_index == 2
    assert q.status == 'active'
    assert q.volume == 900


def test_an_empty_book_is_trusted_rather_than_papered_over():
    """A settled or halted market really has no book, and reporting the cached
    touch instead would reintroduce exactly the bug being fixed."""
    c = FakeClient(book={'orderbook': {'orderbook_fp': {}}})
    q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.yes_bid is None and q.yes_ask is None
    assert q.ask_for('up') is None


def test_a_FAILED_book_request_falls_back_and_says_so(caplog):
    """A transport failure is not an empty book. Falling back to the cached
    touch keeps the loop alive, but it must be logged, because a decision priced
    that way is expected not to fill."""
    c = FakeClient(raise_on_book=KalshiError('503'))
    with caplog.at_level('WARNING'):
        q = asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert q.yes_ask == pytest.approx(0.74), 'the cached touch, as a fallback'
    assert 'CACHED' in caplog.text


def test_it_reads_the_book_exactly_once_per_quote():
    """One extra round trip per symbol per cycle, not one per feature. The
    cycle's own budget is the reason the recorders share a process."""
    c = FakeClient()
    asyncio.run(c.quote_with_live_book('KXSOL15M-26SEP041145-45'))
    assert sum(1 for p in c.calls if p.endswith('/orderbook')) == 1


def test_parse_orderbook_ignores_the_dust_level():
    """Both stacks carry a 0.0010 level with ~100k resting. Reading THAT as the
    touch would price every trade at a tenth of a cent."""
    book = parse_orderbook(LIVE_BOOK)
    assert book['yes_bid'] == pytest.approx(0.75)
    assert book['no_bid'] == pytest.approx(0.24)
