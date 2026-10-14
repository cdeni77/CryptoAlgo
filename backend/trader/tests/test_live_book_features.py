"""Tests that live builds the same book features the backtest scored.

A missing feature silently scores NaN with LightGBM's learned default, which is a
different model.
"""

from __future__ import annotations

import types
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from core.config import Config

from scripts.live import book_feature_row


class _Quote:
    def __init__(self, bid=0.44, ask=0.46, bid_size=250.0, ask_size=150.0):
        self.yes_bid, self.yes_ask = bid, ask
        self.yes_bid_size, self.yes_ask_size = bid_size, ask_size


# YES side: (price, size) bids; NO side: (price, size) in NO terms.
_YES = [(0.44, 250.0), (0.43, 150.0), (0.40, 500.0)]
_NO = [(0.54, 150.0), (0.53, 150.0), (0.50, 300.0)]


def test_the_price_features_come_from_the_touch():
    row = book_feature_row(_Quote(), _YES, _NO, baseline_probability=0.40)
    assert row['market_prob'] == pytest.approx(0.45)
    assert row['market_minus_baseline'] == pytest.approx(0.05)
    assert row['spread'] == pytest.approx(0.02)


def test_imbalance_uses_the_resting_size_at_the_touch():
    row = book_feature_row(_Quote(bid_size=300.0, ask_size=100.0), _YES, _NO,
                           baseline_probability=0.40)
    assert row['imbalance_touch'] == pytest.approx(0.5)


def test_depth_ratio_needs_the_totals_that_live_was_not_computing():
    """`depth_ratio` uses the bid and ask totals, which live now computes."""
    row = book_feature_row(_Quote(), _YES, _NO, baseline_probability=0.40)
    assert np.isfinite(row['depth_ratio'])
    # Bids total 900, asks total 600 -> log(900/600).
    assert row['depth_ratio'] == pytest.approx(np.log(900.0 / 600.0), rel=1e-6)


def test_a_one_sided_book_yields_no_probability():
    """A one-sided book gives no probability, as in the backtest."""
    row = book_feature_row(_Quote(ask=None), _YES, _NO, baseline_probability=0.40)
    assert pd.isna(row['market_prob'])
    assert pd.isna(row['market_minus_baseline'])


def test_an_empty_ladder_does_not_raise():
    """An empty ladder does not raise; top-of-book features are still computed."""
    row = book_feature_row(_Quote(), [], [], baseline_probability=0.40)
    assert row['market_prob'] == pytest.approx(0.45)
    assert pd.isna(row['depth_ratio'])


def test_every_feature_the_model_expects_is_present():
    """Every market_state and market_price column is present, since an absent one scores as NaN."""
    from core.book_features import MARKET_PRICE, MARKET_STATE
    row = book_feature_row(_Quote(), _YES, _NO, baseline_probability=0.40)
    for column in tuple(MARKET_STATE) + tuple(MARKET_PRICE):
        assert column in row, column


# --- the ladder from the stream cache ----------------------------------------
#
# The stream recorder and the trading loop run in one process, so
# `record_stream.CACHE` already holds every subscribed ladder. It is fresher than
# a REST fetch and costs no request. A ladder the socket stopped updating is
# never used.

class _Ladder:
    def __init__(self, yes, no, age=0.05, stale=False):
        self.yes, self.no = yes, no
        self.age_seconds, self.stale = age, stale


class _Cache:
    def __init__(self, ladder=None, gapped=False):
        self._ladder, self._gapped = ladder, gapped

    def ladder(self, ticker):
        return self._ladder

    def gapped(self, ticker):
        return self._gapped


def test_the_ladder_comes_from_the_stream_cache_when_it_is_fresh():
    from scripts.live import ladder_from_cache
    cache = _Cache(_Ladder(_YES, _NO))
    yes, no = ladder_from_cache(cache, 'KXBTC15M-1')
    assert yes == _YES and no == _NO


def test_a_stale_ladder_is_refused():
    """A stale ladder is refused: silence at this frame rate means the transport is unhealthy."""
    from scripts.live import ladder_from_cache
    cache = _Cache(_Ladder(_YES, _NO, age=30.0, stale=True))
    assert ladder_from_cache(cache, 'KXBTC15M-1') == ([], [])


def test_a_gapped_book_is_refused():
    """A gapped book is refused: `seq` is global per subscription, so a gap may drop
    levels anywhere.
    """
    from scripts.live import ladder_from_cache
    cache = _Cache(_Ladder(_YES, _NO), gapped=True)
    assert ladder_from_cache(cache, 'KXBTC15M-1') == ([], [])


def test_no_cache_at_all_is_not_an_error():
    """No cache is not an error: `scripts.live` can run without the stream task."""
    from scripts.live import ladder_from_cache
    assert ladder_from_cache(None, 'KXBTC15M-1') == ([], [])
    assert ladder_from_cache(_Cache(None), 'KXBTC15M-1') == ([], [])


def test_the_full_feature_set_is_computable_from_a_cached_ladder():
    from scripts.live import book_feature_row, ladder_from_cache
    yes, no = ladder_from_cache(_Cache(_Ladder(_YES, _NO)), 'KXBTC15M-1')
    row = book_feature_row(_Quote(), yes, no, baseline_probability=0.40)
    for column in ('market_prob', 'market_minus_baseline', 'spread',
                   'imbalance_touch', 'imbalance_5c', 'depth_ratio',
                   'book_convexity'):
        assert not pd.isna(row[column]), column


@pytest.mark.asyncio
async def test_record_touch_persists_the_resting_totals_it_trades_on():
    """`_record_touch` writes every depth column that live trades on, including the
    totals.

    The next retrain uses these rows, so a column used for trading must be recorded.
    """
    import pandas as pd
    from scripts import live
    from core.quotes import DEPTH_MAP

    captured = []

    class _Store:
        def write(self, table, frame, **kw):
            captured.append((table, frame))

    class _Kalshi:
        async def _request(self, method, path):
            return {'orderbook_fp': {
                'yes_dollars': [['0.60', '100'], ['0.59', '250']],
                'no_dollars': [['0.39', '70'], ['0.38', '130']],
            }}

    quote = types.SimpleNamespace(
        yes_bid=0.60, yes_ask=0.61, yes_bid_size=100.0, yes_ask_size=70.0)
    window_open = pd.Timestamp('2026-08-28 12:00', tz='UTC')
    scored = pd.DataFrame({'symbol': ['BTC']})

    with mock.patch('core.datastore.ResearchStore', lambda *a, **k: _Store()):
        await live._record_touch(
            scored, {'BTC': (quote, 'KXBTC15M-X')}, window_open, 3,
            Config(), _Kalshi())

    assert captured, 'nothing was written'
    row = captured[0][1].iloc[0]
    for column in DEPTH_MAP:
        assert column in row.index, f'{column} is consumed but never recorded'

    # The totals cover the whole ladder, not just levels near the touch.
    assert row['depth_bid_total'] == pytest.approx(350.0)
    assert row['depth_ask_total'] == pytest.approx(200.0)
