"""The live path must build the same book features the backtest scored.

The model's features include `market_prob`, `market_minus_baseline`, `spread`,
`imbalance_touch`, `imbalance_5c`, `depth_ratio` and `book_convexity`. Live
computed none of them: `_record_touch` derives the depth every cycle and writes
it to the store, and the scoring row never saw it.

That does not raise. LightGBM scores a NaN feature using the default direction
it learned, so the live loop would run silently as a DIFFERENT model from the
one whose gates were measured — which is the whole class of failure the
`price_source` distinction exists to prevent, one level deeper.

The columns come from the same `market_state_features` the backtest uses, off
the same quote object the fill will be priced against, so the two cannot drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.live import book_feature_row


class _Quote:
    def __init__(self, bid=0.44, ask=0.46, bid_size=250.0, ask_size=150.0):
        self.yes_bid, self.yes_ask = bid, ask
        self.yes_bid_size, self.yes_ask_size = bid_size, ask_size


# yes side: (price, size) bids; no side: (price, size) quoted in NO terms
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
    """`depth_bid_total`/`depth_ask_total` were the one thing `_record_touch`
    never derived, so `depth_ratio` could not exist live at all."""
    row = book_feature_row(_Quote(), _YES, _NO, baseline_probability=0.40)
    assert np.isfinite(row['depth_ratio'])
    # bids total 900, asks total 600 -> log(900/600)
    assert row['depth_ratio'] == pytest.approx(np.log(900.0 / 600.0), rel=1e-6)


def test_a_one_sided_book_yields_no_probability():
    """The same rule as the backtest: a lone bid is not a probability."""
    row = book_feature_row(_Quote(ask=None), _YES, _NO, baseline_probability=0.40)
    assert pd.isna(row['market_prob'])
    assert pd.isna(row['market_minus_baseline'])


def test_an_empty_ladder_does_not_raise():
    """The ladder fetch is best-effort — top of book still lands when it fails,
    and a decision must still be scoreable."""
    row = book_feature_row(_Quote(), [], [], baseline_probability=0.40)
    assert row['market_prob'] == pytest.approx(0.45)
    assert pd.isna(row['depth_ratio'])


def test_every_feature_the_model_expects_is_present():
    """Absent is the dangerous case: it scores as NaN rather than failing."""
    from core.book_features import MARKET_PRICE, MARKET_STATE
    row = book_feature_row(_Quote(), _YES, _NO, baseline_probability=0.40)
    for column in tuple(MARKET_STATE) + tuple(MARKET_PRICE):
        assert column in row, column
