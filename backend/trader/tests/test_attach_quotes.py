"""Tests for `attach_quotes`: pricing a backtest row against the recorded book.

UP costs the YES ask and DOWN costs `1 - yes_bid`; the mid or the raw bid is wrong by
the spread.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quotes import attach_quotes


def _windows(offsets=(3, 12), symbol='BTC-USD'):
    return pd.DataFrame([
        {'symbol': symbol,
         'window_open': pd.Timestamp('2026-07-01T12:00Z'),
         'offset': o, 'baseline_probability': 0.40}
        for o in offsets])


def _depth(rows, venue='kalshi', symbol='BTC-USD'):
    """Rows are (offset_minutes, yes_bid, yes_ask, age)."""
    return pd.DataFrame([
        {'venue': venue, 'symbol': symbol,
         'window_open': pd.Timestamp('2026-07-01T12:00Z'),
         'offset_minutes': o, 'yes_bid': b, 'yes_ask': a,
         'quote_age_seconds': age, 'source': 'backfill'}
        for o, b, a, age in rows])


def test_the_up_price_is_the_yes_ask():
    """Buying UP lifts the YES offer, so it costs the ask, never the mid."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['ask_up'].iloc[0] == pytest.approx(0.46)


def test_the_down_price_is_one_minus_the_yes_bid():
    """Buying DOWN takes the other side of what YES bidders pay."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['ask_down'].iloc[0] == pytest.approx(0.56)


def test_the_two_sides_sum_to_more_than_one_by_the_spread():
    """The two costs sum to 1 + spread, never to 1. This catches a denomination error."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    total = got['ask_up'].iloc[0] + got['ask_down'].iloc[0]
    assert total == pytest.approx(1.02), 'should be 1 + the 2c spread'


def test_a_one_sided_book_prices_only_the_side_that_exists():
    """A missing ask leaves UP unpriced, so the backtest cannot buy at a price that did
    not exist.
    """
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, np.nan, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])
    assert got['ask_down'].iloc[0] == pytest.approx(0.56)


def test_a_window_with_no_quote_keeps_nan_so_decide_falls_back():
    """Rows with no recorded quote price against the baseline, not a fabricated quote."""
    got = attach_quotes(_windows((3, 12)), _depth([(3, 0.44, 0.46, 1.0)]))
    late = got[got['offset'] == 12].iloc[0]
    assert pd.isna(late['ask_up']) and pd.isna(late['ask_down'])


def test_the_offset_must_match_exactly():
    """A decision at +3m is never priced with the book at +12m."""
    got = attach_quotes(_windows((3,)), _depth([(12, 0.80, 0.82, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])


def test_only_the_venue_we_trade_is_used():
    """Only the traded venue prices the trade. Polymarket is a feature, not the execution book."""
    depth = pd.concat([_depth([(3, 0.10, 0.12, 1.0)], venue='polymarket'),
                       _depth([(3, 0.44, 0.46, 1.0)], venue='kalshi')])
    got = attach_quotes(_windows((3,)), depth, venue='kalshi')
    assert got['ask_up'].iloc[0] == pytest.approx(0.46)


def test_quote_age_rides_along_so_staleness_is_visible():
    """Quote age is attached so a stale price is visible."""
    # 12s is under the 30-second default, which matches
    # `core.metrics.MAX_QUOTE_AGE_SECONDS`.
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 12.0)]))
    # Age at the decision instant: 12s before the minute plus the 3s decision latency.
    assert got['quote_age_seconds'].iloc[0] == pytest.approx(15.0)


def test_a_stale_quote_can_be_refused_outright():
    """Above the age limit the row is priced as if there were no book."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 600.0)]),
                        max_age_seconds=120.0)
    assert pd.isna(got['ask_up'].iloc[0])


def test_crossed_or_nonsense_quotes_are_dropped():
    """A crossed book (ask below bid) is dropped. Kept, it would show a guaranteed profit."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.60, 0.40, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0]) and pd.isna(got['ask_down'].iloc[0])


def test_prices_outside_zero_to_one_are_dropped():
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 1.40, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])


def test_attaching_does_not_reorder_or_drop_window_rows():
    """`decide()` reads the table row by row, so dropping or reordering rows changes
    which windows trade.
    """
    windows = _windows((3, 6, 9, 12))
    got = attach_quotes(windows, _depth([(6, 0.44, 0.46, 1.0)]))
    assert len(got) == len(windows)
    assert list(got['offset']) == [3, 6, 9, 12]


def test_the_market_probability_is_the_mid():
    """The market probability is the mid.

    `model_minus_market` compares forecasts. Scoring the market at its ask would credit
    the model with half the spread on every row.
    """
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['market_probability'].iloc[0] == pytest.approx(0.45)


def test_the_mid_needs_both_sides():
    """A one-sided book has no mid, so the market probability is NaN."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, np.nan, 1.0)]))
    assert pd.isna(got['market_probability'].iloc[0])


def test_the_mid_sits_between_the_two_trade_costs():
    """Structural check: ask_up >= mid >= 1 - ask_down."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)])).iloc[0]
    assert got['ask_up'] >= got['market_probability'] >= 1.0 - got['ask_down']


# --- depth columns that market_state needs ---------------------------------

DEPTH_COLUMNS = ('bid_at_touch', 'ask_at_touch', 'bid_1c', 'ask_1c',
                 'bid_5c', 'ask_5c', 'bid_vol', 'ask_vol')


def _rich_depth(venue='kalshi'):
    return pd.DataFrame([{
        'venue': venue, 'symbol': 'BTC-USD',
        'window_open': pd.Timestamp('2026-07-01T12:00Z'),
        'offset_minutes': 3, 'yes_bid': 0.44, 'yes_ask': 0.46,
        'quote_age_seconds': 1.0, 'source': 'backfill',
        'yes_bid_size': 250.0, 'yes_ask_size': 150.0,
        'depth_bid_1c': 400.0, 'depth_ask_1c': 300.0,
        'depth_bid_5c': 900.0, 'depth_ask_5c': 600.0,
        'depth_bid_total': 2000.0, 'depth_ask_total': 1000.0}])


def test_the_depth_columns_market_state_needs_are_attached():
    got = attach_quotes(_windows((3,)), _rich_depth())
    for column in DEPTH_COLUMNS:
        assert column in got.columns, column
    assert got['bid_at_touch'].iloc[0] == pytest.approx(250.0)
    assert got['bid_5c'].iloc[0] == pytest.approx(900.0)
    assert got['bid_vol'].iloc[0] == pytest.approx(2000.0)


def test_the_other_venue_is_attached_under_its_own_prefix():
    """The other venue's book is attached under its own prefix so the columns do not collide."""
    depth = pd.concat([_rich_depth('kalshi'),
                       _rich_depth('polymarket').assign(yes_bid=0.40, yes_ask=0.41)])
    got = attach_quotes(_windows((3,)), depth, venue='kalshi', other_venue='polymarket')
    assert got['market_probability'].iloc[0] == pytest.approx(0.45)
    assert got['pm_market_probability'].iloc[0] == pytest.approx(0.405)


def test_a_missing_other_venue_leaves_nan_not_zero():
    got = attach_quotes(_windows((3,)), _rich_depth('kalshi'),
                        venue='kalshi', other_venue='polymarket')
    assert pd.isna(got['pm_market_probability'].iloc[0])
