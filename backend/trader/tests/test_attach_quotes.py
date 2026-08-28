"""Pricing a backtest row against the venue's real book.

**This is the circularity the audit named.** `core/decide.py` falls back to
`p_market = baseline_probability` when a row carries no `ask_up`/`ask_down`,
which is every backtest row — so the simulated counterparty is the model's own
training target, and `model_minus_market` cannot be computed at all. Both of the
gates that were supposed to answer "does this beat the market" read NaN.

Attaching the recorded book fixes that, and the denomination is the part that
kills you quietly. A Kalshi YES book quotes what YES costs; the price of the
DOWN side is `1 - yes_bid`, because buying NO means taking the other side of
what YES bidders will pay. Using `yes_bid` directly, or the mid, is wrong by the
spread with the sign inverted on one side — the same class of error as the
`no_levels` denomination trap that put a 0.51 YES ask in a column holding a
0.51 NO bid.
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
    """rows: (offset_minutes, yes_bid, yes_ask, age)"""
    return pd.DataFrame([
        {'venue': venue, 'symbol': symbol,
         'window_open': pd.Timestamp('2026-07-01T12:00Z'),
         'offset_minutes': o, 'yes_bid': b, 'yes_ask': a,
         'quote_age_seconds': age, 'source': 'backfill'}
        for o, b, a, age in rows])


def test_the_up_price_is_the_yes_ask():
    """Buying UP means lifting the YES offer, so it costs the ask — never the
    mid, which is a price nobody will sell you."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['ask_up'].iloc[0] == pytest.approx(0.46)


def test_the_down_price_is_one_minus_the_yes_bid():
    """Buying DOWN is taking the other side of what YES bidders will pay."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['ask_down'].iloc[0] == pytest.approx(0.56)


def test_the_two_sides_sum_to_more_than_one_by_the_spread():
    """A sanity property that catches a denomination slip immediately: the two
    costs must sum to 1 + spread, never to 1."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    total = got['ask_up'].iloc[0] + got['ask_down'].iloc[0]
    assert total == pytest.approx(1.02), 'should be 1 + the 2c spread'


def test_a_one_sided_book_prices_only_the_side_that_exists():
    """A missing ask means nobody is offering UP. Inventing one would let the
    backtest buy at a price that was never available."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, np.nan, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])
    assert got['ask_down'].iloc[0] == pytest.approx(0.56)


def test_a_window_with_no_quote_keeps_nan_so_decide_falls_back():
    """Most of five years predates the venue. Those rows must price against the
    baseline as before, not against a fabricated quote."""
    got = attach_quotes(_windows((3, 12)), _depth([(3, 0.44, 0.46, 1.0)]))
    late = got[got['offset'] == 12].iloc[0]
    assert pd.isna(late['ask_up']) and pd.isna(late['ask_down'])


def test_the_offset_must_match_exactly():
    """A decision at +3m may not be priced with the book at +12m: nine minutes
    of a fifteen-minute window is most of the question."""
    got = attach_quotes(_windows((3,)), _depth([(12, 0.80, 0.82, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])


def test_only_the_venue_we_trade_is_used():
    """Polymarket is a cross-venue FEATURE, not the book we execute against.
    Pricing a Kalshi trade off a Polymarket quote would book a fill at a price
    the venue never showed."""
    depth = pd.concat([_depth([(3, 0.10, 0.12, 1.0)], venue='polymarket'),
                       _depth([(3, 0.44, 0.46, 1.0)], venue='kalshi')])
    got = attach_quotes(_windows((3,)), depth, venue='kalshi')
    assert got['ask_up'].iloc[0] == pytest.approx(0.46)


def test_quote_age_rides_along_so_staleness_is_visible():
    """Predexon serves book CHANGES, so a quiet book carries forward. A
    forecast that cannot be told from a stale price 'beats' it for free."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 47.0)]))
    assert got['quote_age_seconds'].iloc[0] == pytest.approx(47.0)


def test_a_stale_quote_can_be_refused_outright():
    """Above the tolerance the row is priced as if there were no book at all —
    which is the honest state, not a cheap fill."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 600.0)]),
                        max_age_seconds=120.0)
    assert pd.isna(got['ask_up'].iloc[0])


def test_crossed_or_nonsense_quotes_are_dropped():
    """ask below bid is not a book. Kept, it would show a guaranteed profit."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.60, 0.40, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0]) and pd.isna(got['ask_down'].iloc[0])


def test_prices_outside_zero_to_one_are_dropped():
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 1.40, 1.0)]))
    assert pd.isna(got['ask_up'].iloc[0])


def test_attaching_does_not_reorder_or_drop_window_rows():
    """The table is fed to decide() row by row; losing or reordering rows
    silently changes which windows were traded."""
    windows = _windows((3, 6, 9, 12))
    got = attach_quotes(windows, _depth([(6, 0.44, 0.46, 1.0)]))
    assert len(got) == len(windows)
    assert list(got['offset']) == [3, 6, 9, 12]


def test_the_market_probability_is_the_mid():
    """`model_minus_market` compares forecasts, so the market's number is the
    mid. The ask is what a trade costs and belongs in the money, where decide()
    already uses it — scoring the market at its ask would credit the model with
    half the spread as skill on every row."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)]))
    assert got['market_probability'].iloc[0] == pytest.approx(0.45)


def test_the_mid_needs_both_sides():
    """A lone bid says the probability is at LEAST something, which is not a
    probability — and a one-sided mid would be a fabricated benchmark."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, np.nan, 1.0)]))
    assert pd.isna(got['market_probability'].iloc[0])


def test_the_mid_sits_between_the_two_trade_costs():
    """A structural check: ask_up >= mid >= 1 - ask_down, always."""
    got = attach_quotes(_windows((3,)), _depth([(3, 0.44, 0.46, 1.0)])).iloc[0]
    assert got['ask_up'] >= got['market_probability'] >= 1.0 - got['ask_down']
