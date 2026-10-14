"""Tests that `mid_of` selects strikes the way the backfill does.

Only a two-sided quote with size on both sides counts; a zero price or a last trade does
not.
"""
from __future__ import annotations

import pytest

from scripts.record_implied_vol import mid_of


def test_a_zero_bid_is_no_bid():
    """A zero bid is no bid (for example 0.0000 / 0.0300)."""
    assert mid_of({'yes_bid_dollars': '0.0000',
                   'yes_ask_dollars': '0.0300'}) is None


def test_a_zero_ask_is_no_ask():
    assert mid_of({'yes_bid_dollars': '0.9700',
                   'yes_ask_dollars': '0.0000'}) is None


def test_a_real_two_sided_quote_is_the_mid():
    assert mid_of({'yes_bid_dollars': '0.4400',
                   'yes_ask_dollars': '0.4600'}) == pytest.approx(0.45)


def test_last_price_is_not_a_substitute_for_a_quote():
    """`last_price` is not used: it is a past trade, possibly hours old, not the current
    P(above).
    """
    assert mid_of({'last_price_dollars': '0.5000'}) is None


def test_integer_cents_are_still_accepted():
    """Both encodings appear on this venue; a value above 1.0 can only be cents."""
    assert mid_of({'yes_bid': 44, 'yes_ask': 46}) == pytest.approx(0.45)


def test_a_crossed_quote_is_refused():
    assert mid_of({'yes_bid_dollars': '0.6000',
                   'yes_ask_dollars': '0.4000'}) is None


def test_a_price_with_no_size_behind_it_is_not_a_quote():
    """A price with zero size behind it is not a quote.

    Example: bid 0.01 (size 2314) and ask 1.00 (size 0) would give a false mid of 0.505
    for a strike with no market.
    """
    assert mid_of({'yes_bid_dollars': '0.0100', 'yes_ask_dollars': '1.0000',
                   'yes_bid_size_fp': '2314.00',
                   'yes_ask_size_fp': '0.00'}) is None


def test_no_size_on_the_bid_is_also_refused():
    assert mid_of({'yes_bid_dollars': '0.4400', 'yes_ask_dollars': '0.4600',
                   'yes_bid_size_fp': '0.00',
                   'yes_ask_size_fp': '150.00'}) is None


def test_a_genuinely_two_sided_market_is_kept():
    assert mid_of({'yes_bid_dollars': '0.4400', 'yes_ask_dollars': '0.4600',
                   'yes_bid_size_fp': '250.00',
                   'yes_ask_size_fp': '150.00'}) == pytest.approx(0.45)


def test_absent_size_fields_do_not_reject_a_two_sided_quote():
    """Sizes are checked only when present; missing size fields do not reject a quote."""
    assert mid_of({'yes_bid_dollars': '0.4400',
                   'yes_ask_dollars': '0.4600'}) == pytest.approx(0.45)
