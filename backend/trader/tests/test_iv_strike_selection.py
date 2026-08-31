"""Live fitted rungs the backfill would have dropped, and the fit collapsed.

`cross_section` in the backfill drops a strike unless it has a TWO-SIDED quote:
"A one-sided quote is dropped rather than half-invented. The inversion needs
P(above), and a single side does not give one."

`mid_of` did two things that violate it. A zero bid passed the `is not None`
check, so a strike quoting 0.0000/0.0300 contributed a mid of 0.015 — but a
zero bid means there is no bid, which CLAUDE.md states outright ("a zero level
means there is nothing there"). And when there was no two-sided quote at all it
fell back to `last_price`, which on an illiquid strike can be hours stale.

Measured effect — the two paths were fitting different ladders entirely:

    symbol  source    med R2   %R2<0.9   med strikes
    ETH     backfill   0.976     16.1%        5
    ETH     live       0.827     91.4%       50
    SOL     backfill   0.986     19.7%        5
    SOL     live       0.295    100.0%       19

Fitting one sigma across 50 rungs whose mids were half-invented is why ETH fell
to 0.827 and SOL to 0.295. BTC is liquid enough to survive it (0.986) which is
why this hid.
"""
from __future__ import annotations

import pytest

from scripts.record_implied_vol import mid_of


def test_a_zero_bid_is_no_bid():
    """The exact shape seen live: ETH 3209.99 quoting 0.0000 / 0.0300."""
    assert mid_of({'yes_bid_dollars': '0.0000',
                   'yes_ask_dollars': '0.0300'}) is None


def test_a_zero_ask_is_no_ask():
    assert mid_of({'yes_bid_dollars': '0.9700',
                   'yes_ask_dollars': '0.0000'}) is None


def test_a_real_two_sided_quote_is_the_mid():
    assert mid_of({'yes_bid_dollars': '0.4400',
                   'yes_ask_dollars': '0.4600'}) == pytest.approx(0.45)


def test_last_price_is_not_a_substitute_for_a_quote():
    """The backfill drops the rung. A last price is a trade that happened at
    some point, not the ladder's current P(above), and on an illiquid strike it
    can be hours old."""
    assert mid_of({'last_price_dollars': '0.5000'}) is None


def test_integer_cents_are_still_accepted():
    """Both encodings appear on this venue; above 1.0 it can only be cents."""
    assert mid_of({'yes_bid': 44, 'yes_ask': 46}) == pytest.approx(0.45)


def test_a_crossed_quote_is_refused():
    assert mid_of({'yes_bid_dollars': '0.6000',
                   'yes_ask_dollars': '0.4000'}) is None
