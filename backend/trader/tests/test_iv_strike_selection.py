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


def test_a_price_with_no_size_behind_it_is_not_a_quote():
    """The defect that survived the first fix, seen live on SOL:

        KXSOLD-26AUG3113-T99.7499
          yes_bid 0.0100  size 2314
          yes_ask 1.0000  size    0   <- no ask exists
          liquidity 0.00  volume 0  open_interest 0

    Both prices are non-zero and ask >= bid, so it passed and contributed a
    fabricated mid of 0.505 — a "coin flip" for a strike with no market.
    Eighteen of those in one fit produced R2 0.02 and a 6,371 bp/min sigma.

    This is the direct analogue of the backfill's `if snaps:` — Predexon
    recorded no book for such a strike, so it never entered a fit.
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
    """The backfilled path and the fixtures carry prices without sizes. Sizes
    are checked when present; their absence is not evidence of an empty book."""
    assert mid_of({'yes_bid_dollars': '0.4400',
                   'yes_ask_dollars': '0.4600'}) == pytest.approx(0.45)
