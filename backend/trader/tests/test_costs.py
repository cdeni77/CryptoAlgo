"""Kalshi's fee schedule, and the shape that decides the strategy.

`fee = ceil(0.07 * contracts * price * (1 - price) * 100) / 100`, per order, and
settlement is free. The `p(1-p)` term means a confident bet is a cheap bet,
which is the opposite of a perpetual future's fixed toll — and it is why the
barrier framing and this venue fit together at all.

Nothing here has been checked against a filled order ticket. The taker formula is
the published schedule; the maker rate is modelled as a flat per-contract charge
and is unverified. The last venue this repo priced was wrong in both shape and
magnitude, and it was settled by reading three real tickets rather than by
reasoning — so these tests pin the arithmetic, not its truth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from core.config import Config
from core.costs import (
    CONTRACT_PAYOUT, TICK, FeeSchedule, break_even_probability, effective_price,
    expected_value_per_contract, fee_per_contract, required_edge_pp, trade_fee,
    unaffordable_price_band,
)

CFG = Config()


def test_the_fee_follows_the_published_formula():
    for contracts, price in ((1, 0.50), (3, 0.85), (10, 0.20), (7, 0.97)):
        raw = CFG.fee_rate * contracts * price * (1 - price)
        expected = math.ceil(round(raw * 100, 9)) / 100
        assert trade_fee(contracts, price, CFG) == pytest.approx(expected)


def test_the_ceiling_is_per_order_not_per_contract():
    """A one-contract order pays a higher rate than the schedule implies.

    0.07 x 1 x 0.25 = $0.0175, charged as $0.02 — 14% more. At a $100 account
    every order is a small order, so this is the dominant correction rather than
    a rounding detail.
    """
    assert trade_fee(1, 0.50, CFG) == pytest.approx(0.02)
    assert trade_fee(2, 0.50, CFG) == pytest.approx(0.04)
    # Ten contracts: 0.175 -> 0.18, so the per-contract rate falls with size.
    assert trade_fee(10, 0.50, CFG) == pytest.approx(0.18)
    assert trade_fee(10, 0.50, CFG) / 10 < trade_fee(1, 0.50, CFG)


def test_a_confident_bet_is_a_cheap_bet():
    """The whole reason this venue suits a barrier forecast.

    The fee is maximal at 50c and falls toward either extreme, so the
    large-displacement, late-in-the-window predictions — the confident ones — are
    exactly where `p(1-p)` is small.
    """
    prices = np.array([0.50, 0.60, 0.70, 0.80, 0.90, 0.95])
    fees = np.asarray(fee_per_contract(prices, CFG))
    assert np.all(np.diff(fees) < 0)
    share_of_stake = fees / prices
    assert np.all(np.diff(share_of_stake) < 0)
    assert share_of_stake[0] == pytest.approx(0.035, abs=1e-6)
    assert share_of_stake[-1] == pytest.approx(0.0035, abs=1e-6)


def test_the_fee_is_symmetric_about_a_half():
    assert fee_per_contract(0.30, CFG) == pytest.approx(fee_per_contract(0.70, CFG))
    assert fee_per_contract(0.05, CFG) == pytest.approx(fee_per_contract(0.95, CFG))


def test_break_even_is_the_all_in_cost():
    """A contract paying $1 breaks even when the win probability equals its cost."""
    for price in (0.20, 0.50, 0.85, 0.95):
        cost = float(effective_price(price, CFG))
        assert break_even_probability(price, CFG) == pytest.approx(cost)
        assert expected_value_per_contract(cost, price, CFG) == pytest.approx(0.0, abs=1e-12)


def test_required_edge_falls_toward_the_extremes():
    """Derived from the config rather than hardcoded.

    The half-spread moved from an assumed 1.0c to a measured 0.5c, and hardcoded
    expectations then fail for the right reason in the wrong place. Both endpoints
    are computed here, so the shape is asserted and the level tracks whatever the
    schedule actually is.
    """
    prices = np.array([0.50, 0.70, 0.85, 0.95])
    edges = np.asarray(required_edge_pp(prices, CFG))
    assert np.all(np.diff(edges) < 0), edges
    half = CFG.half_spread_cents / 100.0
    for price, edge in zip(prices, edges):
        expected = (float(fee_per_contract(price + half, CFG)) + half) * 100
        assert edge == pytest.approx(expected, abs=0.01)


def test_the_half_spread_overtakes_the_fee_in_the_upper_tail():
    """Where the spread starts to dominate the fee, derived not hardcoded.

    `0.07 * p * (1 - p) = half_spread` has a closed form, and the crossover moves
    when the spread does — it was 83c at the assumed 1.0c and is about 92c at the
    measured 0.5c. Docstrings in this repo have twice stated a stale figure for
    this (60c, then 83c), so it is computed.
    """
    half = CFG.half_spread_cents / 100.0
    crossover = 0.5 * (1 + math.sqrt(1 - 4 * half / CFG.fee_rate))
    assert 0.80 < crossover < 0.98, crossover
    for price in (0.50, 0.70, crossover - 0.05):
        assert half < float(fee_per_contract(price, CFG)), price
    for price in (crossover + 0.02, 0.99):
        assert half > float(fee_per_contract(price, CFG)), price


def test_the_unaffordable_band_is_the_middle_not_the_ends():
    """`p(1-p)` makes the affordable set two disjoint tails.

    Reporting min and max of a disjoint set reads as "everything is affordable",
    which is the opposite of what the schedule says.
    """
    threshold = 1.5
    low, high = unaffordable_price_band(threshold, CFG)
    assert low < 0.5 < high, (low, high)
    # Inside the band the schedule alone demands more than the threshold;
    # outside it, less. That is the whole claim.
    assert float(required_edge_pp(0.5, CFG)) > threshold
    assert float(required_edge_pp(max(low - 0.02, 0.01), CFG)) <= threshold
    assert float(required_edge_pp(min(high + 0.02, 0.99), CFG)) <= threshold


def test_a_maker_schedule_is_flat_per_contract():
    maker = Config(assume_maker=True)
    assert fee_per_contract(0.20, maker) == pytest.approx(maker.maker_fee_rate)
    assert fee_per_contract(0.80, maker) == pytest.approx(maker.maker_fee_rate)
    assert trade_fee(100, 0.5, maker) == pytest.approx(
        math.ceil(round(maker.maker_fee_rate * 100 * 100, 9)) / 100)


def test_effective_price_never_leaves_the_tick_range():
    assert float(effective_price(0.99, CFG)) < 1.0 + 0.05
    assert float(effective_price(0.01, CFG)) > TICK


def test_the_schedule_reports_itself_as_unverified():
    """No filled ticket has been read against this module, and it says so."""
    schedule = FeeSchedule.of(CFG)
    assert schedule.verified_against_ticket is False
    rows = schedule.table()
    assert len(rows) >= 5
    assert all(0 < r['fee_per_contract'] < 0.02 for r in rows)


def test_the_payout_is_one_dollar():
    assert CONTRACT_PAYOUT == 1.0
    # 100 winning contracts bought at 85c return $100 against an $85 outlay.
    assert expected_value_per_contract(1.0, 0.85, CFG) == pytest.approx(
        1.0 - float(effective_price(0.85, CFG)))
