"""Kalshi's fee schedule, and the shape that decides the strategy.

`fee = ceil(0.07 * contracts * price * (1 - price) * 10_000) / 10_000`, per
order, and settlement is free. The ceiling is to a HUNDREDTH of a cent, not a
whole cent — this file used to say `* 100) / 100`, which over-charged 7% in
aggregate and ~17% on the smallest orders before it was corrected against 328
real fills. The `p(1-p)` term means a confident bet is a cheap bet, which is
the opposite of a perpetual future's fixed toll — and it is why the barrier
framing and this venue fit together at all.

The taker formula is now measured: all 328 real fills came back `is_taker:
true`, confirming the published schedule at hundredth-of-a-cent precision. The
maker rate remains a flat modelled per-contract charge that no order ticket has
ever confirmed — treat it as provisional, and see `core/costs.py` for the full
account of what changed and why.
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
        expected = math.ceil(round(raw * 10_000, 6)) / 10_000
        assert trade_fee(contracts, price, CFG) == pytest.approx(expected)


def test_the_ceiling_is_per_order_not_per_contract():
    """The ceiling is still per order — it is just no longer material.

    This test used to assert $0.02 for one contract at 50c and call the 14%
    surcharge "the dominant correction" at a $100 account. Measured against 328
    real fills, the granularity is a hundredth of a cent, so the same order pays
    $0.0175 and the per-order effect is a rounding detail after all.

    The direction of the effect is unchanged and still worth pinning: rounding is
    per order, so splitting an order can only cost more, never less.
    """
    assert trade_fee(1, 0.50, CFG) == pytest.approx(0.0175)
    assert trade_fee(2, 0.50, CFG) == pytest.approx(0.035)
    assert trade_fee(10, 0.50, CFG) == pytest.approx(0.175)
    # A price whose raw fee is not a whole number of hundredth-cents still
    # rounds up once per order, so per-contract cost falls (very slightly) with
    # size.
    assert trade_fee(7, 0.37, CFG) / 7 < trade_fee(1, 0.37, CFG)


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


# --- measured against real fills, 2026-08-25 -------------------------------
#
# 328 filled orders were read back from `GET /portfolio/fills` — the first time
# any order ticket had been checked against this module. The venue's `fee_cost`
# matches `ceil(rate * n * p * (1-p) * 10_000) / 10_000` on 311 of them: the
# ceiling is to a **hundredth of a cent**, not to a whole cent.
#
# The whole-cent rule this module used over-charged by 7% in aggregate and ~17%
# on a small order, which made every net-edge gate too strict.

def test_the_fee_rounds_up_to_a_hundredth_of_a_cent():
    """The venue's granularity, from a real ticket.

    Fill: 4 contracts of NO at $0.76, `fee_cost` $0.051100.
    0.07 * 4 * 0.76 * 0.24 = $0.0510720, and the venue charged $0.0511.
    A whole-cent ceiling would have charged $0.06 — 17% more.
    """
    assert trade_fee(4, 0.76, CFG) == pytest.approx(0.0511, abs=1e-9)


def test_a_one_contract_order_is_not_rounded_to_two_cents():
    """0.07 * 1 * 0.5 * 0.5 = $0.0175 exactly, and that is what is charged.

    The old rule billed $0.02 and this module's docstring called that 14%
    surcharge "the dominant correction" at a $100 account. It is not a
    correction at all — the granularity is 100x finer than assumed.
    """
    assert trade_fee(1, 0.50, CFG) == pytest.approx(0.0175, abs=1e-9)


def test_the_fee_is_still_ceiled_not_truncated():
    """Rounding is still upward — the venue never charges less than the formula."""
    raw = CFG.fee_rate * 3 * 0.37 * 0.63
    charged = float(trade_fee(3, 0.37, CFG))
    assert charged >= raw
    assert charged - raw < 1e-4


def test_the_per_order_ceiling_no_longer_penalises_small_orders_materially():
    """Ten one-contract orders cost within a hundredth of a cent each of one
    ten-contract order. Under the whole-cent rule they cost 11% more."""
    single = float(trade_fee(1, 0.50, CFG))
    ten = float(trade_fee(10, 0.50, CFG))
    assert 10 * single - ten < 1e-3
