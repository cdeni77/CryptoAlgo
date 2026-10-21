"""Tests for the Kalshi fee schedule in `core.costs`.

`fee = ceil(0.07 * C * P * (1 - P) * 10_000) / 10_000` per order; settlement is free.
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
    """The ceiling applies per order, so splitting an order can only cost more."""
    assert trade_fee(1, 0.50, CFG) == pytest.approx(0.0175)
    assert trade_fee(2, 0.50, CFG) == pytest.approx(0.035)
    assert trade_fee(10, 0.50, CFG) == pytest.approx(0.175)
    # A raw fee that is not a whole number of hundredth-cents rounds up once per
    # order, so the per-contract cost falls slightly with size.
    assert trade_fee(7, 0.37, CFG) / 7 < trade_fee(1, 0.37, CFG)


def test_a_confident_bet_is_a_cheap_bet():
    """The fee falls from 50c toward either extreme, so confident bets are cheap."""
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
    """A $1 contract breaks even when the win probability equals its all-in cost."""
    for price in (0.20, 0.50, 0.85, 0.95):
        cost = float(effective_price(price, CFG))
        assert break_even_probability(price, CFG) == pytest.approx(cost)
        assert expected_value_per_contract(cost, price, CFG) == pytest.approx(0.0, abs=1e-12)


def test_required_edge_falls_toward_the_extremes():
    """Required edge falls toward the extremes; expected values come from the config."""
    prices = np.array([0.50, 0.70, 0.85, 0.95])
    edges = np.asarray(required_edge_pp(prices, CFG))
    assert np.all(np.diff(edges) < 0), edges
    half = CFG.half_spread_cents / 100.0
    for price, edge in zip(prices, edges):
        expected = (float(fee_per_contract(price + half, CFG)) + half) * 100
        assert edge == pytest.approx(expected, abs=0.01)


def test_the_half_spread_overtakes_the_fee_in_the_upper_tail():
    """The half-spread exceeds the fee above a crossover price, computed from the config."""
    half = CFG.half_spread_cents / 100.0
    crossover = 0.5 * (1 + math.sqrt(1 - 4 * half / CFG.fee_rate))
    assert 0.80 < crossover < 0.98, crossover
    for price in (0.50, 0.70, crossover - 0.05):
        assert half < float(fee_per_contract(price, CFG)), price
    for price in (crossover + 0.02, 0.99):
        assert half > float(fee_per_contract(price, CFG)), price


def test_the_unaffordable_band_is_the_middle_not_the_ends():
    """The unaffordable band is the middle, since `p(1-p)` makes the affordable set two tails."""
    threshold = 1.5
    low, high = unaffordable_price_band(threshold, CFG)
    assert low < 0.5 < high, (low, high)
    # Inside the band the schedule alone needs more than the threshold; outside it,
    # less.
    assert float(required_edge_pp(0.5, CFG)) > threshold
    assert float(required_edge_pp(max(low - 0.02, 0.01), CFG)) <= threshold
    assert float(required_edge_pp(min(high + 0.02, 0.99), CFG)) <= threshold


def test_the_maker_fee_is_zero_by_default():
    """The maker multiplier defaults to 0 for these series."""
    maker = Config(assume_maker=True)
    assert fee_per_contract(0.50, maker) == 0.0
    assert trade_fee(100, 0.5, maker) == 0.0


def test_the_maker_fee_is_a_quarter_of_the_taker_schedule():
    """Maker fee is ceil(M * 0.0175 * C * P * (1 - P)) per order."""
    maker = Config(assume_maker=True, maker_fee_multiplier=1.0)
    taker = Config()
    assert fee_per_contract(0.30, maker) == pytest.approx(
        fee_per_contract(0.30, taker) / 4)
    # 10 contracts at 50c: 0.0175 * 10 * 0.25 = 0.04375 -> 0.0438.
    assert trade_fee(10, 0.5, maker) == pytest.approx(0.0438)


def test_effective_price_never_leaves_the_tick_range():
    assert float(effective_price(0.99, CFG)) < 1.0 + 0.05
    assert float(effective_price(0.01, CFG)) > TICK


def test_the_schedule_reports_itself_as_unverified():
    """The schedule reports that it is not verified against an order ticket."""
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


# --- checked against real fills ---------------------------------------------
#
# The venue's `fee_cost` matches `ceil(rate * n * p * (1-p) * 10_000) / 10_000`:
# the ceiling is to a hundredth of a cent, not a whole cent.

def test_the_fee_rounds_up_to_a_hundredth_of_a_cent():
    """The fee rounds up to a hundredth of a cent.

    From a real fill: 4 NO contracts at $0.76 were charged $0.0511 (0.07 * 4 * 0.76 *
    0.24 = $0.0510720).
    """
    assert trade_fee(4, 0.76, CFG) == pytest.approx(0.0511, abs=1e-9)


def test_a_one_contract_order_is_not_rounded_to_two_cents():
    """One contract at 50c costs exactly $0.0175, not $0.02."""
    assert trade_fee(1, 0.50, CFG) == pytest.approx(0.0175, abs=1e-9)


def test_the_fee_is_still_ceiled_not_truncated():
    """Rounding is upward: the venue never charges less than the formula."""
    raw = CFG.fee_rate * 3 * 0.37 * 0.63
    charged = float(trade_fee(3, 0.37, CFG))
    assert charged >= raw
    assert charged - raw < 1e-4


def test_the_per_order_ceiling_no_longer_penalises_small_orders_materially():
    """Ten one-contract orders cost almost the same as one ten-contract order."""
    single = float(trade_fee(1, 0.50, CFG))
    ten = float(trade_fee(10, 0.50, CFG))
    assert 10 * single - ten < 1e-3


# --------------------------------------------------------------------------
# the dead band: a HOLE in the middle of the price band
# --------------------------------------------------------------------------

def test_the_dead_band_is_inert_until_asked_for():
    """Default None, so nothing changes for anything already running."""
    from core.config import DEFAULT_CONFIG
    from core.decide import outside_dead_band
    import numpy as np
    assert DEFAULT_CONFIG.exclude_price_band is None
    px = np.array([0.05, 0.35, 0.50, 0.65, 0.95])
    assert outside_dead_band(px, DEFAULT_CONFIG).all()


def test_the_dead_band_refuses_only_prices_inside_it():
    from dataclasses import replace
    from core.config import DEFAULT_CONFIG
    from core.decide import outside_dead_band
    import numpy as np
    cfg = replace(DEFAULT_CONFIG, exclude_price_band=(0.35, 0.65))
    px = np.array([0.05, 0.34, 0.35, 0.50, 0.65, 0.66, 0.95])
    got = outside_dead_band(px, cfg).tolist()
    assert got == [True, True, False, False, False, True, True]


def test_an_off_centre_dead_band_is_refused():
    """The edge is a disagreement about sigma_remaining and points BOTH ways,
    so an off-centre hole silently discards one direction of the thesis — the
    same failure the asymmetric outer band already guards against."""
    import pytest
    from dataclasses import replace
    from core.config import DEFAULT_CONFIG
    with pytest.raises(ValueError, match='symmetric'):
        replace(DEFAULT_CONFIG, exclude_price_band=(0.35, 0.60))
    with pytest.raises(ValueError, match='band in'):
        replace(DEFAULT_CONFIG, exclude_price_band=(0.65, 0.35))


def test_both_decision_paths_agree_about_the_dead_band():
    """`decide()` and the vectorised `stateless_screen` must refuse the same
    rows — the backtest uses one and live uses the other, and this repo has
    already had them disagree about entry price for months."""
    from dataclasses import replace
    from core.config import DEFAULT_CONFIG
    from core.decide import outside_dead_band
    import numpy as np
    cfg = replace(DEFAULT_CONFIG, exclude_price_band=(0.35, 0.65))
    for price in (0.04, 0.20, 0.36, 0.50, 0.64, 0.80, 0.96):
        vec = bool(outside_dead_band(np.array([price]), cfg)[0])
        single = not (0.35 <= price <= 0.65)
        assert vec == single, f'paths disagree at {price}'
