"""Two ways the live order path ignored what the running configuration said.

1. `order_limit_price` read `DEFAULT_CONFIG`, not the config in force. The rule
   is "pay down to the gate, not to a fraction" — everything above `min_edge_pp`
   is spendable because a fill leaving at least the gate is one the system has
   already said yes to. But the gate it subtracted was the DEFAULT 1.5pp while
   live runs 3.0pp, so on a 5pp edge it would spend 3.5c instead of 2.0c and
   could fill below the threshold that admitted the trade.

2. `decide()` caps the stake at `measured_depth * depth_fraction` — "a measured
   depth beats the standing guess" — but `depth_up`/`depth_down` appeared
   nowhere outside tests. `measured_depth` was always None, so sizing was bounded
   only by `max_stake_dollars = $25`, which CLAUDE.md calls an assumption and an
   unmeasured one. Harmless at a $3 stake; at the cap it is 100 contracts against
   a median 212 resting, and the safeguard meant to catch that was not connected.
"""
from __future__ import annotations

import types

import pytest

from core.config import Config
from scripts.live import depth_dollars, order_limit_price


def _decision(edge, price=0.50):
    return types.SimpleNamespace(edge=edge, price=price)


def test_slippage_is_bounded_by_the_gate_actually_in_force():
    """5pp edge behind a 3pp gate leaves 2c to spend, not 3.5c."""
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    limit = order_limit_price(_decision(0.05), config=config)
    assert limit == pytest.approx(0.52, abs=1e-9)


def test_a_looser_gate_leaves_more_to_spend():
    config = Config(min_edge_pp=1.5, max_slippage_cents=3.0)
    assert order_limit_price(_decision(0.05), config=config) == pytest.approx(0.53)


def test_the_slippage_rail_still_caps_a_large_edge():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    # 11pp edge would leave 8c spendable; the rail holds it to 3c.
    assert order_limit_price(_decision(0.11), config=config) == pytest.approx(0.53)


def test_an_edge_at_the_gate_spends_nothing():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    assert order_limit_price(_decision(0.03), config=config) == pytest.approx(0.50)


def test_depth_is_dollars_of_resting_size_on_the_side_being_crossed():
    """Buying UP crosses the YES ask, whose size comes from the NO stack.
    `decide()` reads dollars, so it is contracts x the price paid."""
    quote = types.SimpleNamespace(yes_bid=0.40, yes_ask=0.42)
    yes_levels = [[0.40, 100.0], [0.39, 50.0]]     # YES bids -> capacity to buy NO
    no_levels = [[0.58, 80.0], [0.57, 40.0]]       # NO bids  -> capacity to buy YES
    up, down = depth_dollars(quote, yes_levels, no_levels)
    assert up == pytest.approx(80.0 * 0.42, rel=1e-6)
    assert down == pytest.approx(100.0 * 0.60, rel=1e-6)


def test_an_empty_ladder_gives_no_measurement_not_a_zero():
    """Zero would refuse every trade. Absent depth means fall back to
    max_stake_dollars, which is what the code did before the book was read."""
    quote = types.SimpleNamespace(yes_bid=0.40, yes_ask=0.42)
    up, down = depth_dollars(quote, [], [])
    assert up is None and down is None
