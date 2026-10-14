"""Tests that the live order path uses the running config and the measured depth."""
from __future__ import annotations

import types

import pytest

from core.config import Config
from scripts.live import depth_dollars, order_limit_price


def _decision(edge, price=0.50):
    return types.SimpleNamespace(edge=edge, price=price)


def test_slippage_is_bounded_by_the_gate_actually_in_force():
    """A 5pp edge behind a 3pp gate leaves 2c to spend."""
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    limit = order_limit_price(_decision(0.05), config=config)
    assert limit == pytest.approx(0.52, abs=1e-9)


def test_a_looser_gate_leaves_more_to_spend():
    config = Config(min_edge_pp=1.5, max_slippage_cents=3.0)
    assert order_limit_price(_decision(0.05), config=config) == pytest.approx(0.53)


def test_the_slippage_rail_still_caps_a_large_edge():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    # An 11pp edge would leave 8c to spend; the rail holds it to 3c.
    assert order_limit_price(_decision(0.11), config=config) == pytest.approx(0.53)


def test_an_edge_at_the_gate_spends_nothing():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    assert order_limit_price(_decision(0.03), config=config) == pytest.approx(0.50)


def test_depth_is_dollars_of_resting_size_on_the_side_being_crossed():
    """UP depth is the size on the NO stack times the YES ask, in dollars."""
    quote = types.SimpleNamespace(yes_bid=0.40, yes_ask=0.42)
    yes_levels = [[0.40, 100.0], [0.39, 50.0]]  # YES bids -> capacity to buy NO
    no_levels = [[0.58, 80.0], [0.57, 40.0]]  # NO bids -> capacity to buy YES
    up, down = depth_dollars(quote, yes_levels, no_levels)
    assert up == pytest.approx(80.0 * 0.42, rel=1e-6)
    assert down == pytest.approx(100.0 * 0.60, rel=1e-6)


def test_an_empty_ladder_gives_no_measurement_not_a_zero():
    """An empty ladder gives None, so `decide()` falls back to `max_stake_dollars`."""
    quote = types.SimpleNamespace(yes_bid=0.40, yes_ask=0.42)
    up, down = depth_dollars(quote, [], [])
    assert up is None and down is None
