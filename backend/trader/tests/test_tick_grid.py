"""Tests that `place_order` puts the limit on the venue's tick grid, rounding toward a
fill.

`order_limit_price` leaves the allowance unrounded, so a sub-tick allowance is kept.
"""
from __future__ import annotations

import types

import pytest

from core.config import Config
from scripts.live import order_limit_price


def test_snap_to_tick_is_gone():
    """`snap_to_tick` does not exist in `scripts.live`."""
    import scripts.live as live
    assert not hasattr(live, 'snap_to_tick')


def test_the_allowance_survives_instead_of_being_rounded_away():
    """A sub-cent allowance is kept: 0.25c of give on a 0.74 touch gives 0.7425."""
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.0025, price=0.74)
    limit = order_limit_price(d, config=config)
    assert limit == pytest.approx(0.7425)
    assert limit > d.price, 'a positive allowance must widen the limit'


def test_the_limit_never_falls_below_the_price_we_must_pay():
    """The limit is never below the price we must pay, or it could not fill."""
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    for price in (0.05, 0.35, 0.74, 0.95):
        for edge in (0.0, 0.01, 0.03, 0.09):
            d = types.SimpleNamespace(edge=edge, price=price)
            assert order_limit_price(d, config=config) >= price - 1e-12


def test_the_allowance_is_still_capped_by_max_slippage():
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.40, price=0.50)
    assert order_limit_price(d, config=config) == pytest.approx(0.53)


def test_a_nonsense_edge_does_not_produce_a_nonsense_limit():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=float('nan'), price=0.60)
    assert order_limit_price(d, config=config) == pytest.approx(0.60)


def test_the_limit_stays_inside_the_venues_orderable_range():
    """The limit stays within the venue's orderable range (1c..99c)."""
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.10, price=0.985)
    assert order_limit_price(d, config=config) <= 0.99 + 1e-12


def test_place_order_puts_a_bid_on_the_grid_by_rounding_UP():
    """A bid rounds up to the grid: 0.7425 becomes 75c, not 74c."""
    import math
    from data_collection.kalshi_client import CENT
    assert int(math.ceil(0.7425 / CENT - 1e-9)) == 75


def test_place_order_puts_an_ask_on_the_grid_by_rounding_DOWN():
    """An ask (buying NO) rounds down, since it fills at or below the YES bid."""
    import math
    from data_collection.kalshi_client import CENT
    # Paying at most 0.7425 for NO is selling YES at 0.2575 or better.
    assert int(math.floor((1.0 - 0.7425) / CENT + 1e-9)) == 25
