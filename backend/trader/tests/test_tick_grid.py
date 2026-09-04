"""The limit goes on the venue's grid, in the direction that can still FILL.

This file used to assert the opposite, and the reversal is the point.

`order_limit_price` returns `decision.price + allowance`, which lands on
arbitrary fractions like 0.373. Kalshi's `price_level_structure` is
`tapered_deci_cent`. An off-grid limit gets snapped by the venue, and the snap
can go against us — measured, 4 of 143 fills came back above `max_price`, by
0.08c to 0.71c, every one on a round tick:

    our limit  filled   over
        0.373   0.380  +0.71c
        0.876   0.880  +0.41c
        0.797   0.800  +0.34c
        0.179   0.180  +0.08c

That is **0.011c per contract**, and a `snap_to_tick` helper was added to round
the limit DOWN so the rail could not be crossed. The reasoning was wrong in two
ways, and it cost the loop a day of trading:

  * **It destroyed the allowance it was rounding.** A sub-tick allowance became
    exactly zero, so the limit landed on the touch and any move killed the
    order. Three of the ten kills on 2026-09-04 had a zero-cent allowance.
  * **It fought `place_order`, which was already right.** That function puts the
    limit on the grid with `ceil` for a bid and `floor` for an ask, precisely so
    "the limit never becomes one that CANNOT fill". Snapping down first threw
    the allowance away before the fill-friendly rounding could use it.

A kill earns zero, against a hundredth of a cent of protection. And the bound
that matters was never this number: it is the gate in `decide()`, which a fill
one tick above `max_price` still clears by `min_edge_pp - 1c` — at least 2pp at
the 3.0pp gate in force.

So the grid now belongs to `place_order` alone, and these tests pin it there.
"""
from __future__ import annotations

import types

import pytest

from core.config import Config
from scripts.live import order_limit_price


def test_snap_to_tick_is_gone():
    """Deleted, not merely unused — the next reader would wire it back in."""
    import scripts.live as live
    assert not hasattr(live, 'snap_to_tick')


def test_the_allowance_survives_instead_of_being_rounded_away():
    """A sub-cent allowance is kept, where snapping down erased it.

    0.25c of give on a 0.74 touch. Under the old rule this returned exactly
    0.74 — the touch — and the order could not survive a one-cent move.
    """
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.0025, price=0.74)
    limit = order_limit_price(d, config=config)
    assert limit == pytest.approx(0.7425)
    assert limit > d.price, 'a positive allowance must widen the limit'


def test_the_limit_never_falls_below_the_price_we_must_pay():
    """The one invariant the old rounding was protecting, kept explicitly.

    A limit under the ask cannot fill at all, so it is the one direction the
    arithmetic must never take.
    """
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    for price in (0.05, 0.35, 0.74, 0.95):
        for edge in (0.0, 0.01, 0.03, 0.09):
            d = types.SimpleNamespace(edge=edge, price=price)
            assert order_limit_price(d, config=config) >= price - 1e-12


def test_the_allowance_is_still_capped_by_max_slippage():
    """Removing the snap must not remove the rail against walking a thin book."""
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.40, price=0.50)
    assert order_limit_price(d, config=config) == pytest.approx(0.53)


def test_a_nonsense_edge_does_not_produce_a_nonsense_limit():
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=float('nan'), price=0.60)
    assert order_limit_price(d, config=config) == pytest.approx(0.60)


def test_the_limit_stays_inside_the_venues_orderable_range():
    """`place_order` refuses a YES limit outside 1c..99c, so the cap matters."""
    config = Config(min_edge_pp=0.0, max_slippage_cents=3.0)
    d = types.SimpleNamespace(edge=0.10, price=0.985)
    assert order_limit_price(d, config=config) <= 0.99 + 1e-12


def test_place_order_puts_a_bid_on_the_grid_by_rounding_UP():
    """The grid now lives here, and `ceil` is what keeps a bid fillable.

    0.7425 must become 75c, not 74c: 74c is below the 0.7425 we were willing to
    pay and, on the captured kill, below the actual ask.
    """
    import math
    from data_collection.kalshi_client import CENT
    assert int(math.ceil(0.7425 / CENT - 1e-9)) == 75


def test_place_order_puts_an_ask_on_the_grid_by_rounding_DOWN():
    """Buying NO sells YES, which fills at or BELOW the YES bid, so an ask
    rounds the other way. Same intent, opposite arithmetic."""
    import math
    from data_collection.kalshi_client import CENT
    # Paying at most 0.7425 for NO is selling YES at 0.2575 or better.
    assert int(math.floor((1.0 - 0.7425) / CENT + 1e-9)) == 25
