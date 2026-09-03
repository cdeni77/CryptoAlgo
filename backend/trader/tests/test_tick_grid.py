"""Our limit price was not on the venue's tick grid, so the venue snapped it.

`order_limit_price` returns `decision.price + allowance`, which lands on
arbitrary fractions like 0.373. Kalshi's `price_level_structure` is
`tapered_deci_cent`: a tenth of a cent below 10c and above 90c, a full cent in
between. The venue snaps an off-grid price to a valid tick, and the snap can go
AGAINST us — measured, 4 of 143 fills came back above `max_price`, by 0.08c to
0.71c, every one of them on a round tick:

    our limit  filled   over
        0.373   0.380  +0.71c
        0.876   0.880  +0.41c
        0.797   0.800  +0.34c
        0.179   0.180  +0.08c

About 0.011c per contract across all fills, so the money is nil. But the limit
is the rail that bounds the worst fill, and a rail that can be crossed is not
one. Rounding DOWN is the conservative direction for a buy: it can only ever
give up a fraction of a tick of fill probability, never pay more than intended.
"""
from __future__ import annotations

import pytest

from scripts.live import snap_to_tick


def test_the_mid_range_grid_is_a_whole_cent():
    assert snap_to_tick(0.373) == pytest.approx(0.37)
    assert snap_to_tick(0.797) == pytest.approx(0.79)
    assert snap_to_tick(0.876) == pytest.approx(0.87)


def test_the_tails_are_a_tenth_of_a_cent():
    """Below 10c and above 90c the venue quotes deci-cents, so rounding to a
    whole cent there would throw away nine tenths of the available precision —
    and 41% of live contracts are bought under 15c."""
    assert snap_to_tick(0.0734) == pytest.approx(0.073)
    assert snap_to_tick(0.9567) == pytest.approx(0.956)


def test_it_always_rounds_DOWN_for_a_buy():
    """Up is the direction that crossed the limit. Never round toward paying
    more."""
    for raw in (0.3799, 0.8799, 0.1799, 0.09999):
        assert snap_to_tick(raw) <= raw + 1e-12


def test_a_price_already_on_the_grid_is_unchanged():
    assert snap_to_tick(0.37) == pytest.approx(0.37)
    assert snap_to_tick(0.05) == pytest.approx(0.05)


def test_the_limit_the_order_sends_is_on_the_grid():
    """The whole point: `order_limit_price` must not hand the venue a price it
    has to snap for us."""
    import types
    from core.config import Config
    config = Config(min_edge_pp=3.0, max_slippage_cents=3.0)
    limit = order_limit_price(types.SimpleNamespace(edge=0.0523, price=0.35),
                              config=config)
    assert snap_to_tick(limit) == pytest.approx(limit)


from scripts.live import order_limit_price  # noqa: E402
