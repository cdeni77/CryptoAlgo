"""How far the limit may cross, and why the old rule refused most fills.

Measured on the live account 2026-08-25/26: of 42 order attempts, 16 filled.
The limit allowance was `min(edge * 0.25, 1c)`, so a 3.68pp edge could pay 0.92c
— against a book that moves ~2c in the seconds between reading it and sending
the order. The order then misses on price while the forecast is still perfectly
good, and the window is lost entirely.

**Crossing further is close to free for a taker.** `self_trade_prevention_type`
is `taker_at_cross` and the order is `immediate_or_cancel`, so it fills against
resting size from the touch outward and stops. The limit does not set what we
pay; it sets the worst we would tolerate. Widening it buys fill probability and
only pays more where the touch was too thin to fill at the old one — which is
exactly the case that used to return nothing at all.

So the rule is the gate, not an arbitrary fraction: pay away as much of the edge
as still clears `min_edge_pp`, the threshold that admitted the trade in the first
place. A fill at the gate is by construction a trade this system accepts; no
fill earns zero.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.decide import Decision, Reason, Side
from scripts.live import order_limit_price

WINDOW = pd.Timestamp('2026-08-26 02:00', tz='UTC')


def decision(*, price: float, edge: float) -> Decision:
    return Decision(
        symbol='BTC-USD', window_open=WINDOW,
        settle_time=WINDOW + timedelta(minutes=15), offset=12,
        reason=Reason.TRADED, side=Side.UP, price=price,
        effective_cost=price + 0.02, model_probability=price + edge,
        baseline_probability=price, edge=edge, contracts=5,
        stake=price * 5, fee=0.02, price_source='quote',
        market_ticker='KXBTC15M-26AUG252215-15')


def test_the_limit_pays_down_to_the_gate_that_admitted_the_trade():
    """The live BTC decision: 3.68pp of edge against a 1.5pp gate.

    The old rule allowed 0.92c. Everything above the gate is spendable, so this
    is 2.18c — and the trade that results still clears the same gate.
    """
    gate = DEFAULT_CONFIG.min_edge_pp / 100.0
    # The allowance is spent EXACTLY, not rounded onto the venue's grid.
    # `snap_to_tick` used to floor it here and gave 2.00c instead of 2.18c; the
    # grid now belongs to `place_order`, which rounds a bid UP so the limit
    # stays fillable. See tests/test_tick_grid.py.
    d = decision(price=0.75, edge=0.0368)
    allowance = order_limit_price(d) - d.price
    assert allowance == pytest.approx(0.0368 - gate, abs=1e-9)
    assert allowance > 0.0092, 'must be wider than the old share-of-edge rule'


def test_a_marginal_edge_spends_only_what_it_has():
    """An edge barely above the gate spends exactly that sliver — and keeps it.

    This used to assert the limit sat AT the touch, because `snap_to_tick`
    floored a 0.05c allowance to zero. That is the bug that stopped the loop
    filling: a limit on the touch cannot survive a one-cent move, and three of
    the ten kills on 2026-09-04 had a zero-cent allowance. The sliver is small,
    but it is the trade's own edge and it is not ours to round away.
    """
    d = decision(price=0.40, edge=DEFAULT_CONFIG.min_edge_pp / 100.0 + 0.0005)
    assert order_limit_price(d) == pytest.approx(0.4005, abs=1e-9)


def test_an_edge_at_or_below_the_gate_never_crosses():
    d = decision(price=0.40, edge=DEFAULT_CONFIG.min_edge_pp / 100.0)
    assert order_limit_price(d) == pytest.approx(0.40, abs=1e-9)


def test_a_huge_edge_is_still_capped_in_cents():
    """The cap is a rail against a book so thin the order walks it absurdly."""
    d = decision(price=0.30, edge=0.60)
    allowance = order_limit_price(d) - d.price
    assert allowance == pytest.approx(
        DEFAULT_CONFIG.max_slippage_cents / 100.0, abs=1e-9)


def test_the_limit_never_exceeds_ninety_nine_cents():
    assert order_limit_price(decision(price=0.98, edge=0.60)) <= 0.99
