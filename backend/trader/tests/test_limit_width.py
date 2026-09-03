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
    # On the venue's grid: `snap_to_tick` rounds the limit DOWN to a
    # price Kalshi quotes, so an off-grid input here would measure the
    # rounding rather than the allowance.
    d = decision(price=0.75, edge=0.0368)
    allowance = order_limit_price(d) - d.price
    # Snapped down to the whole-cent grid in the mid range.
    assert allowance == pytest.approx(
        __import__('math').floor((0.75 + 0.0368 - gate) * 100) / 100 - 0.75,
        abs=1e-9)
    assert allowance > 0.0092, 'must be wider than the old share-of-edge rule'


def test_a_marginal_edge_is_not_chased():
    """An edge barely above the gate has nothing to spend, and spends nothing."""
    d = decision(price=0.40, edge=DEFAULT_CONFIG.min_edge_pp / 100.0 + 0.0005)
    # A sub-tick allowance cannot be expressed on a 1c grid, so the limit
    # sits at the touch. Rounding UP would breach the bound the allowance
    # exists to set — measured, the venue's own rounding did exactly that
    # on 4 of 143 fills.
    assert order_limit_price(d) == pytest.approx(d.price, abs=1e-9)


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
