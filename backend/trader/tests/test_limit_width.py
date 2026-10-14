"""Tests for `order_limit_price`: how far a limit may cross.

The limit may spend the edge above `min_edge_pp`, so any fill still clears the gate. A
cents cap still applies.
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
    """A 3.68pp edge against a 1.5pp gate may spend 2.18c."""
    gate = DEFAULT_CONFIG.min_edge_pp / 100.0
    # The allowance is spent exactly, not rounded onto the venue grid here.
    # `place_order` rounds a bid up to the tick, so the limit stays fillable.
    d = decision(price=0.75, edge=0.0368)
    allowance = order_limit_price(d) - d.price
    assert allowance == pytest.approx(0.0368 - gate, abs=1e-9)
    assert allowance > 0.0092, 'must be wider than the old share-of-edge rule'


def test_a_marginal_edge_spends_only_what_it_has():
    """An edge just above the gate spends exactly that small amount."""
    d = decision(price=0.40, edge=DEFAULT_CONFIG.min_edge_pp / 100.0 + 0.0005)
    assert order_limit_price(d) == pytest.approx(0.4005, abs=1e-9)


def test_an_edge_at_or_below_the_gate_never_crosses():
    d = decision(price=0.40, edge=DEFAULT_CONFIG.min_edge_pp / 100.0)
    assert order_limit_price(d) == pytest.approx(0.40, abs=1e-9)


def test_a_huge_edge_is_still_capped_in_cents():
    """`max_slippage_cents` caps a large edge, as a guard against a very thin book."""
    d = decision(price=0.30, edge=0.60)
    allowance = order_limit_price(d) - d.price
    assert allowance == pytest.approx(
        DEFAULT_CONFIG.max_slippage_cents / 100.0, abs=1e-9)


def test_the_limit_never_exceeds_ninety_nine_cents():
    assert order_limit_price(decision(price=0.98, edge=0.60)) <= 0.99
