"""A maker order must rest at the bid and be guaranteed not to take.

Measured 2026-10-25: `order_limit_price` added a slippage allowance that
scales with the claimed edge, on top of a price that under `maker_entry` is
already the BID. At a claimed 0.56pp the allowance was 0.06c and 83% of fills
rested; at 1.99pp it was 1.49c -- through a ~1c spread -- and the maker share
fell to 37%. The allowance bounds how far a TAKER walks the book; a resting
order walks nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.decide import Decision, Reason, Side
from livetrade.execution import order_limit_price

W = pd.Timestamp('2026-10-25 00:00', tz='UTC')


def _decision(price: float, edge: float) -> Decision:
    return Decision(
        symbol='BTC-USD', window_open=W, settle_time=W + pd.Timedelta(minutes=15),
        offset=3, side=Side.UP, reason=Reason.TRADED, contracts=5,
        price=price, edge=edge, model_probability=price + edge,
        baseline_probability=price, effective_cost=price, stake=1.0, fee=0.0)


@pytest.mark.parametrize('edge_pp', [0.6, 1.2, 2.0, 5.0])
def test_a_resting_order_prices_at_the_bid_whatever_the_edge(edge_pp):
    """The failure was edge-dependent: the better the model thought the trade
    was, the further through the ask it priced."""
    config = Config(maker_entry=True, min_edge_pp=0.5, max_slippage_cents=3.0)
    limit = order_limit_price(_decision(0.68, edge_pp / 100.0), config=config)
    assert limit == pytest.approx(0.68), (
        f'a resting order must sit at the bid; got {limit} at {edge_pp}pp')


def test_a_taker_order_still_gets_its_allowance():
    """The allowance is correct for crossing; only the maker path was wrong."""
    config = Config(maker_entry=False, min_edge_pp=0.5, max_slippage_cents=3.0)
    limit = order_limit_price(_decision(0.68, 0.0200), config=config)
    assert limit > 0.68
    assert limit == pytest.approx(0.68 + 0.015, abs=1e-9)


def test_the_taker_allowance_is_still_capped_by_max_slippage():
    config = Config(maker_entry=False, min_edge_pp=0.5, max_slippage_cents=1.0)
    limit = order_limit_price(_decision(0.68, 0.0500), config=config)
    assert limit == pytest.approx(0.68 + 0.01, abs=1e-9)


def test_post_only_is_sent_only_when_resting():
    """`self_trade_prevention_type` governs SELF matches and cannot keep an
    order off the public book; post_only is the guarantee."""
    import inspect

    from data_collection import kalshi_client
    src = inspect.getsource(kalshi_client.KalshiClient.place_order)
    assert "'post_only': True} if rest else {}" in src, (
        'a resting order must be sent post_only')


def test_post_only_is_sent_when_resting_and_not_when_crossing():
    import inspect

    from data_collection import kalshi_client
    src = inspect.getsource(kalshi_client.KalshiClient.place_order)
    assert "**({'post_only': True} if rest else {})" in src, (
        'post_only must be sent only on a resting order')


def test_a_post_only_refusal_is_an_abstention_not_an_error():
    """The venue refuses a crossing post-only order with
    `400 invalid_order: post only cross`. That is the guarantee working: the
    book moved inside the ~0.4s to the venue and the order would have TAKEN.
    Before post_only it filled anyway and paid the fee and the spread."""
    import inspect

    from livetrade import execution
    src = inspect.getsource(execution)
    assert "'post only cross' in str(exc).lower()" in src, (
        'a post-only refusal must be recognised')
    assert "'post_only_cross'" in src, (
        'it must be reported as its own outcome, not as a generic refusal')
