"""Tests that one offset is not decided twice in the same window.

`client_order_id` is fixed per (symbol, window, offset), so a second attempt
always fails with `order_already_exists`. Both the cadence cycle and the decision
planner can start a cycle, so `mark_decided` records every decided offset.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.decide import Decision, Reason, Side
from scripts.live import mark_decided, seconds_until_next_decision

WINDOW = pd.Timestamp('2026-08-26 02:00', tz='UTC')


class Args:
    cycle_seconds = 60


def decision_at(offset: int, *, window: pd.Timestamp = WINDOW) -> Decision:
    return Decision(
        symbol='BTC-USD', window_open=window,
        settle_time=window + timedelta(minutes=15), offset=offset,
        reason=Reason.TRADED, side=Side.UP, price=0.50, effective_cost=0.52,
        model_probability=0.56, baseline_probability=0.50, edge=0.04,
        contracts=5, stake=2.5, fee=0.02, price_source='quote',
        market_ticker='KXBTC15M-26AUG252215-15')


def test_an_offset_a_cycle_already_decided_does_not_fire_again():
    """A cadence cycle at :05 that decided +12m stops the planner refiring at :10."""
    fired: set = set()
    at_05 = datetime(2026, 8, 26, 2, 12, 5, tzinfo=timezone.utc)
    mark_decided(fired, [decision_at(12)])
    delay = seconds_until_next_decision(
        DEFAULT_CONFIG, Args(), now=at_05, already_fired=fired)
    assert delay > 1.0, 'the +12m target was already decided; it must not refire'


def test_an_offset_nobody_has_decided_still_fires_immediately():
    """An offset nobody has decided still fires immediately."""
    at_05 = datetime(2026, 8, 26, 2, 12, 5, tzinfo=timezone.utc)
    delay = seconds_until_next_decision(
        DEFAULT_CONFIG, Args(), now=at_05, already_fired=set())
    assert delay == pytest.approx(0.5)


def test_marking_is_keyed_on_the_window_as_well_as_the_offset():
    """+12m of the next window is a different bet and still fires."""
    fired: set = set()
    mark_decided(fired, [decision_at(12)])
    assert (WINDOW, 12) in fired
    assert (WINDOW + timedelta(minutes=15), 12) not in fired


def test_a_refused_decision_still_counts_as_having_fired():
    """A refused decision still counts as fired, since a retry would reuse the same
    client_order_id.
    """
    fired: set = set()
    refused = decision_at(12)
    object.__setattr__(refused, 'reason', Reason.EDGE_BELOW_GATE)
    assert not refused.traded
    mark_decided(fired, [refused])
    assert (WINDOW, 12) in fired
