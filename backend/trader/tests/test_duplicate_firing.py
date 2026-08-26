"""Two orders for one offset, five seconds apart, and the second always 409s.

Read off the live account 2026-08-25/26. Seventeen of forty-two order attempts
were refused `order_already_exists`, and every one of them came in a pair at
:05 and :10 past the same minute:

    19:12:05  placing down 6 @ 17c  ->  fill_or_kill_insufficient_resting_volume
    19:12:10  placing down 4 @ 14c  ->  order_already_exists

`client_order_id` is deterministic per (symbol, window, offset) — deliberately,
so a POST that times out in flight is deduplicated by the venue rather than
double-filled. So the second attempt could never do anything but collide.

The cause is that two different things can start a cycle. The ordinary
`--cycle-seconds 60` cadence lands at :05, which is already inside
`DECISION_TOLERANCE_SECONDS = 15` of the +12m mark, so that cycle decides and
orders. `seconds_until_next_decision` then fires the *same* offset again,
because `already_fired` only ever recorded targets the planner itself
scheduled — never the cycle that actually traded.

`already_entered` does not save you either: the first attempt books no position
when it fails, so the offset is legitimately retryable as far as exposure is
concerned.
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
    """The exact live sequence: cadence cycle at :05, planner refire at :10."""
    fired: set = set()
    at_05 = datetime(2026, 8, 26, 2, 12, 5, tzinfo=timezone.utc)
    mark_decided(fired, [decision_at(12)])
    delay = seconds_until_next_decision(
        DEFAULT_CONFIG, Args(), now=at_05, already_fired=fired)
    assert delay > 1.0, 'the +12m target was already decided; it must not refire'


def test_an_offset_nobody_has_decided_still_fires_immediately():
    """The guard must not suppress a target that genuinely has not run."""
    at_05 = datetime(2026, 8, 26, 2, 12, 5, tzinfo=timezone.utc)
    delay = seconds_until_next_decision(
        DEFAULT_CONFIG, Args(), now=at_05, already_fired=set())
    assert delay == pytest.approx(0.5)


def test_marking_is_keyed_on_the_window_as_well_as_the_offset():
    """+12m of the NEXT window is a different bet and must still fire."""
    fired: set = set()
    mark_decided(fired, [decision_at(12)])
    assert (WINDOW, 12) in fired
    assert (WINDOW + timedelta(minutes=15), 12) not in fired


def test_a_refused_decision_still_counts_as_having_fired():
    """The offset was evaluated. Re-deciding it sends the same client_order_id.

    A refusal is not a reason to try the identical order again five seconds
    later — that is precisely the 409 pair this exists to stop.
    """
    fired: set = set()
    refused = decision_at(12)
    object.__setattr__(refused, 'reason', Reason.EDGE_BELOW_GATE)
    assert not refused.traded
    mark_decided(fired, [refused])
    assert (WINDOW, 12) in fired
