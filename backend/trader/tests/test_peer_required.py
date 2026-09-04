"""Trading without the peer book trades a model missing most of its edge.

Measured by refitting with the TRAINING peer deliberately lagged:

    peer lag          log-loss skill        t      folds+
    0 min (socket)   +0.00294 +/- 0.00064  +4.57    6/6
    1 min (REST)     +0.00087 +/- 0.00086  +1.01    5/6
    dropped entirely -0.00015              -0.15    4/6

A ONE-MINUTE lag costs 70% of the skill — most of the way from
contemporaneous to absent. So `cross_venue`'s contribution is
CONTEMPORANEITY rather than a durable interaction, which is why it looked
dead alone (+0.000030, below the clock control) and load-bearing under
leave-one-out.

That is not a leak: both books are genuinely readable at the decision instant
over sockets. It is an infrastructure requirement, and it means an outage does
not degrade one feature gracefully — it removes most of the edge while the loop
carries on trading.

`--complete-cases` already requires the peer in TRAINING, so refusing a row
without one live is what makes the two paths agree. Measured over two hours of
live cycles, 0 of 9 would have abstained, so the guard costs nothing while the
socket is healthy — which is the point of a guard.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.decide import Reason, decide


def _row(**over):
    row = {
        'symbol': 'BTC-USD', 'offset': 12,
        'baseline_probability': 0.40, 'model_probability': 0.52,
        'ask_up': 0.44, 'ask_down': 0.58,
        'venue_prob_gap': 0.01,
        'sigma_remaining': 0.0008,
    }
    row.update(over)
    return row


def test_a_row_without_a_peer_gap_is_refused():
    config = Config(require_peer_book=True)
    out = decide(_row(venue_prob_gap=float('nan')), config, bankroll=200.0)
    assert not out.traded
    assert out.reason is Reason.NO_PEER_BOOK


def test_a_row_with_a_peer_gap_is_unaffected():
    config = Config(require_peer_book=True)
    assert decide(_row(), config, bankroll=200.0).reason is not Reason.NO_PEER_BOOK


def test_the_guard_is_off_by_default_so_past_entries_keep_their_meaning():
    """Every ledger entry so far was measured without it. Turning it on
    silently would rewrite what they mean."""
    assert Config().require_peer_book is False
    out = decide(_row(venue_prob_gap=float('nan')), Config(), bankroll=200.0)
    assert out.reason is not Reason.NO_PEER_BOOK


def test_a_row_that_never_carried_the_column_is_not_refused():
    """The column is absent for a configuration that does not use cross_venue
    at all; absence of the feature is not absence of the book."""
    config = Config(require_peer_book=True)
    row = _row()
    row.pop('venue_prob_gap')
    assert decide(row, config, bankroll=200.0).reason is not Reason.NO_PEER_BOOK
