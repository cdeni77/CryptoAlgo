"""Lagging the TRAINING peer to the live tolerance, to settle one question.

`cross_venue` is the only load-bearing group in the model — leave-one-out takes
skill from +0.00282 to -0.00015 — while scoring +0.000030 ALONE, below the clock
control. Useless alone and essential combined is what a genuine interaction
looks like AND what a timing leak looks like, and the pattern cannot separate
them.

Measured, training's gap is a sub-second-simultaneous disagreement (median
|age difference| 0.56s, sd 7.4s, 97% backfill/backfill pairs) while live admits
a peer up to 30s stale — and a stale peer does not add noise, it converts the
feature into Kalshi's own recent move (the induced error correlates 0.724 with
Kalshi's 60s return).

So: refit with the training peer deliberately lagged. If skill survives, the
interaction is real and live merely needs to keep the socket healthy. If it
collapses toward the ablated -0.00015, then what `cross_venue` contributes is
CONTEMPORANEITY — a property live can only supply when the socket is up, and the
model's headline skill is partly an artifact of an alignment production cannot
guarantee.

`peer_lag_minutes` defaults to 0 so nothing changes unless the experiment asks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.quotes import attach_quotes


def _depth(offset, pm_bid, pm_ask):
    common = {
        'symbol': 'BTC-USD',
        'window_open': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
        'event_time': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
        'available_time': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
        'quote_age_seconds': 1.0,
        'yes_bid_size': 100.0, 'yes_ask_size': 100.0,
        'depth_bid_1c': 100.0, 'depth_ask_1c': 100.0,
        'depth_bid_5c': 100.0, 'depth_ask_5c': 100.0,
        'depth_bid_total': 100.0, 'depth_ask_total': 100.0,
    }
    return [
        {**common, 'venue': 'kalshi', 'offset_minutes': offset,
         'yes_bid': 0.50, 'yes_ask': 0.52},
        {**common, 'venue': 'polymarket', 'offset_minutes': offset,
         'yes_bid': pm_bid, 'yes_ask': pm_ask},
    ]


def _windows(offset=12):
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'window_open': [pd.Timestamp('2026-09-03 12:00', tz='UTC')],
        'offset': [offset]})


def test_no_lag_uses_the_contemporaneous_peer():
    rows = _depth(11, 0.30, 0.32) + _depth(12, 0.40, 0.42)
    out = attach_quotes(_windows(12), pd.DataFrame(rows), peer_lag_minutes=0)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.41)


def test_a_one_minute_lag_takes_the_previous_minutes_peer():
    """The live condition: Kalshi now, Polymarket a minute ago."""
    rows = _depth(11, 0.30, 0.32) + _depth(12, 0.40, 0.42)
    out = attach_quotes(_windows(12), pd.DataFrame(rows), peer_lag_minutes=1)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.31)


def test_the_traded_venue_is_never_lagged():
    """Only the PEER moves. Lagging Kalshi too would test something else
    entirely — the decision must still price against the book it could have
    taken."""
    rows = _depth(11, 0.30, 0.32) + _depth(12, 0.40, 0.42)
    out = attach_quotes(_windows(12), pd.DataFrame(rows), peer_lag_minutes=1)
    assert out['ask_up'].iloc[0] == pytest.approx(0.52)


def test_a_lag_with_no_earlier_row_gives_no_peer():
    """Absent, not carried forward — the point is to remove the peer's
    contemporaneity, not to invent a substitute."""
    out = attach_quotes(_windows(12), pd.DataFrame(_depth(12, 0.40, 0.42)),
                        peer_lag_minutes=1)
    assert np.isnan(out['pm_market_probability'].iloc[0])


def test_the_default_is_zero_so_nothing_changes_unless_asked():
    assert Config().peer_lag_minutes == 0
