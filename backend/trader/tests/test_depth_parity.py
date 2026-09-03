"""`decide()` caps the stake on measured depth — but only live could supply it.

Wiring `depth_up`/`depth_down` into `scripts/live.py` closed one gap and opened
another: the live path began sizing against the book while the backtest, which
calls the SAME `decide()`, still sized against `max_stake_dollars` alone. "One
decide()" is the invariant that stops the two from drifting, and a cap that
applies to only one of them breaks it exactly where it matters — at size, where
the backtest would claim fills the book could not support.

`venue_depth` already carries what is needed (`yes_bid_size`, `yes_ask_size` via
DEPTH_MAP), so this is a projection, not new data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.quotes import attach_quotes


def _windows():
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'window_open': [pd.Timestamp('2026-09-02 12:00', tz='UTC')],
        'offset': [12],
    })


def _depth(bid=0.40, ask=0.42, bid_size=100.0, ask_size=80.0):
    return pd.DataFrame({
        'venue': ['kalshi'], 'symbol': ['BTC-USD'],
        'window_open': [pd.Timestamp('2026-09-02 12:00', tz='UTC')],
        'offset_minutes': [12],
        'event_time': [pd.Timestamp('2026-09-02 12:12', tz='UTC')],
        'available_time': [pd.Timestamp('2026-09-02 12:12', tz='UTC')],
        'quote_age_seconds': [2.0],
        'yes_bid': [bid], 'yes_ask': [ask],
        'yes_bid_size': [bid_size], 'yes_ask_size': [ask_size],
        'depth_bid_1c': [bid_size], 'depth_ask_1c': [ask_size],
        'depth_bid_5c': [bid_size], 'depth_ask_5c': [ask_size],
        'depth_bid_total': [bid_size], 'depth_ask_total': [ask_size],
    })


def test_the_backtest_carries_the_same_depth_cap_live_does():
    out = attach_quotes(_windows(), _depth())
    assert 'depth_up' in out.columns and 'depth_down' in out.columns


def test_depth_is_dollars_on_the_side_being_crossed():
    """Buying UP pays the YES ask against the size resting there; buying DOWN
    pays 1 - yes_bid against the YES bid stack. Same convention as
    `scripts.live.depth_dollars`."""
    out = attach_quotes(_windows(), _depth(bid=0.40, ask=0.42,
                                           bid_size=100.0, ask_size=80.0))
    row = out.iloc[0]
    assert row['depth_up'] == pytest.approx(80.0 * 0.42, rel=1e-6)
    assert row['depth_down'] == pytest.approx(100.0 * (1.0 - 0.40), rel=1e-6)


def test_a_row_with_no_size_gives_nan_not_zero():
    """Zero would refuse every trade. Unmeasured means fall back to the standing
    cap, which is what the backtest did before the book was joined."""
    d = _depth()
    d['yes_bid_size'] = np.nan
    d['yes_ask_size'] = np.nan
    row = attach_quotes(_windows(), d).iloc[0]
    assert np.isnan(row['depth_up']) and np.isnan(row['depth_down'])
