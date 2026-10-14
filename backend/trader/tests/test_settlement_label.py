"""Tests that the backtest settles on the venue's result, with our label as fallback.

Our Coinbase label differs from the venue's BRTI in near-ties, which a model could
exploit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest import settlement_outcomes


def _frame(outcome, venue=None):
    f = pd.DataFrame({
        'symbol': ['BTC-USD', 'BTC-USD'],
        'window_open': pd.to_datetime(['2026-07-01T12:00Z', '2026-07-01T12:15Z']),
        'outcome': outcome,
    })
    if venue is not None:
        f['venue_outcome'] = venue
    return f


def test_the_venue_result_wins_where_we_hold_it():
    got = settlement_outcomes(_frame([1, 0], venue=[0.0, 1.0]))
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:00Z'))] is False
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:15Z'))] is True


def test_our_label_fills_in_where_the_venue_has_none():
    """Our label fills windows the venue has no result for, since Kalshi purges older markets."""
    got = settlement_outcomes(_frame([1, 0], venue=[np.nan, 1.0]))
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:00Z'))] is True
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:15Z'))] is True


def test_a_frame_without_the_venue_column_still_settles():
    got = settlement_outcomes(_frame([1, 0]))
    assert set(got.values()) == {True, False}


def test_a_window_with_neither_label_is_absent_rather_than_guessed():
    """A window with neither label is absent, so it is not booked as a win or a loss."""
    got = settlement_outcomes(_frame([np.nan, 0], venue=[np.nan, np.nan]))
    assert ('BTC-USD', pd.Timestamp('2026-07-01T12:00Z')) not in got
    assert len(got) == 1
