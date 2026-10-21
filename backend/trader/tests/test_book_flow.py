"""Tests for `book_flow`: how the book changes over the window, not its state at one
instant.

The main risk is lookahead. A decision at +12m may read minutes 0..12 and nothing after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.book_flow import BOOK_FLOW, book_flow_features


def _depth(offsets, bid_size, ask_size, bid=0.50, ask=0.52):
    return pd.DataFrame({
        'venue': 'kalshi', 'symbol': 'BTC-USD',
        'window_open': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
        'offset_minutes': offsets,
        'yes_bid_size': bid_size, 'yes_ask_size': ask_size,
        'yes_bid': bid, 'yes_ask': ask,
    })


def _windows(offset=12):
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'window_open': [pd.Timestamp('2026-09-03 12:00', tz='UTC')],
        'offset': [offset],
    })


def test_a_decision_never_reads_a_minute_after_itself():
    """A decision at +12m is unchanged when minutes 13 and 14 are added."""
    early = _depth(list(range(13)), [100] * 13, [100] * 13)
    late = pd.concat([early, _depth([13, 14], [1, 1], [9999, 9999])],
                     ignore_index=True)
    a = book_flow_features(_windows(12), early).iloc[0]
    b = book_flow_features(_windows(12), late).iloc[0]
    for col in BOOK_FLOW:
        assert (pd.isna(a[col]) and pd.isna(b[col])) or a[col] == pytest.approx(b[col]), (
            f'{col} changed when minutes after the decision were added')


def test_imbalance_change_reads_the_direction_the_book_moved():
    """Bid size rising while the ask holds gives a positive imbalance change."""
    off = list(range(13))
    bids = [100] * 9 + [150, 200, 250, 300]
    row = book_flow_features(_windows(12), _depth(off, bids, [100] * 13)).iloc[0]
    assert row['imbalance_change_3'] > 0


def test_a_draining_book_reads_negative_depth_trend():
    off = list(range(13))
    draining = [400, 400, 400, 400, 400, 400, 400, 400, 400, 300, 200, 120, 80]
    row = book_flow_features(_windows(12), _depth(off, draining, draining)).iloc[0]
    assert row['depth_trend_3'] < 0


def test_a_window_with_no_history_is_nan_not_zero():
    """A window with no history gives NaN, not zero, since zero claims the book did not move."""
    row = book_flow_features(_windows(12), _depth([12], [100], [100])).iloc[0]
    assert np.isnan(row['imbalance_change_3'])


def test_the_other_venue_is_not_mixed_in():
    """Polymarket rows never enter a Kalshi flow feature."""
    k = _depth(list(range(13)), [100] * 13, [100] * 13)
    pm = _depth(list(range(13)), [9999] * 13, [1] * 13)
    pm['venue'] = 'polymarket'
    both = pd.concat([k, pm], ignore_index=True)
    a = book_flow_features(_windows(12), k).iloc[0]
    b = book_flow_features(_windows(12), both).iloc[0]
    for col in BOOK_FLOW:
        assert (pd.isna(a[col]) and pd.isna(b[col])) or a[col] == pytest.approx(b[col])


# --------------------------------------------------------------------------
# the mid's own revision
#
# The lookahead test above sweeps every column in BOOK_FLOW, so it covers these
# too — but `_depth` holds bid and ask fixed, so the mid never moves and that
# sweep would pass on a feature that read the future. These move it.
# --------------------------------------------------------------------------

def test_a_mid_revision_after_the_decision_is_not_read():
    """The invariant, for the mid specifically. The market re-prices violently
    in minutes 13-14; a decision at +12m must not see a cent of it."""
    offs = list(range(13))
    early = _depth(offs, [100] * 13, [100] * 13,
                   bid=[0.50] * 13, ask=[0.52] * 13)
    late = pd.concat(
        [early, _depth([13, 14], [100, 100], [100, 100],
                       bid=[0.90, 0.95], ask=[0.92, 0.97])],
        ignore_index=True)
    a = book_flow_features(_windows(12), early).iloc[0]
    b = book_flow_features(_windows(12), late).iloc[0]
    for col in ('mid_change_1', 'mid_change_3'):
        assert (pd.isna(a[col]) and pd.isna(b[col])) or \
            a[col] == pytest.approx(b[col]), f'{col} read a minute after the decision'


def test_mid_change_reads_the_direction_the_market_revised():
    """Mid walks 51c -> 61c over the last three minutes. Both horizons report
    the move, in cents, and the three-minute one reports more of it."""
    bid = [0.50] * 10 + [0.54, 0.58, 0.60]
    ask = [0.52] * 10 + [0.56, 0.60, 0.62]
    row = book_flow_features(
        _windows(12), _depth(list(range(13)), [100] * 13, [100] * 13,
                             bid=bid, ask=ask)).iloc[0]
    assert row['mid_change_1'] == pytest.approx(61.0 - 59.0)
    assert row['mid_change_3'] == pytest.approx(61.0 - 51.0)


def test_a_falling_mid_reads_negative():
    bid = [0.60] * 10 + [0.56, 0.52, 0.50]
    ask = [0.62] * 10 + [0.58, 0.54, 0.52]
    row = book_flow_features(
        _windows(12), _depth(list(range(13)), [100] * 13, [100] * 13,
                             bid=bid, ask=ask)).iloc[0]
    assert row['mid_change_3'] < 0


def test_a_one_sided_book_has_no_mid_and_reports_nan_not_zero():
    """Zero would say "the market did not move", which is a claim. A missing
    side is not a mid."""
    row = book_flow_features(
        _windows(12), _depth(list(range(13)), [100] * 13, [100] * 13,
                             bid=[np.nan] * 13, ask=[0.52] * 13)).iloc[0]
    assert pd.isna(row['mid_change_1'])
    assert pd.isna(row['mid_change_3'])


def test_the_short_horizon_reaches_an_offset_the_three_minute_one_cannot():
    """At +2m there is no minute -1, so every three-minute column is NaN. The
    measured skill in this project peaks at the EARLIEST offset, which is
    exactly where the three-minute family cannot speak."""
    row = book_flow_features(
        _windows(2), _depth([0, 1, 2], [100] * 3, [100] * 3,
                            bid=[0.50, 0.52, 0.54], ask=[0.52, 0.54, 0.56])).iloc[0]
    assert pd.isna(row['mid_change_3'])
    assert pd.isna(row['imbalance_change_3'])
    assert row['mid_change_1'] == pytest.approx(55.0 - 53.0)
