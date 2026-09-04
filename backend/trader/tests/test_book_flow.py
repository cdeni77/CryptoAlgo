"""Book DYNAMICS: how the ladder is changing, not how it stands.

`market_state` reads the book as a snapshot at the decision instant — spread,
imbalance, depth ratios. Nothing reads the direction it is moving, and
`venue_depth` carries a row at every minute 0..15 (16,345 of 18,321 windows have
ten or more), so the history is there and unused.

This is the one mechanism the archive left open rather than rejected:

    "Book imbalance: still fails, but it moved. Daily corr went +0.0074
     (t=+0.30) to +0.0253 (t=+1.84) over 67 days as the sample grew. Does not
     clear the bar the trade tape failed at, but unlike the tape it is positive
     and strengthening. Re-test when the backfill completes."

The trade tape IS rejected — sixteen cells, not one significant, largest
t=-1.90 on 1,800 windows — and nothing here touches it.

**The lookahead guard is the whole risk.** A decision at +12m may read minutes
0..12 and nothing after. A one-minute leak in a fifteen-minute window is 7% of
the question and reads exactly like skill.
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
    """The invariant. Minutes 13 and 14 carry a wild imbalance swing; a decision
    at +12m must be identical with and without them."""
    early = _depth(list(range(13)), [100] * 13, [100] * 13)
    late = pd.concat([early, _depth([13, 14], [1, 1], [9999, 9999])],
                     ignore_index=True)
    a = book_flow_features(_windows(12), early).iloc[0]
    b = book_flow_features(_windows(12), late).iloc[0]
    for col in BOOK_FLOW:
        assert (pd.isna(a[col]) and pd.isna(b[col])) or a[col] == pytest.approx(b[col]), (
            f'{col} changed when minutes after the decision were added')


def test_imbalance_change_reads_the_direction_the_book_moved():
    """Bid size doubling while the ask holds is buying pressure building."""
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
    """Zero would say 'the book did not move', which is a claim. Absent is
    absent — the model was fitted with these missing on some rows."""
    row = book_flow_features(_windows(12), _depth([12], [100], [100])).iloc[0]
    assert np.isnan(row['imbalance_change_3'])


def test_the_other_venue_is_not_mixed_in():
    """A Polymarket ladder in a Kalshi flow feature is the `no_levels`
    denomination trap wearing a different name."""
    k = _depth(list(range(13)), [100] * 13, [100] * 13)
    pm = _depth(list(range(13)), [9999] * 13, [1] * 13)
    pm['venue'] = 'polymarket'
    both = pd.concat([k, pm], ignore_index=True)
    a = book_flow_features(_windows(12), k).iloc[0]
    b = book_flow_features(_windows(12), both).iloc[0]
    for col in BOOK_FLOW:
        assert (pd.isna(a[col]) and pd.isna(b[col])) or a[col] == pytest.approx(b[col])
