"""Tests for which observer's quote wins when several recorded the same minute.

Ages are not comparable across sources, so rows rank by source first, then by age within
a source.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.quotes import QUOTE_SOURCE_PRIORITY, attach_quotes

OPEN = pd.Timestamp('2026-07-01T12:00Z')


def _windows(offset=12):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': OPEN,
                          'offset': offset, 'baseline_probability': 0.40}])


def _row(source, bid, ask, age, offset=12):
    return {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
            'offset_minutes': offset, 'yes_bid': bid, 'yes_ask': ask,
            'quote_age_seconds': age, 'source': source}


def test_the_decision_instant_quote_beats_the_reconstruction():
    """The live_touch quote beats the backfill, even though the backfill has the smaller age."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live_touch', 0.60, 0.62, np.nan),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61
    assert out['ask_up'].iloc[0] == 0.62


def test_a_recorder_seen_before_the_decision_beats_the_reconstruction():
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live', 0.60, 0.62, 2.0),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61


def test_a_recorder_seen_after_the_decision_never_prices_it():
    """A recorder's age counts forward from the minute; at 29s it saw the book after the
    decision, so the backfill prices the row.
    """
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live', 0.60, 0.62, 29.0),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_the_priority_order_is_honoured_between_the_two_live_recorders():
    depth = pd.DataFrame([
        _row('live', 0.30, 0.32, 1.0),
        _row('live_ws', 0.60, 0.62, 2.0),
    ])
    out = attach_quotes(_windows(), depth)
    assert QUOTE_SOURCE_PRIORITY.index('live_ws') < \
        QUOTE_SOURCE_PRIORITY.index('live')
    assert out['market_probability'].iloc[0] == 0.61


def test_an_unusable_preferred_row_does_not_displace_a_usable_one():
    """An unusable preferred row (crossed or stale) does not displace a usable one.

    Unusable rows sort last, so a source preference cannot turn into data loss.
    """
    depth = pd.DataFrame([
        _row('live_touch', 0.80, 0.20, np.nan),  # crossed: ask below bid
        _row('backfill', 0.30, 0.32, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_a_stale_preferred_row_does_not_displace_a_fresh_one():
    depth = pd.DataFrame([
        _row('live_ws', 0.80, 0.82, 4000.0),  # far past max_age_seconds
        _row('backfill', 0.30, 0.32, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_age_still_breaks_ties_within_one_source():
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 20.0),
        _row('backfill', 0.60, 0.62, 0.5),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.61


def test_an_unknown_source_ranks_behind_every_known_one():
    depth = pd.DataFrame([
        _row('some_future_recorder', 0.30, 0.32, 0.0),
        _row('backfill', 0.60, 0.62, 900.0 - 1),
    ])
    out = attach_quotes(_windows(), depth, max_age_seconds=1000.0)
    assert out['market_probability'].iloc[0] == 0.61


def test_a_depth_frame_with_no_source_column_still_prices():
    """A depth frame without a `source` column still prices."""
    depth = pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)]).drop(
        columns=['source'])
    out = attach_quotes(_windows(), depth)
    assert out['market_probability'].iloc[0] == 0.31


def test_the_winning_observer_is_recorded_on_the_row():
    """The winning observer is recorded in `quote_observer`; `quote_source` holds the venue."""
    depth = pd.DataFrame([
        _row('backfill', 0.30, 0.32, 0.5),
        _row('live_touch', 0.60, 0.62, np.nan),
    ])
    out = attach_quotes(_windows(), depth)
    assert out['quote_source'].iloc[0] == 'kalshi'  # the venue
    assert out['quote_observer'].iloc[0] == 'live_touch'  # the observer


def test_an_unpriced_window_names_no_observer():
    out = attach_quotes(_windows(offset=3),
                        pd.DataFrame([_row('backfill', 0.30, 0.32, 0.5)]))
    assert pd.isna(out['market_probability'].iloc[0])
    assert out['quote_observer'].iloc[0] is None or \
        pd.isna(out['quote_observer'].iloc[0])
