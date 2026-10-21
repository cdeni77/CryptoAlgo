"""Tests that the market gates measure the candidate, not the deployed model.

The candidate's own recorded-quote rows come first; the live `predictions` table is the
fallback.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from core.metrics import LIVE_OBSERVERS, market_rows_from_scored


def _scored(observers, n_each=3):
    rows = []
    for i, obs in enumerate(observers):
        for k in range(n_each):
            rows.append({
                'symbol': 'BTC-USD',
                'window_open': pd.Timestamp('2026-09-01T12:00Z')
                + pd.Timedelta(minutes=15 * (i * n_each + k)),
                'offset': 12, 'market_probability': 0.55,
                'baseline_probability': 0.50, 'model_probability': 0.60,
                'outcome': 1.0, 'quote_observer': obs,
            })
    return pd.DataFrame(rows)


def test_recorded_only_keeps_the_watched_observers():
    frame = _scored(['live_touch', 'live_ws', 'live', 'backfill'])
    rows = market_rows_from_scored(frame, recorded_only=True)
    assert len(rows) == 9, 'three live observers at three rows each'


def test_recorded_only_drops_the_reconstruction():
    frame = _scored(['backfill'])
    assert market_rows_from_scored(frame, recorded_only=True) == []
    # It is still a valid backtest price when not restricted.
    assert len(market_rows_from_scored(frame, recorded_only=False)) == 3


def test_a_frame_with_no_observer_column_is_refused_not_assumed():
    """A frame without an observer column is refused, not assumed to be recorded."""
    frame = _scored(['live_touch']).drop(columns=['quote_observer'])
    assert market_rows_from_scored(frame, recorded_only=True) == []


def test_every_live_observer_is_a_real_source():
    from core.quotes import QUOTE_SOURCE_PRIORITY

    for name in LIVE_OBSERVERS:
        assert name in QUOTE_SOURCE_PRIORITY
    assert 'backfill' not in LIVE_OBSERVERS


def test_the_candidate_is_preferred_over_the_live_table():
    """`market_measurement` tries the candidate's recorded rows before the live table."""
    from scripts.promote import market_measurement

    src = inspect.getsource(market_measurement)
    candidate_at = src.index('recorded_only=True')
    live_at = src.index('scored_against_market()')
    assert candidate_at < live_at, (
        "the candidate's own rows must be tried before the deployed system's "
        'live record, or a new model is gated on its predecessor'
    )


def test_the_live_table_remains_the_fallback():
    """The live table remains the fallback when the candidate has no recorded quotes."""
    from scripts.promote import market_measurement

    src = inspect.getsource(market_measurement)
    assert 'scored_against_market()' in src
    assert 'cannot influence' in src, 'the fallback must say what it is'
