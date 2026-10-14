"""Tests that `venue_gap_change_5` means the same thing in training and live.

Both use the configured offset grid, since live cannot see which rows training dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import gap_change_column

OPEN = pd.Timestamp('2026-07-01T12:00Z')


def _table(offsets_and_gaps):
    return pd.DataFrame([
        {'symbol': 'BTC-USD', 'window_open': OPEN, 'offset': o,
         'venue_prob_gap': g, 'pm_market_probability': 0.5,
         'market_probability': 0.5, 'baseline_probability': 0.5}
        for o, g in offsets_and_gaps])


def _gap_change(offsets_and_gaps, decision_offsets=(3, 6, 9, 12)):
    table = _table(offsets_and_gaps)
    return dict(zip(table['offset'],
                    gap_change_column(table, decision_offsets)))


def test_a_complete_window_differences_one_offset():
    got = _gap_change([(3, 0.01), (6, 0.03), (9, 0.06), (12, 0.10)])
    assert np.isnan(got[3])
    assert got[6] == pytest.approx(0.02)
    assert got[9] == pytest.approx(0.03)
    assert got[12] == pytest.approx(0.04)


def test_a_missing_middle_offset_yields_NaN_not_a_longer_step():
    got = _gap_change([(3, 0.01), (6, 0.03), (12, 0.10)])
    assert np.isnan(got[12]), (
        'the previous CONFIGURED offset is absent, so there is no one-offset '
        'change to report'
    )
    assert got[6] == pytest.approx(0.02), 'the intact step still works'


def test_the_first_offset_is_always_NaN():
    got = _gap_change([(3, 0.01), (6, 0.03)])
    assert np.isnan(got[3])


def test_a_NaN_gap_upstream_propagates():
    got = _gap_change([(3, 0.01), (6, float('nan')), (9, 0.06)])
    assert np.isnan(got[6])
    assert np.isnan(got[9]), 'differencing against a NaN is not a number'


def test_it_follows_the_configured_grid_not_the_rows_present():
    """The step follows the configured grid, so on a (3, 12) policy it spans 9 minutes."""
    got = _gap_change([(3, 0.01), (12, 0.10)], decision_offsets=(3, 12))
    assert got[12] == pytest.approx(0.09)


def test_build_features_uses_the_helper():
    """`build_features` uses `gap_change_column` instead of an inline `shift(1)`."""
    import inspect

    from core.features import build_features

    src = inspect.getsource(build_features)
    assert 'gap_change_column(' in src
    assert "['venue_prob_gap'].shift(1)" not in src
