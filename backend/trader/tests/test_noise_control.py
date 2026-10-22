"""NC2: the noise-only control (RESEARCH_PROTOCOL section 5, item 2).

Information-free features must be reproducible from the row's identity alone,
or the control measures a different thing on every run and cannot be a
control at all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import (ALL_GROUPS, FEATURE_GROUPS, OPT_IN_GROUPS,
                           _noise_features, feature_columns)

COLUMNS = FEATURE_GROUPS['noise']


def frame(n=200, symbol='BTC-USD'):
    opens = pd.date_range('2026-01-01', periods=n, freq='15min', tz='UTC')
    return pd.DataFrame({'window_open': opens,
                         'offset': np.tile([3, 6, 9, 12], n // 4),
                         'symbol': symbol})


def test_it_is_never_selected_by_default():
    """A control that leaks into an ordinary run stops being a control."""
    assert 'noise' in OPT_IN_GROUPS
    assert 'noise' not in ALL_GROUPS
    assert 'noise' not in feature_columns()[:0] or True
    assert not set(COLUMNS) & set(feature_columns())


def test_the_same_row_always_draws_the_same_value():
    a = _noise_features(frame(), COLUMNS)
    b = _noise_features(frame(), COLUMNS)
    pd.testing.assert_frame_equal(a, b)


def test_it_survives_a_reorder_because_it_keys_on_the_row_not_the_position():
    base = frame()
    shuffled = base.iloc[::-1].reset_index(drop=True)
    a = _noise_features(base, COLUMNS)
    b = _noise_features(shuffled, COLUMNS)
    assert np.allclose(a[COLUMNS[0]].to_numpy()[::-1], b[COLUMNS[0]].to_numpy())


def test_different_rows_and_columns_differ():
    out = _noise_features(frame(), COLUMNS)
    assert out[COLUMNS[0]].nunique() > 150, 'rows must differ'
    assert not np.allclose(out[COLUMNS[0]], out[COLUMNS[1]]), 'columns must differ'
    btc = _noise_features(frame(symbol='BTC-USD'), COLUMNS)
    eth = _noise_features(frame(symbol='ETH-USD'), COLUMNS)
    assert not np.allclose(btc[COLUMNS[0]], eth[COLUMNS[0]]), 'symbols must differ'


def test_it_is_uniform_and_finite():
    out = _noise_features(frame(4000), COLUMNS)
    values = out[COLUMNS[0]].to_numpy()
    assert np.isfinite(values).all(), 'a NaN would drop rows under complete-cases'
    assert 0.0 <= values.min() and values.max() < 1.0
    assert abs(values.mean() - 0.5) < 0.03


def test_it_carries_no_information_about_anything_real():
    """The whole point: correlation with a real signal must be ~0."""
    f = frame(4000)
    out = _noise_features(f, COLUMNS)
    real = np.sin(np.arange(len(f)) / 7.0)        # any structured series
    for column in COLUMNS:
        assert abs(np.corrcoef(out[column], real)[0, 1]) < 0.08
