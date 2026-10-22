"""NC2: the noise-only control (RESEARCH_PROTOCOL section 5, item 2).

Information-free features must be reproducible from the row's identity alone,
or the control measures a different thing on every run and cannot be a
control at all.
"""

from __future__ import annotations

import warnings

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


# --- the multi-seed calibration ------------------------------------------

from core.features import NOISE_GROUPS, _noise_features as _draw  # noqa: E402


def test_every_noise_group_is_an_independent_draw():
    """Without a per-group salt all five groups hold identical columns, and
    the 'distribution' would be one number repeated five times."""
    f = frame(400)
    drawn = {g: _draw(f, FEATURE_GROUPS[g], salt=g) for g in NOISE_GROUPS}
    first = {g: d[FEATURE_GROUPS[g][0]].to_numpy() for g, d in drawn.items()}
    seen = list(NOISE_GROUPS)
    for i, a in enumerate(seen):
        for b in seen[i + 1:]:
            assert not np.allclose(first[a], first[b]), f'{a} == {b}'
            r = abs(np.corrcoef(first[a], first[b])[0, 1])
            assert r < 0.12, f'{a} vs {b} correlated at {r:.3f}'


def test_all_five_are_opt_in():
    for g in NOISE_GROUPS:
        assert g in OPT_IN_GROUPS and g not in ALL_GROUPS


def test_the_groups_share_no_column_names():
    names = [c for g in NOISE_GROUPS for c in FEATURE_GROUPS[g]]
    assert len(names) == len(set(names)) == 55


# --- the attach must not fragment the frame ------------------------------

from core.features import attach_noise  # noqa: E402


def _wide(n_cols=60, n_rows=48):
    """A frame as wide as the real one, where fragmentation shows up."""
    base = frame(n_rows)
    for i in range(n_cols):
        base[f'pad_{i:03d}'] = np.arange(n_rows, dtype=float)
    return base.copy()


def test_attaching_noise_raises_no_fragmentation_warning():
    """The defect: 55 per-column assignments logged 15,700 warnings and cost
    real time on every feature build."""
    from pandas.errors import PerformanceWarning
    table = _wide()
    with warnings.catch_warnings():
        warnings.simplefilter('error', PerformanceWarning)
        out = attach_noise(table, list(NOISE_GROUPS))
    assert len(out.columns) == len(table.columns) + 55


def test_the_attached_values_are_the_drawn_ones():
    table = _wide()
    out = attach_noise(table, ['noise', 'noise3'])
    for group in ('noise', 'noise3'):
        expected = _draw(table, FEATURE_GROUPS[group], salt=group)
        for column in FEATURE_GROUPS[group]:
            assert np.allclose(out[column], expected[column])
    # groups not selected must not appear
    assert not set(FEATURE_GROUPS['noise2']) & set(out.columns)


def test_no_noise_group_selected_returns_the_frame_untouched():
    table = _wide()
    assert attach_noise(table, ['geometry']) is table
    assert attach_noise(table, None) is table


def test_the_order_does_not_depend_on_set_iteration():
    table = _wide()
    a = attach_noise(table, ['noise5', 'noise2', 'noise'])
    b = attach_noise(table, ['noise', 'noise2', 'noise5'])
    assert list(a.columns) == list(b.columns)
