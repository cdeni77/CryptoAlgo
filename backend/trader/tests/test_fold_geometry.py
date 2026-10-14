"""Tests for walk-forward fold geometry.

`fold_block_days` is both the test span and the step, and matches the weekly
retrain cadence. `fold_train_days` sets a rolling training window; expanding is
the default because the retrain fits on all history.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.cv import purged_walk_forward

# 15-minute grid over eight months.
INDEX = pd.DatetimeIndex(pd.date_range('2026-01-01', '2026-09-01',
                                       freq='15min', tz='UTC'))


def _folds(**kw):
    return purged_walk_forward(INDEX, embargo_minutes=1440,
                               min_train_windows=10, min_test_windows=10, **kw)


def test_expanding_is_still_the_default():
    """Expanding training is the default; the weekly retrain must match it."""
    folds = _folds(fold_block_days=21.0)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes), 'training set must never shrink when expanding'
    assert sizes[-1] > sizes[0] * 3


def test_a_rolling_window_holds_its_length_instead_of_growing():
    folds = _folds(fold_block_days=21.0, train_days=35.0)
    spans = [(f.train.max() - f.train.min()).total_seconds() / 86400.0
             for f in folds]
    assert max(spans) <= 35.0 + 1e-9
    # Flat, not growing. The first fold may be shorter because less history exists.
    assert spans[0] <= 35.0 + 1e-9
    assert all(abs(s - 35.0) < 1.0 for s in spans[1:]), spans


def test_the_rolling_window_is_measured_from_the_embargo_not_the_test_start():
    """The rolling window is measured back from the embargo, not from the test start."""
    folds = _folds(fold_block_days=21.0, train_days=35.0)
    f = folds[-1]
    embargo = pd.Timedelta(minutes=1440)
    floor = (f.test.min() - embargo) - pd.Timedelta(days=35.0)
    assert f.train.min() >= floor
    # It reaches back the full 35 days.
    assert (f.train.min() - floor) < pd.Timedelta(days=1)


def test_a_rolling_window_never_crosses_the_embargo():
    for f in _folds(fold_block_days=21.0, train_days=35.0):
        gap = (f.test.min() - f.train.max()).total_seconds() / 60.0
        assert gap >= 1440, f'fold {f.index} leaked across the embargo'


def test_a_shorter_block_gives_the_cadence_live_actually_runs():
    """Seven-day blocks match the weekly retrain, so the model is never staler in test than
    live.
    """
    weekly = _folds(fold_block_days=7.0)
    triweekly = _folds(fold_block_days=21.0)
    assert len(weekly) > len(triweekly) * 2
    for f in weekly:
        span = (f.test.max() - f.test.min()).total_seconds() / 86400.0
        assert span <= 7.0


def test_blocks_do_not_overlap_so_no_window_is_scored_twice():
    """Blocks do not overlap, so no window is scored twice."""
    seen: set = set()
    for f in _folds(fold_block_days=7.0):
        assert not (seen & set(f.test)), f'fold {f.index} re-tests scored windows'
        seen |= set(f.test)


def test_a_rolling_window_trains_on_less_than_expanding():
    rolling = _folds(fold_block_days=21.0, train_days=35.0)
    expanding = _folds(fold_block_days=21.0)
    assert len(rolling[-1].train) < len(expanding[-1].train) / 3


@pytest.mark.parametrize('days', [7.0, 35.0, 90.0])
def test_every_geometry_is_leak_free(days):
    from core.cv import assert_no_leakage
    for f in _folds(fold_block_days=7.0, train_days=days):
        assert_no_leakage(f)
