"""Tests for calendar-based folds in `purged_walk_forward`.

Count-based folds have very unequal spans when data density varies; calendar folds give
each fold an equal span.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.cv import purged_walk_forward


def _index(days=180, per_day_early=4, per_day_late=200):
    """A timeline whose density rises sharply, like the real store."""
    start = pd.Timestamp('2026-01-01', tz='UTC')
    stamps = []
    for d in range(days):
        n = per_day_early if d < days // 2 else per_day_late
        day = start + pd.Timedelta(days=d)
        stamps += [day + pd.Timedelta(minutes=15 * i) for i in range(n)]
    return pd.DatetimeIndex(stamps)


def test_calendar_folds_have_comparable_spans():
    folds = purged_walk_forward(_index(), n_folds=6, scheme='calendar')
    spans = [(f.test.max() - f.test.min()).days for f in folds]
    assert max(spans) / max(min(spans), 1) < 1.5, (
        f'calendar folds should be near-equal in days, got {spans}')


def test_calendar_spans_are_equal_by_construction_whatever_the_density():
    for early, late in ((4, 200), (50, 60), (200, 4)):
        folds = purged_walk_forward(
            _index(per_day_early=early, per_day_late=late), n_folds=6,
            scheme='calendar')
        spans = [(f.test.max() - f.test.min()).days for f in folds]
        assert max(spans) / max(min(spans), 1) < 1.5, (
            f'density {early}->{late} gave spans {spans}')


def test_a_thin_early_block_costs_a_fold_and_that_must_be_visible():
    """A sparse start can leave fewer than `n_folds` folds, so the fold count must be
    visible.

    The sign-agreement gate holds its significance at any fold count, so a lost fold
    costs power rather than causing an automatic failure.
    """
    thin = purged_walk_forward(_index(per_day_early=4, per_day_late=200),
                               n_folds=6, scheme='calendar')
    even = purged_walk_forward(_index(per_day_early=100, per_day_late=100),
                               n_folds=6, scheme='calendar')
    assert len(thin) < len(even), (
        f'expected a sparse start to cost folds: {len(thin)} vs {len(even)}')
    # Not `== 6`: anchored calendar folds accumulate, so an even timeline yields every
    # complete block it contains.
    assert len(even) >= 6


def test_the_spans_stay_comparable_as_data_arrives():
    """Fold spans stay comparable to each other as the data grows.

    Equal blocks over a growing span still move the boundaries, but proportionally.
    Fixed boundaries would need fixed-width blocks and a growing fold count.
    """
    for days in (150, 180, 210):
        folds = purged_walk_forward(_index(days=days), n_folds=6,
                                    scheme='calendar')
        spans = [(f.test.max() - f.test.min()).days for f in folds]
        assert max(spans) / max(min(spans), 1) < 1.5, (
            f'{days}d: spans drifted apart, {spans}')


def test_the_embargo_still_purges_training_windows():
    folds = purged_walk_forward(_index(), n_folds=6, scheme='calendar',
                                embargo_minutes=1440)
    for f in folds:
        gap = (f.test.min() - f.train.max()).total_seconds() / 60.0
        assert gap > 1440, f'fold {f.index} embargo is only {gap:.0f} minutes'


def test_count_remains_available_and_is_what_past_entries_used():
    """The count scheme is still available. Provenance records which scheme a run used."""
    folds = purged_walk_forward(_index(), n_folds=6, scheme='count')
    assert len(folds) >= 5


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match='scheme'):
        purged_walk_forward(_index(), n_folds=6, scheme='nearest')
