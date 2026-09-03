"""Folds cut on window COUNT re-cut every boundary when data arrives.

`purged_walk_forward` used `np.linspace(0, len(index), ...)` — equal counts.
With data density varying 8x across the span (30 to 244 windows/day, as
`--complete-cases` intersects three sources whose coverage improved through
2026), equal counts means wildly unequal calendar spans: measured 21 to 78 days,
3.7x.

Two consequences, and the second is why this changes:

  * The folds are not the same experiment. Fold 3's single fitted model was
    tested across 78 days of drift, fold 4's across 21, and the pooled mean
    treats them as interchangeable.
  * Nothing is trackable. Appending a week of data re-cut every boundary and
    moved every fold's skill, which reads like the model changing when only the
    splitting did. With continuous collection and periodic retraining that
    happens on every run.

Measured on 12,909 complete-case windows, the scheme is not flattering the
result — balanced folds are STRONGER:

    equal-count      +0.00276 +/- 0.00088  t=+3.15  6/6
    equal-calendar   +0.00312 +/- 0.00075  t=+4.18  6/6

The trade is real and unavoidable: you can balance time or count, never both,
because the density varies. Equal-calendar put 491 windows in one fold and
5,671 in another, so folds carry equal WEIGHT at unequal precision. That is
inefficient but unbiased, and it is the price of a boundary that stays put.
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
    """The invariant, as against the measured contrast.

    Whether COUNT folds skew depends on the density profile — a clean step puts
    every count-block in the dense half and they come out even. On the real
    store, where density rises gradually from 30 to 244 windows/day, they came
    out 21 to 78 days. Calendar blocks are equal regardless, which is the
    property being relied on rather than a fact about one dataset.
    """
    for early, late in ((4, 200), (50, 60), (200, 4)):
        folds = purged_walk_forward(
            _index(per_day_early=early, per_day_late=late), n_folds=6,
            scheme='calendar')
        spans = [(f.test.max() - f.test.min()).days for f in folds]
        assert max(spans) / max(min(spans), 1) < 1.5, (
            f'density {early}->{late} gave spans {spans}')


def test_a_thin_early_block_costs_a_fold_and_that_must_be_visible():
    """Equal calendar blocks over a timeline that STARTS sparse can leave the
    first blocks with too little training data, so fewer than `n_folds` come
    back — 3 of 6 on a 4/day start.

    That is not a bug but it is a trap: `folds_skill_positive >= 5` needs six
    folds, so a candidate could fail that gate for a splitting reason rather
    than a forecasting one. The count scheme cannot do this, because its blocks
    are sized by the data itself.
    """
    thin = purged_walk_forward(_index(per_day_early=4, per_day_late=200),
                               n_folds=6, scheme='calendar')
    even = purged_walk_forward(_index(per_day_early=100, per_day_late=100),
                               n_folds=6, scheme='calendar')
    assert len(thin) < len(even), (
        f'expected a sparse start to cost folds: {len(thin)} vs {len(even)}')
    assert len(even) == 6


def test_the_spans_stay_comparable_as_data_arrives():
    """What equal-calendar actually buys, stated honestly.

    It does NOT pin the boundaries: dividing a GROWING span into a fixed number
    of equal blocks still moves every edge, just proportionally rather than
    erratically. What it guarantees is that the folds remain comparable to each
    other on every run, which is what makes the pooled mean and its fold
    dispersion meaningful.

    Pinning the boundaries outright needs fixed-WIDTH blocks and therefore a
    fold count that grows with the span — a different trade, and the natural
    frame for the rolling-window question rather than for this change.
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
    """Every ledger entry before this change was cut on count. The scheme is
    recorded in provenance so an entry says which it was."""
    folds = purged_walk_forward(_index(), n_folds=6, scheme='count')
    assert len(folds) >= 5


def test_an_unknown_scheme_is_refused():
    with pytest.raises(ValueError, match='scheme'):
        purged_walk_forward(_index(), n_folds=6, scheme='nearest')
