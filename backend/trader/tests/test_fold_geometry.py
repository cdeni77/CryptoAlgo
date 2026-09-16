"""The walk-forward's geometry must be the one that is deployed.

Two knobs, and they answer different questions.

**`fold_block_days` is the cadence.** Blocks do not overlap, so it is both the
test span and the step. The live loop retrains every SEVEN days (the Sunday
cron) while the evaluation tested TWENTY-ONE-day blocks with a single model —
so the backtest let a model go three weeks stale before anyone refitted it,
while live never exceeds one. That is an evaluation of a deployment nobody
runs, the same defect as fitting to a backfilled quote and trading a recorded
one, on a different axis.

**`fold_train_days` is the training window**, and expanding remains the default
because expanding is what `scripts/promote.py` actually installs — the Sunday
retrain fits on all history. Setting it here WITHOUT matching it in the retrain
recreates the very mismatch this file is about, which is why the flag says so.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.cv import purged_walk_forward

# ~54 complete-case windows/day is the measured density; 15-minute grid.
INDEX = pd.DatetimeIndex(pd.date_range('2026-01-01', '2026-09-01',
                                       freq='15min', tz='UTC'))


def _folds(**kw):
    return purged_walk_forward(INDEX, embargo_minutes=1440,
                               min_train_windows=10, min_test_windows=10, **kw)


def test_expanding_is_still_the_default():
    """The deployed geometry. If this changes, the Sunday retrain must too."""
    folds = _folds(fold_block_days=21.0)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes), 'training set must never shrink when expanding'
    assert sizes[-1] > sizes[0] * 3


def test_a_rolling_window_holds_its_length_instead_of_growing():
    folds = _folds(fold_block_days=21.0, train_days=35.0)
    spans = [(f.train.max() - f.train.min()).total_seconds() / 86400.0
             for f in folds]
    assert max(spans) <= 35.0 + 1e-9
    # Flat, not growing: that is the whole difference from expanding. The FIRST
    # fold is the exception and legitimately so — it has only ~20 days of
    # history behind it, and a rolling window cannot invent data it does not
    # have. It is capped by the request, never padded to meet it.
    assert spans[0] <= 35.0 + 1e-9
    assert all(abs(s - 35.0) < 1.0 for s in spans[1:]), spans


def test_the_rolling_window_is_measured_from_the_embargo_not_the_test_start():
    """A day out of 35 is 3% of the sample. A parameter that means 34 when it
    says 35 is exactly the silent drift this module exists to prevent."""
    folds = _folds(fold_block_days=21.0, train_days=35.0)
    f = folds[-1]
    embargo = pd.Timedelta(minutes=1440)
    floor = (f.test.min() - embargo) - pd.Timedelta(days=35.0)
    assert f.train.min() >= floor
    # And it really does reach back the full 35 days of usable history.
    assert (f.train.min() - floor) < pd.Timedelta(days=1)


def test_a_rolling_window_never_crosses_the_embargo():
    for f in _folds(fold_block_days=21.0, train_days=35.0):
        gap = (f.test.min() - f.train.max()).total_seconds() / 60.0
        assert gap >= 1440, f'fold {f.index} leaked across the embargo'


def test_a_shorter_block_gives_the_cadence_live_actually_runs():
    """Seven days is the Sunday retrain. The point is not more folds for their
    own sake — it is that the model is never staler in the test than it is in
    production."""
    weekly = _folds(fold_block_days=7.0)
    triweekly = _folds(fold_block_days=21.0)
    assert len(weekly) > len(triweekly) * 2
    for f in weekly:
        span = (f.test.max() - f.test.min()).total_seconds() / 86400.0
        assert span <= 7.0


def test_blocks_do_not_overlap_so_no_window_is_scored_twice():
    """`fold_block_days` is the step as well as the span. If that stopped being
    true, pooled metrics would double-count and every money number would be
    inflated by the overlap."""
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
