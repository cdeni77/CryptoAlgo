"""Fold boundaries must not move when data is appended.

`purged_walk_forward`'s docstring gave this as the whole reason `'calendar'`
exists — "earlier boundaries then stay put as data arrives, so fold-level
results are comparable run to run" — and the implementation did not deliver it.
It cut with `pd.date_range(index.min(), index.max(), periods=n_folds + 2)`,
dividing the span between the first and LAST window, so every interior boundary
slid whenever anything was appended. Measured on a 245-day span, adding three
days moved the cuts by +10h, +21h, +31h, +41h, +51h, +62h and +72h; fold 5's
test block shifted more than two days.

It stayed invisible while evaluation was occasional. Weekly promotion exposed
it immediately: the SAME configuration on three extra days took `max_drawdown`
0.182 -> 0.388 and `calibration_vs_market` -0.014 -> +0.009, failing three
gates it had passed three days earlier. Almost none of that was the model, and
on that evidence the candidate was refused while the incumbent — which would
also have failed, had anything re-tested it — stayed deployed.

A gate is only a measurement if the thing under it holds still.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.cv import purged_walk_forward

ORIGIN = pd.Timestamp('2026-01-09', tz='UTC')
BLOCK = 21.0  # matches Config.fold_block_days


def _folds(days: float, scheme: str = 'calendar', block: float = BLOCK):
    index = pd.DatetimeIndex(
        pd.date_range(ORIGIN, periods=int(days * 96), freq='15min', tz='UTC'))
    return purged_walk_forward(index, n_folds=6, embargo_minutes=1440,
                               min_train_windows=100, scheme=scheme,
                               fold_block_days=block)


def test_appending_data_within_a_block_moves_no_boundary():
    """The defect, directly. Three more days used to slide all six."""
    before, after = _folds(240), _folds(243)
    assert len(before) == len(after)
    assert [f.test_start for f in before] == [f.test_start for f in after]


def test_within_a_block_the_folds_are_IDENTICAL_end_to_end():
    """Stronger than "boundaries did not move": nothing moves at all.

    Only COMPLETE blocks are tested, so data arriving mid-block lands in the
    untested trailing stub and the evaluation is bit-for-bit the same run to
    run. That is what makes a weekly gate a comparison. The cost is the stub:
    up to `fold_block_days` of the newest data is not in the test set until its
    block completes.
    """
    before, after = _folds(240), _folds(243)
    assert len(before) == len(after)
    for x, y in zip(before, after):
        assert x.test_start == y.test_start
        assert x.test_end == y.test_end
        assert len(x.test) == len(y.test)


def test_every_boundary_sits_on_the_anchored_grid():
    for days in (200, 240, 243, 248, 260, 300):
        for fold in _folds(days):
            offset = (fold.test_start - ORIGIN).total_seconds() / 86400
            assert abs(offset % BLOCK) < 1e-6, (days, fold.test_start)


def test_a_new_block_ADDS_a_fold_and_evicts_nothing():
    """Folds accumulate; they do not roll off.

    Capping at the most recent `n_folds` blocks halved the evaluation —
    `windows_evaluated` 21,307 -> 10,488, failing its own 20,000 bar — because
    six 21-day blocks cover 126 days of a 245-day span. Testing every complete
    block costs nothing in stability, since the boundaries are anchored either
    way, and it is why gates counting folds had to become proportions.
    """
    # 21-day blocks: 210d is exactly 10 blocks, 231d is 11.
    before = [f.test_start for f in _folds(210)]
    after = [f.test_start for f in _folds(231)]
    assert set(before) < set(after), 'every earlier block must survive'
    assert len(after) == len(before) + 1, 'exactly one new block should enter'


def test_a_longer_block_gives_wider_folds_anchored_the_same_way():
    wide = _folds(300, block=70.0)
    for fold in wide:
        offset = (fold.test_start - ORIGIN).total_seconds() / 86400
        assert abs(offset % 70.0) < 1e-6


def test_the_count_scheme_is_still_NOT_anchored():
    """Recorded, not fixed. `'count'` blocks are defined by data volume, so
    they cannot be anchored to a time grid — appending re-cuts them by
    construction. This matters because `Config.fold_scheme` defaults to
    'count' for GATING, so the anchoring above does not reach the gates until
    that default changes. Asserted so nobody assumes it was fixed too.
    """
    before = [f.test_start for f in _folds(245, scheme='count')]
    after = [f.test_start for f in _folds(248, scheme='count')]
    assert before != after


def test_folds_never_overlap_and_train_precedes_test():
    """The anchoring must not have broken the property the module exists for."""
    from core.cv import assert_no_leakage
    for fold in _folds(300):
        assert_no_leakage(fold)
        assert fold.train.max() < fold.test.min()


def test_a_span_too_short_to_anchor_falls_back_and_WARNS(caplog):
    """A fresh store has weeks, not months. Refusing outright would break every
    short-span research run, so it subdivides — but the boundaries are then
    span-dependent again, which is the whole property being claimed, so it has
    to say so rather than quietly hand back unstable folds."""
    with caplog.at_level('WARNING'):
        folds = _folds(30, block=35.0)
    assert folds, 'a short span must still yield folds'
    assert 'boundaries WILL move' in caplog.text
