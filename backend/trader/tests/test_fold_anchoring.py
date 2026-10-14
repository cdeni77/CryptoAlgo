"""Tests that calendar fold boundaries stay fixed when data is appended.

Blocks are anchored to a fixed origin, so adding data within a block changes no fold.
Only then is a weekly gate run comparable with the previous one.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.cv import purged_walk_forward

ORIGIN = pd.Timestamp('2026-01-09', tz='UTC')
# An explicit block width. It does not track `Config.fold_block_days`, because
# anchoring must hold at any width.
BLOCK = 21.0


def _folds(days: float, scheme: str = 'calendar', block: float = BLOCK):
    index = pd.DatetimeIndex(
        pd.date_range(ORIGIN, periods=int(days * 96), freq='15min', tz='UTC'))
    return purged_walk_forward(index, n_folds=6, embargo_minutes=1440,
                               min_train_windows=100, scheme=scheme,
                               fold_block_days=block)


def test_appending_data_within_a_block_moves_no_boundary():
    before, after = _folds(240), _folds(243)
    assert len(before) == len(after)
    assert [f.test_start for f in before] == [f.test_start for f in after]


def test_within_a_block_the_folds_are_IDENTICAL_end_to_end():
    """Within a block the folds are identical end to end.

    Only complete blocks are tested, so new data waits in an untested trailing stub
    until its block completes.
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
    """A new complete block adds a fold and removes none."""
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
    """The 'count' scheme is not anchored, since its blocks follow data volume."""
    before = [f.test_start for f in _folds(245, scheme='count')]
    after = [f.test_start for f in _folds(248, scheme='count')]
    assert before != after


def test_folds_never_overlap_and_train_precedes_test():
    """Anchored folds never overlap and train precedes test."""
    from core.cv import assert_no_leakage
    for fold in _folds(300):
        assert_no_leakage(fold)
        assert fold.train.max() < fold.test.min()


def test_a_span_too_short_to_anchor_falls_back_and_WARNS(caplog):
    """A span too short to anchor falls back to subdivision and logs a warning."""
    with caplog.at_level('WARNING'):
        folds = _folds(30, block=35.0)
    assert folds, 'a short span must still yield folds'
    assert 'boundaries WILL move' in caplog.text
