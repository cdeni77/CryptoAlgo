"""Tests for `folds_for_config`, which builds folds from the whole Config.

Callers that pass only some fields to `purged_walk_forward` silently use its
other defaults. Taking the Config makes that impossible.
"""

from __future__ import annotations

import inspect

import pandas as pd

from core.config import Config
from core.cv import folds_for_config, purged_walk_forward

INDEX = pd.DatetimeIndex(pd.date_range('2026-01-01', '2026-06-01',
                                       freq='15min', tz='UTC'))


def _cfg(**kw):
    return Config(**kw)


def test_the_block_width_comes_from_the_config_not_the_function_default():
    """The block width comes from the Config (7), not the function default (21)."""
    sig = inspect.signature(purged_walk_forward)
    assert sig.parameters['fold_block_days'].default == 21.0
    assert Config().fold_block_days == 7.0

    folds = folds_for_config(INDEX, _cfg())
    spans = {round((f.test.max() - f.test.min()).total_seconds() / 86400.0)
             for f in folds}
    assert max(spans) <= 7


def test_a_wider_block_is_honoured():
    wide = folds_for_config(INDEX, _cfg(fold_block_days=21.0))
    narrow = folds_for_config(INDEX, _cfg(fold_block_days=7.0))
    assert len(narrow) > len(wide) * 2


def test_the_scheme_is_honoured():
    cal = folds_for_config(INDEX, _cfg(fold_scheme='calendar'))
    cnt = folds_for_config(INDEX, _cfg(fold_scheme='count', n_folds=4))
    assert len(cnt) <= 4
    assert len(cal) != len(cnt)


def test_the_rolling_train_window_is_honoured():
    rolling = folds_for_config(INDEX, _cfg(fold_train_days=35.0))
    for f in rolling:
        span = (f.train.max() - f.train.min()).total_seconds() / 86400.0
        assert span <= 35.0 + 1e-9


def test_every_fold_building_script_uses_the_helper():
    """The scripts build folds through `folds_for_config`, not `purged_walk_forward`
    directly.
    """
    import scripts.ablate, scripts.baseline, scripts.train, core.backtest

    for mod in (scripts.ablate, scripts.baseline, scripts.train):
        src = inspect.getsource(mod)
        assert 'purged_walk_forward(' not in src, (
            f'{mod.__name__} builds folds directly; use folds_for_config so the '
            f'whole geometry is carried')
        assert 'folds_for_config(' in src

    # backtest.py forwards every field explicitly and is the reference.
    bt = inspect.getsource(core.backtest)
    for field in ('fold_block_days', 'min_test_windows', 'train_days', 'scheme'):
        assert field in bt, f'backtest stopped forwarding {field}'
