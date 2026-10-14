"""Tests that the default embargo covers the longest feature lookback.

`assert_no_leakage` checks a fold against its own embargo, so it cannot catch an embargo
that is too small.
"""

from __future__ import annotations

import inspect

import pandas as pd

from core.config import Config
from core.cv import assert_no_leakage, purged_walk_forward


def window_index(days=200):
    return pd.date_range('2025-01-01', periods=days * 96, freq='15min', tz='UTC')


def test_the_configured_embargo_covers_the_longest_feature_lookback():
    """The configured embargo covers the longest lookback in `vol_lookbacks_minutes`."""
    config = Config()
    longest = max(config.vol_lookbacks_minutes)
    assert config.embargo_minutes >= longest, (
        f'embargo is {config.embargo_minutes} minutes but the longest feature '
        f'lookback is {longest}; a training row that close to the test block '
        f'computes its features from test-period bars'
    )
    assert config.embargo_minutes >= config.window_minutes


def test_the_module_default_matches_the_configured_default():
    """The `purged_walk_forward` default matches the config default."""
    default = inspect.signature(purged_walk_forward).parameters['embargo_minutes'].default
    assert default == Config().embargo_minutes, (
        f'core.cv default is {default}, core.config default is '
        f'{Config().embargo_minutes}'
    )


def test_folds_built_from_the_config_alone_honour_a_day():
    """Folds built without an explicit embargo (the production path) have at least a day's gap."""
    config = Config()
    folds = purged_walk_forward(window_index(), n_folds=config.n_folds)
    # `>=`, not `==`: anchored calendar blocks accumulate, so `n_folds` is a cap, not a
    # count.
    assert len(folds) >= config.n_folds
    for fold in folds:
        assert_no_leakage(fold)
        gap = fold.test_start - fold.train_end
        assert gap > pd.Timedelta(minutes=max(config.vol_lookbacks_minutes)), (
            f'fold {fold.index}: only {gap} between the last training window and '
            f'the first test window'
        )


def test_the_leakage_check_is_measured_against_the_lookback_not_the_fold():
    """A fold built with a zero embargo passes `assert_no_leakage` but fails the lookback check."""
    config = Config()
    folds = purged_walk_forward(window_index(), n_folds=4, embargo_minutes=0)
    lookback = pd.Timedelta(minutes=max(config.vol_lookbacks_minutes))
    offenders = [f.index for f in folds if (f.test_start - f.train_end) <= lookback]
    assert offenders, 'a zero embargo produced no offending fold; the fixture is wrong'
    # Every one passes the module's own check, which is why this test exists.
    for fold in folds:
        assert_no_leakage(fold)
