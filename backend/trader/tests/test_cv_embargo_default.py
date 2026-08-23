"""PROPOSED. The embargo default, and a leakage check that is not self-referential.

Drop into `backend/trader/tests/`.

Measured: setting `core/config.py` `embargo_minutes = 1440` -> `0` and
`core/cv.py`'s own default to `0` leaves all 230 tests passing. Every existing
embargo test passes `embargo_minutes=` explicitly, so the default that
`core/backtest.py`, `scripts/train.py`, `scripts/baseline.py` and
`scripts/evaluate.py` actually use is unguarded.

Worse, `assert_no_leakage` compares `fold.gap_minutes` against
`fold.embargo_minutes` — the same number that built the fold — so it is
structurally incapable of catching a too-small embargo. The check below is
against the *feature lookback*, which is the quantity that determines what the
embargo has to be.
"""

from __future__ import annotations

import inspect

import pandas as pd

from core.config import Config
from core.cv import assert_no_leakage, purged_walk_forward


def window_index(days=200):
    return pd.date_range('2025-01-01', periods=days * 96, freq='15min', tz='UTC')


def test_the_configured_embargo_covers_the_longest_feature_lookback():
    """`log_rv_1440` needs a day of bars, so a training row inside a day of the
    test block computes a feature from test-period data.

    Stated against `vol_lookbacks_minutes` rather than against the literal 1440,
    so adding a longer lookback fails here instead of silently leaking.
    """
    config = Config()
    longest = max(config.vol_lookbacks_minutes)
    assert config.embargo_minutes >= longest, (
        f'embargo is {config.embargo_minutes} minutes but the longest feature '
        f'lookback is {longest}; a training row that close to the test block '
        f'computes its features from test-period bars'
    )
    assert config.embargo_minutes >= config.window_minutes


def test_the_module_default_matches_the_configured_default():
    """Two defaults that can drift are one silent leak.

    `purged_walk_forward` is also called directly, so its own default has to be
    the safe one too.
    """
    default = inspect.signature(purged_walk_forward).parameters['embargo_minutes'].default
    assert default == Config().embargo_minutes, (
        f'core.cv default is {default}, core.config default is '
        f'{Config().embargo_minutes}'
    )


def test_folds_built_from_the_config_alone_honour_a_day():
    """No `embargo_minutes=` argument anywhere — this is the production path."""
    config = Config()
    folds = purged_walk_forward(window_index(), n_folds=config.n_folds)
    assert len(folds) == config.n_folds
    for fold in folds:
        assert_no_leakage(fold)
        gap = fold.test_start - fold.train_end
        assert gap > pd.Timedelta(minutes=max(config.vol_lookbacks_minutes)), (
            f'fold {fold.index}: only {gap} between the last training window and '
            f'the first test window'
        )


def test_the_leakage_check_is_measured_against_the_lookback_not_the_fold():
    """A fold that carries its own too-small embargo must still be refused.

    `assert_no_leakage` reads `fold.embargo_minutes`, so a fold built with an
    embargo of zero passes it. The requirement is absolute, and this is where it
    is stated.
    """
    config = Config()
    folds = purged_walk_forward(window_index(), n_folds=4, embargo_minutes=0)
    lookback = pd.Timedelta(minutes=max(config.vol_lookbacks_minutes))
    offenders = [f.index for f in folds if (f.test_start - f.train_end) <= lookback]
    assert offenders, 'a zero embargo produced no offending fold; the fixture is wrong'
    # Every one of them passes the module's own check, which is the point.
    for fold in folds:
        assert_no_leakage(fold)
