"""Folds must be built from the WHOLE Config, in one place.

`purged_walk_forward` carries its own defaults, and three scripts called it
passing only `n_folds` and `embargo_minutes`:

    scripts/train.py:55   scripts/baseline.py:57   scripts/ablate.py:80

So `--fold-scheme`, `--fold-block-days` and `--fold-train-days` were parsed,
folded into the Config, echoed in the run header and recorded in provenance --
and then ignored, while `core/backtest.py` forwarded all of them. Two different
experiments from one command line, with the header describing the one that did
not run.

It went from latent to live when `Config.fold_block_days` became 7.0 while this
module's default stayed 21.0: every gated number was cut into 7-day blocks
while `scripts/ablate.py` -- the tool that decides which feature groups survive
-- cut 21-day ones.

Adding an argument at three call sites fixes today and leaves the fourth caller
to be written wrong later. `folds_for_config` takes the Config, so it cannot be
called with half of it.
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
    """The live instance of the bug: Config says 7, the function default says
    21, and a caller that forwards neither gets 21."""
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
    """A seam test. The bug was three call sites each passing a SUBSET, and
    `purged_walk_forward` was correct throughout -- so testing it proved
    nothing. What must be asserted is that nobody builds folds from a Config by
    hand again.
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
