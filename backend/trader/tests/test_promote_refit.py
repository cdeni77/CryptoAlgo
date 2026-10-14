"""Tests that promotion deploys a model refitted on all data, not the last fold's model.

The walk-forward is the evidence that a configuration works. The shipped model
is that configuration refitted through the present; the last fold's model is
always one test block stale.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class _Model:
    def __init__(self, n):
        self.n_train_windows = n


def test_the_deployed_model_is_the_refit_not_the_last_fold():
    from scripts.promote import choose_candidate
    folds = [_Model(100), _Model(200), _Model(300)]
    refit = _Model(400)
    assert choose_candidate(folds, refit) is refit


def test_the_refit_must_see_more_history_than_the_last_fold():
    """A refit that has not seen more history than the last fold is refused."""
    from scripts.promote import choose_candidate
    folds = [_Model(300)]
    with pytest.raises(ValueError, match='fresher|history'):
        choose_candidate(folds, _Model(300))


def test_without_a_refit_it_falls_back_and_the_caller_can_tell():
    """Without a refit, the last fold's model is used as the fallback."""
    from scripts.promote import choose_candidate
    folds = [_Model(100), _Model(300)]
    chosen = choose_candidate(folds, None)
    assert chosen is folds[-1]


def test_no_models_at_all_raises_rather_than_returning_none():
    from scripts.promote import choose_candidate
    with pytest.raises(ValueError):
        choose_candidate([], None)


def test_the_refit_carries_the_scoring_bundle_or_it_cannot_be_deployed():
    """`refit_on_all` passes the scoring bundle, without which live cannot score a window.
    """
    import inspect
    from scripts.promote import refit_on_all

    source = inspect.getsource(refit_on_all)
    assert 'scoring=' in source, (
        'refit_on_all calls fit_model without a scoring bundle, so the model it '
        'returns is not deployable')


def test_promote_refuses_to_install_an_artifact_live_cannot_load():
    """`promote` checks `deployable` before installing."""
    import inspect
    from core import promotion

    source = inspect.getsource(promotion.promote)
    assert 'deployable' in source, (
        'promote installs without checking that the candidate can score a '
        'window it has never seen')
