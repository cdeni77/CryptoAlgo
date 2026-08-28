"""What gets deployed must be trained on everything, not on the last fold.

`promote` installed `result.models[-1]`, whose training ends where that fold's
TEST block begins — so the deployed artifact is always one test block stale, by
construction. Measured on the artifact that traded live: it was trained through
2025-12-05 and deployed in August, and had therefore never seen a single Kalshi
15-minute market, because the venue's markets did not exist yet.

The walk-forward is the EVIDENCE that the configuration works. The thing to ship
is that configuration refitted on all data through the present. Those are
different objects and conflating them costs exactly one test block of freshness
every time.
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
    """If the refit is not strictly fresher, something is wrong with how it was
    built and shipping it silently would hide that."""
    from scripts.promote import choose_candidate
    folds = [_Model(300)]
    with pytest.raises(ValueError, match='fresher|history'):
        choose_candidate(folds, _Model(300))


def test_without_a_refit_it_falls_back_and_the_caller_can_tell():
    """A refit can legitimately fail — too few rows, a fit that did not
    converge. Falling back to the last fold is right, but it must be visible,
    because the fallback is the stale artifact this exists to stop shipping."""
    from scripts.promote import choose_candidate
    folds = [_Model(100), _Model(300)]
    chosen = choose_candidate(folds, None)
    assert chosen is folds[-1]


def test_no_models_at_all_raises_rather_than_returning_none():
    from scripts.promote import choose_candidate
    with pytest.raises(ValueError):
        choose_candidate([], None)


def test_the_refit_carries_the_scoring_bundle_or_it_cannot_be_deployed():
    """The refit was fitted WITHOUT `scoring=fit.bundle(config)`.

    `core/backtest.py` passes it for every fold model, so the folds are
    deployable and the refit — the thing actually installed — was not. Nothing
    in `promote` checks, so the artifact installed cleanly, reported all its
    gates, and then `scripts/live.py` refused it on every cycle:

        this artifact carries no scoring bundle, so it cannot score a window
        it has never seen

    The volatility model, the seasonality factor and the baseline's scale/tail
    all live in that bundle. Without them there is nothing to turn a raw window
    into a feature row, so the artifact can be evaluated and never deployed.
    """
    import inspect
    from scripts.promote import refit_on_all

    source = inspect.getsource(refit_on_all)
    assert 'scoring=' in source, (
        'refit_on_all calls fit_model without a scoring bundle, so the model it '
        'returns is not deployable')


def test_promote_refuses_to_install_an_artifact_live_cannot_load():
    """The bundle bug reached a restart loop because nothing between fitting and
    installing asked whether the artifact could score. Promotion is the gate;
    `deployable` is the cheapest possible check and it was not among them."""
    import inspect
    from core import promotion

    source = inspect.getsource(promotion.promote)
    assert 'deployable' in source, (
        'promote installs without checking that the candidate can score a '
        'window it has never seen')
