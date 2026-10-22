"""The model seed is a config field, so an arm can move only the seed.

That arm's true effect is zero by construction, so whatever spread it reports
is the booster's own stochasticity -- the share of measurement noise that
ensembling removes and more data does not.
"""

from __future__ import annotations

import inspect

from core.config import DEFAULT_CONFIG
from core import model as model_module


def test_the_default_is_unchanged():
    """17 was hard-coded; the field must not move the shipped model."""
    assert DEFAULT_CONFIG.model_seed == 17


def test_it_can_be_overridden():
    assert DEFAULT_CONFIG.with_overrides(model_seed=18).model_seed == 18
    assert DEFAULT_CONFIG.model_seed == 17, 'the config is frozen'


def test_an_override_is_recorded_for_the_trial_ledger():
    armed = DEFAULT_CONFIG.with_overrides(model_seed=18)
    assert 'model_seed' in armed.cli_overrides


def test_the_booster_reads_it_and_still_offsets_per_member():
    """Ensemble members must differ from each other, as they did before."""
    src = inspect.getsource(model_module.fit_model)
    assert "getattr(config, 'model_seed', 17)" in src
    assert '+ k' in src, 'members must still differ in their bagging draws'
    assert "'seed': 17 + k" not in src, 'the hard-coded seed must be gone'
