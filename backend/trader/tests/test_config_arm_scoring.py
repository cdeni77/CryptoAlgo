"""The arm scoring path, serial and parallel.

`--help` passing is not evidence `main()` works: the equivalent path in
`scripts/ablate` raised `KeyError` on every serial run because only the
parallel branch filled the shared inputs, and the cell-list tests never
touched it.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from scripts import config_arm


class _Fold:
    def __init__(self, i):
        self.train, self.test = f'tr{i}', f'te{i}'


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(config_arm, 'fit_fold',
                        lambda d, tr, c, groups=(): (
                            types.SimpleNamespace(baseline=0.0), tr))
    monkeypatch.setattr(config_arm, 'apply_fold',
                        lambda d, f, te, c, groups=(): {
                            'fold': int(str(te)[-1]), 'seed': c.model_seed})
    monkeypatch.setattr(config_arm, 'fit_model',
                        lambda tr, b, c, groups=(): types.SimpleNamespace(
                            best_iteration=7,
                            predict=lambda t: np.array([float(t['fold'])])))
    monkeypatch.setattr(config_arm, '_graded_label',
                        lambda t, c: np.array([0.0]))
    # the null carries the arm value, so each arm differs
    monkeypatch.setattr(config_arm, '_null_for',
                        lambda t, c: np.array([float(t['seed'])]))
    monkeypatch.setattr(config_arm, 'log_loss', lambda y, p: float(p[0]))
    monkeypatch.setattr(config_arm, 'effective_observations', lambda t: 100.0)


def _run(jobs, stub_config):
    folds = [_Fold(i) for i in range(3)]
    return config_arm._score_arms(                             # noqa: SLF001
        [17, 18], folds, 'DATASET', stub_config, ('g',), 'model_seed', jobs)


@pytest.fixture
def stub_config():
    from core.config import DEFAULT_CONFIG
    return DEFAULT_CONFIG


def test_serial_scoring_fills_the_shared_inputs(stubbed, stub_config):
    config_arm._SHARED.clear()                                 # noqa: SLF001
    out = _run(1, stub_config)
    assert set(out) == {17, 18}
    for value in (17, 18):
        skills, sizes, trees = out[value]
        assert None not in skills, 'every fold must be filled'
        assert sizes == [100.0] * 3 and trees == [7] * 3
        # skill = log_loss(null) - log_loss(model) = seed - fold
        assert skills == [float(value) - f for f in range(3)]


def test_the_arms_differ_by_the_field_under_test(stubbed, stub_config):
    out = _run(1, stub_config)
    assert out[18][0] != out[17][0], 'the config override must reach the fit'
    delta = np.asarray(out[18][0]) - np.asarray(out[17][0])
    assert np.allclose(delta, 1.0), 'seed 18 minus seed 17 is a constant here'


def test_an_unknown_field_is_refused_before_any_fitting(stub_config):
    with pytest.raises(ValueError, match='unknown config fields'):
        stub_config.with_overrides(not_a_real_field=1)
