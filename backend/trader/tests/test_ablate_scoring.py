"""The scoring path itself, serial and parallel.

The existing tests exercise `trials`, which only builds the cell list. The
wiring between it and the workers went untested, and a serial run raised
`KeyError: 'dataset'` because only the parallel branch populated the shared
inputs. A cell list that is correct is not a run that works.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from scripts import ablate


class _Fold:
    def __init__(self, i):
        self.train, self.test = f'train{i}', f'test{i}'


@pytest.fixture
def stubbed(monkeypatch):
    """Replace the three heavy calls with arithmetic on the fold index."""
    def fit_fold(dataset, train, config, groups=()):
        return types.SimpleNamespace(baseline=0.0), train

    def apply_fold(dataset, fit, test, config, groups=()):
        return {'fold': int(str(test)[-1]), 'groups': tuple(groups)}

    def fit_model(train, baseline, config, groups=()):
        return types.SimpleNamespace(
            best_iteration=len(groups),
            predict=lambda test: np.array([float(test['fold'])]))

    monkeypatch.setattr(ablate, 'fit_fold', fit_fold)
    monkeypatch.setattr(ablate, 'apply_fold', apply_fold)
    monkeypatch.setattr(ablate, 'fit_model', fit_model)
    # `y` cancels in log_loss(y,b) - log_loss(y,m), so the skill a fold
    # reports is m - b = fold index - number of groups.
    monkeypatch.setattr(ablate, '_graded_label',
                        lambda test, cfg: np.array([float(test['fold'])]))
    monkeypatch.setattr(ablate, '_null_for',
                        lambda test, cfg: np.array([float(len(test['groups']))]))
    monkeypatch.setattr(ablate, 'log_loss', lambda y, p: float(y[0] - p[0]))
    monkeypatch.setattr(ablate, 'effective_observations', lambda test: 100.0)


def test_serial_scoring_populates_the_shared_inputs(stubbed):
    """The regression: jobs=1 used the worker without filling `_SHARED`."""
    ablate._SHARED.clear()                                     # noqa: SLF001
    folds = [_Fold(i) for i in range(3)]
    cells = {'all groups': ('a', 'b'), 'all minus b': ('a',)}
    out = ablate._score_cells(cells, folds, 'DATASET', 'CONFIG', jobs=1)  # noqa: SLF001

    assert set(out) == set(cells)
    skills, trees, sizes = out['all groups']
    assert skills == [-2.0, -1.0, 0.0]     # fold - len(groups)
    assert trees == [2, 2, 2] and sizes == [100.0] * 3
    assert out['all minus b'][0] == [-1.0, 0.0, 1.0]
    assert None not in skills, 'every fold must be filled'


def test_the_paired_delta_reads_the_scored_cells(stubbed):
    folds = [_Fold(i) for i in range(3)]
    cells = {'all groups': ('a', 'b'), 'all minus b': ('a',)}
    out = ablate._score_cells(cells, folds, 'D', 'C', jobs=1)  # noqa: SLF001
    per_fold = {k: (np.asarray(v[0], float), np.asarray(v[2], float))
                for k, v in out.items()}
    deltas = ablate._paired_deltas(per_fold)                   # noqa: SLF001
    assert list(deltas['group']) == ['b']
    # all groups is exactly 1.0 lower on every fold
    assert deltas['delta'].iloc[0] == pytest.approx(-1.0)
    assert deltas['se'].iloc[0] == pytest.approx(0.0, abs=1e-12)
    assert deltas['folds'].iloc[0] == 3
