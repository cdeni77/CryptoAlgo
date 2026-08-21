"""Promotion: the gate between a trained model and real money.

The property under test is not "does the model work" — synthetic data cannot
answer that. It is "can a model that failed its gates reach the live path". The
previous system promoted anything that finished training, so this is the guard
that replaces that behaviour, and it needs to hold whatever the candidate looks
like.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.model import ForecastModel
from core.promotion import (
    MODEL_FILENAME,
    PromotionRecord,
    current_record,
    load_records,
    new_version,
    promote,
    promotions_dir,
    report,
    trials_to_date,
    write_record,
)

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


@pytest.fixture
def model() -> ForecastModel:
    """A minimal artifact. Promotion cares that it saves, not what it predicts."""
    return ForecastModel(
        heads={},
        feature_columns=('a', 'b'),
        symbol_categories=('BIP', 'ETP'),
        feature_set_hash='deadbeef',
        horizon_bars=24,
    )


def _record(*, promoted: bool, version: str | None = None) -> PromotionRecord:
    gates = [
        {'name': 'pbo', 'value': 0.1, 'threshold': 0.3, 'comparison': 'max', 'passed': True},
        {'name': 'walk_forward_median_sharpe', 'value': 0.9 if promoted else -0.4,
         'threshold': 0.5, 'comparison': 'min', 'passed': promoted},
    ]
    return PromotionRecord(
        version=version or new_version(),
        promoted=promoted,
        gates=gates,
        backtest={'sharpe': 1.2 if promoted else -0.8, 'trades': 250},
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_a_blocked_candidate_is_not_installed(tmp_path, model):
    """The whole point. A failed gate set must not reach the live artifact path."""
    installed, record = promote(model, _record(promoted=False), models_dir=tmp_path)

    assert not installed
    assert not (tmp_path / MODEL_FILENAME).exists(), 'a blocked model was installed'
    assert current_record(tmp_path) is None, 'a blocked model became the live pointer'
    assert record.failed_gates == ['walk_forward_median_sharpe']


def test_a_passing_candidate_is_installed_and_becomes_current(tmp_path, model):
    record = _record(promoted=True)
    installed, record = promote(model, record, models_dir=tmp_path)

    assert installed
    assert (tmp_path / MODEL_FILENAME).exists()

    live = current_record(tmp_path)
    assert live is not None and live.version == record.version
    assert live.promoted


def test_a_blocked_candidate_is_still_recorded(tmp_path, model):
    """Rejections stay in the ledger, because the trial count depends on them.

    The deflated Sharpe ratio discounts an observed Sharpe by how many
    configurations were tried. A ledger holding only successes makes every
    survivor look better than the evidence supports, so throwing the failures
    away is not a tidiness choice — it changes the statistics.
    """
    promote(model, _record(promoted=False), models_dir=tmp_path)
    promote(model, _record(promoted=False), models_dir=tmp_path)
    promote(model, _record(promoted=True), models_dir=tmp_path)

    records = load_records(tmp_path)
    assert len(records) == 3
    assert sum(r.promoted for r in records) == 1
    assert trials_to_date(tmp_path) == 3


def test_forcing_requires_a_reason(tmp_path, model):
    with pytest.raises(ValueError, match='reason'):
        promote(model, _record(promoted=False), models_dir=tmp_path, force=True)

    assert not (tmp_path / MODEL_FILENAME).exists()


def test_a_forced_model_is_visibly_forced(tmp_path, model):
    """An override is recorded, not silent, for as long as the model is live."""
    installed, record = promote(
        model, _record(promoted=False), models_dir=tmp_path,
        force=True, force_reason='cost model understates the DOGE fee floor',
    )

    assert installed
    assert record.forced
    assert 'DOGE' in (record.force_reason or '')
    assert record.failed_gates == ['walk_forward_median_sharpe'], (
        'forcing must not rewrite which gates failed'
    )

    live = current_record(tmp_path)
    assert live is not None and live.forced
    assert 'FORCED' in report(live)


def test_forcing_a_passing_candidate_does_not_mark_it_forced(tmp_path, model):
    """`--force` on a candidate that would have passed anyway is not an override."""
    _, record = promote(
        model, _record(promoted=True), models_dir=tmp_path,
        force=True, force_reason='belt and braces',
    )

    assert record.promoted
    assert not record.forced


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------


def test_records_come_back_newest_first(tmp_path, model):
    versions = ['20260101T000000Z', '20260301T000000Z', '20260201T000000Z']
    for version in versions:
        write_record(_record(promoted=False, version=version), tmp_path)

    assert [r.version for r in load_records(tmp_path)] == sorted(versions, reverse=True)
    assert [r.version for r in load_records(tmp_path, limit=2)] == [
        '20260301T000000Z', '20260201T000000Z'
    ]


def test_a_corrupt_record_does_not_break_the_ledger(tmp_path, model):
    """One unreadable file must not hide the rest of the history."""
    write_record(_record(promoted=True, version='20260101T000000Z'), tmp_path)
    (promotions_dir(tmp_path) / '20260102T000000Z.json').write_text('{not json')

    records = load_records(tmp_path)

    assert [r.version for r in records] == ['20260101T000000Z']


def test_a_record_round_trips_through_json(tmp_path, model):
    original = _record(promoted=True)
    path = write_record(original, tmp_path)

    reloaded = json.loads(path.read_text())
    reloaded.pop('failed_gates')
    restored = PromotionRecord(**reloaded)

    assert restored.version == original.version
    assert restored.promoted
    assert restored.gates == original.gates


def test_trials_is_at_least_one_on_an_empty_ledger(tmp_path):
    """The model in front of you is itself a trial, so zero is never the answer."""
    assert trials_to_date(tmp_path) == 1


def test_promotion_replaces_the_previous_model(tmp_path, model):
    first = _record(promoted=True, version='20260101T000000Z')
    second = _record(promoted=True, version='20260201T000000Z')

    promote(model, first, models_dir=tmp_path)
    promote(model, second, models_dir=tmp_path)

    live = current_record(tmp_path)
    assert live is not None and live.version == '20260201T000000Z'
    # The staging directory is transient: a promotion must not leave one behind
    # for the next run to trip over.
    assert not list((tmp_path / '.staging').glob('*')) if (tmp_path / '.staging').exists() else True


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _dataset_for(config, seed, *, drift, funding_mean):
    from core.dataset import Dataset
    from tests.test_backtest import _features_targets

    features, targets, bars_by, funding_by, profiles = _features_targets(
        config, seed, drift=drift, funding_mean=funding_mean
    )
    return Dataset(
        features=features, targets=targets, bars=bars_by, funding=funding_by,
        profiles=profiles, venue='synthetic', reference_venue=None,
        as_of=None, horizon_bars=48,
    )


@pytest.mark.slow
def test_driftless_random_walks_do_not_promote(config):
    """A candidate fitted to noise must be blocked on the merits.

    The version of this test that shipped could not fail. It called
    `evaluate_candidate(..., full=False)`, which skips the synthetic panels, so
    `synthetic_positive_fraction` was never measured — and an unmeasured gate
    fails by construction. The verdict was therefore structural and completely
    independent of the data: replacing the driftless walks with 0.4% drift per
    bar and 50bp/hour of carry, an enormous exploitable edge, still passed.

    So this now runs the full evaluation and asserts on the gates that actually
    reflect the data.
    """
    from core.promotion import evaluate_candidate

    dataset = _dataset_for(config, 0, drift=0.0, funding_mean=0.0)

    model, record = evaluate_candidate(
        dataset, config, n_periods=3, full=True, synthetic_paths=2, trials=1,
    )

    assert model is not None, 'the candidate should train; it just should not promote'
    assert not record.promoted, (
        f'driftless random walks cleared the gates: '
        f'{json.dumps(record.measurements, default=str)}'
    )
    # Every gate was measured, so the rejection cannot be an artefact of a
    # skipped simulation.
    unmeasured = [k for k, v in record.measurements.items() if v is None]
    assert not unmeasured, f'gates not measured, so the verdict is structural: {unmeasured}'

    # And the rejection has to rest on a gate that reads the data, not on
    # bookkeeping. These are the ones noise should fail.
    data_dependent = {
        'walk_forward_median_sharpe', 'walk_forward_p05_sharpe',
        'deflated_sharpe', 'bootstrap_p05_sharpe', 'probability_positive',
        'synthetic_positive_fraction',
    }
    assert data_dependent & set(record.failed_gates), (
        f'blocked, but only by bookkeeping gates: {record.failed_gates}'
    )


@pytest.mark.slow
def test_a_real_edge_scores_better_than_noise(config):
    """The differential the previous test lacked.

    A test that only ever sees noise cannot distinguish "the gates work" from
    "the gates always fail". Running the same evaluation on data with a genuine
    drift-and-carry edge must move the measurements in the right direction, or
    the gate arithmetic is not reading the data at all.
    """
    from core.promotion import evaluate_candidate

    _, noise = evaluate_candidate(
        _dataset_for(config, 3, drift=0.0, funding_mean=0.0),
        config, n_periods=3, full=True, synthetic_paths=2, trials=1,
    )
    _, edge = evaluate_candidate(
        _dataset_for(config, 3, drift=0.004, funding_mean=5e-4),
        config, n_periods=3, full=True, synthetic_paths=2, trials=1,
    )

    noisy = noise.measurements.get('walk_forward_median_sharpe')
    real = edge.measurements.get('walk_forward_median_sharpe')
    assert noisy is not None and real is not None
    assert real > noisy, (
        f'a 0.4%/bar drift with 50bp/hour carry scored no better than noise '
        f'({real:.3f} vs {noisy:.3f}) — the gates are not reading the data'
    )


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------


def test_the_cost_schedule_is_reachable_without_a_cwd_assumption():
    """The fee schedule must resolve from anywhere, including inside the image.

    This is the regression test for a deployment bug that silently mispriced
    every containerised run: `configs/` lived at the repository root while the
    trader's Docker build context is `backend/trader`, so the file was never
    copied into the image. `_common.py` computed one relative path from
    `__file__`, found nothing, logged a warning nobody read, and fell through to
    the hardcoded 10bp/side — wrong for every Coinbase CDE contract by 0.06x to
    2.5x, in both directions.
    """
    import os

    from core.config import DEFAULT_COST_CONFIG_NAME, Config, find_cost_config

    resolved = find_cost_config()
    assert resolved is not None, (
        f'{DEFAULT_COST_CONFIG_NAME} is not on any search path; a container '
        f'built from this tree would misprice every contract'
    )
    assert resolved.exists()

    # Under the trader package, so the Docker build context includes it.
    assert 'backend/trader' in resolved.as_posix()

    # And it has to load into a Config that actually differs from the default,
    # or "loaded" would mean nothing.
    loaded = Config().with_cost_assumptions(resolved)
    assert loaded.cost_config_version != Config().cost_config_version
    assert loaded.min_fee_per_contract_by_symbol, 'no per-symbol fees came through'


def test_a_missing_cost_config_does_not_silently_pass(tmp_path):
    """A name that resolves to nothing returns None rather than a wrong file."""
    from core.config import find_cost_config

    assert find_cost_config('no_such_venue_v1999.json') is None


def test_a_full_evaluation_measures_every_gate(config):
    """No gate may be structurally unmeasurable, or nothing can ever promote.

    `pbo`, `deflated_sharpe` and `parameter_plateau` were in `DEFAULT_GATES` but
    nothing computed them, so every candidate came back with three gates reading
    "not measured" — which fails by design. The effect was that `--force` was the
    only route to live, which defeats the entire point of having gates.

    This does not check that the values are *good* — on synthetic data they should
    not be. It checks that they are numbers.
    """
    from core.dataset import Dataset
    from core.metrics import DEFAULT_GATES
    from core.promotion import evaluate_candidate
    from tests.test_backtest import _features_targets

    features, targets, bars_by, funding_by, profiles = _features_targets(
        config, 3, drift=0.0004, funding_mean=2e-5
    )
    dataset = Dataset(
        features=features, targets=targets, bars=bars_by, funding=funding_by,
        profiles=profiles, venue='synthetic', reference_venue=None,
        as_of=None, horizon_bars=48,
    )

    _, record = evaluate_candidate(
        dataset, config, n_periods=3, full=True, synthetic_paths=2, trials=4,
    )

    unmeasured = [
        name for name, value in record.measurements.items() if value is None
    ]
    assert not unmeasured, (
        f'gates with no measurement, which fail by construction: {unmeasured}. '
        f'A gate nothing computes makes --force the only route to live.'
    )
    assert set(record.measurements) == set(DEFAULT_GATES), (
        'the report and the gate table describe different sets'
    )


def test_a_quick_evaluation_cannot_promote(config):
    """`--quick` skips the slow simulations, and a skipped gate must fail."""
    from core.dataset import Dataset
    from core.promotion import evaluate_candidate
    from tests.test_backtest import _features_targets

    features, targets, bars_by, funding_by, profiles = _features_targets(
        config, 3, drift=0.0004, funding_mean=2e-5
    )
    dataset = Dataset(
        features=features, targets=targets, bars=bars_by, funding=funding_by,
        profiles=profiles, venue='synthetic', reference_venue=None,
        as_of=None, horizon_bars=48,
    )

    _, record = evaluate_candidate(dataset, config, n_periods=3, full=False, trials=2)

    assert not record.promoted
    assert 'synthetic_positive_fraction' in record.failed_gates
    assert 'parameter_plateau' in record.failed_gates
