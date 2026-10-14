"""Tests for opt-in model extensions: the ensemble and the venue label.

Every extension is off by default, so the first tests check that the defaults
fit and score exactly as before.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import pytest

from core.baseline import BarrierBaseline, clip_prob, expit
from core.config import Config
from core.model import (
    BASELINE_LOGIT, fit_model,
)

FAST = Config(n_estimators=80, early_stopping_rounds=15, min_child_samples=50,
              learning_rate=0.1)
BASELINE = BarrierBaseline(distribution='normal', nu=30.0, scale={})
NOISE = ('abs_z_score', 'path_efficiency', 'excursion_span_z')


def synthetic(n_windows=3000, *, signal=0.5, seed=0, per_window=3,
              start='2026-01-01'):
    """Rows whose truth is the init logit plus `signal * z_score`."""
    rng = np.random.default_rng(seed)
    opens = pd.date_range(start, periods=n_windows, freq='15min', tz='UTC')
    n = n_windows * per_window
    x = rng.normal(size=n)
    init = rng.normal(0.0, 1.0, n)
    truth = expit(init + signal * x)
    table = pd.DataFrame({
        'window_open': np.repeat(opens, per_window),
        'outcome': (rng.random(n) < truth).astype(float),
        BASELINE_LOGIT: init,
        'baseline_probability': expit(init),
        'z_score': x,
    })
    for column in NOISE:
        table[column] = rng.normal(size=n)
    return table


def _config(**changes):
    values = {**FAST.__dict__, **changes}
    values.pop('cli_overrides', None)
    return Config(**values)


def fit(table, config=FAST, **kw):
    return fit_model(table, BASELINE, config, groups=('geometry',), **kw)


@pytest.fixture(scope='module')
def train_table():
    return synthetic()


@pytest.fixture(scope='module')
def test_table():
    return synthetic(n_windows=1500, seed=1, start='2026-03-01')


# ------------------------------------------------------------ defaults

def test_the_defaults_are_the_historical_model(train_table, test_table):
    config = Config()
    assert (config.ensemble_size, config.label_source) == (1, 'proxy')
    assert config.edge_lower_bound_z == 0.0
    assert config.require_member_agreement is False

    model = fit(train_table)
    assert model.members == () and model.ensemble_size == 1
    assert (model.init_coefficient, model.intercept) == (1.0, 0.0)
    base = test_table[BASELINE_LOGIT].to_numpy()
    raw = model.booster.predict(test_table[model.features].to_numpy(),
                                raw_score=True)
    expected = clip_prob(expit(base + model.residual_scale * raw))
    np.testing.assert_array_equal(model.predict(test_table), expected)
    scored = model.score(test_table)
    assert list(scored.columns) == ['model_probability']
    np.testing.assert_array_equal(scored['model_probability'], expected)


def test_an_artifact_pickled_before_these_fields_still_predicts(
        train_table, test_table, tmp_path):
    model = fit(train_table)
    expected = model.predict(test_table)
    for name in ('init_coefficient', 'intercept', 'members'):
        del model.__dict__[name]
    path = tmp_path / 'old.joblib'
    joblib.dump(model, path)
    old = joblib.load(path)
    np.testing.assert_array_equal(old.predict(test_table), expected)
    assert list(old.score(test_table).columns) == ['model_probability']
    assert (old.provenance()['init_coefficient'],
            old.provenance()['intercept']) == (1.0, 0.0)
    old.verify()


def test_an_artifact_pickled_with_a_blend_layer_still_predicts_through_it(
        train_table, test_table, tmp_path):
    model = fit(train_table)
    model.__dict__.update(final_layer='blend', init_coefficient=0.8,
                          intercept=0.1)
    path = tmp_path / 'blend.joblib'
    joblib.dump(model, path)
    old = joblib.load(path)
    base = test_table[BASELINE_LOGIT].to_numpy()
    expected = clip_prob(expit(0.8 * base
                               + old.residual_scale * old.raw_correction(test_table)
                               + 0.1))
    np.testing.assert_array_equal(old.predict(test_table), expected)
    old.verify()


# ------------------------------------------------------------ ensemble

@pytest.fixture(scope='module')
def ensemble(train_table):
    return fit(train_table, _config(ensemble_size=3))


def test_an_ensemble_carries_every_member(ensemble, test_table):
    assert ensemble.ensemble_size == 3
    assert ensemble.booster is ensemble.members[0]
    per_member = ensemble.member_corrections(test_table)
    assert per_member.shape == (3, len(test_table))
    assert not np.allclose(per_member[0], per_member[1]), 'members must differ'
    np.testing.assert_allclose(ensemble.raw_correction(test_table),
                               per_member.mean(axis=0))
    assert len(ensemble.importance()) == len(ensemble.features)
    assert ensemble.provenance()['ensemble_size'] == 3


def test_the_ensemble_scores_its_spread(ensemble, test_table):
    scored = ensemble.score(test_table)
    assert list(scored.columns) == ['model_probability', 'model_probability_sd',
                                    'model_probability_min',
                                    'model_probability_max']
    p = scored['model_probability']
    assert (scored['model_probability_min'] <= p + 1e-12).all()
    assert (p <= scored['model_probability_max'] + 1e-12).all()
    assert (scored['model_probability_sd'] >= 0).all()
    assert (scored['model_probability_sd'] > 0).any()


def test_a_row_with_no_init_has_no_spread_either(ensemble, test_table):
    table = test_table.head(5).copy()
    table.loc[table.index[0], BASELINE_LOGIT] = np.nan
    scored = ensemble.score(table)
    assert scored.iloc[0].isna().all()
    assert scored.iloc[1:].notna().all().all()


# ------------------------------------------------------------ venue label

def test_label_source_venue_trains_on_the_venue_label():
    """With `label_source='venue'` the fitted coefficient follows the venue label."""
    table = synthetic(signal=0.0, seed=21)
    rng = np.random.default_rng(22)
    x = table['z_score'].to_numpy()
    init = table[BASELINE_LOGIT].to_numpy()
    table['venue_outcome'] = (rng.random(len(x)) < expit(init + 0.6 * x)).astype(float)
    table['outcome'] = (rng.random(len(x)) < expit(init - 0.6 * x)).astype(float)
    table.loc[table.sample(frac=0.1, random_state=0).index, 'venue_outcome'] = np.nan
    venue = fit(table, _config(label_source='venue'))
    proxy = fit(table, _config(label_source='proxy'))
    assert np.corrcoef(venue.raw_correction(table), x)[0, 1] > 0.3
    assert np.corrcoef(proxy.raw_correction(table), x)[0, 1] < -0.3


def test_label_source_venue_falls_back_to_the_proxy_where_absent(train_table):
    table = train_table.assign(venue_outcome=np.nan)
    venue = fit(table, _config(label_source='venue'))
    proxy = fit(table)
    np.testing.assert_array_equal(venue.raw_correction(table),
                                  proxy.raw_correction(table))


# ------------------------------------------------------------ config and CLI

@pytest.mark.parametrize('bad', [dict(label_source='kalshi'), dict(ensemble_size=0),
                                 dict(edge_lower_bound_z=-1.0)])
def test_a_bad_setting_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_every_new_field_reaches_provenance_and_the_cli():
    from scripts._common import add_data_arguments, config_from_args
    parser = add_data_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        '--ensemble-size', '5', '--label-source', 'venue',
        '--edge-lower-bound-z', '1.5', '--require-member-agreement'])
    config = config_from_args(args)
    expected = dict(ensemble_size=5, label_source='venue',
                    edge_lower_bound_z=1.5, require_member_agreement=True)
    record = config.provenance()
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert name in config.cli_overrides
        recorded = record[name]
        assert (tuple(recorded) if isinstance(recorded, list) else recorded) == value

    untouched = config_from_args(parser.parse_args([]))
    assert untouched.ensemble_size == 1 and not untouched.cli_overrides & set(expected)
