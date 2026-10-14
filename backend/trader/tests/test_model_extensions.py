"""Opt-in model extensions: blend layer, ensemble, mirror, venue label.

Every extension is off by default, so the first tests pin that the defaults
fit and score exactly as before.
"""

from __future__ import annotations

import argparse

import joblib
import numpy as np
import pandas as pd
import pytest

from core.baseline import BarrierBaseline, clip_prob, expit, log_loss
from core.config import Config
from core.model import (
    BASELINE_LOGIT, MIN_SHRINKAGE_ROWS, _fit_blend, fit_model,
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
    assert (config.final_layer, config.ensemble_size, config.mirror_augment,
            config.label_source) == ('scale', 1, False, 'proxy')
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
    for name in ('final_layer', 'init_coefficient', 'intercept', 'members'):
        del model.__dict__[name]
    path = tmp_path / 'old.joblib'
    joblib.dump(model, path)
    old = joblib.load(path)
    np.testing.assert_array_equal(old.predict(test_table), expected)
    assert list(old.score(test_table).columns) == ['model_probability']
    assert old.provenance()['final_layer'] == 'scale'
    old.verify()


# ------------------------------------------------------------ blend layer

def _blend_sample(a=0.8, b=0.6, c=0.0, n=40_000, seed=3):
    rng = np.random.default_rng(seed)
    init = rng.normal(0.0, 1.5, n)
    corr = rng.normal(0.0, 1.0, n)
    y = (rng.random(n) < expit(a * init + b * corr + c)).astype(float)
    return init, corr, y


def test_the_blend_recovers_a_planted_init_and_correction_weight():
    init, corr, y = _blend_sample()
    a, b, c = _fit_blend(init, corr, y, min_edge=0.015)
    assert a == pytest.approx(0.8, abs=0.06)
    assert b == pytest.approx(0.6, abs=0.06)
    assert c == 0.0


def test_the_blend_recovers_a_planted_intercept_only_when_allowed():
    init, corr, y = _blend_sample(c=0.25)
    a, b, c = _fit_blend(init, corr, y, min_edge=0.015, intercept=True)
    assert (a, b, c) == pytest.approx((0.8, 0.6, 0.25), abs=0.07)
    assert _fit_blend(init, corr, y, min_edge=0.015)[2] == 0.0


def test_the_blend_respects_its_bounds():
    init, corr, y = _blend_sample(a=3.0, b=3.0)
    a, b, _ = _fit_blend(init, corr, y, min_edge=0.015, max_scale=1.0)
    assert a <= 1.5 + 1e-9 and b <= 1.0 + 1e-9


def test_the_blend_falls_back_to_every_row_when_few_would_trade(caplog):
    init, corr, y = _blend_sample(n=2 * MIN_SHRINKAGE_ROWS)
    corr = corr * 1e-4                     # nothing moves the price 1.5pp
    _fit_blend(init, corr, y, min_edge=0.015)
    assert 'fitted on all' in caplog.text


def test_the_ridge_holds_a_small_sample_near_the_identity():
    init, corr, y = _blend_sample(a=1.0, b=0.6, n=MIN_SHRINKAGE_ROWS, seed=11)
    a, _, c = _fit_blend(init, corr, y, min_edge=0.0, intercept=True)
    assert abs(a - 1.0) < 0.3 and abs(c) < 0.3


def test_a_blended_model_scores_through_its_own_layer(train_table, test_table):
    model = fit(train_table, _config(final_layer='blend'))
    assert model.final_layer == 'blend'
    assert 0.5 <= model.init_coefficient <= 1.5
    assert model.intercept == 0.0
    base = test_table[BASELINE_LOGIT].to_numpy()
    raw = model.raw_correction(test_table)
    expected = clip_prob(expit(model.init_coefficient * base
                               + model.residual_scale * raw))
    np.testing.assert_allclose(model.predict(test_table), expected, rtol=0, atol=1e-15)
    record = model.provenance()
    assert record['final_layer'] == 'blend'
    assert record['init_coefficient'] == model.init_coefficient
    assert record['config']['final_layer'] == 'blend'


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


# ------------------------------------------------------------ mirror

def test_mirroring_is_applied_to_inner_train_only(train_table, monkeypatch):
    import core.model as model_module
    seen = []
    real = model_module.mirror_rows

    def spy(frame):
        seen.append(frame)
        return real(frame)

    monkeypatch.setattr(model_module, 'mirror_rows', spy)
    fit(train_table, _config(mirror_augment=True))
    assert len(seen) == 1
    windows = np.sort(train_table['window_open'].unique())
    cut = windows[int(len(windows) * 0.8)]
    assert seen[0]['window_open'].max() < cut


def test_a_symmetric_signal_is_learned_at_least_as_well_with_mirroring(test_table):
    small = synthetic(n_windows=2400, signal=0.6, seed=5)
    y = test_table['outcome'].to_numpy()
    plain = fit(small, _config(mirror_augment=False))
    mirrored = fit(small, _config(mirror_augment=True))
    plain_loss = log_loss(y, plain.predict(test_table))
    mirrored_loss = log_loss(y, mirrored.predict(test_table))
    base_loss = log_loss(y, test_table['baseline_probability'].to_numpy())
    assert mirrored_loss < base_loss
    assert mirrored_loss <= plain_loss + 0.002


# ------------------------------------------------------------ venue label

def test_label_source_venue_trains_on_the_venue_label():
    """Proxy and venue disagree in sign; the fitted coefficient follows the source."""
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

@pytest.mark.parametrize('bad', [dict(final_layer='mean'),
                                 dict(label_source='kalshi'), dict(ensemble_size=0),
                                 dict(edge_lower_bound_z=-1.0)])
def test_a_bad_setting_is_refused_at_construction(bad):
    with pytest.raises(ValueError):
        Config(**bad)


def test_every_new_field_reaches_provenance_and_the_cli():
    from scripts._common import add_data_arguments, config_from_args
    parser = add_data_arguments(argparse.ArgumentParser())
    args = parser.parse_args([
        '--final-layer', 'blend', '--final-layer-intercept', '--ensemble-size', '5',
        '--mirror-augment', '--label-source', 'venue',
        '--edge-lower-bound-z', '1.5', '--require-member-agreement'])
    config = config_from_args(args)
    expected = dict(final_layer='blend', final_layer_intercept=True,
                    ensemble_size=5, mirror_augment=True, label_source='venue',
                    edge_lower_bound_z=1.5, require_member_agreement=True)
    record = config.provenance()
    for name, value in expected.items():
        assert getattr(config, name) == value
        assert name in config.cli_overrides
        recorded = record[name]
        assert (tuple(recorded) if isinstance(recorded, list) else recorded) == value

    untouched = config_from_args(parser.parse_args([]))
    assert untouched.final_layer == 'scale' and not untouched.cli_overrides & set(expected)
