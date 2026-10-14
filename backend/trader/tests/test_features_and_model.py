"""Tests for the alignment canary and the null control.

A feature equal to the outcome must be recovered, and must fail when shifted by one
window. Bars with no structure must give no skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.baseline import log_loss, reliability
from core.config import Config
from core.dataset import Dataset, ScoringBundle, apply_fold, fit_fold, score_live
from core.features import (
    ALL_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS, feature_columns, population_report,
)
from core.model import BASELINE_LOGIT, fit_model
from tests.conftest import make_bars

FAST = Config(n_estimators=120, early_stopping_rounds=15, n_folds=3,
              seasonality_min_days=5)


@pytest.fixture(scope='module')
def prepared():
    """One dataset and one train/test split, reused by the slow tests here."""
    bars = make_bars(days=45, lead=0.35, close_noise=0.15)
    dataset = Dataset.build(bars, FAST)
    index = dataset.window_index
    cut = int(len(index) * 0.75)
    fit, train = fit_fold(dataset, index[:cut], FAST)
    test = apply_fold(dataset, fit, index[cut:], FAST)
    return dataset, fit, train, test


# ------------------------------------------------------------- the canary

def test_a_perfect_feature_is_recovered_end_to_end(prepared):
    """A feature equal to the outcome gives large skill. Otherwise the panel and target
    are misaligned.
    """
    _, fit, train, test = prepared
    column = FEATURE_GROUPS['geometry'][0]
    train = train.copy()
    test = test.copy()
    train[column] = train['outcome'].astype(float)
    test[column] = test['outcome'].astype(float)

    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    probability = model.predict(test)
    y = test['outcome'].to_numpy(dtype=float)
    skill = log_loss(y, test['baseline_probability'].to_numpy()) - log_loss(y, probability)
    assert skill > 0.15, (
        f'a feature equal to the outcome produced only {skill:+.5f} of log loss '
        f'skill; the panel and the target are not aligned'
    )
    assert model.residual_scale > 0.5


def test_the_canary_fails_when_the_target_is_shifted(prepared):
    """The canary fails when the feature answers the next window instead."""
    _, fit, train, test = prepared
    column = FEATURE_GROUPS['geometry'][0]
    train = train.copy()
    test = test.copy()
    # The feature is the answer to the next window, not this one.
    for frame in (train, test):
        shifted = frame.groupby(['symbol', 'offset'])['outcome'].shift(-1)
        frame[column] = shifted.astype(float)
    train = train.dropna(subset=[column])
    test = test.dropna(subset=[column])

    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    y = test['outcome'].to_numpy(dtype=float)
    skill = (log_loss(y, test['baseline_probability'].to_numpy())
             - log_loss(y, model.predict(test)))
    assert skill < 0.05, (
        f'a feature answering the *next* window still produced {skill:+.5f} of '
        f'skill, so the canary is not measuring alignment'
    )


# --------------------------------------------------------- the null control

def test_the_pipeline_finds_nothing_on_a_null():
    """On independent random walks the pipeline finds no skill."""
    rng = np.random.default_rng(99)
    bars = make_bars(days=40, lead=0.0, seed=7)
    for symbol in bars:
        n = len(bars[symbol])
        returns = rng.normal(0, 1.4e-4, n)
        price = float(bars[symbol]['open'].iloc[0]) * np.exp(returns.cumsum())
        bars[symbol] = bars[symbol].assign(
            open=price, close=price, high=price * 1.00005, low=price * 0.99995)

    dataset = Dataset.build(bars, FAST)
    index = dataset.window_index
    cut = int(len(index) * 0.75)
    fit, train = fit_fold(dataset, index[:cut], FAST)
    test = apply_fold(dataset, fit, index[cut:], FAST)
    model = fit_model(train, fit.baseline, FAST)

    y = test['outcome'].to_numpy(dtype=float)
    skill = (log_loss(y, test['baseline_probability'].to_numpy())
             - log_loss(y, model.predict(test)))
    assert skill < 0.002, (
        f'the pipeline found {skill:+.5f} of skill on independent random walks'
    )


# ----------------------------------------------------------------- shape

def test_every_declared_feature_is_produced(prepared):
    """Every declared feature is populated, since an all-NaN column looks like a working one."""
    _, _, _, test = prepared
    report = population_report(test)
    empty = report[report['populated'] == 0]
    assert empty.empty, f'never populated: {empty["feature"].tolist()}'
    thin = report[(report['share'] < 0.85) & (report['populated'] > 0)]
    assert thin.empty, f'under 85% populated: {thin[["feature", "share"]].to_dict("records")}'


def test_the_control_group_is_declared_as_one():
    """The calendar control group is declared and contains the time-of-day columns."""
    assert CONTROL_GROUPS, 'no control group is declared'
    for group in CONTROL_GROUPS:
        assert group in FEATURE_GROUPS
    controls = set(feature_columns(CONTROL_GROUPS))
    assert 'hour_sin' in controls and 'dow_sin' in controls


def test_selecting_groups_selects_columns():
    assert set(feature_columns(('geometry',))) == set(FEATURE_GROUPS['geometry'])
    # The default matrix is `ALL_GROUPS`, which excludes the book groups because most
    # bar history has no book.
    from core.features import ALL_GROUPS as _DEFAULTS
    assert len(feature_columns()) == sum(len(FEATURE_GROUPS[g]) for g in _DEFAULTS)
    with pytest.raises(ValueError, match='unknown feature groups'):
        feature_columns(('nonexistent',))


def test_the_baseline_logit_is_required_before_prediction(prepared):
    """Prediction requires the baseline logit column instead of recomputing it."""
    _, fit, train, test = prepared
    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    stripped = test.drop(columns=[BASELINE_LOGIT])
    with pytest.raises(ValueError, match=BASELINE_LOGIT):
        model.predict(stripped)


def test_an_untrained_correction_reproduces_the_baseline(prepared):
    """With `residual_scale` at zero the model reproduces the baseline exactly."""
    _, fit, train, test = prepared
    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    model.residual_scale = 0.0
    assert np.allclose(model.predict(test),
                       test['baseline_probability'].to_numpy(), atol=1e-9)


# ------------------------------------------------------- the scoring bundle

def test_the_artifact_carries_what_the_live_path_needs(prepared):
    """The artifact carries the scoring bundle the live path needs, so it is deployable."""
    _, fit, train, _ = prepared
    bundle = fit.bundle(FAST)
    model = fit_model(train, fit.baseline, FAST, scoring=bundle)
    assert model.deployable
    assert model.provenance()['deployable'] is True
    for symbol in FAST.symbols:
        assert bundle.covers(symbol), symbol
    # The frames are left out on purpose: they are hundreds of megabytes.
    assert not hasattr(bundle, 'states')


def test_score_live_scores_one_window_per_symbol(prepared):
    dataset, fit, train, _ = prepared
    bundle = fit.bundle(FAST)
    window = dataset.window_index[-3]
    scored = score_live(
        {s: g.reset_index().rename(columns={'index': 'event_time'})
         for s, g in dataset.grids.items()},
        bundle, FAST, window_open=window, offset=FAST.decision_offsets[-1])
    assert len(scored) == len(FAST.symbols)
    assert set(scored['window_open']) == {window}
    assert 'baseline_probability' in scored


def test_score_live_reports_no_outcome_for_an_unsettled_window(prepared):
    """An unsettled window has NaN outcome, not zero."""
    dataset, fit, _, _ = prepared
    scored = score_live(
        {s: g.reset_index().rename(columns={'index': 'event_time'})
         for s, g in dataset.grids.items()},
        fit.bundle(FAST), FAST,
        window_open=dataset.window_index[-3], offset=FAST.decision_offsets[0])
    assert scored['outcome'].isna().all()
    assert scored['settle_price'].isna().all()


def test_the_shrinkage_reads_near_zero_on_a_null():
    """On a null dataset `residual_scale` stays below the 0.25 gate.

    Early stopping and the shrinkage fit must not share rows, and a failed optimiser
    must not return its bracket point 0.7639320225 as a fitted value.
    """
    from core.config import Config
    from core.dataset import Dataset, fit_fold
    from core.model import fit_model
    from tests.conftest import make_bars

    config = Config(n_estimators=120, early_stopping_rounds=15, n_folds=3,
                    seasonality_min_days=5)
    bars = make_bars(days=70, lead=0.0, seed=11)  # lead=0: nothing to find
    dataset = Dataset.build(bars, config)
    index = dataset.window_index
    fit, train_table = fit_fold(dataset, index[:int(len(index) * 0.85)], config)
    model = fit_model(train_table, fit.baseline, config)

    assert model.residual_scale < 0.25, (
        f'alpha is {model.residual_scale:.4f} on a null; the gate it guards is '
        f'0.25, so this would promote'
    )
    assert model.residual_scale != pytest.approx(0.7639320225, abs=1e-6), (
        'that is scipy\'s golden-section bracket seed, not a fitted value — '
        '_fit_residual_scale is swallowing a non-convergence again'
    )


def test_the_shrinkage_excludes_unscoreable_rows_rather_than_dying():
    """The shrinkage fit drops unscoreable (NaN) rows and needs enough rows to remain."""
    import numpy as np

    from core.model import MIN_SHRINKAGE_ROWS, _fit_residual_scale

    rng = np.random.default_rng(0)
    n = MIN_SHRINKAGE_ROWS * 4
    logit = rng.normal(0, 1.0, n)
    correction = rng.normal(0, 0.1, n)
    outcome = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(float)

    clean = _fit_residual_scale(logit, correction, outcome)
    assert 0.0 <= clean <= 2.0
    assert clean != pytest.approx(0.7639320225, abs=1e-6), (
        "that is scipy's bracket seed, not a fitted value"
    )

    # A few NaNs (as from an outage) change the answer only slightly.
    holed = correction.copy()
    holed[:20] = np.nan
    with_holes = _fit_residual_scale(logit, holed, outcome)
    assert 0.0 <= with_holes <= 2.0
    assert abs(with_holes - clean) < 0.25, (
        f'dropping 20 of {n} rows moved alpha from {clean:.4f} to '
        f'{with_holes:.4f}; that is not an exclusion, that is a different fit'
    )

    # Too few scoreable rows is a data problem and raises.
    mostly_nan = correction.copy()
    mostly_nan[MIN_SHRINKAGE_ROWS // 2:] = np.nan
    with pytest.raises(ValueError, match='scoreable rows remain'):
        _fit_residual_scale(logit, mostly_nan, outcome)


def test_the_shrinkage_raises_when_the_optimiser_does_not_converge(monkeypatch):
    """A non-converged optimiser raises instead of returning its bracket point."""
    import numpy as np
    from scipy import optimize

    from core.model import MIN_SHRINKAGE_ROWS, _fit_residual_scale

    n = MIN_SHRINKAGE_ROWS * 2
    rng = np.random.default_rng(1)
    logit = rng.normal(0, 1.0, n)
    correction = rng.normal(0, 0.1, n)
    outcome = (rng.random(n) < 0.5).astype(float)

    class Abandoned:
        x = 0.7639320225002102
        success = False
        message = 'stopped early'

    monkeypatch.setattr(optimize, 'minimize_scalar', lambda *a, **k: Abandoned())
    with pytest.raises(ValueError, match='did not converge'):
        _fit_residual_scale(logit, correction, outcome)
