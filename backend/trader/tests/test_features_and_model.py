"""The alignment canary, and the null control.

**The canary.** A feature set equal to the realised outcome must be recovered end
to end. Nothing in the previous suite checked this, and it is the first thing any
"why is the skill zero" investigation has to rule out: a one-bar shift between
panel and target destroys signal silently and reads exactly like no edge.
Lookahead tests assert the model cannot see the *future*; this asserts it can see
the *present*, which is the opposite failure and had no guard. A mutation test
sits beside it — shift the target by one window and the canary must fail — so the
canary cannot pass for the wrong reason.

**The null control.** On bars with no exploitable structure the pipeline must
find nothing. A suite that only checks a model can find signal cannot catch a
pipeline that manufactures it, and manufacturing it is the failure mode this
whole project exists to avoid.
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
    """Inject the answer as a feature; the model must find it.

    If this fails, no measurement in the system means anything — a broken join
    and a genuinely unpredictable market are indistinguishable from the outside.
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
    """The mutation. Without it the canary could pass for the wrong reason."""
    _, fit, train, test = prepared
    column = FEATURE_GROUPS['geometry'][0]
    train = train.copy()
    test = test.copy()
    # The feature is the answer to the *next* window, not this one.
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
    """Independent random walks. Any skill here is manufactured."""
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
    """A group that produced nothing arrives as an all-NaN column of the same shape.

    Which a column-name hash cannot distinguish from a working one, so this is
    the only place it is visible.
    """
    _, _, _, test = prepared
    report = population_report(test)
    empty = report[report['populated'] == 0]
    assert empty.empty, f'never populated: {empty["feature"].tolist()}'
    thin = report[(report['share'] < 0.85) & (report['populated'] > 0)]
    assert thin.empty, f'under 85% populated: {thin[["feature", "share"]].to_dict("records")}'


def test_the_control_group_is_declared_as_one():
    """Hour of day cannot forecast direction, and the suite has to know that.

    The previous incarnation of this project ran a 27-cell survey whose best cell
    was its own control, and that was the most useful result it produced.
    """
    assert CONTROL_GROUPS, 'no control group is declared'
    for group in CONTROL_GROUPS:
        assert group in FEATURE_GROUPS
    controls = set(feature_columns(CONTROL_GROUPS))
    assert 'hour_sin' in controls and 'dow_sin' in controls


def test_selecting_groups_selects_columns():
    assert set(feature_columns(('geometry',))) == set(FEATURE_GROUPS['geometry'])
    assert len(feature_columns()) == sum(len(v) for v in FEATURE_GROUPS.values())
    with pytest.raises(ValueError, match='unknown feature groups'):
        feature_columns(('nonexistent',))


def test_the_baseline_logit_is_required_before_prediction(prepared):
    """Recomputing it here would silently use a different baseline."""
    _, fit, train, test = prepared
    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    stripped = test.drop(columns=[BASELINE_LOGIT])
    with pytest.raises(ValueError, match=BASELINE_LOGIT):
        model.predict(stripped)


def test_an_untrained_correction_reproduces_the_baseline(prepared):
    """The residual architecture, stated as a property.

    With alpha at zero the model *is* the baseline, so every tree it grows is
    incremental skill by construction rather than by comparison.
    """
    _, fit, train, test = prepared
    model = fit_model(train, fit.baseline, FAST, groups=('geometry',))
    model.residual_scale = 0.0
    assert np.allclose(model.predict(test),
                       test['baseline_probability'].to_numpy(), atol=1e-9)


# ------------------------------------------------------- the scoring bundle

def test_the_artifact_carries_what_the_live_path_needs(prepared):
    """An artifact that can be evaluated and not deployed is a trap.

    Nothing said so until the live path tried to score a window and found no
    volatility model.
    """
    _, fit, train, _ = prepared
    bundle = fit.bundle(FAST)
    model = fit_model(train, fit.baseline, FAST, scoring=bundle)
    assert model.deployable
    assert model.provenance()['deployable'] is True
    for symbol in FAST.symbols:
        assert bundle.covers(symbol), symbol
    # And the frames are deliberately left out: they are hundreds of megabytes.
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
    """A window being decided has not settled.

    Writing a plausible zero would make an unresolved bet look like a loss.
    """
    dataset, fit, _, _ = prepared
    scored = score_live(
        {s: g.reset_index().rename(columns={'index': 'event_time'})
         for s, g in dataset.grids.items()},
        fit.bundle(FAST), FAST,
        window_open=dataset.window_index[-3], offset=FAST.decision_offsets[0])
    assert scored['outcome'].isna().all()
    assert scored['settle_price'].isna().all()
