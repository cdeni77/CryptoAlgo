"""Cross-validation and significance-test invariants.

Two properties matter most here and both are checkable exactly, which is why
these tests use constructed indices rather than market data:

* Purging removes every training row whose label resolves inside the test set.
  A purge shorter than the label horizon leaks silently and produces a
  validation score that cannot be reproduced live.
* Effective sample size reflects label overlap. 2,880 hourly rows at a 72-hour
  horizon carry about 40 independent outcomes, and every significance test needs
  that number rather than the row count.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from core.cv import (
    CPCVLayout,
    assemble_paths,
    average_uniqueness,
    combinatorial_purged_splits,
    effective_sample_size,
    label_concurrency,
    path_test_index,
    purged_walk_forward,
    recency_weights,
    sample_weights,
    assert_no_leakage,
)
from core.metrics import (
    DEFAULT_GATES,
    deflated_sharpe,
    drawdown_profile,
    evaluate_gates,
    expected_max_sharpe,
    gate_report,
    probability_of_backtest_overfitting,
    probabilistic_sharpe,
    sharpe_ratio,
    summarise_paths,
)

HORIZON = 72
N_BARS = 2_880          # 120 days of hourly bars


@pytest.fixture(scope='module')
def index() -> pd.DatetimeIndex:
    return pd.date_range('2026-01-01', periods=N_BARS, freq='1h', tz='UTC')


# ---------------------------------------------------------------------------
# CPCV
# ---------------------------------------------------------------------------


def test_cpcv_geometry():
    """12 groups holding out 2 gives 66 splits and 11 paths."""
    layout = CPCVLayout.for_(12, 2)

    assert layout.n_splits == math.comb(12, 2) == 66
    assert layout.n_paths == math.comb(11, 1) == 11


def test_every_split_is_leakage_free(index):
    folds = combinatorial_purged_splits(index, purge_bars=HORIZON, embargo_bars=HORIZON)

    assert len(folds) == 66
    for fold in folds:
        assert_no_leakage(fold, horizon_bars=HORIZON)
        assert not fold.train_idx.intersection(fold.test_idx).size


def test_short_purge_is_detected(index):
    """A purge shorter than the label horizon must not pass silently."""
    folds = combinatorial_purged_splits(index, purge_bars=4, embargo_bars=4)

    flagged = 0
    for fold in folds:
        try:
            assert_no_leakage(fold, horizon_bars=HORIZON)
        except ValueError:
            flagged += 1

    assert flagged > 0, 'a 4-bar purge against a 72-bar horizon went unnoticed'


def test_paths_cover_the_timeline_exactly_once(index):
    """Each assembled path is one complete out-of-sample history."""
    folds = combinatorial_purged_splits(index, purge_bars=HORIZON, embargo_bars=HORIZON)
    paths = assemble_paths(folds, n_groups=12)

    assert len(paths) == CPCVLayout.for_(12, 2).n_paths

    for path in paths:
        covered = path_test_index(path, index, n_groups=12)
        assert covered.equals(index), 'a path did not cover the whole timeline'
        assert covered.is_unique, 'a path covered a bar twice'
        for _, fold in path:
            assert not fold.train_idx.intersection(fold.test_idx).size


def test_walk_forward_never_trains_on_the_future(index):
    folds = purged_walk_forward(
        index, n_folds=6, min_train_bars=720, purge_bars=HORIZON, embargo_bars=HORIZON
    )

    assert folds
    for fold in folds:
        assert fold.train_idx.max() < fold.test_idx.min()
        assert_no_leakage(fold, horizon_bars=HORIZON)


# ---------------------------------------------------------------------------
# Label overlap
# ---------------------------------------------------------------------------


def test_unit_horizon_recovers_the_full_sample(index):
    """Consecutive one-bar labels share no returns, so each is fully unique.

    Getting the span off by one bar halves the apparent sample here, which then
    propagates into every significance test as inflated confidence.
    """
    uniqueness = average_uniqueness(index, 1)
    resolvable = uniqueness > 0

    assert effective_sample_size(index, 1) == pytest.approx(len(index), abs=1.5)
    assert uniqueness[resolvable].min() > 0.99


def test_effective_sample_scales_with_the_horizon(index):
    """A 72-hour horizon on hourly bars carries roughly n/72 real observations."""
    effective = effective_sample_size(index, HORIZON)

    assert effective == pytest.approx(len(index) / HORIZON, rel=0.05)
    assert 35 < effective < 45, f'expected ~40 independent observations, got {effective:.1f}'


def test_uniqueness_is_bounded(index):
    uniqueness = average_uniqueness(index, HORIZON)
    resolvable = uniqueness > 0

    assert uniqueness[resolvable].min() > 0.0
    assert uniqueness.max() <= 1.0 + 1e-12


def test_unresolvable_labels_carry_no_weight(index):
    """A label opened within one horizon of the end has no outcome.

    Counting it would overstate the evidence at exactly the recent edge of the
    sample, which is where a search is most tempted to trust a result.
    """
    uniqueness = average_uniqueness(index, HORIZON)

    assert (uniqueness[-HORIZON:] == 0.0).all()
    assert (uniqueness[:-HORIZON] > 0.0).all()


def test_concurrency_matches_the_horizon(index):
    """Away from the edges, a horizon of h means h overlapping labels."""
    concurrency = label_concurrency(index, HORIZON)
    interior = concurrency[HORIZON:-HORIZON]

    assert interior.min() == pytest.approx(HORIZON, abs=1)
    assert interior.max() == pytest.approx(HORIZON, abs=1)


def test_weights_are_normalised_and_favour_recent_data(index):
    """Normalised to mean 1, and heavier toward the present.

    Compared across labelled rows only: the trailing horizon carries zero weight
    because those labels cannot resolve, so the very last row is not the heaviest.
    """
    weights = sample_weights(index, horizon_bars=HORIZON, half_life_days=50)
    labelled = weights[weights > 0]

    assert weights.mean() == pytest.approx(1.0)
    assert labelled[-1] > labelled[0]
    assert (weights[-HORIZON:] == 0.0).all()
    assert recency_weights(index, 0).min() == 1.0      # decay disabled


# ---------------------------------------------------------------------------
# Significance
# ---------------------------------------------------------------------------


def test_deflated_sharpe_handles_a_single_trial():
    """One configuration has nothing to deflate against.

    The previous implementation raised StatisticsError here, because
    inv_cdf(1 - 1/1) is inv_cdf(0).
    """
    result = deflated_sharpe(sharpe=1.0, observations=250, trials=1)

    assert result.valid
    assert result.detail['expected_max_sharpe'] == 0.0


def test_expected_max_sharpe_grows_with_the_search():
    """The bar a real edge must clear rises with the number of tries."""
    few = expected_max_sharpe(10, 40)
    many = expected_max_sharpe(3_000, 40)

    assert 0 < few < many
    # and falls as evidence accumulates
    assert expected_max_sharpe(3_000, 2_880) < many


def test_deflated_sharpe_rejects_a_lucky_winner():
    """A Sharpe that survives on 2,880 rows can fail on 40 real observations."""
    lucky = deflated_sharpe(sharpe=0.55, observations=40, trials=3_000)
    pooled = deflated_sharpe(sharpe=0.55, observations=650, trials=3_000)

    assert lucky.detail['p_value'] > 0.10, 'a coin-flip winner passed'
    assert pooled.detail['p_value'] < lucky.detail['p_value']


def test_probabilistic_sharpe_needs_two_observations():
    assert not probabilistic_sharpe(sharpe=1.0, observations=1).valid
    assert probabilistic_sharpe(sharpe=1.0, observations=40).valid


def test_pbo_detects_pure_noise():
    """With no real edge, selection should land near chance."""
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 1, (40, 12))

    result = probability_of_backtest_overfitting(noise)

    assert result.valid
    assert 0.2 < result.pbo < 0.8, f'noise gave pbo={result.pbo}'


def test_pbo_detects_a_genuine_edge():
    """One candidate that is truly better should rank high out of sample."""
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 0.3, (40, 12))
    scores[0] += 3.0                       # a real, persistent edge

    result = probability_of_backtest_overfitting(scores)

    assert result.pbo == pytest.approx(0.0, abs=1e-9)


def test_pbo_requires_a_matrix():
    assert not probability_of_backtest_overfitting(np.zeros((1, 5))).valid
    assert not probability_of_backtest_overfitting(np.zeros((5, 1))).valid


# ---------------------------------------------------------------------------
# Descriptive
# ---------------------------------------------------------------------------


def test_sharpe_of_a_flat_curve_is_zero():
    assert sharpe_ratio([0.0] * 100) == 0.0
    assert sharpe_ratio([0.001] * 100) == 0.0      # no variance


def test_drawdown_reports_an_unrecovered_trough():
    """A curve that never recovers must not report a recovery time."""
    falling = list(np.linspace(100, 60, 50))

    profile = drawdown_profile(falling)

    assert profile.max_drawdown == pytest.approx(0.4, rel=0.01)
    assert profile.time_to_recovery is None


def test_drawdown_reports_recovery_when_it_happens():
    curve = list(np.linspace(100, 80, 20)) + list(np.linspace(80, 120, 40))

    profile = drawdown_profile(curve)

    assert profile.time_to_recovery is not None
    assert profile.max_drawdown == pytest.approx(0.2, rel=0.02)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def test_path_distribution_exposes_the_bad_tail():
    """A good mean with a negative 5th percentile must remain visible."""
    distribution = summarise_paths([1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.5, 0.2, -0.3, -0.8])

    assert distribution.median > 0.8
    assert distribution.p05 < 0
    assert distribution.positive_fraction < 1.0


def test_unmeasured_gates_fail_rather_than_pass():
    """"We did not run that test" is not evidence of safety."""
    promoted, gates = evaluate_gates({'walk_forward_median_sharpe': 0.9})

    assert not promoted
    unmeasured = [g for g in gates if g.value is None]
    assert unmeasured
    assert all(not g.passed for g in unmeasured)


def test_all_gates_passing_promotes():
    measurements = {
        'walk_forward_median_sharpe': 0.8, 'walk_forward_p05_sharpe': 0.1, 'pbo': 0.15,
        'deflated_sharpe': 1.9, 'bootstrap_positive_fraction': 0.95,
        'synthetic_positive_fraction': 0.7, 'stressed_median_sharpe': 0.2,
        'parameter_plateau': 0.8, 'oos_trades': 250,
        'max_exit_participation': 0.06,
        'proxy_funding_symbols': 0.0,   # carry measured on the traded venue
        'ic_covers_cost': 1.4,          # the forecast clears its own round trip
    }

    promoted, gates = evaluate_gates(measurements)

    assert set(measurements) == set(DEFAULT_GATES)
    assert promoted
    assert 'PROMOTED' in gate_report(gates)


def test_one_failing_gate_blocks():
    measurements = {
        'walk_forward_median_sharpe': 0.8, 'walk_forward_p05_sharpe': -0.2,      # fails
        'pbo': 0.15, 'deflated_sharpe': 1.9,
        'bootstrap_positive_fraction': 0.95, 'synthetic_positive_fraction': 0.7,
        'stressed_median_sharpe': 0.2, 'parameter_plateau': 0.8, 'oos_trades': 250,
        'max_exit_participation': 0.06, 'proxy_funding_symbols': 0.0,
        'ic_covers_cost': 1.4,
    }

    promoted, gates = evaluate_gates(measurements)

    assert not promoted
    assert 'BLOCKED by 1 gate' in gate_report(gates)


# ---------------------------------------------------------------------------
# Per-fold preprocessing
# ---------------------------------------------------------------------------


def test_scaler_is_fitted_on_training_rows_only():
    """A scaler fitted on the whole sample leaks test statistics into training.

    The tell is that the training block, standardised correctly, has mean zero
    and unit variance while the test block does not — because the test block was
    never part of the fit.
    """
    from sklearn.preprocessing import StandardScaler

    from core.cv import FoldPreprocessor, preprocess_fold

    train = pd.DataFrame({'x': np.arange(100, dtype=float)})
    test = pd.DataFrame({'x': np.arange(100, 150, dtype=float)})

    scaled_train, scaled_test, fitted = preprocess_fold(
        train, test, preprocessor=FoldPreprocessor(scaler_factory=StandardScaler)
    )

    assert scaled_train['x'].mean() == pytest.approx(0.0, abs=1e-9)
    assert scaled_train['x'].std(ddof=0) == pytest.approx(1.0, abs=1e-9)
    # The test block sits entirely above the training range, so its scaled mean
    # must be well above zero. If it came out near zero the scaler saw it.
    assert scaled_test['x'].mean() > 2.0
    assert fitted.scaler.mean_[0] == pytest.approx(train['x'].mean())


def test_preprocessor_is_a_noop_without_factories():
    from core.cv import preprocess_fold

    train = pd.DataFrame({'x': [1.0, 2.0, 3.0]})
    test = pd.DataFrame({'x': [4.0, 5.0]})

    scaled_train, scaled_test, _ = preprocess_fold(train, test)

    pd.testing.assert_frame_equal(scaled_train, train)
    pd.testing.assert_frame_equal(scaled_test, test)


def test_the_reported_head_ic_is_a_holdout_measurement():
    """A validation IC labelled "in-sample" inverts what it tells you.

    `train_forecast_model` fits on `x_train` and measures every head's `ic` on
    `x_val` — the chronological, horizon-purged holdout. preflight printed that as
    "in-sample IC", so a price IC of +0.004 read as "the model cannot fit its own
    training data, something is broken" rather than "this generalises to no
    directional edge". The second is a finding; the first sends you looking for a
    bug that is not there.

    Pinned two ways: the metric must come from the holdout, and no surface may
    call it in-sample.
    """
    import inspect
    from pathlib import Path

    from core import model as model_module

    source = inspect.getsource(model_module.train_forecast_model)
    # The forecast the IC is computed from must be predicted on the validation
    # split, not the training one.
    assert "model.predict(x_val)" in source, (
        'the head IC is no longer measured on the holdout; any surface calling it '
        'a holdout measurement is now lying'
    )

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in list(root.rglob('*.py')) + list(root.rglob('*.md')):
        if '__pycache__' in path.parts or path.name == Path(__file__).name:
            continue
        if 'in-sample IC' in path.read_text():
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f'holdout IC described as in-sample in: {offenders}'


def test_net_ic_is_reported_against_what_a_skill_free_forecast_scores():
    """`expected_net` and `realised_net` share the cost term, so net IC starts positive.

        expected_net = forecast_price + forecast_carry - cost
        realised_net = realised_price + realised_carry - cost

    `-cost` is identical in both, and cost is known at decision time rather than
    forecast, so correlating the two credits the fee schedule as signal. Measured
    on this store at h=4h a forecast predicting price = 0 and carry = 0 scores
    net IC +0.0714 pooled and +0.1094 cross-sectionally, with every fold
    positive — which read against zero looks exactly like a stable edge.

    It was being read against zero. A `volatility,trend,market_factor` feature
    set at h=4h reported net IC +0.0461 across six folds, all positive, while
    its price IC was -0.0107: the model had no directional skill and the metric
    said six-of-six.

    So the floor is measured per fold and `net_ic_skill` is the difference. This
    test builds a forecast with no skill by construction and asserts that the
    floor catches it — that net IC is comfortably positive, that the floor is
    just as positive, and that the skill is not.
    """
    import numpy as np
    import pandas as pd

    from core.model import cross_sectional_ic, information_coefficient

    rng = np.random.default_rng(11)
    n_times, symbols = 400, ('BIP', 'ETP', 'XPP', 'DOP')
    index = pd.MultiIndex.from_product(
        [pd.date_range('2026-01-01', periods=n_times, freq='4h', tz='UTC'), symbols],
        names=['event_time', 'symbol'],
    )
    # Costs differ per instrument the way the real schedule does: one percentage
    # and one commission, so the spread comes from notional per contract.
    per_symbol = {'BIP': 0.0027, 'ETP': 0.0034, 'XPP': 0.0027, 'DOP': 0.0029}
    cost = np.array([per_symbol[s] for _, s in index])
    # Realised price moves are pure noise, and the forecast is identically zero.
    realised_price = rng.normal(0.0, 0.0144, len(index))
    forecast = np.zeros(len(index))

    realised_net = realised_price - cost
    expected_net = forecast - cost

    net_ic = information_coefficient(expected_net, realised_net)
    floor = information_coefficient(-cost, realised_net)
    net_ic_xs = cross_sectional_ic(expected_net, realised_net, index)
    floor_xs = cross_sectional_ic(-cost, realised_net, index)

    # The metric is positive on a forecast that predicts nothing at all.
    assert net_ic > 0.02, (
        f'net IC {net_ic:+.4f} on a zero forecast — if this is near zero the '
        f'fixture no longer reproduces the defect and the floor is untested'
    )
    # And the floor accounts for all of it, which is the whole point.
    assert net_ic == pytest.approx(floor, abs=1e-9)
    assert net_ic_xs == pytest.approx(floor_xs, abs=1e-9)
    assert net_ic - floor == pytest.approx(0.0, abs=1e-9)

    # The realised price forecast really was skill-free, so nothing was thrown away.
    assert abs(information_coefficient(forecast, realised_price)) < 1e-9 or np.isnan(
        information_coefficient(forecast, realised_price)
    )


def test_the_cv_report_flags_a_net_ic_that_is_only_the_cost_term():
    """The floor has to reach the report, not just the fold.

    A per-fold number nobody aggregates is a number nobody reads. `net_ic_skill`
    is the difference against the floor and `net_ic_is_cost_only` is the flag,
    and `__str__` has to say so — the string is what a research run prints and
    what someone pastes into a commit message.
    """
    from core.metrics import summarise_paths
    from core.model import CVReport

    def report(net_ic: float, floor: float) -> CVReport:
        return CVReport(
            folds=[],
            price_ic=summarise_paths([0.001]),
            carry_ic=summarise_paths([0.0]),
            net_ic=summarise_paths([net_ic]),
            price_ic_xs=summarise_paths([0.001]),
            net_ic_xs=summarise_paths([net_ic]),
            net_ic_cost_only=summarise_paths([floor]),
            net_ic_xs_cost_only=summarise_paths([floor]),
            identity_ceiling=summarise_paths([0.01]),
            total_effective_observations=1_000.0,
        )

    # The case that actually happened: net IC positive, floor higher.
    fake = report(net_ic=0.0461, floor=0.0714)
    assert fake.net_ic_skill < 0
    assert fake.net_ic_is_cost_only
    assert 'NET IC IS THE COST TERM' in str(fake)

    real = report(net_ic=0.12, floor=0.0714)
    assert real.net_ic_skill == pytest.approx(0.0486, abs=1e-4)
    assert not real.net_ic_is_cost_only
    assert 'NET IC IS THE COST TERM' not in str(real)
