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
    promoted, gates = evaluate_gates({'cpcv_median_sharpe': 0.9})

    assert not promoted
    unmeasured = [g for g in gates if g.value is None]
    assert unmeasured
    assert all(not g.passed for g in unmeasured)


def test_all_gates_passing_promotes():
    measurements = {
        'cpcv_median_sharpe': 0.8, 'cpcv_p05_sharpe': 0.1, 'pbo': 0.15,
        'deflated_sharpe': 1.9, 'bootstrap_positive_fraction': 0.95,
        'synthetic_positive_fraction': 0.7, 'stressed_median_sharpe': 0.2,
        'parameter_plateau': 0.8, 'oos_trades': 250,
        'max_exit_participation': 0.06,
    }

    promoted, gates = evaluate_gates(measurements)

    assert set(measurements) == set(DEFAULT_GATES)
    assert promoted
    assert 'PROMOTED' in gate_report(gates)


def test_one_failing_gate_blocks():
    measurements = {
        'cpcv_median_sharpe': 0.8, 'cpcv_p05_sharpe': -0.2,      # fails
        'pbo': 0.15, 'deflated_sharpe': 1.9,
        'bootstrap_positive_fraction': 0.95, 'synthetic_positive_fraction': 0.7,
        'stressed_median_sharpe': 0.2, 'parameter_plateau': 0.8, 'oos_trades': 250,
        'max_exit_participation': 0.06,
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
