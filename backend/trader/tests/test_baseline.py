"""Tests for the barrier baseline `F(displacement / sigma_remaining)`, the null.

It may fit a scale and a tail thickness, never a drift: drift is the alpha under test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.baseline import (
    MAX_NU, MIN_NU, PROB_EPS, BarrierBaseline, attach_baseline, brier, clip_prob,
    expit, log_loss, logit, reliability,
)
from core.config import Config


def barrier_table(n=120_000, seed=11, nu=5.0, inflation=None, drift=0.0):
    """A world where the barrier arithmetic is exactly true.

    Known tail thickness, a known per-offset inflation of the reported sigma, and an
    optional drift. The fit must recover the first two and never the third.
    """
    rng = np.random.default_rng(seed)
    inflation = inflation or {3: 1.25, 6: 1.20, 9: 1.15, 12: 1.10}
    offset = rng.choice(sorted(inflation), n)
    remaining = 15 - offset
    sigma_true = 1.6e-4 * np.sqrt(remaining)
    reported = sigma_true * np.array([inflation[int(o)] for o in offset])
    shock = (stats.t.rvs(nu, size=n, random_state=seed + 1)
             / np.sqrt(nu / (nu - 2)) * sigma_true)
    displacement = rng.normal(0, 1.5e-4, n) * np.sqrt(offset)
    outcome = (displacement + shock + drift * sigma_true > 0).astype(int)
    return pd.DataFrame({'displacement': displacement, 'sigma_remaining': reported,
                         'offset': offset, 'outcome': outcome})


def test_the_fit_recovers_a_planted_tail_thickness():
    table = barrier_table(nu=5.0)
    fitted = BarrierBaseline.fit(table, Config(baseline_distribution='student_t'))
    assert 3.0 < fitted.nu < 9.0, fitted.nu


def test_the_fitted_barrier_beats_an_uncalibrated_one():
    """The fitted barrier calibrates better than a naive Gaussian at unit scale.

    Binary outcomes identify only the composite `z -> P(up)`, not scale and tail
    separately, so the test compares calibration instead of parameter values.
    """
    table = barrier_table(inflation={3: 1.40, 6: 1.30, 9: 1.25, 12: 1.20},
                          n=200_000)
    fitted = BarrierBaseline.fit(table, Config())
    naive = BarrierBaseline(distribution='normal', nu=MAX_NU,
                            scale={o: 1.0 for o in (3, 6, 9, 12)})

    y = table['outcome'].to_numpy()
    assert log_loss(y, fitted.probability_for(table)) < log_loss(
        y, naive.probability_for(table))
    assert (reliability(y, fitted.probability_for(table)).expected_calibration_error
            < reliability(y, naive.probability_for(table)).expected_calibration_error)


def test_a_more_inflated_sigma_yields_a_smaller_scale():
    """More sigma inflation yields a smaller fitted scale (uniform inflation, so no confound)."""
    mild = BarrierBaseline.fit(
        barrier_table(inflation={o: 1.05 for o in (3, 6, 9, 12)}, n=120_000),
        Config(baseline_fit_scale_per_offset=False))
    severe = BarrierBaseline.fit(
        barrier_table(inflation={o: 1.60 for o in (3, 6, 9, 12)}, n=120_000),
        Config(baseline_fit_scale_per_offset=False))
    assert severe.default_scale < mild.default_scale, (
        f'a sigma inflated 1.60x got scale {severe.default_scale:.4f}, one '
        f'inflated 1.05x got {mild.default_scale:.4f}'
    )


def test_fat_tails_calibrate_better_than_a_gaussian_when_the_tails_are_fat():
    table = barrier_table(nu=4.0, n=200_000)
    gaussian = BarrierBaseline.fit(table, Config(baseline_distribution='normal'))
    student = BarrierBaseline.fit(table, Config(baseline_distribution='student_t'))
    assert student.fitted_log_loss < gaussian.fitted_log_loss
    y = table['outcome'].to_numpy()
    ece_gaussian = reliability(y, gaussian.probability_for(table)).expected_calibration_error
    ece_student = reliability(y, student.probability_for(table)).expected_calibration_error
    assert ece_student < ece_gaussian


def test_the_baseline_has_no_drift_to_fit():
    """A planted drift stays visible as a gap between mean prediction and base rate."""
    drifted = barrier_table(drift=1.2, n=120_000)
    fitted = BarrierBaseline.fit(drifted, Config())
    predicted = fitted.probability_for(drifted)
    realised = drifted['outcome'].mean()
    assert realised > 0.60, 'the planted drift did not take'
    assert predicted.mean() < realised - 0.05, (
        f'the baseline absorbed the drift (predicted {predicted.mean():.3f} vs '
        f'realised {realised:.3f}); a null that fits the alpha hides it'
    )


def test_a_zero_displacement_is_a_coin_flip_whatever_the_sigma():
    fitted = BarrierBaseline(distribution='student_t', nu=5.0, scale={9: 1.0})
    probability = fitted.probability(np.zeros(3), np.array([1e-5, 1e-3, 1e-1]),
                                     np.array([9, 9, 9]))
    assert probability == pytest.approx(0.5, abs=1e-9)


def test_the_probability_is_monotone_in_displacement():
    fitted = BarrierBaseline(distribution='student_t', nu=5.0, scale={9: 1.0})
    displacement = np.linspace(-5e-3, 5e-3, 50)
    probability = fitted.probability(displacement, np.full(50, 1e-3), np.full(50, 9))
    assert np.all(np.diff(probability) > 0)


def test_probabilities_are_clipped_so_one_wrong_certainty_is_not_infinite():
    assert clip_prob(np.array([0.0, 1.0])).tolist() == [PROB_EPS, 1 - PROB_EPS]
    loss = log_loss(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(loss) and loss > 10


def test_logit_and_expit_round_trip():
    p = np.array([0.01, 0.25, 0.5, 0.75, 0.99])
    assert expit(logit(p)) == pytest.approx(p, abs=1e-9)


def test_the_scale_is_fitted_per_offset_only_when_asked():
    table = barrier_table(n=60_000)
    shared = BarrierBaseline.fit(
        table, Config(baseline_fit_scale_per_offset=False))
    assert len(shared.scale) == 1
    per_offset = BarrierBaseline.fit(table, Config(baseline_fit_scale_per_offset=True))
    assert len(per_offset.scale) == 4


def test_a_student_t_is_standardised_to_unit_variance():
    """The Student-t is standardised to unit variance (divided by `sqrt(nu/(nu-2))`).

    Without it every probability moves toward one half, which looks conservative instead
    of wrong.
    """
    for nu in (3.0, 5.0, 10.0):
        fitted = BarrierBaseline(distribution='student_t', nu=nu, scale={9: 1.0})
        standardised = float(
            fitted.probability(np.array([1.0]), np.array([1.0]), np.array([9]))[0])
        raw = float(stats.t.cdf(1.0, df=nu))
        assert standardised > raw + 0.01, (nu, standardised, raw)
        # The value also stays a probability above one half, in the right direction.
        assert 0.80 < standardised < 0.95, (nu, standardised)

    # In the limit it is the Gaussian.
    gaussian_like = BarrierBaseline(distribution='student_t', nu=MAX_NU, scale={9: 1.0})
    assert float(gaussian_like.probability(
        np.array([1.0]), np.array([1.0]), np.array([9]))[0]) == pytest.approx(
        float(stats.norm.cdf(1.0)), abs=0.01)


def test_attach_baseline_adds_the_logit_the_model_consumes():
    table = barrier_table(n=5_000)
    fitted = BarrierBaseline.fit(table, Config())
    attached = attach_baseline(table, fitted)
    assert 'baseline_probability' in attached
    assert 'baseline_probability_logit' in attached
    assert np.allclose(expit(attached['baseline_probability_logit']),
                       attached['baseline_probability'], atol=1e-9)


def test_reliability_bins_are_fixed_not_quantile():
    """Reliability bins use fixed edges, so two runs' calibration tables are comparable."""
    y = np.array([0, 1] * 500)
    p = np.linspace(0.01, 0.99, 1000)
    first = reliability(y, p)
    second = reliability(y, p ** 2)
    assert np.array_equal(first.edges, second.edges)


def test_the_fit_refuses_a_table_with_no_usable_rows():
    empty = pd.DataFrame({'displacement': [np.nan], 'sigma_remaining': [0.0],
                          'offset': [9], 'outcome': [1]})
    with pytest.raises(ValueError, match='no usable rows'):
        BarrierBaseline.fit(empty, Config())


def test_brier_agrees_with_log_loss_on_which_forecast_is_better():
    y = np.array([1, 1, 0, 0] * 250)
    good = np.where(y == 1, 0.8, 0.2)
    bad = np.where(y == 1, 0.55, 0.45)
    assert log_loss(y, good) < log_loss(y, bad)
    assert brier(y, good) < brier(y, bad)
