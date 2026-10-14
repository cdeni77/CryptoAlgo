"""Tests for the noise null subtracted from `calibration_vs_market`.

ECE uses absolute deviations, so any unbiased perturbation of a calibrated
forecast raises it. A market-initialised model therefore scores above zero just
by moving. The gate compares the model with a random correction of the same size.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import calibration_noise_null, market_gate_values


def _market_rows(n=4000, seed=0):
    """A well-calibrated market: P(up) drawn, outcomes drawn from it."""
    rng = np.random.default_rng(seed)
    k = np.clip(rng.normal(0.5, 0.18, n), 0.05, 0.95)
    y = (rng.random(n) < k).astype(float)
    return k, y


def test_the_null_is_positive_for_any_real_correction():
    """Pure noise scores above zero, so zero is the wrong bar."""
    k, y = _market_rows()
    null = calibration_noise_null(k, y, 0.25, draws=60)
    assert null > 0, 'if noise scored zero there would be nothing to correct'


def test_a_bigger_correction_has_a_bigger_null():
    """The null grows with the correction size, so it is computed at the candidate's own
    magnitude.
    """
    k, y = _market_rows()
    small = calibration_noise_null(k, y, 0.10, draws=60)
    large = calibration_noise_null(k, y, 0.50, draws=60)
    assert large > small


def test_no_correction_means_no_null():
    k, y = _market_rows()
    assert calibration_noise_null(k, y, 0.0, draws=20) == 0.0


def test_the_null_is_deterministic_for_a_given_seed():
    """The same inputs give the same null."""
    k, y = _market_rows()
    a = calibration_noise_null(k, y, 0.25, draws=40)
    b = calibration_noise_null(k, y, 0.25, draws=40)
    assert a == b


def test_a_model_that_merely_perturbs_the_market_scores_near_zero():
    """A correction that is pure noise scores near zero after the null is subtracted."""
    k, y = _market_rows(n=6000)
    rng = np.random.default_rng(7)
    lg = np.log(k/(1-k))
    noisy = 1/(1+np.exp(-(lg + rng.normal(0, 0.25, len(k)))))
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, mm, yy)
            for i, (kk, mm, yy) in enumerate(zip(k, noisy, y))]
    v = market_gate_values(rows)
    assert abs(v['calibration_vs_market']) < 0.002, (
        f"pure noise should sit near the null, got "
        f"{v['calibration_vs_market']:+.5f}")


def test_a_systematically_overconfident_model_still_fails():
    """A systematically overconfident model still fails the gate."""
    k, y = _market_rows(n=6000)
    lg = np.log(k/(1-k))
    overconfident = 1/(1+np.exp(-lg*1.6))  # pushed away from 0.5, no noise
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, mm, yy)
            for i, (kk, mm, yy) in enumerate(zip(k, overconfident, y))]
    v = market_gate_values(rows)
    assert v['calibration_vs_market'] > 0, 'systematic overconfidence must fail'


def test_the_null_is_reported_alongside_the_gate():
    """The null is reported beside the gate, so the margin over chance is visible."""
    k, y = _market_rows(n=1000)
    rows = [('BTC-USD', f'w{i}', 12, kk, 0.5, kk, yy)
            for i, (kk, yy) in enumerate(zip(k, y))]
    assert 'calibration_noise_null' in market_gate_values(rows)
