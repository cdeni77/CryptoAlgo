"""Tests for `implied_sigma`, which inverts a Kalshi strike ladder to the sigma that priced
it.

Under the zero-drift barrier model, P(S_T > K) = F(ln(S/K) / (sigma * sqrt(t))).
Applying the inverse CDF makes the ladder linear in ln(K/S) with slope
-1 / (sigma * sqrt(t)). One regression recovers sigma, and R^2 shows whether the
ladder is consistent. This is the market's forward estimate of sigma, while
`core/vol.py` uses past returns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.record_implied_vol import implied_sigma


def ladder(sigma_per_min: float, minutes: float, spot: float,
           zs=(-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2)):
    """Quotes a zero-drift model would post, with strikes spaced in sigma units.

    Fixed basis-point spacing puts the outer rungs at 0 and 1, where they carry no
    slope. A real ladder brackets the money.
    """
    from statistics import NormalDist
    normal = NormalDist()
    scale = sigma_per_min * math.sqrt(minutes)
    return [(spot * math.exp(-z * scale), normal.cdf(z)) for z in zs]


def test_it_recovers_the_sigma_that_priced_the_ladder():
    fit = implied_sigma(ladder(0.00035, 60.0, 77200.0), 60.0)
    assert fit is not None
    assert fit.sigma_per_min == pytest.approx(0.00035, rel=1e-6)
    assert fit.r2 > 0.999


def test_it_recovers_sigma_at_a_different_horizon():
    """The sqrt(t) scaling is correct at another horizon."""
    fit = implied_sigma(ladder(0.0005, 12.0, 60000.0), 12.0)
    assert fit.sigma_per_min == pytest.approx(0.0005, rel=1e-6)


def test_the_atm_strike_is_the_one_nearest_a_coin_flip():
    fit = implied_sigma(ladder(0.00035, 60.0, 77200.0), 60.0)
    assert abs(fit.atm_strike - 77200.0) < 1.0


def test_a_noisy_ladder_still_fits_but_reports_a_lower_r2():
    rng = np.random.default_rng(20260826)
    noisy = [(k, float(np.clip(p + rng.normal(0, 0.02), 1e-3, 1 - 1e-3)))
             for k, p in ladder(0.00035, 60.0, 77200.0)]
    fit = implied_sigma(noisy, 60.0)
    assert fit is not None
    assert fit.sigma_per_min == pytest.approx(0.00035, rel=0.25)
    assert fit.r2 < 0.999


def test_degenerate_quotes_are_refused_rather_than_fitted():
    """All-0 or all-1 quotes carry no slope, so no sigma is returned."""
    assert implied_sigma([(70000.0, 1.0), (75000.0, 1.0), (80000.0, 1.0)], 60.0) is None


def test_far_rungs_are_dropped_rather_than_allowed_to_dominate():
    """Far rungs are dropped: near 0 or 1 they are mostly tick noise after the inverse CDF.
    """
    rungs = ladder(0.00035, 60.0, 77200.0)
    with_tails = rungs + [(60000.0, 0.999), (95000.0, 0.001)]
    assert implied_sigma(with_tails, 60.0).n_strikes == len(rungs)


def test_too_few_usable_strikes_is_refused():
    assert implied_sigma([(77000.0, 0.55), (77500.0, 0.45)], 60.0) is None


def test_zero_minutes_remaining_is_refused_not_divided_by():
    assert implied_sigma(ladder(0.00035, 60.0, 77200.0), 0.0) is None


def test_an_inverted_ladder_is_refused():
    """A ladder whose price rises with strike is refused, since sigma would be negative.
    """
    rungs = [(k, p) for k, p in ladder(0.00035, 60.0, 77200.0)]
    flipped = [(k, p) for (k, _), (_, p) in zip(rungs, reversed(rungs))]
    assert implied_sigma(flipped, 60.0) is None
