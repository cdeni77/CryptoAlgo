"""Tests for `focus_quantile` and `max_scale` in `_fit_residual_scale`.

Trades are the largest-|correction| rows, where the model tends to be overconfident, so
the shrinkage can be fitted on them.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.model import MIN_SHRINKAGE_ROWS, _fit_residual_scale


# 20k rows, so the top decile (2,000) clears MIN_SHRINKAGE_ROWS. At 4,000 the
# subset is 400 and the fit backs off instead.
def _sample(n=20000, *, tail_overconfident=True, seed=0):
    """Rows whose large corrections are too confident and small ones honest."""
    rng = np.random.default_rng(seed)
    base = np.zeros(n)
    correction = rng.normal(0, 1.0, n)
    truth = correction.copy()
    if tail_overconfident:
        big = np.abs(correction) >= np.quantile(np.abs(correction), 0.90)
        truth[big] = correction[big] * 0.3  # only 30% of the tail is real
    p = 1.0 / (1.0 + np.exp(-(base + truth)))
    return base, correction, (rng.random(n) < p).astype(float)


def test_focusing_shrinks_harder_when_the_tail_is_overconfident():
    base, corr, y = _sample()
    everywhere = _fit_residual_scale(base, corr, y)
    on_the_tail = _fit_residual_scale(base, corr, y, focus_quantile=0.90)
    assert on_the_tail < everywhere, (
        'the tail is where the model is wrong, so fitting there must shrink '
        f'more (got {on_the_tail:.3f} vs {everywhere:.3f})'
    )


def test_an_honest_model_is_not_shrunk_by_focusing():
    """If the tail is as good as the body, focusing gives about the same alpha."""
    base, corr, y = _sample(tail_overconfident=False)
    everywhere = _fit_residual_scale(base, corr, y)
    on_the_tail = _fit_residual_scale(base, corr, y, focus_quantile=0.90)
    assert on_the_tail == pytest.approx(everywhere, abs=0.15)


def test_zero_keeps_the_historical_behaviour():
    base, corr, y = _sample()
    assert _fit_residual_scale(base, corr, y, focus_quantile=0.0) == \
        _fit_residual_scale(base, corr, y)


def test_it_backs_off_rather_than_raising_on_a_thin_subset():
    """On too thin a subset the fit falls back to all rows instead of raising."""
    n = MIN_SHRINKAGE_ROWS * 2
    base, corr, y = _sample(n=n)
    # The top 1% of 1,000 rows is 10, far under the minimum.
    focused = _fit_residual_scale(base, corr, y, focus_quantile=0.99)
    assert focused == pytest.approx(_fit_residual_scale(base, corr, y))


def test_the_ranking_does_not_depend_on_alpha():
    """Alpha scales the correction, so the order of |correction| does not change."""
    _, corr, _ = _sample()
    order = np.argsort(np.abs(corr))
    for alpha in (0.25, 1.0, 1.9):
        assert np.array_equal(np.argsort(np.abs(alpha * corr)), order)


def test_the_ceiling_is_respected():
    """`max_scale` caps alpha; a value above 1 would amplify the correction, a sign of
    overfitting.
    """
    # A sample whose correction is too small, so an uncapped fit wants to amplify it.
    rng = np.random.default_rng(1)
    n = 20000
    base = np.zeros(n)
    correction = rng.normal(0, 0.4, n)
    p = 1.0 / (1.0 + np.exp(-(base + correction * 2.5)))  # truth is 2.5x
    y = (rng.random(n) < p).astype(float)

    free = _fit_residual_scale(base, correction, y)
    assert free > 1.0, 'the fixture must actually want amplification'

    capped = _fit_residual_scale(base, correction, y, max_scale=1.0)
    assert capped <= 1.0 + 1e-9
    assert capped == pytest.approx(1.0, abs=1e-3), 'it should sit at the cap'


def test_the_ceiling_does_not_disturb_a_shrinking_fit():
    """The cap does not change an alpha that is already below 1."""
    base, corr, y = _sample()  # tail is overconfident
    free = _fit_residual_scale(base, corr, y)
    if free < 1.0:
        assert _fit_residual_scale(base, corr, y, max_scale=1.0) == \
            pytest.approx(free, abs=1e-6)


def test_the_default_ceiling_is_one():
    """The default ceiling is 1: a shrinkage that can exceed 1 is not a shrinkage."""
    from core.config import Config

    assert Config().max_residual_scale == 1.0


def test_a_fitted_alpha_cannot_exceed_the_default_ceiling():
    """With the default config, a fitted alpha never exceeds 1."""
    rng = np.random.default_rng(2)
    n = 20000
    base = np.zeros(n)
    corr = rng.normal(0, 0.4, n)
    p = 1.0 / (1.0 + np.exp(-(base + corr * 3.0)))  # truth is 3x the correction
    y = (rng.random(n) < p).astype(float)
    from core.config import Config

    alpha = _fit_residual_scale(base, corr, y,
                                max_scale=Config().max_residual_scale)
    assert alpha <= 1.0 + 1e-9
