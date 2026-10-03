"""The shrinkage must be fitted where the money is.

`residual_scale` (alpha) answers "how much of the claimed correction survives
out of sample". Fitted over ALL validation rows it optimises the AVERAGE — and
the average is fine: measured on 7,611 live rows the model's pooled ECE is only
0.0032 worse than the market's.

But the ~9% of rows that become trades are a selected sample. `decide()` takes
them precisely where the model most disagrees with the price, which is where it
is most likely to be wrong, and there it claimed **65.07% and delivered 58.53%**
over 217 live trades — a 6.7pp overconfidence (90% CI [+0.73, +12.55]) that
accounts for essentially the whole gap between a +4.90pp predicted edge and a
−1.32pp realised one. The backtest shows the same shape at 2.55pp, so this is a
property of the model rather than of live luck.

Alpha never saw those rows separately, so it had no reason to correct them.

The subset is chosen by |correction| rather than by simulating `decide()`,
which matters: alpha scales the correction monotonically, so the RANKING is
alpha-invariant and there is no fixed point to solve.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.model import MIN_SHRINKAGE_ROWS, _fit_residual_scale


# 20k rows so the top decile (2,000) clears MIN_SHRINKAGE_ROWS comfortably;
# at 4,000 the subset is 400 and the fit correctly backs off instead.
def _sample(n=20000, *, tail_overconfident=True, seed=0):
    """Rows whose large corrections are too confident, small ones honest."""
    rng = np.random.default_rng(seed)
    base = np.zeros(n)
    correction = rng.normal(0, 1.0, n)
    truth = correction.copy()
    if tail_overconfident:
        big = np.abs(correction) >= np.quantile(np.abs(correction), 0.90)
        truth[big] = correction[big] * 0.3     # only 30% of the tail is real
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
    """The guard against the change being a blanket penalty: if the tail is as
    good as the body, focusing should land in the same place."""
    base, corr, y = _sample(tail_overconfident=False)
    everywhere = _fit_residual_scale(base, corr, y)
    on_the_tail = _fit_residual_scale(base, corr, y, focus_quantile=0.90)
    assert on_the_tail == pytest.approx(everywhere, abs=0.15)


def test_zero_keeps_the_historical_behaviour():
    base, corr, y = _sample()
    assert _fit_residual_scale(base, corr, y, focus_quantile=0.0) == \
        _fit_residual_scale(base, corr, y)


def test_it_backs_off_rather_than_raising_on_a_thin_subset():
    """A noisy alpha fitted on 40 rows is worse than an honest one fitted on
    all of them, and an exception here would kill the whole evaluation."""
    n = MIN_SHRINKAGE_ROWS * 2
    base, corr, y = _sample(n=n)
    # Top 1% of 1,000 rows is 10 — far under the minimum.
    focused = _fit_residual_scale(base, corr, y, focus_quantile=0.99)
    assert focused == pytest.approx(_fit_residual_scale(base, corr, y))


def test_the_ranking_does_not_depend_on_alpha():
    """Why there is no fixed point: alpha scales the correction, so the ORDER
    of |correction| is unchanged and the chosen subset is stable."""
    _, corr, _ = _sample()
    order = np.argsort(np.abs(corr))
    for alpha in (0.25, 1.0, 1.9):
        assert np.array_equal(np.argsort(np.abs(alpha * corr)), order)


def test_the_ceiling_is_respected():
    """Alpha answers "how much of the correction SURVIVES out of sample", so a
    value above 1 is the validation split asking to AMPLIFY it — an
    overfitting signature rather than a finding.

    Measured over 28 weekly refits at the historical 2.0 ceiling: 0.532 to
    2.000, median 1.231, amplifying on 19 of 28, and it does NOT settle with
    more data (sd 0.49 on the larger folds against 0.35 on the smaller).
    """
    # A sample whose correction is genuinely too small, so the free fit wants
    # to amplify it.
    rng = np.random.default_rng(1)
    n = 20000
    base = np.zeros(n)
    correction = rng.normal(0, 0.4, n)
    p = 1.0 / (1.0 + np.exp(-(base + correction * 2.5)))   # truth is 2.5x
    y = (rng.random(n) < p).astype(float)

    free = _fit_residual_scale(base, correction, y)
    assert free > 1.0, 'the fixture must actually want amplification'

    capped = _fit_residual_scale(base, correction, y, max_scale=1.0)
    assert capped <= 1.0 + 1e-9
    assert capped == pytest.approx(1.0, abs=1e-3), 'it should sit at the cap'


def test_the_ceiling_does_not_disturb_a_shrinking_fit():
    """The guard: capping must not change alpha where it was already below 1,
    or it is a blanket penalty rather than a bound on an incoherent value."""
    base, corr, y = _sample()                 # tail is overconfident
    free = _fit_residual_scale(base, corr, y)
    if free < 1.0:
        assert _fit_residual_scale(base, corr, y, max_scale=1.0) == \
            pytest.approx(free, abs=1e-6)


def test_the_default_ceiling_is_one():
    """A shrinkage that may exceed 1 is not a shrinkage.

    Changed from 2.0 on 2026-10-03 on the backtest, same config and data:
    edge/contract 2.276 -> 3.410pp, Sharpe 3.42 -> 3.63, alpha sd 0.426 ->
    0.143, and amplifying refits 19/28 -> 0/28. The cost is volume (2,543 ->
    2,054 trades, total return +111% -> +89%).
    """
    from core.config import Config

    assert Config().max_residual_scale == 1.0


def test_a_fitted_alpha_cannot_exceed_the_default_ceiling():
    """The property the default exists to guarantee, end to end."""
    rng = np.random.default_rng(2)
    n = 20000
    base = np.zeros(n)
    corr = rng.normal(0, 0.4, n)
    p = 1.0 / (1.0 + np.exp(-(base + corr * 3.0)))     # truth is 3x the correction
    y = (rng.random(n) < p).astype(float)
    from core.config import Config

    alpha = _fit_residual_scale(base, corr, y,
                                max_scale=Config().max_residual_scale)
    assert alpha <= 1.0 + 1e-9
