"""The null hypothesis, stated precisely enough to lose to.

**The benchmark is not 50%. It is `F(x / sigma_n)`.**

This is the single most important line in the project, and getting it wrong
would have produced the most convincing false positive it has ever generated.
A model fed mid-window state will report 70-90% accuracy on 15-minute up/down
windows. None of that is alpha. At nine minutes into a window with three
minutes of movement left, a displacement of twenty basis points against a
remaining sigma of eight is already 99% settled — anyone with a clock and a
volatility estimate knows it, and the market prices it. Measuring such a model
against a 50% coin flip would show a 40-point edge that does not exist.

So the baseline here is not a constant. It is the best probability obtainable
from the barrier arithmetic alone: displacement, remaining time, and a
volatility forecast. Every measurement in this system is *incremental* against
it — log loss skill, Brier skill, edge in probability points — and a model that
cannot beat it has found nothing.

**Scale and tail are jointly fitted and are not separately identified.** This is
worth stating because the obvious reading of the two parameters is wrong. From
binary outcomes alone only the composite mapping `z -> P(up)` is determined, and
a thicker tail with a larger scale mimics a thinner tail with a smaller one
through the bulk of the distribution. Measured: against a sigma inflated by
1.2-1.4x, the fit returned `scale ~ 1.001` and `nu = 2.93` — it absorbed the
whole inflation into the tail parameter and left the scale alone.

So do not read `scale` as "the sigma inflation" or `nu` as "the tail thickness of
returns". Read the pair as one calibration of the barrier map, and judge it the
only way it can be judged: by whether the resulting probabilities are calibrated
out of sample. `tests/test_baseline.py` asserts that property rather than either
parameter's value.

**What the baseline deliberately does not contain.** Its drift is structurally
zero and is never fitted. A non-zero drift is exactly the alpha under test; if
the null were allowed to fit one, the null would absorb the finding and report
no skill. What the baseline *does* fit is the two things that are pure
arithmetic:

* **A scale factor per decision offset.** One-minute returns carry bid-ask
  bounce, which inflates realised variance; the last observed price is up to a
  minute stale; and the `sqrt(n)` law is an approximation at n = 3. All three
  bias `sigma_remaining` by roughly a constant per offset, and correcting a
  known bias against realised outcomes is not skill, it is calibration. Leaving
  it uncorrected would hand the model an easy win that says nothing about
  forecasting. (Which parameter the correction lands in is not determined — see
  above — only that the composite is corrected.)
* **The tail thickness.** One-minute crypto returns are fat-tailed, so a
  Gaussian barrier is overconfident at large displacements — it would assign
  0.999 where 0.99 is right, and a model that merely knew this would look
  skilful. Which distribution calibrates better is measured out of sample, not
  assumed.

Both are fitted on training folds only. A scale factor fitted on the whole
sample is a leak, and it leaks in the direction that makes the baseline look
strong and the model look weak — which is the one direction nobody checks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Rows used to fit the scale and tail. Five parameters do not need millions of
# observations: at 40,000 rows the standard error on a fitted probability is
# about 0.0025, an order of magnitude finer than the calibration error the fit is
# correcting. Nelder-Mead evaluates the objective several hundred times, so this
# cap is the difference between three seconds and twenty per fold — and at five
# years of history, between minutes and an hour. Sampled deterministically so
# two runs of the same configuration agree exactly.
MAX_FIT_ROWS = 40_000
FIT_SEED = 23

# Probabilities are clipped before any log is taken. A single 0-or-1 prediction
# that turns out wrong makes log loss infinite and every comparison meaningless,
# and a clip at 1e-6 caps the penalty for one such row at 13.8 nats.
PROB_EPS = 1e-6
MIN_NU = 2.05          # below this a Student-t has no finite variance
MAX_NU = 200.0         # above this it is a Gaussian in every measurable respect


def clip_prob(p: np.ndarray | pd.Series) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), PROB_EPS, 1.0 - PROB_EPS)


def log_loss(outcome: np.ndarray, probability: np.ndarray, weights: Optional[np.ndarray] = None) -> float:
    """Mean binary cross-entropy, in nats. Lower is better."""
    p = clip_prob(probability)
    y = np.asarray(outcome, dtype=float)
    terms = -(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
    if weights is None:
        return float(np.mean(terms))
    w = np.asarray(weights, dtype=float)
    return float(np.sum(terms * w) / np.sum(w))


def brier(outcome: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((clip_prob(probability) - np.asarray(outcome, dtype=float)) ** 2))


def logit(p: np.ndarray | pd.Series) -> np.ndarray:
    q = clip_prob(p)
    return np.log(q / (1.0 - q))


def expit(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _standardised_cdf(z: np.ndarray, distribution: str, nu: float) -> np.ndarray:
    """CDF of a zero-mean, unit-variance distribution evaluated at `z`.

    Unit variance is the whole point: `z` is already `displacement / sigma`, so
    the distribution must not carry a scale of its own. A raw Student-t has
    variance `nu / (nu - 2)`, and forgetting that inflates every probability
    toward 0.5 — which reads as a well-behaved conservative baseline rather than
    as a bug.
    """
    if distribution == 'normal':
        return stats.norm.cdf(z)
    if distribution == 'student_t':
        nu = float(np.clip(nu, MIN_NU, MAX_NU))
        # `special.stdtr` is the same CDF as `stats.t.cdf` without the frozen
        # distribution machinery, worth about 20%. The cost that mattered was
        # never the call — it was making it 921 times, which is what MAX_FIT_ROWS
        # and the tolerances below address.
        return special.stdtr(nu, z * np.sqrt(nu / (nu - 2.0)))
    raise ValueError(f'unknown baseline distribution {distribution!r}')


@dataclass
class BarrierBaseline:
    """`P(settle above strike) = F(displacement / (scale[offset] * sigma_remaining))`."""

    distribution: str
    nu: float
    scale: dict[int, float]
    default_scale: float = 1.0
    n_fitted: int = 0
    fitted_log_loss: float = float('nan')
    # The same fit with every scale pinned at one, at the *fitted* tail. Reported
    # for context and not as a fair baseline: because scale and tail are jointly
    # fitted, the tail can already have absorbed a scale misspecification, and
    # then these two numbers agree to eight decimals while both are correct.
    unscaled_log_loss: float = float('nan')

    # ---- prediction -----------------------------------------------------
    def scale_for(self, offsets: np.ndarray) -> np.ndarray:
        lookup = np.vectorize(lambda o: self.scale.get(int(o), self.default_scale))
        return lookup(np.asarray(offsets)).astype(float)

    def z_score(
        self,
        displacement: np.ndarray,
        sigma_remaining: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        sigma = np.asarray(sigma_remaining, dtype=float) * self.scale_for(offsets)
        sigma = np.where(sigma > 0, sigma, np.nan)
        return np.asarray(displacement, dtype=float) / sigma

    def probability(
        self,
        displacement: np.ndarray,
        sigma_remaining: np.ndarray,
        offsets: np.ndarray,
    ) -> np.ndarray:
        z = self.z_score(displacement, sigma_remaining, offsets)
        return clip_prob(_standardised_cdf(z, self.distribution, self.nu))

    def probability_for(self, table: pd.DataFrame) -> np.ndarray:
        """Convenience over a window table carrying `sigma_remaining`."""
        return self.probability(
            table['displacement'].to_numpy(),
            table['sigma_remaining'].to_numpy(),
            table['offset'].to_numpy(),
        )

    # ---- fitting --------------------------------------------------------
    @classmethod
    def fit(
        cls,
        table: pd.DataFrame,
        config: Config = DEFAULT_CONFIG,
        *,
        weights: Optional[np.ndarray] = None,
    ) -> 'BarrierBaseline':
        """Fit scale (per offset) and tail thickness by minimising log loss.

        Both parameters are calibration, not skill — see the module docstring.
        Fit this on training rows only.
        """
        needed = {'displacement', 'sigma_remaining', 'offset', 'outcome'}
        missing = needed - set(table.columns)
        if missing:
            raise ValueError(f'baseline fit needs {sorted(missing)}')
        frame = table.loc[
            np.isfinite(table['displacement'])
            & np.isfinite(table['sigma_remaining'])
            & (table['sigma_remaining'] > 0)
        ]
        if frame.empty:
            raise ValueError('baseline fit: no usable rows')
        n_available = len(frame)
        if n_available > MAX_FIT_ROWS:
            frame = frame.sample(MAX_FIT_ROWS, random_state=FIT_SEED)

        displacement = frame['displacement'].to_numpy(dtype=float)
        sigma = frame['sigma_remaining'].to_numpy(dtype=float)
        outcome = frame['outcome'].to_numpy(dtype=float)
        offsets = frame['offset'].to_numpy()
        w = None if weights is None else np.asarray(weights, dtype=float)[frame.index.to_numpy()] \
            if weights is not None and len(weights) != len(frame) else weights

        unique_offsets = sorted({int(o) for o in offsets})
        if not config.baseline_fit_scale_per_offset:
            unique_offsets = unique_offsets[:1]
        index_of = {o: i for i, o in enumerate(unique_offsets)}
        which = np.array([index_of.get(int(o), 0) for o in offsets])

        fit_nu = (config.baseline_distribution == 'student_t' and config.baseline_nu is None)
        n_scale = len(unique_offsets)

        def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
            scales = np.exp(theta[:n_scale])
            if fit_nu:
                nu = MIN_NU + np.exp(theta[n_scale])
            else:
                nu = config.baseline_nu if config.baseline_nu is not None else MAX_NU
            return scales, float(nu)

        def objective(theta: np.ndarray) -> float:
            scales, nu = unpack(theta)
            z = displacement / (sigma * scales[which])
            p = _standardised_cdf(z, config.baseline_distribution, nu)
            return log_loss(outcome, p, w)

        theta0 = np.zeros(n_scale + (1 if fit_nu else 0))
        if fit_nu:
            theta0[n_scale] = np.log(4.0 - MIN_NU)   # start near nu = 4
        # The parameters are log-scales, so 1e-3 is a tenth of a percent on the
        # scale factor — far finer than anything downstream can distinguish, and
        # a third of the function evaluations of the tolerances this had before.
        result = optimize.minimize(
            objective, theta0, method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-3, 'fatol': 1e-7},
        )
        scales, nu = unpack(result.x)
        unscaled = log_loss(
            outcome,
            _standardised_cdf(displacement / sigma, config.baseline_distribution, nu),
            w,
        )
        baseline = cls(
            distribution=config.baseline_distribution,
            nu=nu,
            scale={o: float(scales[index_of[o]]) for o in unique_offsets},
            default_scale=float(scales[0]),
            n_fitted=n_available,
            fitted_log_loss=float(result.fun),
            unscaled_log_loss=float(unscaled),
        )
        logger.info(baseline.summary())
        return baseline

    # ---- reporting ------------------------------------------------------
    def summary(self) -> str:
        scales = ', '.join(f'{o}m={s:.3f}' for o, s in sorted(self.scale.items()))
        nu = 'gaussian' if self.distribution == 'normal' else f'nu={self.nu:.2f}'
        return (
            f'baseline: {self.distribution} ({nu}) | scale {scales} | '
            f'log loss {self.fitted_log_loss:.5f} '
            f'(unscaled {self.unscaled_log_loss:.5f}) on {self.n_fitted:,} rows'
        )

    def provenance(self) -> dict:
        return {
            'distribution': self.distribution,
            'nu': self.nu,
            'scale': {str(k): v for k, v in self.scale.items()},
            'n_fitted': self.n_fitted,
            'fitted_log_loss': self.fitted_log_loss,
        }


# ---- calibration diagnostics ---------------------------------------------

@dataclass
class Reliability:
    """Observed frequency against predicted probability, in bins.

    The one diagnostic that cannot be faked by a good average: a model can hit
    the base rate exactly while being wrong at every level of confidence, and
    since this system only trades its confident predictions, a miscalibration
    concentrated in the 85-95% band matters far more than the headline number.
    """

    edges: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    count: np.ndarray

    @property
    def expected_calibration_error(self) -> float:
        total = self.count.sum()
        if total == 0:
            return float('nan')
        return float(np.sum(self.count * np.abs(self.predicted - self.observed)) / total)

    @property
    def max_deviation(self) -> float:
        populated = self.count > 0
        if not populated.any():
            return float('nan')
        return float(np.max(np.abs(self.predicted[populated] - self.observed[populated])))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            'bin_low': self.edges[:-1], 'bin_high': self.edges[1:],
            'predicted': self.predicted, 'observed': self.observed, 'count': self.count,
        })

    def table(self) -> str:
        lines = ['  band        predicted   observed      n']
        for row in self.frame().itertuples():
            if row.count == 0:
                continue
            lines.append(
                f'  {row.bin_low:.2f}-{row.bin_high:.2f}   {row.predicted:8.4f}   '
                f'{row.observed:8.4f}  {int(row.count):6,}'
            )
        return '\n'.join(lines)


def reliability(
    outcome: np.ndarray,
    probability: np.ndarray,
    *,
    edges: Optional[np.ndarray] = None,
) -> Reliability:
    """Bin predictions and compare to realised frequency.

    Fixed edges rather than quantiles, because the bands that matter are the
    ones the trading gate uses — a quantile binning moves the boundaries when
    the prediction distribution shifts, and then two runs' calibration tables
    are not comparable.
    """
    if edges is None:
        edges = np.array([0.0, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0])
    p = clip_prob(probability)
    y = np.asarray(outcome, dtype=float)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    predicted = np.full(len(edges) - 1, np.nan)
    observed = np.full(len(edges) - 1, np.nan)
    count = np.zeros(len(edges) - 1, dtype=int)
    for b in range(len(edges) - 1):
        mask = index == b
        count[b] = int(mask.sum())
        if count[b]:
            predicted[b] = p[mask].mean()
            observed[b] = y[mask].mean()
    return Reliability(edges=edges, predicted=predicted, observed=observed, count=count)


def attach_baseline(
    table: pd.DataFrame,
    baseline: BarrierBaseline,
    *,
    column: str = 'baseline_probability',
) -> pd.DataFrame:
    """Return `table` with the baseline probability and its logit attached.

    The logit is what the classifier consumes as an `init_score` offset, so an
    untrained model reproduces the baseline exactly and every tree it grows is
    incremental skill by construction.
    """
    out = table.copy()
    out[column] = baseline.probability_for(table)
    out[f'{column}_logit'] = logit(out[column])
    return out
