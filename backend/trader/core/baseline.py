"""The null hypothesis: `F(x / sigma_n)`, not 50%.

Mid-window state alone makes outcomes predictable, so every measurement is
incremental against this baseline: log-loss skill, Brier skill, edge in
probability points.

The baseline has no drift and never fits one: a drift is the alpha under test.
It fits only arithmetic corrections:

* a scale per decision offset (bid-ask bounce, stale last price, `sqrt(n)` error);
* a tail thickness, because one-minute returns are fat-tailed.

Scale and tail are not separately identified from binary outcomes; only the
composite map `z -> P(up)` is. Judge the pair by out-of-sample calibration,
never read either parameter alone. Fit on training folds only.
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

# Rows used to fit scale and tail. At 40,000 rows the standard error on a fitted
# probability is ~0.0025, far finer than the calibration error being corrected.
# Sampled deterministically, so two runs of one configuration agree.
MAX_FIT_ROWS = 40_000
FIT_SEED = 23

# Probabilities are clipped before any log. A clip at 1e-6 caps the penalty for
# one wrong 0-or-1 prediction at 13.8 nats.
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

    `z` is already `displacement / sigma`, so the distribution carries no scale.
    A raw Student-t has variance `nu / (nu - 2)` and is rescaled here.
    """
    if distribution == 'normal':
        return stats.norm.cdf(z)
    if distribution == 'student_t':
        nu = float(np.clip(nu, MIN_NU, MAX_NU))
        # `special.stdtr` is `stats.t.cdf` without the frozen-distribution overhead.
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
    # The same fit with every scale pinned at one, at the fitted tail. Context
    # only: the tail can absorb a scale error, so this is not a fair baseline.
    unscaled_log_loss: float = float('nan')
    # A sigma multiplier fitted out of sample, on a held-out recent tail of the
    # training rows. The in-sample fit is over-confident on later rows because
    # volatility drifts. Above 1.0 it widens sigma and pulls probabilities toward 0.5.
    # The baseline's analogue of `residual_scale`.
    calibration_scale: float = 1.0
    n_calibration_rows: int = 0

    # ---- prediction -----------------------------------------------------
    def scale_for(self, offsets: np.ndarray) -> np.ndarray:
        lookup = np.vectorize(lambda o: self.scale.get(int(o), self.default_scale))
        return lookup(np.asarray(offsets)).astype(float) * float(self.calibration_scale)

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

        Calibration, not skill. Fit on training rows only.
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
        # Parameters are log-scales, so 1e-3 is a tenth of a percent on the scale.
        result = optimize.minimize(
            objective, theta0, method='Nelder-Mead',
            options={'maxiter': 600, 'xatol': 1e-3, 'fatol': 1e-7},
        )
        scales, nu = unpack(result.x)

        # Calibrate on rows the fit did not see: the most recent slice of
        # training, held out from the scale fit. Recent, not random, because
        # volatility drift is what is being corrected.
        # Off by default (`baseline_calibration_holdout = 0.0`).
        calibration_scale, n_calibration = 1.0, 0
        holdout = float(getattr(config, 'baseline_calibration_holdout', 0.0) or 0.0)
        order = (pd.to_datetime(frame['window_open'], utc=True)
                 if 'window_open' in frame.columns else None)
        if holdout > 0.0 and order is not None and len(frame) >= 200:
            # Through pandas, not numpy: `astype('datetime64[ns]')` on a
            # tz-aware column warns and drops the zone.
            cut = order.quantile(1.0 - holdout)
            tail = (order >= cut).to_numpy()
            if tail.sum() >= 100:
                d_t, s_t = displacement[tail], sigma[tail]
                y_t, w_t = outcome[tail], (None if w is None else np.asarray(w)[tail])
                base_t = s_t * scales[which[tail]]

                def calib_objective(theta: np.ndarray) -> float:
                    k = float(np.exp(theta[0]))
                    z = d_t / np.where(base_t * k > 0, base_t * k, np.nan)
                    return log_loss(y_t, _standardised_cdf(
                        z, config.baseline_distribution, nu), w_t)

                found = optimize.minimize(
                    calib_objective, np.zeros(1), method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-7})
                calibration_scale = float(np.exp(found.x[0]))
                n_calibration = int(tail.sum())
                logger.info(
                    'baseline calibration on %d held-out rows: sigma x %.4f '
                    '(>1 widens sigma and pulls probabilities toward 0.5)',
                    n_calibration, calibration_scale)
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
            calibration_scale=calibration_scale,
            n_calibration_rows=n_calibration,
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

    The system trades its confident predictions, so miscalibration in a band
    matters more than the average.
    """

    edges: np.ndarray
    predicted: np.ndarray
    observed: np.ndarray
    count: np.ndarray
    # Rows excluded because the prediction or the outcome was not finite.
    n_non_finite: int = 0

    @property
    def expected_calibration_error(self) -> float:
        """Count-weighted mean |predicted - observed| over populated bins.

        Empty bins carry `predicted = nan`; including them would make the metric
        NaN, and `nan > 0.02` is False, so a max-gate would pass.
        """
        populated = self.count > 0
        total = self.count[populated].sum()
        if not total:
            return float('nan')
        return float(np.sum(self.count[populated]
                            * np.abs(self.predicted[populated]
                                     - self.observed[populated])) / total)

    @property
    def max_deviation(self) -> float:
        populated = self.count > 0
        if not populated.any():
            return float('nan')
        return float(np.max(np.abs(self.predicted[populated] - self.observed[populated])))

    def worst_deviation(self, *, min_count: int = 500) -> float:
        """The largest bin deviation, over bins with at least `min_count` rows.

        A bin of `min_count` rows at p=0.9 has a standard error near 1.3pp, so
        this gate bounds damage; it cannot certify a sub-pp edge.
        """
        populated = self.count >= int(min_count)
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

    Fixed edges, not quantiles, so two runs' tables stay comparable.
    """
    if edges is None:
        # Symmetric, and 2pp wide in the tails: wide bins average away opposite
        # errors in the band this system trades.
        edges = np.array([
            0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50,
            0.60, 0.70, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0,
        ])
    raw = np.asarray(probability, dtype=float)
    y_all = np.asarray(outcome, dtype=float)
    # `np.digitize(nan, ...)` returns the last index, which would count a NaN
    # prediction in the most confident bin. Exclude non-finite rows and count them.
    finite = np.isfinite(raw) & np.isfinite(y_all)
    n_non_finite = int((~finite).sum())
    p = clip_prob(raw[finite])
    y = y_all[finite]
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
    return Reliability(edges=edges, predicted=predicted, observed=observed,
                       count=count, n_non_finite=n_non_finite)


def attach_baseline(
    table: pd.DataFrame,
    baseline: BarrierBaseline,
    *,
    column: str = 'baseline_probability',
) -> pd.DataFrame:
    """Return `table` with the baseline probability and its logit attached.

    The logit is the classifier's `init_score`, so an untrained model reproduces
    the baseline exactly and every tree is incremental skill.
    """
    out = table.copy()
    out[column] = baseline.probability_for(table)
    out[f'{column}_logit'] = logit(out[column])
    # The market's correction to the arithmetic. Computed here, not in
    # `build_features`, because that runs before the baseline exists.
    if 'market_prob' in out.columns:
        out['market_minus_baseline'] = (
            pd.to_numeric(out['market_prob'], errors='coerce') - out[column])
    # The market's own logit, for a market-initialised model. `core.model`
    # raises if it is missing, rather than silently using the baseline.
    if 'market_probability' in out.columns:
        from core.model import attach_market_logit
        out = attach_market_logit(out)
    return out
