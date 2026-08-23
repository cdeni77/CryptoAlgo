"""The remaining-variance forecast — the one quantity the barrier needs.

The barrier probability is `F(displacement / sigma_remaining)`. The
displacement is known exactly, so everything this system can be right or wrong
about lives in `sigma_remaining`. That is a deliberate concentration of risk:
one estimand, measurable against realised outcomes, rather than a return
forecast that five years of this project failed to find.

It is also the one thing already known to be forecastable here. The previous
formulation's dispersion head scored an out-of-sample IC of +0.34 while every
direction head sat at +0.02. Volatility clusters, mean-reverts, and has a
strong and stable time-of-day shape; returns do none of those things.

**Everything is indexed by `as_of`, not by bar time.** A row stamped 10:04 uses
bars up to and including the one covering `[10:03, 10:04)` and nothing later,
so a decision at 10:04 joins on equality. Indexing by bar time and shifting at
the join site is the same information organised so that one forgotten shift
leaks a minute of the future — and a one-minute leak in a fifteen-minute
window is enormous.

**The model is HAR, not GARCH.** A heterogeneous autoregression on log realised
volatility at 15/60/240/1440-minute lookbacks captures the long-memory
clustering that matters at this horizon, fits by least squares on millions of
rows in seconds, and cannot silently diverge the way a fitted GARCH can. It is
refitted inside every cross-validation fold, seasonality included, because a
seasonal factor estimated on the whole sample is a leak that looks like skill.

**Scaling to the remaining span is `sqrt(n)`, and the error is absorbed
downstream.** One-minute returns carry bid-ask bounce, which inflates realised
variance, and the last observed price is up to a minute stale — both bias
`sigma_remaining` by roughly a constant factor. `core/baseline.py` fits one
scale per decision offset against realised outcomes, which is the honest place
to put a bias correction: it is measured against what actually happened rather
than assumed away here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

MINUTES_PER_DAY = 1440
# Sub-annualised throughout: everything here is per-minute standard deviation
# of log return, in natural units. Annualising a 15-minute quantity and then
# de-annualising it is two chances to lose a sqrt.


def log_returns(grid: pd.DataFrame) -> pd.Series:
    """One-minute log returns on a gap-free minute grid, indexed by `as_of`.

    A missing minute yields NaN rather than a two-minute return stitched
    together, because a two-minute return in a squared-return average reads as
    a volatility spike that never happened.
    """
    close = grid['close']
    step = np.log(close / close.shift(1))
    step[grid['close'].isna() | grid['close'].shift(1).isna()] = np.nan
    step.index = step.index + pd.Timedelta(minutes=1)
    return step.rename('r')


def realised_vol(returns: pd.Series, lookback: int, *, min_fraction: float = 0.5) -> pd.Series:
    """Per-minute realised volatility over a trailing `lookback` of minutes."""
    squared = returns.pow(2)
    mean_sq = squared.rolling(lookback, min_periods=max(2, int(lookback * min_fraction))).mean()
    return np.sqrt(mean_sq).rename(f'rv_{lookback}')


def parkinson_vol(grid: pd.DataFrame, lookback: int) -> pd.Series:
    """Range-based per-minute volatility, five times more efficient than
    squared returns for the same sample — and blind to the bid-ask bounce that
    inflates them, which is why it is carried alongside rather than instead."""
    span = np.log(grid['high'] / grid['low'])
    var = span.pow(2) / (4.0 * np.log(2.0))
    out = np.sqrt(var.rolling(lookback, min_periods=max(2, lookback // 2)).mean())
    out.index = out.index + pd.Timedelta(minutes=1)
    return out.rename(f'pk_{lookback}')


def _circular_smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Moving average that wraps at midnight, so 23:58 and 00:02 are neighbours."""
    if window <= 1:
        return values
    pad = window // 2
    wrapped = np.concatenate([values[-pad:], values, values[:pad]])
    kernel = np.ones(window) / window
    return np.convolve(wrapped, kernel, mode='valid')[:values.size]


@dataclass
class Seasonality:
    """A multiplicative minute-of-day volatility factor, mean one.

    Estimated as the average of `log|r|` per minute of day, de-meaned, smoothed
    circularly and exponentiated. Log space because volatility is multiplicative
    and a mean of absolute returns is dominated by its largest few; smoothing
    because a per-minute factor from even five years of data is mostly noise —
    1,440 bins over 1,825 days is 1,825 observations per bin, and the standard
    error of a mean log-absolute-return on that is not small next to the
    seasonal amplitude it is trying to measure.
    """

    factor: np.ndarray                    # length 1440, mean 1
    days_observed: float
    smoothed_over: int

    @classmethod
    def fit(cls, returns: pd.Series, config: Config = DEFAULT_CONFIG) -> 'Seasonality':
        clean = returns.dropna()
        days = len(clean) / MINUTES_PER_DAY
        flat = np.ones(MINUTES_PER_DAY)
        if config.seasonality_smooth_minutes <= 0 or days < config.seasonality_min_days:
            if config.seasonality_smooth_minutes > 0:
                logger.warning(
                    'seasonality: %.1f days is under the %d-day minimum, using a flat factor',
                    days, config.seasonality_min_days,
                )
            return cls(factor=flat, days_observed=days, smoothed_over=0)
        # `as_of` is the minute after the return, so shift back to attribute the
        # move to the minute it happened in.
        minute = ((clean.index - pd.Timedelta(minutes=1)).hour * 60
                  + (clean.index - pd.Timedelta(minutes=1)).minute)
        log_abs = np.log(clean.abs().clip(lower=1e-12))
        by_minute = pd.Series(log_abs.to_numpy(), index=minute).groupby(level=0).mean()
        raw = by_minute.reindex(range(MINUTES_PER_DAY)).to_numpy()
        raw = pd.Series(raw).interpolate(limit_direction='both').to_numpy()
        smoothed = _circular_smooth(raw - np.nanmean(raw), config.seasonality_smooth_minutes)
        factor = np.exp(smoothed)
        factor = factor / factor.mean()
        return cls(factor=factor, days_observed=days,
                   smoothed_over=config.seasonality_smooth_minutes)

    def at(self, times: pd.DatetimeIndex) -> np.ndarray:
        """The factor for the minute each timestamp falls in."""
        idx = (times.hour * 60 + times.minute).to_numpy()
        return self.factor[idx]

    def mean_over(self, start: pd.DatetimeIndex, minutes: int) -> np.ndarray:
        """Root-mean-square factor over the `minutes` starting at each timestamp.

        The remaining span of a window can straddle a seasonal ramp — a window
        opening at 13:28 covers the New York equity open — and variance adds, so
        the correct aggregate is the root mean of the squared factors over the
        minutes actually remaining, not the factor at the decision minute.
        """
        if minutes <= 0:
            return np.ones(len(start))
        base = (start.hour * 60 + start.minute).to_numpy()
        offsets = np.arange(minutes)
        idx = (base[:, None] + offsets[None, :]) % MINUTES_PER_DAY
        return np.sqrt((self.factor[idx] ** 2).mean(axis=1))

    @property
    def amplitude(self) -> float:
        """Ratio of the busiest minute of the day to the quietest."""
        return float(self.factor.max() / self.factor.min())


# The HAR design matrix, in the order the coefficients are stored.
def _design(features: pd.DataFrame, lookbacks: tuple[int, ...]) -> tuple[np.ndarray, list[str]]:
    names = [f'log_rv_{lb}' for lb in lookbacks] + ['log_pk_60', 'log_seasonal']
    columns = [features[name].to_numpy(dtype=float) for name in names]
    matrix = np.column_stack([np.ones(len(features))] + columns)
    return matrix, ['const'] + names


@dataclass
class VolModel:
    """HAR regression for log per-minute volatility over the next window.

    `fit` and `predict` take the per-minute feature frame produced by
    `vol_features`, so the seasonality and the regression are fitted on exactly
    the rows a fold is allowed to see.
    """

    lookbacks: tuple[int, ...]
    coefficients: Optional[np.ndarray] = None
    names: list[str] = field(default_factory=list)
    residual_sd: float = float('nan')
    r_squared: float = float('nan')
    min_sigma: float = 0.0

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        target: pd.Series,
        config: Config = DEFAULT_CONFIG,
    ) -> 'VolModel':
        matrix, names = _design(features, tuple(config.vol_lookbacks_minutes))
        raw = target.to_numpy(dtype=float)
        # A zero realised volatility is a stretch with no trades, not a
        # measurement of calm. Excluded before the log rather than after, so a
        # degenerate series is dropped rather than warned about once per fit.
        positive = np.isfinite(raw) & (raw > 0)
        y = np.full_like(raw, np.nan)
        y[positive] = np.log(raw[positive])
        usable = np.isfinite(y) & np.isfinite(matrix).all(axis=1)
        if usable.sum() < 10 * matrix.shape[1]:
            raise ValueError(
                f'vol model: {usable.sum()} usable rows for {matrix.shape[1]} '
                f'coefficients — refusing to fit'
            )
        coef, *_ = np.linalg.lstsq(matrix[usable], y[usable], rcond=None)
        fitted = matrix[usable] @ coef
        residual = y[usable] - fitted
        total = y[usable] - y[usable].mean()
        return cls(
            lookbacks=tuple(config.vol_lookbacks_minutes),
            coefficients=coef,
            names=names,
            residual_sd=float(residual.std(ddof=matrix.shape[1])),
            r_squared=float(1.0 - residual.var() / total.var()) if total.var() > 0 else float('nan'),
            min_sigma=config.min_sigma_bps_per_minute / 10_000.0,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Per-minute volatility forecast, floored.

        The floor is not cosmetic: a dead-quiet stretch drives the forecast to
        zero, the barrier then divides by it, and the baseline returns 0 or 1
        with total confidence on a window that is genuinely a coin flip. One
        such row can dominate a log loss.
        """
        if self.coefficients is None:
            raise ValueError('vol model is not fitted')
        matrix, _ = _design(features, self.lookbacks)
        log_sigma = matrix @ self.coefficients
        sigma = np.exp(log_sigma)
        return pd.Series(np.maximum(sigma, self.min_sigma), index=features.index, name='sigma_per_min')

    def summary(self) -> str:
        if self.coefficients is None:
            return 'vol model: unfitted'
        terms = ', '.join(f'{n}={c:+.3f}' for n, c in zip(self.names, self.coefficients))
        return f'vol model: R2(log)={self.r_squared:.3f} resid_sd={self.residual_sd:.3f} | {terms}'


def vol_features(
    grid: pd.DataFrame,
    seasonality: Seasonality,
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Per-minute volatility state, indexed by `as_of`.

    Every column is knowable at its own index: a row stamped 10:04 uses the bar
    covering `[10:03, 10:04)` and everything before it.
    """
    returns = log_returns(grid)
    parts = {f'rv_{lb}': realised_vol(returns, lb) for lb in config.vol_lookbacks_minutes}
    parts['pk_60'] = parkinson_vol(grid, 60)
    frame = pd.DataFrame(parts)
    frame['seasonal'] = seasonality.at(frame.index)
    floor = config.min_sigma_bps_per_minute / 10_000.0 / 10.0
    for column in list(parts):
        frame[f'log_{column}'] = np.log(frame[column].clip(lower=floor))
    frame['log_seasonal'] = np.log(frame['seasonal'])
    frame['r'] = returns
    return frame


def forward_realised_vol(grid: pd.DataFrame, minutes: int) -> pd.Series:
    """Per-minute realised volatility over the *next* `minutes`, indexed by `as_of`.

    The vol model's target. It overlaps across adjacent rows, which inflates any
    in-sample statistic computed on it — that is tolerable here and nowhere
    else, because the quantity that matters is the calibration of the resulting
    barrier probability, and that is measured against realised binary outcomes
    out of sample rather than against this.
    """
    returns = log_returns(grid)
    squared = returns.pow(2)
    # `returns` is indexed by `as_of`, so the minute after `a` sits at `a + 1`
    # and the span this looks forward over is `(a, a + minutes]`. A trailing
    # rolling mean lands that span at index `a + minutes`, so one backward
    # shift of exactly `minutes` puts it at `a`.
    #
    # This was written as `.shift(-minutes).rolling(...).mean().shift(minutes)`
    # and the two shifts cancelled: the "forward" target was the *trailing*
    # window, the HAR fit came back with R2 1.000 and a unit coefficient on
    # rv_15, and it would have produced a beautifully calibrated baseline that
    # could not have been computed live. Assert the direction rather than
    # reading it off the expression — `tests/test_vol.py` pins it.
    forward = squared.rolling(minutes, min_periods=minutes).mean().shift(-minutes)
    return np.sqrt(forward).rename('forward_vol')


def sigma_remaining(
    sigma_per_min: np.ndarray,
    remaining_minutes: np.ndarray | int,
    seasonal_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale a per-minute volatility to the span left in the window.

    `sqrt(n)`, times the root-mean seasonal factor over those n minutes when one
    is supplied. Any systematic bias in this scaling — microstructure inflation,
    the sub-minute staleness of the last observed price, the sqrt law itself
    being wrong at this horizon — is corrected by the per-offset scale factor
    that `core/baseline.py` fits against realised outcomes.
    """
    n = np.asarray(remaining_minutes, dtype=float)
    scaled = np.asarray(sigma_per_min, dtype=float) * np.sqrt(np.maximum(n, 0.0))
    if seasonal_scale is not None:
        scaled = scaled * np.asarray(seasonal_scale, dtype=float)
    return scaled
