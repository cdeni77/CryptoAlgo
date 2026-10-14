"""The remaining-volatility forecast: the one quantity the barrier needs.

The barrier probability is `F(displacement / sigma_remaining)`, and the
displacement is known exactly.

Point-in-time: everything is indexed by `as_of`, not bar time. A row stamped
10:04 uses bars up to `[10:03, 10:04)` and nothing later, so a decision joins on equality.

The model is HAR on log realised volatility at several lookbacks, fitted by
least squares. It and the seasonality are refitted inside every fold.
Scaling to the remaining span is `sqrt(n)`. `core/baseline.py` fits a per-offset
scale that absorbs the bias from bid-ask bounce and stale prices.
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
# Units: per-minute standard deviation of log return, never annualised.


def log_returns(grid: pd.DataFrame) -> pd.Series:
    """One-minute log returns on a gap-free minute grid, indexed by `as_of`.

    A missing minute yields NaN, not a stitched two-minute return, which would
    read as a false volatility spike.
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
    """Range-based per-minute volatility.

    Blind to the bid-ask bounce that inflates squared returns, so it is carried alongside them."""
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

    The mean of `log|r|` per minute of day, de-meaned, smoothed circularly and
    exponentiated. Log space because volatility is multiplicative; smoothing
    because a per-minute factor is mostly noise.
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
        # `as_of` is the minute after the return, so shift back to attribute
        # the move to the minute it happened in.
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

        The remaining span can straddle a seasonal ramp, and variance adds, so this
        averages squared factors over the remaining minutes.
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

    `fit` and `predict` take the frame from `vol_features`, so the fit sees only
    the rows a fold may see.
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
        # A zero realised volatility is a stretch with no trades, not calm.
        # Drop it before the log.
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

        Without the floor, a dead-quiet stretch drives sigma to zero and the
        baseline returns 0 or 1 with full confidence.
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

    A row stamped 10:04 uses the bar covering `[10:03, 10:04)` and everything before it.
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
    """Per-minute realised volatility over the next `minutes`, indexed by `as_of`.

    The vol model's target. Rows overlap, which inflates in-sample statistics;
    calibration is judged out of sample on binary outcomes instead.
    """
    returns = log_returns(grid)
    squared = returns.pow(2)
    # `returns` is indexed by `as_of`, so the span after `a` is `(a, a + minutes]`.
    # A trailing rolling mean lands it at `a + minutes`; one backward shift of
    # `minutes` puts it at `a`. `tests/test_vol.py` pins the direction.
    forward = squared.rolling(minutes, min_periods=minutes).mean().shift(-minutes)
    return np.sqrt(forward).rename('forward_vol')


def sigma_remaining(
    sigma_per_min: np.ndarray,
    remaining_minutes: np.ndarray | int,
    seasonal_scale: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Scale a per-minute volatility to the span left in the window.

    `sqrt(n)`, times the root-mean seasonal factor when one is supplied.
    `core/baseline.py` fits a per-offset scale that corrects bias in this scaling.
    """
    n = np.asarray(remaining_minutes, dtype=float)
    scaled = np.asarray(sigma_per_min, dtype=float) * np.sqrt(np.maximum(n, 0.0))
    if seasonal_scale is not None:
        scaled = scaled * np.asarray(seasonal_scale, dtype=float)
    return scaled
