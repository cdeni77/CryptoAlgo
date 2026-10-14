"""Tests for the volatility layer in `core.vol`, starting with a forward target that
looks forward.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.vol import (
    MINUTES_PER_DAY, Seasonality, VolModel, forward_realised_vol, log_returns,
    parkinson_vol, realised_vol, sigma_remaining, vol_features,
)
from core.windows import minute_grid
from tests.conftest import make_bars


def test_the_forward_target_looks_forward():
    """A forward target just before a volatility regime change already sees the new regime."""
    n = 4000
    times = pd.date_range('2025-01-01', periods=n, freq='1min', tz='UTC')
    rng = np.random.default_rng(1)
    sigma = np.where(np.arange(n) < n // 2, 1e-5, 1e-3)
    price = 100 * np.exp(np.cumsum(rng.normal(0, sigma)))
    bars = pd.DataFrame({'event_time': times, 'open': price, 'high': price,
                         'low': price, 'close': price, 'volume': 1.0,
                         'trade_count': 1})
    grid = minute_grid(bars)
    forward = forward_realised_vol(grid, 15).dropna()

    change = times[n // 2]
    before = forward.loc[forward.index < change - pd.Timedelta(minutes=20)]
    just_before = forward.loc[
        (forward.index >= change - pd.Timedelta(minutes=14))
        & (forward.index < change)]
    assert before.median() < 5e-5, 'quiet regime is not quiet'
    assert just_before.median() > 1e-4, (
        'a target measured just before the volatility jump does not see it, so '
        'it is looking backwards'
    )


def test_the_forward_target_is_not_the_trailing_one():
    """The forward target is not the trailing one; a unit coefficient on rv_15 would show it."""
    bars = make_bars(days=25)['BTC-USD']
    grid = minute_grid(bars)
    config = Config()
    seasonality = Seasonality.fit(log_returns(grid), config)
    features = vol_features(grid, seasonality, config)
    target = forward_realised_vol(grid, config.window_minutes)
    frame = pd.concat([features, target], axis=1).dropna()
    correlation = np.corrcoef(frame['log_rv_15'], np.log(frame['forward_vol']))[0, 1]
    assert correlation < 0.98, (
        f'trailing rv_15 explains the "forward" target at r={correlation:.4f}; '
        f'the two shifts have cancelled again'
    )


def test_realised_volatility_only_uses_the_past():
    """Realised volatility at `as_of` uses the bar ending at `as_of` and nothing later."""
    bars = make_bars(days=2)['BTC-USD']
    grid = minute_grid(bars)
    returns = log_returns(grid)
    rv = realised_vol(returns, 15)
    # Corrupt the tail; nothing before the corruption may change.
    corrupted = grid.copy()
    cut = len(corrupted) // 2
    corrupted.iloc[cut:, corrupted.columns.get_loc('close')] *= 4.0
    rv_after = realised_vol(log_returns(corrupted), 15)
    head = slice(0, cut - 1)
    assert np.allclose(rv.iloc[head].dropna(),
                       rv_after.iloc[head].dropna(), equal_nan=True)


def test_seasonality_recovers_a_planted_shape():
    bars = make_bars(days=120)['BTC-USD']
    grid = minute_grid(bars)
    seasonality = Seasonality.fit(log_returns(grid), Config())
    # The generator's shape is 1 +/- 0.5, so the peak-to-trough ratio is 3.
    assert 2.0 < seasonality.amplitude < 5.0, seasonality.amplitude
    assert seasonality.factor.mean() == pytest.approx(1.0, abs=1e-6)
    assert len(seasonality.factor) == MINUTES_PER_DAY


def test_seasonality_falls_back_to_flat_on_a_short_sample():
    """On too short a sample the seasonal factor is flat."""
    bars = make_bars(days=3)['BTC-USD']
    grid = minute_grid(bars)
    seasonality = Seasonality.fit(log_returns(grid), Config(seasonality_min_days=60))
    assert np.allclose(seasonality.factor, 1.0)
    assert seasonality.smoothed_over == 0


def test_the_seasonal_factor_wraps_at_midnight():
    """The seasonal factor wraps at midnight: 23:58 and 00:02 are two minutes apart."""
    factor = np.zeros(MINUTES_PER_DAY)
    factor[0] = 1.0
    seasonality = Seasonality(factor=np.ones(MINUTES_PER_DAY), days_observed=999,
                              smoothed_over=31)
    late = pd.DatetimeIndex(['2025-01-01 23:58:00'], tz='UTC')
    spans = seasonality.mean_over(late, 10)
    assert np.isfinite(spans).all(), 'a span crossing midnight produced no factor'


def test_the_volatility_forecast_is_floored():
    """The volatility forecast is floored at `min_sigma_bps_per_minute`.

    Tested on the model directly, since a series flat enough to forecast zero is one
    `VolModel.fit` refuses.
    """
    config = Config(min_sigma_bps_per_minute=0.5)
    floor = config.min_sigma_bps_per_minute / 10_000.0
    lookbacks = tuple(config.vol_lookbacks_minutes)
    # An intercept far below the floor and no slope: every prediction is tiny.
    model = VolModel(
        lookbacks=lookbacks,
        coefficients=np.array([-30.0] + [0.0] * (len(lookbacks) + 2)),
        names=['const'], min_sigma=floor,
    )
    index = pd.date_range('2025-01-01', periods=5, freq='1min', tz='UTC')
    features = pd.DataFrame(
        {name: np.zeros(5) for name in
         [f'log_rv_{lb}' for lb in lookbacks] + ['log_pk_60', 'log_seasonal']},
        index=index)
    predicted = model.predict(features)
    assert np.allclose(predicted, floor), predicted.to_list()
    assert (predicted > 0).all()


def test_a_degenerate_series_is_refused_rather_than_fitted():
    """A series with no price movement is refused instead of fitted."""
    n = 2000
    times = pd.date_range('2025-01-01', periods=n, freq='1min', tz='UTC')
    price = np.full(n, 100.0)
    bars = pd.DataFrame({'event_time': times, 'open': price, 'high': price,
                         'low': price, 'close': price, 'volume': 1.0,
                         'trade_count': 1})
    grid = minute_grid(bars)
    config = Config()
    features = vol_features(grid, Seasonality.fit(log_returns(grid), config), config)
    target = forward_realised_vol(grid, config.window_minutes)
    frame = pd.concat([features, target], axis=1)
    with pytest.raises(ValueError, match='refusing to fit'):
        VolModel.fit(frame, frame['forward_vol'], config)


def test_sigma_scales_as_the_square_root_of_the_remaining_span():
    per_minute = np.array([1e-4, 1e-4, 1e-4])
    scaled = sigma_remaining(per_minute, np.array([1, 4, 9]))
    assert scaled == pytest.approx([1e-4, 2e-4, 3e-4])
    assert sigma_remaining(per_minute, 0)[0] == 0.0


def test_a_seasonal_ramp_multiplies_the_remaining_sigma():
    per_minute = np.array([1e-4])
    plain = sigma_remaining(per_minute, np.array([9]))
    ramped = sigma_remaining(per_minute, np.array([9]), np.array([1.5]))
    assert ramped[0] == pytest.approx(plain[0] * 1.5)


def test_the_har_fit_refuses_too_few_rows():
    bars = make_bars(days=1)['BTC-USD']
    grid = minute_grid(bars)
    config = Config()
    features = vol_features(grid, Seasonality.fit(log_returns(grid), config), config)
    with pytest.raises(ValueError, match='refusing to fit'):
        VolModel.fit(features.head(10), pd.Series([1e-4] * 10, index=features.index[:10]),
                     config)


def test_parkinson_is_indexed_like_everything_else():
    """Every per-minute series is stamped `as_of` = bar time + one minute."""
    bars = make_bars(days=1)['BTC-USD']
    grid = minute_grid(bars)
    pk = parkinson_vol(grid, 60)
    assert pk.index[0] == grid.index[0] + pd.Timedelta(minutes=1)
    assert log_returns(grid).index[0] == grid.index[0] + pd.Timedelta(minutes=1)
