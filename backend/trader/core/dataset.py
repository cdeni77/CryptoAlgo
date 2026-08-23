"""Loading the panel, and keeping every fitted thing inside its fold.

Three objects in this system are *fitted*: the volatility model, the intraday
seasonality factor, and the barrier baseline's scale and tail. All three are
fitted against realised outcomes, so all three leak if they see a test fold.
None of them is a headline number, which is exactly why the leak would go
unnoticed — a seasonality factor estimated on the full sample makes the
baseline stronger and the model look weaker, and nobody audits a result in that
direction.

So the split here is deliberate:

* **`Dataset` holds only what is computable from trailing bars.** The minute
  grid, the per-minute state (rolling statistics, all backward-looking), and the
  window table with its outcomes. Building it is the expensive step and it
  happens once.
* **`FoldFit` holds the three fitted objects**, and is constructed from a
  training slice. Applying it to any slice is cheap.

The seasonality shortcut matters for this to be affordable: only three of the
forty-two feature columns depend on the fitted factor, and all three are a
minute-of-day lookup, so `apply_seasonality` re-derives them per fold in
milliseconds instead of rebuilding five years of rolling windows six times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.baseline import BarrierBaseline, attach_baseline
from core.config import Config, DEFAULT_CONFIG
from core.datastore import ResearchStore
from core.features import apply_seasonality, attach_cross_asset, build_features, minute_state
from core.vol import (
    MINUTES_PER_DAY, Seasonality, VolModel, forward_realised_vol, log_returns,
    sigma_remaining as scale_sigma,
)
from core.windows import GridReport, build_window_panel, minute_grid

logger = logging.getLogger(__name__)

MINUTE_DATASET = 'minute_bars'
REFERENCE_SYMBOL = 'BTC-USD'


class DatasetError(RuntimeError):
    pass


def load_minute_bars(
    config: Config = DEFAULT_CONFIG,
    *,
    store: Optional[ResearchStore] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    symbols: Optional[Sequence[str]] = None,
) -> dict[str, pd.DataFrame]:
    """Read one-minute bars for the universe out of the research store."""
    store = store or ResearchStore()
    wanted = tuple(symbols) if symbols else config.symbols
    bars: dict[str, pd.DataFrame] = {}
    for symbol in wanted:
        frame = store.read(
            MINUTE_DATASET, venue=config.venue, symbols=[symbol],
            start=start, end=end,
        )
        if frame is None or frame.empty:
            logger.error('%s: no %s rows in the store for venue %r',
                         symbol, MINUTE_DATASET, config.venue)
            continue
        bars[symbol] = frame.sort_values('event_time', ignore_index=True)
    if not bars:
        raise DatasetError(
            f'no one-minute bars for {list(wanted)} on venue {config.venue!r}. '
            f'Run `python -m scripts.scrape --backfill-days 1825` and then '
            f'`python -m scripts.sync_store`.'
        )
    missing = sorted(set(wanted) - set(bars))
    if missing:
        logger.warning('universe is short of %s — the cross_asset group will be '
                       'thinner than it looks', missing)
    return bars


@dataclass
class Dataset:
    """The parts of the panel that no fit has touched."""

    config: Config
    grids: dict[str, pd.DataFrame]
    states: dict[str, pd.DataFrame]
    windows: pd.DataFrame
    reports: dict[str, GridReport]
    forward_vol: dict[str, pd.Series]

    @classmethod
    def build(
        cls,
        bars_by_symbol: dict[str, pd.DataFrame],
        config: Config = DEFAULT_CONFIG,
        *,
        reference: str = REFERENCE_SYMBOL,
    ) -> 'Dataset':
        grids = {s: minute_grid(b) for s, b in bars_by_symbol.items()}
        flat = Seasonality(factor=np.ones(MINUTES_PER_DAY), days_observed=0.0, smoothed_over=0)
        states = {s: minute_state(g, flat, config) for s, g in grids.items()}
        states = attach_cross_asset(states, reference, config)
        windows, reports = build_window_panel(bars_by_symbol, config)
        forward = {s: forward_realised_vol(g, config.window_minutes) for s, g in grids.items()}
        return cls(config=config, grids=grids, states=states, windows=windows,
                   reports=reports, forward_vol=forward)

    # ---- shape ---------------------------------------------------------
    @property
    def symbols(self) -> list[str]:
        return sorted(self.grids)

    @property
    def window_index(self) -> pd.DatetimeIndex:
        """The distinct window opens, sorted. Folds split on this, never on rows."""
        return pd.DatetimeIndex(sorted(self.windows['window_open'].unique()))

    @property
    def span_days(self) -> float:
        index = self.window_index
        if len(index) < 2:
            return 0.0
        return (index[-1] - index[0]).total_seconds() / 86400.0

    def coverage(self) -> pd.DataFrame:
        rows = []
        for symbol, report in sorted(self.reports.items()):
            rows.append({
                'symbol': symbol,
                'first_minute': report.first_minute,
                'last_minute': report.last_minute,
                'minutes_expected': report.minutes_expected,
                'minutes_present': report.minutes_present,
                'minute_coverage': report.minute_coverage,
                'windows': report.windows_total,
                'dropped_boundary': report.windows_dropped_boundary,
                'boundary_drop_rate': report.boundary_drop_rate,
                'interior_gaps': report.windows_with_interior_gaps,
            })
        return pd.DataFrame(rows)

    def trailing(self, days: Optional[float]) -> 'Dataset':
        """A hard cut to the last `days` of windows.

        Distinct from the recency half-life, which is a soft weighting. Both are
        useful and they are not substitutes: this one changes what the model can
        see, the other changes how much it cares.
        """
        if not days:
            return self
        index = self.window_index
        if len(index) == 0:
            return self
        cutoff = index[-1] - pd.Timedelta(days=days)
        return Dataset(
            config=self.config, grids=self.grids, states=self.states,
            windows=self.windows.loc[self.windows['window_open'] >= cutoff].reset_index(drop=True),
            reports=self.reports, forward_vol=self.forward_vol,
        )


@dataclass
class FoldFit:
    """The three fitted objects, plus the rows they were fitted on."""

    seasonality: dict[str, Seasonality]
    vol_models: dict[str, VolModel]
    baseline: BarrierBaseline
    train_windows: int
    states: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        lines = [f'fold fit on {self.train_windows:,} windows']
        for symbol in sorted(self.vol_models):
            season = self.seasonality[symbol]
            lines.append(
                f'  {symbol}: {self.vol_models[symbol].summary()} | '
                f'seasonal amplitude {season.amplitude:.2f} over {season.days_observed:.0f}d'
            )
        lines.append(f'  {self.baseline.summary()}')
        return '\n'.join(lines)


def fit_fold(
    dataset: Dataset,
    train_window_opens: pd.DatetimeIndex,
    config: Optional[Config] = None,
    *,
    groups: Optional[Sequence[str]] = None,
) -> tuple[FoldFit, pd.DataFrame]:
    """Fit seasonality, the volatility model and the baseline on training windows.

    Returns the fit and the *training* feature table. Scoring any other slice
    goes through `apply_fold`, which shares this fit — so a test row can never
    be scored against a model that saw it.
    """
    config = config or dataset.config
    train_end = train_window_opens.max() if len(train_window_opens) else None
    if train_end is None:
        raise DatasetError('empty training window set')

    seasonality: dict[str, Seasonality] = {}
    vol_models: dict[str, VolModel] = {}
    states: dict[str, pd.DataFrame] = {}
    for symbol, grid in dataset.grids.items():
        # Only bars strictly before the end of training may inform either fit.
        cut = grid.loc[grid.index < train_end]
        returns = log_returns(cut)
        seasonality[symbol] = Seasonality.fit(returns, config)
        state = apply_seasonality(dataset.states[symbol], seasonality[symbol])
        states[symbol] = state
        target = dataset.forward_vol[symbol]
        train_rows = state.loc[state.index < train_end]
        aligned = target.reindex(train_rows.index)
        usable = train_rows.loc[aligned.notna() & (aligned > 0)]
        vol_models[symbol] = VolModel.fit(usable, aligned.loc[usable.index], config)

    train_table = _score_windows(
        dataset, states, vol_models, seasonality,
        dataset.windows.loc[dataset.windows['window_open'].isin(train_window_opens)],
        config, groups=groups,
    )
    baseline = BarrierBaseline.fit(train_table, config)
    fit = FoldFit(seasonality=seasonality, vol_models=vol_models, baseline=baseline,
                  train_windows=len(train_window_opens), states=states)
    return fit, attach_baseline(train_table, baseline)


def apply_fold(
    dataset: Dataset,
    fit: FoldFit,
    window_opens: pd.DatetimeIndex,
    config: Optional[Config] = None,
    *,
    groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Score a slice of windows with a fit made elsewhere."""
    config = config or dataset.config
    table = _score_windows(
        dataset, fit.states, fit.vol_models, fit.seasonality,
        dataset.windows.loc[dataset.windows['window_open'].isin(window_opens)],
        config, groups=groups,
    )
    return attach_baseline(table, fit.baseline)


def _score_windows(
    dataset: Dataset,
    states: dict[str, pd.DataFrame],
    vol_models: dict[str, VolModel],
    seasonalities: dict[str, Seasonality],
    windows: pd.DataFrame,
    config: Config,
    *,
    groups: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach the volatility forecast, then the features, to a window slice."""
    if windows.empty:
        raise DatasetError('no windows in this slice')
    parts = []
    for symbol, part in windows.groupby('symbol', sort=True):
        state = states.get(symbol)
        model = vol_models.get(symbol)
        if state is None or model is None:
            logger.warning('%s: no volatility model, slice dropped', symbol)
            continue
        sigma = model.predict(state)
        decision = pd.DatetimeIndex(part['decision_time'])
        part = part.copy()
        part['sigma_per_min'] = sigma.reindex(decision).to_numpy()
        remaining = (config.window_minutes - part['offset']).to_numpy()

        # The HAR forecasts volatility *at the decision minute*, seasonality
        # included. The remaining span is a different set of minutes and can
        # straddle a seasonal ramp — a window opening at 13:28 covers the New
        # York cash open — so scale by the ratio of the root-mean seasonal
        # factor over the minutes actually left to the factor at the decision
        # minute. Variance adds, which is why it is a root mean and not a mean.
        seasonality = seasonalities[symbol]
        now_factor = seasonality.at(decision)
        ramp = np.ones(len(part))
        for span in np.unique(remaining):
            mask = remaining == span
            ramp[mask] = (seasonality.mean_over(decision[mask], int(span))
                          / np.maximum(now_factor[mask], 1e-9))
        part['seasonal_ramp'] = np.log(np.maximum(ramp, 1e-9))
        part['sigma_remaining'] = scale_sigma(
            part['sigma_per_min'].to_numpy(), remaining, ramp)
        part['log_sigma_per_min'] = np.log(np.maximum(part['sigma_per_min'], 1e-9))
        parts.append(part)
    if not parts:
        raise DatasetError('every symbol lacked a volatility model')
    scored = pd.concat(parts, ignore_index=True)
    return build_features(scored, states, config, groups=groups)
