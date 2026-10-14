"""Loading the panel, and keeping every fitted thing inside its fold.

Three objects are fitted against realised outcomes, so all three leak if they
see a test fold: the volatility model, the intraday seasonality factor, and the
baseline's scale and tail.

* `Dataset` holds only what trailing bars compute. It is built once.
* `FoldFit` holds the three fitted objects, built from a training slice.

Only a few feature columns depend on the seasonality factor, and
`apply_seasonality` re-derives them per fold cheaply.
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


def peer_quote_limits(config: Config) -> dict:
    """The peer venue's staleness cap and null-age rule.

    Unset, the cap follows `max_quote_age_seconds` and a missing peer age stays
    fresh. `null_quote_age` reaches the peer only once its own cap is set."""
    peer = getattr(config, 'peer_max_quote_age_seconds', None)
    return {'peer_max_age_seconds': peer,
            'peer_null_age': ('fresh' if peer is None
                              else getattr(config, 'null_quote_age', 'fresh'))}


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
        depth: Optional[pd.DataFrame] = None,
        ladder_fits: Optional[pd.DataFrame] = None,
        index_proxy: Optional[pd.DataFrame] = None,
    ) -> 'Dataset':
        grids = {s: minute_grid(b) for s, b in bars_by_symbol.items()}
        flat = Seasonality(factor=np.ones(MINUTES_PER_DAY), days_observed=0.0, smoothed_over=0)
        states = {s: minute_state(g, flat, config) for s, g in grids.items()}
        states = attach_cross_asset(states, reference, config)
        windows, reports = build_window_panel(bars_by_symbol, config)
        # Quotes and ladder fits are observed, not fitted, so they attach here,
        # not per fold. Only `iv_minus_realised` waits for `build_features`,
        # because its denominator is the fitted sigma.
        if depth is not None and len(depth):
            from core.quotes import attach_quotes
            windows = attach_quotes(
                windows, depth,
                max_age_seconds=float(config.max_quote_age_seconds),
                sources=tuple(getattr(config, 'quote_sources', ()) or ()),
                null_age=getattr(config, 'null_quote_age', 'fresh'),
                decision_latency_seconds=float(
                    getattr(config, 'decision_latency_seconds', 3.0)),
                **peer_quote_limits(config))
            # Book dynamics from the same depth table. Observed, so attached here.
            # `book_flow_features` reads only minutes at or before the decision.
            from core.book_flow import book_flow_features
            from core.quotes import filter_sources
            flow = book_flow_features(
                windows,
                filter_sources(depth,
                               tuple(getattr(config, 'quote_sources', ()) or ())),
                decision_latency_seconds=float(
                    getattr(config, 'decision_latency_seconds', 3.0)))
            for col in flow.columns:
                windows[col] = flow[col].to_numpy()
        if ladder_fits is not None and len(ladder_fits):
            from core.book_features import attach_implied_vol
            windows = attach_implied_vol(
                windows, ladder_fits,
                decision_latency_seconds=float(
                    getattr(config, 'decision_latency_seconds', 3.0)))
        # The settlement-index proxy, as per-minute summaries
        # (`core.index_features.minute_summary` / `load_proxy_minutes`).
        # Observed, so it attaches here; its z-score waits for the fold.
        if index_proxy is not None and len(index_proxy):
            from core.index_features import attach_index_proxy
            windows = attach_index_proxy(
                windows, index_proxy,
                max_age_seconds=config.index_proxy_max_age_seconds)
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

        Changes what the model can see; the recency half-life changes only the weighting.
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


# One representative column per book family that a complete row must carry.
# Requiring every column would drop rows for an absent depth level, not a missing book.
COMPLETE_CASE_COLUMNS = {
    'market_state': ('ask_up', 'market_probability', 'bid_at_touch'),
    'cross_venue': ('pm_market_probability',),
    'implied_vol': ('implied_sigma_per_min',),
    'label': ('venue_outcome',),
}


def complete_cases(windows: pd.DataFrame,
                   groups: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Rows where every field the model claims to use is present.

    LightGBM learns a default direction for NaN, so a mixed panel trains a blend
    of two models. `venue_outcome` is required, so rows are graded on the venue's
    own settlement, not our Coinbase label.
    """
    wanted = tuple(groups) + ('label',) if groups else tuple(COMPLETE_CASE_COLUMNS)
    needed = [c for g in wanted for c in COMPLETE_CASE_COLUMNS.get(g, ())]
    present = [c for c in needed if c in windows.columns]
    missing = sorted(set(needed) - set(present))
    if missing:
        logger.warning('complete_cases: %s never joined, so it cannot be required',
                       ', '.join(missing))
    if not present or not len(windows):
        return windows.iloc[0:0] if not len(windows) else windows
    return windows[windows[present].notna().all(axis=1)]


@dataclass
class ScoringBundle:
    """Everything needed to score a window that has never been seen.

    Carries the fitted seasonality and volatility models with the baseline, so a
    promoted artifact can score live. Excludes the per-minute state frames: live
    rebuilds them from the last day of bars.
    """

    seasonality: dict[str, Seasonality]
    vol_models: dict[str, VolModel]
    baseline: BarrierBaseline
    symbols: tuple[str, ...]
    window_minutes: int
    decision_offsets: tuple[int, ...]

    def covers(self, symbol: str) -> bool:
        return symbol in self.seasonality and symbol in self.vol_models

    def summary(self) -> str:
        return (f'scoring bundle: {len(self.vol_models)} symbols, '
                f'{self.window_minutes}min windows at '
                f'{", ".join(f"+{o}m" for o in self.decision_offsets)}, '
                f'{self.baseline.distribution} baseline')


@dataclass
class FoldFit:
    """The three fitted objects, plus the rows they were fitted on."""

    seasonality: dict[str, Seasonality]
    vol_models: dict[str, VolModel]
    baseline: BarrierBaseline
    train_windows: int
    states: dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def bundle(self, config: Config) -> ScoringBundle:
        """The deployable subset: fits, no frames."""
        return ScoringBundle(
            seasonality=self.seasonality, vol_models=self.vol_models,
            baseline=self.baseline, symbols=tuple(sorted(self.vol_models)),
            window_minutes=config.window_minutes,
            decision_offsets=tuple(config.decision_offsets),
        )

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

    Returns the fit and the training feature table. Other slices go through
    `apply_fold`, so a test row is never scored by a model that saw it.
    """
    config = config or dataset.config
    train_end = train_window_opens.max() if len(train_window_opens) else None
    if train_end is None:
        raise DatasetError('empty training window set')

    seasonality: dict[str, Seasonality] = {}
    vol_models: dict[str, VolModel] = {}
    states: dict[str, pd.DataFrame] = {}
    for symbol, grid in dataset.grids.items():
        # Only bars strictly before the end of training inform either fit.
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
    deferred: Sequence[str] = (),
) -> pd.DataFrame:
    """Attach the volatility forecast, then the features, to a window slice.

    `deferred` names feature columns the caller fills afterwards (the live
    book groups). They are created empty without a warning, so a real gap still shows.
    """
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
        # Variance-minutes, not wall-clock minutes: the settlement value is a
        # one-minute average, so remaining variance at offset m is
        # `(window - delta - m) + delta/3` (2.33 at m=12, not 3).
        remaining = np.array(
            [config.remaining_variance_minutes(o) for o in part['offset']],
            dtype=float)

        # The HAR forecasts volatility at the decision minute. Scale by the
        # root-mean seasonal factor over the remaining minutes divided by the
        # factor now, because the remaining span can straddle a seasonal ramp.
        seasonality = seasonalities[symbol]
        now_factor = seasonality.at(decision)
        ramp = np.ones(len(part))
        # Rounded up to whole minutes for the per-minute seasonal table: the
        # longer span is the conservative reading.
        for span in np.unique(np.ceil(remaining).astype(int)):
            mask = np.ceil(remaining).astype(int) == span
            ramp[mask] = (seasonality.mean_over(decision[mask], max(int(span), 1))
                          / np.maximum(now_factor[mask], 1e-9))
        part['seasonal_ramp'] = np.log(np.maximum(ramp, 1e-9))
        part['sigma_remaining'] = scale_sigma(
            part['sigma_per_min'].to_numpy(), remaining, ramp)
        part['log_sigma_per_min'] = np.log(np.maximum(part['sigma_per_min'], 1e-9))
        parts.append(part)
    if not parts:
        raise DatasetError('every symbol lacked a volatility model')
    scored = pd.concat(parts, ignore_index=True)
    return build_features(scored, states, config, groups=groups,
                          deferred=deferred)


def score_live(
    bars_by_symbol: dict[str, pd.DataFrame],
    bundle: ScoringBundle,
    config: Config,
    *,
    window_open: pd.Timestamp,
    offset: int,
    groups: Optional[Sequence[str]] = None,
    deferred: Sequence[str] = (),
) -> pd.DataFrame:
    """Score exactly one decision point per symbol, from freshly fetched bars.

    Uses the same code path as the backtest; only the requested windows differ.
    The outcome is NaN: the window has not settled.
    """
    if offset not in bundle.decision_offsets:
        logger.warning('offset +%dm is not one the model was fitted at (%s)',
                       offset, bundle.decision_offsets)
    grids, states = {}, {}
    for symbol, bars in bars_by_symbol.items():
        if not bundle.covers(symbol):
            logger.error('%s has no fitted volatility model in this artifact, skipped', symbol)
            continue
        grid = minute_grid(bars)
        grids[symbol] = grid
        flat = Seasonality(factor=np.ones(MINUTES_PER_DAY), days_observed=0.0, smoothed_over=0)
        state = minute_state(grid, flat, config)
        states[symbol] = apply_seasonality(state, bundle.seasonality[symbol])
    if not states:
        raise DatasetError('no symbol could be scored with this artifact')
    states = attach_cross_asset(states, REFERENCE_SYMBOL, config)

    # `include_unsettled=True`: the window being decided has not settled.
    # Without it, `build_windows` trims the in-progress window away.
    windows, _ = build_window_panel(
        {s: bars_by_symbol[s] for s in states}, config, offsets=(offset,),
        include_unsettled=True)
    slice_ = windows.loc[windows['window_open'] == window_open]
    if slice_.empty:
        raise DatasetError(
            f'no window opens at {window_open} for offset +{offset}m. Either the '
            f'feed does not reach {window_open} + {offset}m - 1min yet (a stale '
            f'or partial fetch, which is withheld rather than forward-filled), '
            f'or the previous window has no settlement minute so this one has no '
            f'strike.'
        )
    dataset = Dataset(config=config, grids=grids, states=states, windows=windows,
                      reports={}, forward_vol={})
    table = _score_windows(dataset, states, bundle.vol_models, bundle.seasonality,
                           slice_, config, groups=groups, deferred=deferred)
    scored = attach_baseline(table, bundle.baseline)
    # The window has not settled; clear values derived from a settle price.
    scored['outcome'] = np.nan
    scored['settle_price'] = np.nan
    scored['settle_return'] = np.nan
    return scored
