"""Features, grouped by the mechanism each one exploits.

A feature earns its place only by naming a way the baseline is wrong:

1. `vol_state`: the baseline's sigma is mis-estimated, predictably.
2. `microstructure`: the baseline's drift is zero and reality's is not
   (bid-ask bounce, order-flow bursts).
3. `cross_asset`: Bitcoin leads Ether and Solana at the minute scale; the
   barrier sees one symbol.
4. `geometry`: the baseline is Markov and the path is not.
5. `time_of_day`: the control. Calendar position cannot forecast direction, so
   a good score here means the measurement is wrong.

Point-in-time: every column is knowable at `decision_time`. The per-minute frame
is indexed by `as_of` (bar time plus one minute), so the join is on equality.
"""

from __future__ import annotations

import logging
from typing import Iterable, Optional, Sequence

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.vol import MINUTES_PER_DAY, Seasonality, log_returns, realised_vol, vol_features

logger = logging.getLogger(__name__)

# `us_equity_hours` uses New York local time, so DST is handled.
NEW_YORK = ZoneInfo('America/New_York')

# The canonical column list, per group. `build_features` reindexes to exactly
# this, so a saved model always scores the same matrix: a group that produced
# nothing arrives as an all-NaN column, not a missing one.
# Column names are imported from the code that computes them.
from core.book_flow import BOOK_FLOW
from core.index_features import INDEX_PROXY                       # noqa: E402
from core.book_features import (                                # noqa: E402
    CROSS_VENUE as BOOK_CROSS_VENUE,
    IMPLIED_VOL as BOOK_IMPLIED_VOL,
    MARKET_PRICE as BOOK_MARKET_PRICE,
    MARKET_STATE as BOOK_MARKET_STATE,
)

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    'vol_state': (
        'log_rv_15', 'log_rv_60', 'log_rv_240', 'log_rv_1440',
        'rv_slope_short', 'rv_slope_long', 'range_to_return', 'vol_of_vol',
        'log_sigma_per_min', 'rv_surprise', 'seasonal_ramp',
    ),
    'microstructure': (
        'autocorr_60', 'run_length', 'body_ratio_15', 'volume_z_15',
        'vwap_gap_15', 'signed_volume_15', 'zero_return_share_60',
    ),
    'cross_asset': (
        'peer_displacement', 'peer_return_5', 'peer_return_15',
        'lead_residual', 'beta_1440', 'universe_dispersion',
    ),
    'geometry': (
        'z_score', 'abs_z_score', 'excursion_up_z', 'excursion_down_z',
        'excursion_span_z', 'excursion_asymmetry', 'path_efficiency',
        'displacement_vs_elapsed', 'touched_opposite',
    ),
    # The decision offset. Legitimately informative: it lets one pooled model
    # behave like per-offset models.
    'offset': (
        'elapsed_fraction', 'remaining_minutes',
    ),
    # The control. Calendar position cannot forecast direction.
    # `quarter_of_hour` is the quarter of the hour, so it is calendar, not offset.
    'time_of_day': (
        'quarter_of_hour', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'us_equity_hours',
    ),
    # --- the venue's own book ---
    #
    # Declared as groups so each can run alone (`--groups market_state`).
    # LightGBM gain share says where splits went, not what predicts; the
    # group-alone run is the real test.
    # `market_minus_baseline` can echo the quote, so it sits in `market_price`
    # and the structure-only variant runs without it.
    'market_state': BOOK_MARKET_STATE,
    'market_price': BOOK_MARKET_PRICE,
    'cross_venue': BOOK_CROSS_VENUE,
    'implied_vol': BOOK_IMPLIED_VOL,
    # Book dynamics, as against the snapshot `market_state` reads. A candidate
    # under test, so opt-in.
    'book_flow': BOOK_FLOW,
    # The settlement index, estimated live from its constituent exchanges
    # (`core/index_proxy.py`). Short history, so opt-in.
    'index_proxy': INDEX_PROXY,
    # **The book groups split by what the DATA requires, not by what they
    # mean.** `market_state` and `book_flow` each mix two kinds of feature:
    # some read only the quote, some read resting SIZES. That distinction is
    # invisible in the names and decides whether a feature can exist at all
    # before 2026-08-25, because the venue serves no historical order book and
    # Predexon's minute-stale touch sizes are not a substitute (ratio 1.22,
    # correlation 0.26 against the live book, measured 2026-10-21).
    #
    # Selecting the whole group therefore drags four perfectly recoverable
    # features down with seven unrecoverable ones — and under
    # `--complete-cases` that costs the entire pre-August history. Same reason
    # `market_price` was split out of `market_state`.
    #
    # These OVERLAP the groups above by design: pick one framing or the other.
    'book_quote': ('spread', 'spread_change_3', 'mid_change_1', 'mid_change_3'),
    'book_depth': ('imbalance_touch', 'imbalance_5c', 'depth_ratio',
                   'book_convexity', 'imbalance_change_3', 'depth_trend_3',
                   'imbalance_persistence'),
}

# The venue's book covers a small share of the bar history, so these groups
# are selectable, not default: in the default matrix they would be mostly NaN.
BOOK_GROUPS = ('market_state', 'market_price', 'cross_venue', 'implied_vol',
               'book_flow', 'book_quote', 'book_depth')
# The two framings of the eleven book columns. Declared, so `feature_columns`
# can refuse a selection that mixes them and so the no-column-in-two-groups
# invariant can be asserted WITHIN a framing rather than abandoned.
QUOTE_DEPTH_FRAMING = ('book_quote', 'book_depth')
MEANING_FRAMING = ('market_state', 'book_flow')
# Selectable, never default: everything with a short history.
OPT_IN_GROUPS = BOOK_GROUPS + ('index_proxy',)
ALL_GROUPS = tuple(g for g in FEATURE_GROUPS if g not in OPT_IN_GROUPS)
# The control. Named so a survey cannot quietly omit it.
CONTROL_GROUPS = ('time_of_day',)


def feature_columns(groups: Optional[Sequence[str]] = None) -> list[str]:
    chosen = ALL_GROUPS if groups is None else tuple(groups)
    unknown = sorted(set(chosen) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f'unknown feature groups: {unknown}')
    # **Mixing the two framings is refused, because it is a silent no-op.**
    # `book_quote | book_depth` and `market_state | book_flow` cover the SAME
    # eleven columns, cut by data requirement in one case and by meaning in
    # the other. Dedup below makes selecting both harmless, which is exactly
    # the problem: an ablation that drops `book_depth` while `market_state` is
    # still selected gets every depth column back and reports no effect. That
    # reads as "the feature does not matter" when the feature was never
    # removed — the most expensive kind of wrong answer this repo can produce.
    both = sorted(set(chosen) & set(QUOTE_DEPTH_FRAMING)), sorted(
        set(chosen) & set(MEANING_FRAMING))
    if both[0] and both[1]:
        raise ValueError(
            'feature groups %s and %s are two framings of the same columns; '
            'select one framing, not both' % (both[0], both[1]))
    # De-duplicated anyway: a framing's own groups are disjoint, but this
    # keeps `feature_columns` order-stable and total.
    seen, out = set(), []
    for g in chosen:
        for c in FEATURE_GROUPS[g]:
            if c not in seen:
                seen.add(c)
                out.append(c)
    return out


# --------------------------------------------------------------------------
# per-minute state
# --------------------------------------------------------------------------

def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback, min_periods=lookback // 4).mean()
    sd = series.rolling(lookback, min_periods=lookback // 4).std()
    return (series - mean) / sd.replace(0.0, np.nan)


def apply_seasonality(state: pd.DataFrame, seasonality: Seasonality) -> pd.DataFrame:
    """Re-derive the three seasonality-dependent columns.

    Seasonality is fitted, so it stays inside the fold. `core/dataset.py` builds
    the state once against a flat factor and calls this per fold.
    """
    # Shallow copy: only three columns are rewritten, and copy-on-write keeps
    # the rest shared. A deep copy costs hundreds of megabytes per fold.
    state = state.copy(deep=False)
    state['seasonal'] = seasonality.at(state.index)
    state['log_seasonal'] = np.log(state['seasonal'])
    state['rv_surprise'] = state['log_rv_15'] - (state['log_rv_1440'] + state['log_seasonal'])
    return state


def minute_state(
    grid: pd.DataFrame,
    seasonality: Seasonality,
    config: Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Per-minute features for one symbol, indexed by `as_of`.

    `as_of` is the minute after the bar, so a row stamped 10:04 is exactly what
    a decision at 10:04 may use.
    """
    frame = vol_features(grid, seasonality, config)
    returns = frame['r']

    # -- vol_state ---------------------------------------------------------
    frame['rv_slope_short'] = frame['log_rv_15'] - frame['log_rv_60']
    frame['rv_slope_long'] = frame['log_rv_240'] - frame['log_rv_1440']
    frame['range_to_return'] = frame['log_pk_60'] - frame['log_rv_60']
    frame['vol_of_vol'] = frame['log_rv_15'].rolling(240, min_periods=60).std()
    # `rv_surprise`: short-lookback vol against what the long lookback and the
    # seasonal factor predict.
    predicted = frame['log_rv_1440'] + frame['log_seasonal']
    frame['rv_surprise'] = frame['log_rv_15'] - predicted

    # -- microstructure ----------------------------------------------------
    frame['autocorr_60'] = (
        returns.rolling(60, min_periods=30).corr(returns.shift(1))
    )
    sign = np.sign(returns.fillna(0.0))
    # Length of the current run of same-signed minutes, signed by its direction.
    # Vectorised with groupby-cumcount over run ids.
    run_id = (sign != sign.shift(1)).cumsum()
    frame['run_length'] = (sign.groupby(run_id).cumcount() + 1) * sign

    body = (grid['close'] - grid['open']).abs()
    span = (grid['high'] - grid['low']).replace(0.0, np.nan)
    body_ratio = (body / span)
    body_ratio.index = body_ratio.index + pd.Timedelta(minutes=1)
    frame['body_ratio_15'] = body_ratio.rolling(15, min_periods=5).mean()

    volume = grid['volume'].copy()
    volume.index = volume.index + pd.Timedelta(minutes=1)
    log_volume = np.log1p(volume)
    frame['volume_z_15'] = _zscore(log_volume.rolling(15, min_periods=5).mean(), MINUTES_PER_DAY)
    # No trade-count feature: Coinbase candles carry no trade count.

    close = grid['close'].copy()
    close.index = close.index + pd.Timedelta(minutes=1)
    typical = close * volume
    vwap = (typical.rolling(15, min_periods=5).sum()
            / volume.rolling(15, min_periods=5).sum().replace(0.0, np.nan))
    frame['vwap_gap_15'] = (close - vwap) / close
    frame['signed_volume_15'] = (
        (np.sign(returns.fillna(0.0)) * log_volume).rolling(15, min_periods=5).mean()
    )
    frame['zero_return_share_60'] = (
        (returns.fillna(0.0) == 0).astype(float).rolling(60, min_periods=20).mean()
    )

    frame['close'] = close
    return frame


def attach_cross_asset(
    states: dict[str, pd.DataFrame],
    reference: str,
    config: Config = DEFAULT_CONFIG,
) -> dict[str, pd.DataFrame]:
    """Add peer-return and lead-lag columns.

    The reference is Bitcoin, the deepest book and the leader. For Bitcoin itself
    the reference is the mean of the other two, so the columns mean "how far the
    rest of the universe moved" for every symbol.
    """
    returns = {s: f['r'] for s, f in states.items()}
    out: dict[str, pd.DataFrame] = {}
    for symbol, frame in states.items():
        peers = [s for s in states if s != symbol]
        if not peers:
            for column in ('peer_return_5', 'peer_return_15', 'beta_1440', 'lead_residual',
                           'universe_dispersion'):
                frame[column] = np.nan
            out[symbol] = frame
            continue
        if symbol == reference or reference not in states:
            peer_r = pd.concat([returns[s] for s in peers], axis=1).mean(axis=1)
        else:
            peer_r = returns[reference]
        own_r = frame['r']
        frame['peer_return_5'] = peer_r.rolling(5, min_periods=3).sum()
        frame['peer_return_15'] = peer_r.rolling(15, min_periods=8).sum()
        cov = own_r.rolling(MINUTES_PER_DAY, min_periods=240).cov(peer_r)
        var = peer_r.rolling(MINUTES_PER_DAY, min_periods=240).var()
        beta = cov / var.replace(0.0, np.nan)
        frame['beta_1440'] = beta
        # Own 15-minute move minus what the peer's move implies. Negative means
        # this symbol has not caught up yet: the tradeable direction.
        frame['lead_residual'] = (
            own_r.rolling(15, min_periods=8).sum() - beta * frame['peer_return_15']
        )
        all_r = pd.concat([returns[s].rolling(15, min_periods=8).sum() for s in states], axis=1)
        frame['universe_dispersion'] = all_r.std(axis=1)
        out[symbol] = frame
    return out


# --------------------------------------------------------------------------
# window-level assembly
# --------------------------------------------------------------------------

def _clock_features(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    decision = pd.DatetimeIndex(frame['decision_time'])
    window_open = pd.DatetimeIndex(frame['window_open'])
    out = pd.DataFrame(index=frame.index)
    out['elapsed_fraction'] = frame['offset'] / config.window_minutes
    # Variance-minutes, matching what `sigma_remaining` was scaled by.
    out['remaining_minutes'] = [
        config.remaining_variance_minutes(o) for o in frame['offset']]
    out['quarter_of_hour'] = window_open.minute // config.window_minutes
    minute_of_day = decision.hour * 60 + decision.minute
    out['hour_sin'] = np.sin(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    out['hour_cos'] = np.cos(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    out['dow_sin'] = np.sin(2 * np.pi * decision.dayofweek / 7)
    out['dow_cos'] = np.cos(2 * np.pi * decision.dayofweek / 7)
    # The US cash session in New York local time, so it holds in both DST regimes.
    local = decision.tz_convert(NEW_YORK) if decision.tz is not None else decision
    local_minute = local.hour * 60 + local.minute
    out['us_equity_hours'] = ((local_minute >= 9 * 60 + 30)
                              & (local_minute < 16 * 60)
                              & (local.dayofweek < 5)).astype(float)
    return out


def _geometry_features(frame: pd.DataFrame, config: Config) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    sigma = frame['sigma_remaining'].replace(0.0, np.nan)
    out['z_score'] = frame['displacement'] / sigma
    out['abs_z_score'] = out['z_score'].abs()
    out['excursion_up_z'] = frame['excursion_up'] / sigma
    out['excursion_down_z'] = frame['excursion_down'] / sigma
    out['excursion_span_z'] = (frame['excursion_up'] - frame['excursion_down']) / sigma
    span = (frame['excursion_up'] - frame['excursion_down']).replace(0.0, np.nan)
    # Where the price sits in its own realised range: -1 at the low, +1 at the
    # high. `F(x/sigma)` cannot see this.
    out['excursion_asymmetry'] = (
        (frame['excursion_up'] + frame['excursion_down']) / span
    )
    out['path_efficiency'] = frame['displacement'].abs() / span
    # The move so far against what its elapsed minutes normally produce.
    # Independent of the remaining span, so not a restatement of z_score.
    elapsed_sigma = frame['sigma_per_min'] * np.sqrt(frame['offset'].clip(lower=1))
    out['displacement_vs_elapsed'] = frame['displacement'] / elapsed_sigma.replace(0.0, np.nan)
    # Has the window already been on the other side of the strike? A barrier
    # has no memory of that.
    out['touched_opposite'] = np.where(
        frame['displacement'] >= 0, (frame['excursion_down'] < 0).astype(float),
        (frame['excursion_up'] > 0).astype(float),
    )
    return out


def gap_change_column(table: pd.DataFrame, decision_offsets) -> list:
    """`venue_prob_gap` differenced against the previous configured offset.

    Not a row shift over surviving rows: `--complete-cases` can drop individual
    offsets, so a row shift could step back several offsets. Live
    (`scripts/live.py::gap_change`) uses `offsets[i-1]` or nothing, and cannot
    see which rows training dropped, so the configured grid is the shared definition.
    """
    grid = sorted(int(o) for o in decision_offsets)
    back = {o: grid[i - 1] for i, o in enumerate(grid) if i}
    gap = table['venue_prob_gap']
    lookup = {(sym, win, int(off)): val for sym, win, off, val in zip(
        table['symbol'], table['window_open'], table['offset'], gap)}
    out = []
    for sym, win, off, val in zip(table['symbol'], table['window_open'],
                                  table['offset'], gap):
        key = (sym, win, back.get(int(off)))
        prior = lookup.get(key)
        out.append(float(val) - float(prior)
                   if key[2] is not None and prior is not None
                   and pd.notna(prior) and pd.notna(val)
                   else float('nan'))
    return out


def build_features(
    windows: pd.DataFrame,
    minute_states: dict[str, pd.DataFrame],
    config: Config = DEFAULT_CONFIG,
    *,
    groups: Optional[Sequence[str]] = None,
    deferred: Sequence[str] = (),
) -> pd.DataFrame:
    """Join per-minute state onto the window table and add window-level features.

    Returns `windows` with `sigma_per_min`, `sigma_remaining` and every selected
    feature column attached, reindexed to the canonical list so the matrix shape
    does not depend on which groups happened to populate.
    """
    from core.vol import sigma_remaining as scale_sigma

    if 'sigma_per_min' not in windows.columns:
        raise ValueError(
            'build_features needs `sigma_per_min` on the window table — attach it '
            'with core.dataset.attach_volatility first, which is where the fitted '
            'VolModel lives so the fit stays inside the fold'
        )

    frames = []
    for symbol, part in windows.groupby('symbol', sort=True):
        state = minute_states.get(symbol)
        if state is None:
            logger.warning('%s: no minute state, features will be NaN', symbol)
            frames.append(part)
            continue
        joined = part.merge(
            state.drop(columns=[c for c in ('r', 'close') if c in state.columns]),
            left_on='decision_time', right_index=True, how='left', suffixes=('', '_state'),
        )
        frames.append(joined)
    table = pd.concat(frames, ignore_index=True).sort_values(
        ['decision_time', 'symbol', 'offset'], ignore_index=True)

    # `peer_displacement` is window-level: what the other symbols did in this
    # same window, the form the lead-lag mechanism takes here.
    pivot = table.pivot_table(
        index=['window_open', 'offset'], columns='symbol', values='displacement', aggfunc='first')
    totals = pivot.sum(axis=1, min_count=1)
    counts = pivot.notna().sum(axis=1)
    peer_frame = pivot.copy()
    for symbol in pivot.columns:
        others = counts - pivot[symbol].notna().astype(int)
        peer_frame[symbol] = (totals - pivot[symbol].fillna(0.0)) / others.replace(0, np.nan)
    peer_long = peer_frame.stack(future_stack=True).rename('peer_displacement').reset_index()
    table = table.merge(peer_long, on=['window_open', 'offset', 'symbol'], how='left')

    table = pd.concat([
        table,
        _clock_features(table, config),
        _geometry_features(table, config),
    ], axis=1)

    # --- the book families ---
    if 'ask_up' in table.columns or 'bid_at_touch' in table.columns:
        from core.book_features import market_state_features
        # `market_state_features` reads the raw snapshot shape: prices in cents
        # under best_bid/best_ask. The joined table carries dollars under
        # ask_up/ask_down, so rebuild the snapshot here.
        #   ask_down = 1 - yes_bid  ->  yes_bid = 1 - ask_down
        snap = table.copy()
        snap['best_bid'] = (1.0 - pd.to_numeric(
            table.get('ask_down'), errors='coerce')) * 100.0
        snap['best_ask'] = pd.to_numeric(
            table.get('ask_up'), errors='coerce') * 100.0
        snap['n_snapshots'] = np.nan
        book = market_state_features(snap)
        for column in book.columns:
            table[column] = book[column].values
        if 'pm_market_probability' in table.columns:
            gap = (pd.to_numeric(table.get('market_probability'), errors='coerce')
                   - pd.to_numeric(table['pm_market_probability'], errors='coerce'))
            table['venue_prob_gap'] = gap
            table['pm_available'] = np.where(
                pd.to_numeric(table['pm_market_probability'],
                              errors='coerce').notna(), 1.0, 0.0)
            k_spread = pd.to_numeric(table.get('spread'), errors='coerce')
            p_spread = pd.to_numeric(table.get('pm_spread'), errors='coerce')
            ratio = k_spread / p_spread.replace(0.0, np.nan)
            table['venue_spread_ratio'] = np.where(ratio > 0, np.log(ratio), np.nan)
            # Change since the previous configured decision offset, within the
            # same window and symbol, so it never crosses a window boundary.
            # The `_5` in the name is a misnomer: on the (3, 6, 9, 12) grid it is
            # a three-minute step. Do not change the arithmetic without a refit:
            # live (`scripts/live.py::gap_change`) and the artifact use this name.
            # NaN at a window's first offset is correct on both sides.
            table['venue_gap_change_5'] = gap_change_column(
                table, config.decision_offsets)

    # `iv_minus_realised` divides by the fitted `sigma_per_min`, which exists
    # only here, inside the fold. Computing it earlier would leak.
    if 'implied_sigma_per_min' in table.columns:
        implied = pd.to_numeric(table['implied_sigma_per_min'], errors='coerce')
        realised = pd.to_numeric(table.get('sigma_per_min'), errors='coerce')
        ratio = implied / realised.replace(0.0, np.nan)
        table['iv_minus_realised'] = np.where(ratio > 0, np.log(ratio), np.nan)

    # `proxy_displacement_z` divides by the fitted `sigma_remaining`, so it is
    # finished here for the same reason.
    if 'proxy_displacement' in table.columns:
        from core.index_features import proxy_displacement_z
        table['proxy_displacement_z'] = proxy_displacement_z(
            table['proxy_displacement'], table['sigma_remaining'])

    return reindex_to_features(table, feature_columns(groups), deferred=deferred)


def reindex_to_features(table: pd.DataFrame, wanted: Sequence[str], *,
                        deferred: Sequence[str] = ()) -> pd.DataFrame:
    """Create every selected column, warning about the ones nobody will fill.

    `deferred` names columns the caller attaches afterwards (the live book
    groups and `iv_minus_realised`); those get no warning, so a real gap stays visible.
    The column is always created: the matrix is built by name from the artifact's list.
    """
    deferred = set(deferred or ())
    for column in wanted:
        if column not in table.columns:
            if column not in deferred:
                logger.warning('feature %s was not produced; carried as all-NaN',
                               column)
            table[column] = np.nan
    return table


def population_report(table: pd.DataFrame, groups: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """How much of each feature is populated.

    An all-NaN group has the same shape as a working one; this says which is which.
    """
    wanted = feature_columns(groups)
    lookup = {c: g for g, cs in FEATURE_GROUPS.items() for c in cs}
    rows = []
    for column in wanted:
        series = table[column] if column in table.columns else pd.Series(dtype=float)
        finite = int(np.isfinite(series.to_numpy(dtype=float)).sum()) if len(series) else 0
        rows.append({
            'feature': column, 'group': lookup[column],
            'populated': finite,
            'share': finite / len(table) if len(table) else 0.0,
            'is_control': lookup[column] in CONTROL_GROUPS,
        })
    return pd.DataFrame(rows).sort_values(['group', 'feature'], ignore_index=True)
