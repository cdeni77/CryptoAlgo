"""Features, grouped by the mechanism each one is supposed to exploit.

The baseline already knows the arithmetic. A feature earns its place here only
by naming a way the baseline is *wrong*, and there are exactly four:

1. **`vol_state` — the baseline's sigma is mis-estimated.** This is the only
   input the barrier has, so any predictable error in it is predictable error
   in the probability. Volatility term structure, the range-to-return ratio,
   vol of vol, and whether the remaining span of the window runs into a
   seasonal ramp.

2. **`microstructure` — the baseline's drift is zero and reality's is not.**
   One-minute returns are not a martingale at the tick level: bid-ask bounce
   makes them negatively autocorrelated, order-flow bursts make them
   positively autocorrelated, and either one is a drift the barrier ignores.

3. **`cross_asset` — the three symbols do not move at the same instant.**
   Bitcoin leads Ether and Solana at the minute scale. If Bitcoin has already
   moved in this window and Solana has not, Solana's remaining minutes carry a
   drift, and the barrier — which sees only Solana — cannot know it.

4. **`geometry` — the baseline is Markov and the path is not.** A window that
   spiked forty basis points and came back to two is not the same as one that
   drifted to two, but `F(x/sigma)` cannot tell them apart: it sees only where
   price is now. Excursions and path efficiency are the correction, and this is
   the group with the clearest theoretical claim on being non-zero.

5. **`clock` — a control, and labelled as one.** Hour of day, day of week, the
   quarter-hour the window opens on. Time of day cannot forecast direction, so
   if this group scores well on its own something is wrong with the
   measurement, not with the market. The last incarnation of this project ran a
   27-cell survey in which `seasonality,cost` — the control — beat every real
   feature set, and that is the single most useful result it produced. Keep a
   control in the grid.

**Every column is knowable at `decision_time`.** The per-minute frame is
indexed by `as_of` (bar time plus one minute), so the join is on equality and
there is no shift to forget. `tests/test_features.py` asserts the canary: a
feature set to the realised outcome must be recovered end to end, and the same
feature shifted one minute into the past must not be.
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

# The exchange `us_equity_hours` is about. `zoneinfo` handles DST, which a
# fixed UTC band cannot: 13:30-20:00 UTC is the session in EDT and misses the
# last hour of it in EST.
NEW_YORK = ZoneInfo('America/New_York')

# The canonical column list, per group. `build_features` reindexes to exactly
# this so a saved model always scores against the same matrix — a group that
# produced nothing arrives as an all-NaN column rather than an absent one, which
# is a shape the model can be told about instead of a KeyError at scoring time.
# Column names live with the code that computes them, so a rename cannot
# leave this dict pointing at a column nothing produces.
from core.book_flow import BOOK_FLOW
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
    # `clock` was two different things under one name, and measuring them
    # together made the control look like it carried the model.
    #
    # These ARE the decision offset, and they are legitimately informative:
    # they are why one pooled model can behave like four offset-specific ones,
    # and why dropping `clock` wholesale used to cost the most in ablation
    # despite clock-alone contributing exactly nothing.
    'offset': (
        'elapsed_fraction', 'remaining_minutes',
    ),
    # **The control.** Calendar position cannot forecast direction. Named so a
    # survey cannot quietly omit it — the previous incarnation of this project
    # ran a 27-cell grid whose best cell was its own control, and that was the
    # most useful result it produced. `quarter_of_hour` is which quarter of the
    # HOUR a window sits in, so it is calendar, not offset.
    'time_of_day': (
        'quarter_of_hour', 'hour_sin', 'hour_cos',
        'dow_sin', 'dow_cos', 'us_equity_hours',
    ),
    # --- the venue's own book, from eight months of collection ---------------
    #
    # Declared here rather than inline so `--groups market_state` can run each
    # ALONE. That is the test that matters: a group which forecasts nothing
    # scores `alpha 0.000` and reproduces the baseline exactly, which is how
    # `vol_state` and the `clock` control were both shown to be null despite
    # `clock` carrying a 27.9% LightGBM gain share. Gain share says where splits
    # were spent, not what predicts.
    #
    # `market_minus_baseline` is the most informative column available AND the
    # one that invites echo: given a well-calibrated quote, copying it is the
    # cheapest route to a low log loss, which scores well on `log_loss_skill`
    # and reads ~0 on `model_minus_market`. It stays in its own PRICE_COLUMNS
    # subset so the structure-only variant can be run without it.
    'market_state': BOOK_MARKET_STATE,
    'market_price': BOOK_MARKET_PRICE,
    'cross_venue': BOOK_CROSS_VENUE,
    'implied_vol': BOOK_IMPLIED_VOL,
    # Book DYNAMICS, as against the snapshot `market_state` reads. The one
    # mechanism the archive left open rather than rejected — imbalance went
    # t=+0.30 to t=+1.84 over 67 days, "positive and strengthening". Off by
    # default: it is a candidate under test, not part of the deployed model.
    'book_flow': BOOK_FLOW,
}

# The venue's book only exists from 2026-01-08, against five years of bars, so
# these are SELECTABLE but not DEFAULT. In the default matrix they would be
# all-NaN for ~90% of rows — which `population_report` exists to catch, and
# which would silently widen every feature matrix the project has ever built.
# `--groups market_state` selects them explicitly; that is also how each is
# ablated alone, which is the only test that says whether it forecasts anything.
BOOK_GROUPS = ('market_state', 'market_price', 'cross_venue', 'implied_vol',
               'book_flow')
ALL_GROUPS = tuple(g for g in FEATURE_GROUPS if g not in BOOK_GROUPS)
# The control. Named so a survey cannot quietly omit it.
CONTROL_GROUPS = ('time_of_day',)


def feature_columns(groups: Optional[Sequence[str]] = None) -> list[str]:
    chosen = ALL_GROUPS if groups is None else tuple(groups)
    unknown = sorted(set(chosen) - set(FEATURE_GROUPS))
    if unknown:
        raise ValueError(f'unknown feature groups: {unknown}')
    return [c for g in chosen for c in FEATURE_GROUPS[g]]


# --------------------------------------------------------------------------
# per-minute state
# --------------------------------------------------------------------------

def _zscore(series: pd.Series, lookback: int) -> pd.Series:
    mean = series.rolling(lookback, min_periods=lookback // 4).mean()
    sd = series.rolling(lookback, min_periods=lookback // 4).std()
    return (series - mean) / sd.replace(0.0, np.nan)


def apply_seasonality(state: pd.DataFrame, seasonality: Seasonality) -> pd.DataFrame:
    """Re-derive the three seasonality-dependent columns in place.

    Seasonality is *fitted*, so it belongs inside the cross-validation fold —
    but only three of the forty-two columns depend on it, and all three are a
    minute-of-day lookup. Rebuilding the whole per-minute state six times to
    change three columns would cost hours per run; this costs milliseconds and
    keeps the fit where it has to be. `core/dataset.py` builds the state once
    against a flat factor and calls this per fold.
    """
    # Shallow: only the three assigned columns are rewritten, and pandas'
    # copy-on-write leaves the other forty sharing memory with the original. A
    # deep copy here is a few hundred megabytes per symbol per fold.
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

    `as_of` is the minute *after* the bar, so a row stamped 10:04 is exactly
    what a decision at 10:04 may use. Nothing here shifts at the join site.
    """
    frame = vol_features(grid, seasonality, config)
    returns = frame['r']

    # -- vol_state ---------------------------------------------------------
    frame['rv_slope_short'] = frame['log_rv_15'] - frame['log_rv_60']
    frame['rv_slope_long'] = frame['log_rv_240'] - frame['log_rv_1440']
    frame['range_to_return'] = frame['log_pk_60'] - frame['log_rv_60']
    frame['vol_of_vol'] = frame['log_rv_15'].rolling(240, min_periods=60).std()
    # `rv_surprise` is the short-lookback vol against what the long lookback and
    # the seasonal factor together predict — the part of current volatility that
    # is genuinely news rather than level or time of day.
    predicted = frame['log_rv_1440'] + frame['log_seasonal']
    frame['rv_surprise'] = frame['log_rv_15'] - predicted

    # -- microstructure ----------------------------------------------------
    frame['autocorr_60'] = (
        returns.rolling(60, min_periods=30).corr(returns.shift(1))
    )
    sign = np.sign(returns.fillna(0.0))
    # Length of the current run of same-signed minutes, signed by its direction.
    # groupby-cumcount over run ids is the vectorised form; a loop over five
    # years of minutes is not viable.
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
    # `trade_count_z_15` used to live here and has been removed rather than
    # repaired. Coinbase's candles endpoint returns open/high/low/close/volume and
    # nothing else, so `trade_count` was NULL on 100% of 2,617,876 stored rows and
    # the feature could never fire. It looked alive only because
    # `tests/conftest.py` fabricated the column, so
    # `test_every_declared_feature_is_produced` passed on synthetic data that the
    # real store cannot produce. A declared feature that is structurally
    # unavailable is worse than an absent one: it occupies a slot in the matrix,
    # dilutes every importance share, and reads as a measurement.
    #
    # If a trade count is wanted, it has to come from a source that has one — the
    # Exchange `/products/{id}/candles` endpoint does not carry it either, so that
    # means trades or ticker aggregation, which is a data-collection change and not
    # a feature change.

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
    """Add peer-return and lead-lag columns, in place of a per-symbol loop.

    The reference is Bitcoin: it is the deepest book of the three and the one
    the others follow, so the residual of a symbol's move against Bitcoin's is
    the part of its displacement that is idiosyncratic. For Bitcoin itself the
    reference is the mean of the other two, so the column means the same thing
    in every row — "how far has the rest of the universe moved" — rather than
    being identically zero for one symbol and informative for the others.
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
        # Own 15-minute move minus what the peer's move implies. Positive means
        # this symbol has already run ahead of the lead; negative means it has
        # not caught up yet, which is the tradeable direction of the mechanism.
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
    # Variance-minutes, matching what `sigma_remaining` was scaled by. Reporting
    # wall-clock minutes here while the barrier divides by something else gives
    # the model two inconsistent views of the same clock.
    out['remaining_minutes'] = [
        config.remaining_variance_minutes(o) for o in frame['offset']]
    out['quarter_of_hour'] = window_open.minute // config.window_minutes
    minute_of_day = decision.hour * 60 + decision.minute
    out['hour_sin'] = np.sin(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    out['hour_cos'] = np.cos(2 * np.pi * minute_of_day / MINUTES_PER_DAY)
    out['dow_sin'] = np.sin(2 * np.pi * decision.dayofweek / 7)
    out['dow_cos'] = np.cos(2 * np.pi * decision.dayofweek / 7)
    # The US cash session, in New York local time so that it is the session in
    # both daylight regimes rather than in one.
    #
    # The comment here used to claim 13:30-20:00 UTC "covers the US cash session
    # across both daylight regimes". It does not: that is 09:30-16:00 EDT exactly,
    # and in EST the session is 14:30-21:00 UTC — so for roughly four months a
    # year the flag was on for the hour before the open and off for the last hour
    # of trading. Converting is cheaper than a wide band and removes the
    # judgement call.
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
    # Where in its own realised range the price sits. -1 at the low, +1 at the
    # high. This is the one number `F(x/sigma)` structurally cannot see.
    out['excursion_asymmetry'] = (
        (frame['excursion_up'] + frame['excursion_down']) / span
    )
    out['path_efficiency'] = frame['displacement'].abs() / span
    # How large the move so far is against what its own elapsed minutes would
    # normally produce. Independent of the remaining span, so it is not a
    # restatement of z_score.
    elapsed_sigma = frame['sigma_per_min'] * np.sqrt(frame['offset'].clip(lower=1))
    out['displacement_vs_elapsed'] = frame['displacement'] / elapsed_sigma.replace(0.0, np.nan)
    # Has the window already been on the other side of the strike? A barrier
    # model has no memory of that; a mean-reverting market does.
    out['touched_opposite'] = np.where(
        frame['displacement'] >= 0, (frame['excursion_down'] < 0).astype(float),
        (frame['excursion_up'] > 0).astype(float),
    )
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

    # `peer_displacement` is a window-level quantity, not a per-minute one: it is
    # what the other symbols have done *in this same window*, which is the form
    # the lead-lag mechanism actually takes here.
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

    # --- the book families ---------------------------------------------------
    #
    # Declared in FEATURE_GROUPS and computed in `core.book_features`, but for a
    # while nothing CALLED them: a full run reported "46 features (12 empty),
    # 2 trees, skill +0.00001" because a quarter of the matrix was all-NaN. The
    # model's own empty-feature warning was the only thing that said so, and a
    # group that produced nothing arrives with the same shape as one that
    # worked.
    if 'ask_up' in table.columns or 'bid_at_touch' in table.columns:
        from core.book_features import market_state_features
        # `market_state_features` reads the raw snapshot shape: prices in CENTS
        # under best_bid/best_ask. The joined table carries dollars under
        # ask_up/ask_down. Reconstructing here rather than carrying prices in two
        # units on one row, which is precisely how a 0.51 YES ask ended up in a
        # column holding a 0.51 NO bid.
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
            # Change since the PREVIOUS DECISION OFFSET, within the same window
            # and symbol. The shift is by OFFSET, so it never reaches across a
            # window boundary — consecutive windows chain, and a gap that
            # crossed one would look entirely correct and be wrong.
            #
            # **The `_5` in the name is a misnomer: on the (3, 6, 9, 12) grid
            # `shift(1)` is a THREE-minute step, not five.** Do not "correct"
            # this into a real five-minute lookback. Five minutes is not on the
            # decision grid, `scripts/live.py::gap_change` reproduces this exact
            # one-offset step, and the artifact carries the name — so changing
            # the arithmetic silently fits one feature and scores another, which
            # is the defect `724e9d04` and `68ef2ae9` were already about. Rename
            # only alongside a refit.
            #
            # NaN at the first offset of a window is therefore correct on both
            # sides: it is the first row of each group. Live logs it as an
            # all-NaN column at +3m, which is the feature working.
            ordered = table.sort_values(['symbol', 'window_open', 'offset'])
            prev = ordered.groupby(['symbol', 'window_open'])['venue_prob_gap'].shift(1)
            table['venue_gap_change_5'] = (
                ordered['venue_prob_gap'] - prev).reindex(table.index)

    # `iv_minus_realised` is the implied-vol mechanism and the only book column
    # that cannot be attached with the rest: it divides the market's forward
    # sigma by the baseline's `sigma_per_min`, which is FITTED and therefore
    # only exists here, inside the fold. Computing it earlier would divide by a
    # sigma fitted on the whole sample — a leak that makes the baseline stronger
    # and the model look weaker, the direction nobody audits.
    if 'implied_sigma_per_min' in table.columns:
        implied = pd.to_numeric(table['implied_sigma_per_min'], errors='coerce')
        realised = pd.to_numeric(table.get('sigma_per_min'), errors='coerce')
        ratio = implied / realised.replace(0.0, np.nan)
        table['iv_minus_realised'] = np.where(ratio > 0, np.log(ratio), np.nan)

    return reindex_to_features(table, feature_columns(groups), deferred=deferred)


def reindex_to_features(table: pd.DataFrame, wanted: Sequence[str], *,
                        deferred: Sequence[str] = ()) -> pd.DataFrame:
    """Create every selected column, warning about the ones nobody will fill.

    `deferred` names columns the CALLER attaches after this returns. The live
    path has to: the book is read at the decision instant, and
    `iv_minus_realised` needs the FITTED `sigma_per_min`, which only exists
    inside the fold. Warning about those made every live cycle log nine
    complaints about features populated a moment later — and a real gap then
    looked exactly like the noise.

    The column is still created either way. The feature matrix is built by name
    from the artifact's list, so a missing column and an empty one are different
    failures and only one of them is recoverable.
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
    """How much of each feature is actually populated.

    A feature group that silently produced nothing arrives as an all-NaN column
    with the same shape as a working one, and a column-name hash cannot tell the
    difference. This is the check that says so out loud.
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
