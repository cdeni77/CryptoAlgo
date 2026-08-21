"""Feature construction, grouped by economic mechanism.

Each group answers one question about the market, and each is a pure function of
a symbol's inputs. Nothing here is coin-specific: a feature that made sense only
for one instrument was a naming artifact, not a signal.

    carry          What does it cost to hold, and is that cost trending?
    cross_venue    Is the venue we trade lagging the venue that sets the price?
    volatility     How much movement, and how much of it is jump vs diffusion?
    liquidity      What will it cost to get in and out?
    positioning    Is leverage building or unwinding?
    trend          Direction and persistence over multiple horizons.
    market_factor  How much of this is just beta to BTC?
    seasonality    Hour-of-day and weekday effects.
    cost           Does the expected move clear this instrument's fee hurdle?

Two structural choices matter more than any individual feature:

**Cross-sectional standardisation.** Groups marked `standardize` are converted to
z-scores across the universe at each timestamp, so "strong momentum" means strong
*relative to the other instruments this hour*. That is what lets one pooled model
span sixteen contracts instead of fitting each separately on ~40 independent
events. Absolute quantities — the fee hurdle, the hour of day — are left alone.

**No lookahead.** Every rolling window looks backward only. Funding and open
interest are lagged one bar because they are published after the fact, and the
research store's point-in-time reads (`core.datastore`) are the other half of
that guarantee.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import fee_floor, get_contract_spec

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ('open', 'high', 'low', 'close', 'volume')

# Longest rolling window used anywhere, in bars. Rows before this are dropped:
# a feature built from a partial window is a different feature.
MAX_WARMUP_BARS = 336      # 14 days of hourly bars


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


@dataclass
class SymbolInputs:
    """Everything the feature groups need for one instrument.

    `bars` is the venue we trade. `reference_bars` is a deeper venue quoting the
    same underlying, used for basis and lead-lag — not as a substitute for the
    traded price. `market_bars` is BTC on the traded venue, for beta.
    """

    symbol: str
    bars: pd.DataFrame
    funding: Optional[pd.DataFrame] = None
    open_interest: Optional[pd.DataFrame] = None
    reference_bars: Optional[pd.DataFrame] = None
    market_bars: Optional[pd.DataFrame] = None

    def __post_init__(self) -> None:
        missing = set(OHLCV_COLUMNS) - set(self.bars.columns)
        if missing:
            raise ValueError(f"{self.symbol}: bars missing {sorted(missing)}")
        if not isinstance(self.bars.index, pd.DatetimeIndex):
            raise ValueError(f"{self.symbol}: bars must be indexed by timestamp")
        self.bars = self.bars.sort_index()


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _safe(series: pd.Series) -> pd.Series:
    """Replace zeros with NaN so a division can't explode into infinity."""
    return series.replace(0, np.nan)


def _rolling_z(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Backward-looking z-score. Never uses a future mean."""
    mp = min_periods or max(window // 4, 2)
    mean = series.rolling(window, min_periods=mp).mean()
    std = series.rolling(window, min_periods=mp).std()
    return (series - mean) / _safe(std)


def _align(source: Optional[pd.DataFrame], index: pd.DatetimeIndex, column: str) -> Optional[pd.Series]:
    """Reindex a lower-frequency series onto the bar index and lag it one bar.

    The lag is the point: funding and open interest describe a window that has
    already closed, so using the value stamped at t to predict t would be
    reading the answer.
    """
    if source is None or source.empty or column not in source.columns:
        return None
    series = source[column]
    if not isinstance(series.index, pd.DatetimeIndex):
        return None
    return series.sort_index().reindex(index, method='ffill').shift(1)


def _signed_volume(bars: pd.DataFrame) -> pd.Series:
    """Buy-minus-sell volume estimated from where the bar closed in its range.

    A close at the high implies buyers absorbed the range; at the low, sellers.
    This is the standard OHLCV stand-in for trade-level signing.
    """
    span = _safe(bars['high'] - bars['low'])
    position = (bars['close'] - bars['low']) / span
    return bars['volume'] * (2.0 * position - 1.0)


def _dollar_volume(bars: pd.DataFrame) -> pd.Series:
    return bars['volume'] * bars['close']


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def carry_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Funding cost and its dynamics — the perp-specific edge.

    Coinbase settles funding hourly, so on hourly bars each bar carries exactly
    one settlement. That makes the level directly comparable to the bar's return
    rather than needing to be amortised across an 8-hour window.
    """
    index = inputs.bars.index
    out = pd.DataFrame(index=index)

    rate = _align(inputs.funding, index, 'rate')
    if rate is None:
        return out

    out['carry_bps'] = rate * 10_000
    out['carry_z_168h'] = _rolling_z(rate, 168)

    # Term structure: is the immediate cost above or below its recent run rate?
    mean_8h = rate.rolling(8, min_periods=3).mean()
    mean_24h = rate.rolling(24, min_periods=6).mean()
    out['carry_term_1h_8h'] = (rate - mean_8h) * 10_000
    out['carry_term_8h_24h'] = (mean_8h - mean_24h) * 10_000

    # What a position actually pays over a typical hold.
    out['carry_cum_24h'] = rate.rolling(24, min_periods=6).sum() * 10_000
    out['carry_cum_72h'] = rate.rolling(72, min_periods=18).sum() * 10_000

    # Persistence: crowded positioning shows up as funding that keeps its sign.
    positive = (rate > 0).astype(float)
    out['carry_persistence_24h'] = positive.rolling(24, min_periods=6).mean()
    out['carry_sign_flip_24h'] = positive.diff().abs().rolling(24, min_periods=6).sum()

    # Carry relative to the volatility it is being paid to endure. A high number
    # means you are paying a lot for a market that isn't moving.
    realized = inputs.bars['close'].pct_change().rolling(24, min_periods=6).std()
    out['carry_to_vol'] = (rate.abs() * 24.0) / _safe(realized)
    return out


def cross_venue_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Basis and lead-lag against a deeper venue quoting the same underlying.

    A thinner venue that lags a deeper one is a mechanical edge, and it is the
    reason to keep Binance data around after switching training to Coinbase:
    as a signal, not as a price.
    """
    index = inputs.bars.index
    out = pd.DataFrame(index=index)
    if inputs.reference_bars is None or inputs.reference_bars.empty:
        return out

    ref = inputs.reference_bars['close'].sort_index().reindex(index).ffill()
    own = inputs.bars['close']

    basis = (own / _safe(ref)) - 1.0
    out['basis_bps'] = basis * 10_000
    out['basis_z_168h'] = _rolling_z(basis, 168)
    out['basis_change_1h'] = basis.diff() * 10_000

    own_ret = own.pct_change()
    ref_ret = ref.pct_change()

    # The reference venue's last move, which we may not have followed yet.
    out['ref_return_1h'] = ref_ret
    out['ref_return_4h'] = ref.pct_change(4)

    # Positive lead-lag means the reference leads us: its past return predicts
    # our next one, which is directly tradable.
    out['lead_lag_corr_72h'] = own_ret.rolling(72, min_periods=24).corr(ref_ret.shift(1))
    out['contemp_corr_72h'] = own_ret.rolling(72, min_periods=24).corr(ref_ret)
    return out


def volatility_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Range-based volatility estimators, plus the jump/diffusion split.

    Close-to-close volatility throws away the bar's range. The three estimators
    here use it, and each is robust to something different: Parkinson to nothing
    in particular but efficient, Garman-Klass to the open, Rogers-Satchell to
    drift. Separating jumps matters because a barrier strategy is stopped out by
    jumps, not by diffusion.
    """
    bars = inputs.bars
    out = pd.DataFrame(index=bars.index)

    hl = np.log(_safe(bars['high']) / _safe(bars['low']))
    co = np.log(_safe(bars['close']) / _safe(bars['open']))
    ho = np.log(_safe(bars['high']) / _safe(bars['open']))
    lo = np.log(_safe(bars['low']) / _safe(bars['open']))
    hc = np.log(_safe(bars['high']) / _safe(bars['close']))
    lc = np.log(_safe(bars['low']) / _safe(bars['close']))

    for window in (24, 72):
        mp = max(window // 4, 4)
        out[f'rv_parkinson_{window}h'] = np.sqrt(
            (hl ** 2).rolling(window, min_periods=mp).mean() / (4.0 * np.log(2.0))
        )
        out[f'rv_garman_klass_{window}h'] = np.sqrt(
            (0.5 * hl ** 2 - (2.0 * np.log(2.0) - 1.0) * co ** 2)
            .rolling(window, min_periods=mp).mean().clip(lower=0)
        )
        out[f'rv_rogers_satchell_{window}h'] = np.sqrt(
            (ho * hc + lo * lc).rolling(window, min_periods=mp).mean().clip(lower=0)
        )

    returns = bars['close'].pct_change()

    # Bipower variation estimates the diffusive part; what realised variance has
    # above it is jump. The scaling constant is pi/2.
    realized_var = (returns ** 2).rolling(24, min_periods=6).sum()
    bipower = (np.pi / 2.0) * (returns.abs() * returns.abs().shift(1)).rolling(
        24, min_periods=6
    ).sum()
    out['jump_share_24h'] = ((realized_var - bipower).clip(lower=0) / _safe(realized_var))

    vol_24 = returns.rolling(24, min_periods=6).std()
    vol_168 = returns.rolling(168, min_periods=48).std()
    out['vol_term_24_168'] = vol_24 / _safe(vol_168)
    out['vol_of_vol_168h'] = vol_24.rolling(168, min_periods=48).std() / _safe(vol_24)
    out['vol_z_168h'] = _rolling_z(vol_24, 168, min_periods=48)

    # Downside vs upside dispersion: barrier strategies care which tail is fat.
    up = returns.where(returns > 0, 0.0).rolling(72, min_periods=18).std()
    down = returns.where(returns < 0, 0.0).abs().rolling(72, min_periods=18).std()
    out['vol_asymmetry_72h'] = up / _safe(down)
    return out


def liquidity_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Execution-cost proxies inferred from bars.

    These are what turn a static slippage assumption into a state-dependent one.
    All three spread estimators are standard in the microstructure literature
    precisely because they need only OHLCV.
    """
    bars = inputs.bars
    out = pd.DataFrame(index=bars.index)
    returns = bars['close'].pct_change()
    dollar_volume = _dollar_volume(bars)

    # Amihud: price impact per dollar traded.
    out['amihud_24h'] = (returns.abs() / _safe(dollar_volume)).rolling(
        24, min_periods=6
    ).mean() * 1e9

    # Roll: spread implied by negative autocovariance of returns. Only defined
    # when that covariance is negative, which is the bid-ask bounce signature.
    cov = returns.rolling(48, min_periods=12).cov(returns.shift(1))
    out['roll_spread_48h'] = 2.0 * np.sqrt((-cov).clip(lower=0))

    # Corwin-Schultz: spread from the ratio of single-bar to two-bar ranges.
    k = 3.0 - 2.0 * np.sqrt(2.0)
    log_hl_sq = np.log(_safe(bars['high']) / _safe(bars['low'])) ** 2
    beta = log_hl_sq + log_hl_sq.shift(1)
    two_bar_high = bars['high'].rolling(2).max()
    two_bar_low = bars['low'].rolling(2).min()
    gamma = np.log(_safe(two_bar_high) / _safe(two_bar_low)) ** 2
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    out['corwin_schultz_spread'] = spread.clip(lower=0).rolling(24, min_periods=6).mean()

    # Kyle's lambda: how far price moves per unit of signed flow.
    signed = _signed_volume(bars) * bars['close']
    flow_cov = returns.rolling(72, min_periods=24).cov(signed)
    flow_var = signed.rolling(72, min_periods=24).var()
    out['kyle_lambda_72h'] = (flow_cov / _safe(flow_var)) * 1e9

    out['signed_volume_24h'] = (
        signed.rolling(24, min_periods=6).sum() / _safe(dollar_volume.rolling(24, min_periods=6).sum())
    )
    out['volume_z_168h'] = _rolling_z(bars['volume'], 168, min_periods=48)
    out['dollar_volume_z_168h'] = _rolling_z(dollar_volume, 168, min_periods=48)
    return out


def positioning_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Open interest: is leverage building into the move or unwinding out of it?"""
    index = inputs.bars.index
    out = pd.DataFrame(index=index)

    oi = _align(inputs.open_interest, index, 'oi_contracts')
    if oi is None:
        oi = _align(inputs.open_interest, index, 'oi_usd')
    if oi is None:
        return out

    out['oi_change_1h'] = oi.pct_change()
    out['oi_change_24h'] = oi.pct_change(24)
    out['oi_z_168h'] = _rolling_z(oi, 168, min_periods=48)

    returns_24h = inputs.bars['close'].pct_change(24)
    oi_change_24h = oi.pct_change(24)

    # Price up on rising OI is new money; price up on falling OI is a squeeze
    # closing out. The sign product separates them.
    out['oi_price_divergence_24h'] = np.sign(oi_change_24h) * np.sign(returns_24h)
    out['oi_return_interaction_24h'] = oi_change_24h * returns_24h

    # Cascade signature: OI falling hard while volatility and volume spike.
    vol_24h = inputs.bars['close'].pct_change().rolling(24, min_periods=6).std()
    volume_ratio = inputs.bars['volume'] / _safe(
        inputs.bars['volume'].rolling(168, min_periods=48).mean()
    )
    out['liquidation_cascade_24h'] = (
        (oi_change_24h < -0.05).astype(float)
        + (volume_ratio > 3.0).astype(float)
        + (vol_24h > vol_24h.rolling(168, min_periods=48).mean()).astype(float)
    )
    return out


def trend_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Direction and persistence across horizons."""
    close = inputs.bars['close']
    out = pd.DataFrame(index=inputs.bars.index)

    for horizon in (1, 4, 12, 24, 72, 168):
        out[f'return_{horizon}h'] = close.pct_change(horizon)

    # Efficiency ratio: net move over summed absolute moves. High means a clean
    # trend, low means chop covering the same ground.
    for window in (24, 72):
        mp = max(window // 4, 6)
        net = close.diff(window).abs()
        gross = close.diff().abs().rolling(window, min_periods=mp).sum()
        out[f'efficiency_ratio_{window}h'] = net / _safe(gross)

    for window in (24, 72, 168):
        mp = max(window // 4, 6)
        mean = close.rolling(window, min_periods=mp).mean()
        std = close.rolling(window, min_periods=mp).std()
        out[f'price_z_{window}h'] = (close - mean) / _safe(std)

    # Distance to the recent extremes, which is where barriers get hit.
    for window in (24, 168):
        mp = max(window // 4, 6)
        high = inputs.bars['high'].rolling(window, min_periods=mp).max()
        low = inputs.bars['low'].rolling(window, min_periods=mp).min()
        out[f'range_position_{window}h'] = (close - low) / _safe(high - low)
        out[f'dist_from_high_{window}h'] = (close - high) / _safe(high)

    returns = close.pct_change()
    out['ret_skew_72h'] = returns.rolling(72, min_periods=18).skew()
    out['ret_kurt_72h'] = returns.rolling(72, min_periods=18).kurt()
    out['ret_autocorr_48h'] = returns.rolling(48, min_periods=12).corr(returns.shift(1))
    return out


def market_factor_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Beta to BTC and the return left over after hedging it out.

    Without this the model relearns "the market went up" separately in every
    instrument. Residual momentum is the part that is actually about this coin.
    """
    out = pd.DataFrame(index=inputs.bars.index)
    if inputs.market_bars is None or inputs.market_bars.empty:
        return out

    market = inputs.market_bars['close'].sort_index().reindex(inputs.bars.index).ffill()
    own_ret = inputs.bars['close'].pct_change()
    market_ret = market.pct_change()

    if own_ret.equals(market_ret):        # BTC against itself
        return out

    for window in (24, 72):
        mp = max(window // 4, 6)
        cov = own_ret.rolling(window, min_periods=mp).cov(market_ret)
        var = market_ret.rolling(window, min_periods=mp).var()
        beta = cov / _safe(var)
        out[f'btc_beta_{window}h'] = beta
        out[f'btc_corr_{window}h'] = own_ret.rolling(window, min_periods=mp).corr(market_ret)
        residual = own_ret - beta * market_ret
        out[f'btc_residual_mom_{window}h'] = residual.rolling(window, min_periods=mp).sum()

    for horizon in (4, 24, 72):
        out[f'btc_rel_return_{horizon}h'] = (
            inputs.bars['close'].pct_change(horizon) - market.pct_change(horizon)
        )
    return out


def seasonality_features(inputs: SymbolInputs) -> pd.DataFrame:
    """Cyclical time-of-day and day-of-week.

    Encoded as sine/cosine pairs so hour 23 and hour 0 are adjacent, which an
    integer hour column would hide from a tree split.
    """
    index = inputs.bars.index
    out = pd.DataFrame(index=index)
    hour = index.hour.to_numpy(dtype=float)
    dow = index.dayofweek.to_numpy(dtype=float)
    out['hour_sin'] = np.sin(2 * np.pi * hour / 24.0)
    out['hour_cos'] = np.cos(2 * np.pi * hour / 24.0)
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7.0)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7.0)
    out['is_weekend'] = (dow >= 5).astype(float)
    return out


def cost_features(inputs: SymbolInputs, config: Optional[Config] = None) -> pd.DataFrame:
    """Whether the expected move clears this instrument's fee hurdle.

    This is the group that makes the cost model visible to the model itself.
    Because Coinbase charges per contract rather than per dollar, the hurdle is
    size-dependent: one BTC contract is ~$600 of notional carrying the same
    $0.75 commission as a $1,750 DOGE contract. So both the marginal
    (one-contract) and a reference-notional hurdle are exposed — the gap between
    them is exactly the small-account penalty.
    """
    config = config or Config()
    bars = inputs.bars
    out = pd.DataFrame(index=bars.index)

    spec = get_contract_spec(inputs.symbol)
    close = _safe(bars['close'])
    notional_per_contract = spec.units * close

    pct_side = float(config.fee_pct_per_side)
    floor = fee_floor(inputs.symbol, config)
    slippage_round_trip = 2.0 * float(config.slippage_bps) / 10_000.0

    # A per-contract commission is a fixed fraction of notional, because notional
    # per contract is fixed. So the hurdle does not depend on position size, and
    # there is no large-order discount to model. What size does change is
    # granularity: one contract is the smallest position available, and
    # `contract_notional_usd` is how coarse that is.
    fee_per_side = np.maximum(pct_side, floor / _safe(notional_per_contract))
    hurdle = 2.0 * fee_per_side + slippage_round_trip

    # The hurdle level is kept because it is the economically meaningful
    # threshold a move has to clear, and it moves with price. Notional per
    # contract is deliberately *not* a feature: it is near-constant per
    # instrument, so a tree splits on it to recover instrument identity with a
    # continuous variable — more expressive, and more overfittable, than the
    # symbol category the pooled model already has. The model does not size
    # positions either; `core.execution.size_from_forecast` does.
    out['fee_hurdle_bps'] = hurdle * 10_000

    vol_24h = bars['close'].pct_change().rolling(24, min_periods=6).std()
    expected_move = vol_24h * 1.5
    out['expected_move_bps'] = expected_move * 10_000
    out['edge_over_hurdle'] = expected_move / _safe(hurdle)
    out['hurdle_to_vol'] = hurdle / _safe(vol_24h)
    return out


# ---------------------------------------------------------------------------
# Group registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureGroup:
    """A named block of features.

    `standardize` says whether the values are only meaningful relative to the
    rest of the universe. Momentum is: being up 3% matters relative to what
    everything else did. A fee hurdle is not: 25 basis points is 25 basis
    points regardless of what other contracts cost.
    """

    name: str
    fn: Callable[..., pd.DataFrame]
    standardize: bool
    needs_config: bool = False


GROUPS: tuple[FeatureGroup, ...] = (
    FeatureGroup('carry', carry_features, standardize=True),
    FeatureGroup('cross_venue', cross_venue_features, standardize=True),
    FeatureGroup('volatility', volatility_features, standardize=True),
    FeatureGroup('liquidity', liquidity_features, standardize=True),
    FeatureGroup('positioning', positioning_features, standardize=True),
    FeatureGroup('trend', trend_features, standardize=True),
    FeatureGroup('market_factor', market_factor_features, standardize=True),
    FeatureGroup('seasonality', seasonality_features, standardize=False),
    FeatureGroup('cost', cost_features, standardize=False, needs_config=True),
)

GROUPS_BY_NAME = {group.name: group for group in GROUPS}


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def build_symbol_features(
    inputs: SymbolInputs,
    *,
    config: Optional[Config] = None,
    groups: Optional[Sequence[str]] = None,
    trim_warmup: bool = True,
) -> pd.DataFrame:
    """Wide feature frame for one instrument."""
    selected = [GROUPS_BY_NAME[name] for name in groups] if groups else list(GROUPS)
    blocks: list[pd.DataFrame] = []

    for group in selected:
        block = group.fn(inputs, config=config) if group.needs_config else group.fn(inputs)
        if block is None or block.empty:
            continue
        blocks.append(block.add_prefix(''))

    if not blocks:
        return pd.DataFrame(index=inputs.bars.index)

    frame = pd.concat(blocks, axis=1)
    frame = frame.loc[:, ~frame.columns.duplicated()]
    frame = frame.replace([np.inf, -np.inf], np.nan)

    if trim_warmup and len(frame) > MAX_WARMUP_BARS:
        frame = frame.iloc[MAX_WARMUP_BARS:]
    return frame.astype('float64')


def build_panel(
    inputs: Iterable[SymbolInputs],
    *,
    config: Optional[Config] = None,
    groups: Optional[Sequence[str]] = None,
    standardize: bool = True,
    min_universe: int = 3,
) -> pd.DataFrame:
    """Feature panel for the whole universe, MultiIndexed by (time, symbol).

    This is the shape the pooled model consumes: one row per instrument per
    timestamp, so a cross-sectional operation is a groupby on the time level.
    """
    frames: dict[str, pd.DataFrame] = {}
    for item in inputs:
        frame = build_symbol_features(item, config=config, groups=groups)
        if not frame.empty:
            frames[item.symbol] = frame

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, names=['symbol', 'event_time'])
    panel = panel.reorder_levels(['event_time', 'symbol']).sort_index()

    # Force the canonical column order. Without this, the panel's schema depends
    # on which symbol happened to be built first: a group can be legitimately
    # empty for one instrument — BTC has no beta to BTC — and `concat` unions
    # columns in first-seen order, so the same universe in a different order
    # produces a differently-shaped matrix. A model saved against one ordering
    # would then score against another.
    canonical = feature_columns(groups)
    panel = panel.reindex(columns=canonical)

    if standardize:
        panel = cross_sectional_standardize(
            panel, groups=groups, min_universe=min_universe
        )
    return panel


def standardizable_columns(
    panel_columns: Iterable[str],
    groups: Optional[Sequence[str]] = None,
) -> list[str]:
    """Columns belonging to groups whose values are only meaningful relatively."""
    selected = [GROUPS_BY_NAME[name] for name in groups] if groups else list(GROUPS)
    absolute: set[str] = set()
    for group in selected:
        if group.standardize:
            continue
        # Re-derive the group's column names from a throwaway build.
        absolute |= set(_group_column_names(group))
    return [c for c in panel_columns if c not in absolute]


def _stub_inputs() -> SymbolInputs:
    """Minimal inputs that exercise every group, for column discovery.

    Every optional input is supplied: a group given no funding, no open interest
    or no reference venue returns an empty frame, and its columns would go
    undiscovered.
    """
    index = pd.date_range('2026-01-01', periods=MAX_WARMUP_BARS + 8, freq='1h', tz='UTC')
    price = pd.Series(np.linspace(100.0, 110.0, len(index)), index=index)
    bars = pd.DataFrame(
        {
            'open': price, 'high': price * 1.002, 'low': price * 0.998,
            'close': price, 'volume': 1_000.0,
        },
        index=index,
    )
    market = bars.copy()
    market['close'] = price * 1.01        # distinct from `bars`, so beta is defined
    return SymbolInputs(
        symbol='BIP',
        bars=bars,
        funding=pd.DataFrame({'rate': 0.00001}, index=index),
        open_interest=pd.DataFrame({'oi_contracts': 1_000.0}, index=index),
        reference_bars=bars.assign(close=price * 1.001),
        market_bars=market,
    )


def _group_column_names(group: FeatureGroup) -> list[str]:
    """Column names a group emits, discovered by running it on stub inputs."""
    stub = _stub_inputs()
    block = group.fn(stub, config=Config()) if group.needs_config else group.fn(stub)
    return list(block.columns) if block is not None else []


# A cross-section whose spread is this small relative to its own magnitude
# carries no ranking information; dividing by it turns float noise into large
# spurious z-scores.
DEGENERATE_SPREAD_RATIO = 1e-8

# Even a valid cross-section can produce extreme values on a thin universe.
# Clipping keeps one outlier from dominating a tree split.
MAX_ABS_ZSCORE = 8.0


def cross_sectional_standardize(
    panel: pd.DataFrame,
    *,
    groups: Optional[Sequence[str]] = None,
    min_universe: int = 3,
) -> pd.DataFrame:
    """Convert relative features to z-scores across the universe at each bar.

    Three guards, each for a way this goes wrong on real panels:

    * Timestamps with fewer than `min_universe` instruments reporting are left
      NaN rather than standardised against one or two peers.
    * A degenerate cross-section — every instrument reporting nearly the same
      value — is left NaN. Its spread is float noise, and dividing by it would
      manufacture large z-scores out of nothing.
    * Surviving values are clipped, so a thin universe cannot hand the model a
      single dominating outlier.
    """
    if panel.empty:
        return panel

    columns = standardizable_columns(panel.columns, groups)
    if not columns:
        return panel

    out = panel.copy()
    block = out[columns]
    grouped = block.groupby(level='event_time')

    mean = grouped.transform('mean')
    std = grouped.transform('std')
    count = grouped.transform('count')
    scale = block.abs().groupby(level='event_time').transform('mean')

    informative = std > (DEGENERATE_SPREAD_RATIO * (scale + 1.0))
    standardized = (block - mean) / std.where(informative)
    standardized = standardized.clip(-MAX_ABS_ZSCORE, MAX_ABS_ZSCORE)

    out[columns] = standardized.where(count >= min_universe)
    return out.replace([np.inf, -np.inf], np.nan)


def feature_columns(groups: Optional[Sequence[str]] = None) -> list[str]:
    """Every column the builder can emit, in build order."""
    selected = [GROUPS_BY_NAME[name] for name in groups] if groups else list(GROUPS)
    names: list[str] = []
    for group in selected:
        for column in _group_column_names(group):
            if column not in names:
                names.append(column)
    return names
