"""Triple-barrier labelling with cost-aware take-profit.

A label answers "if I had entered here, would the trade have won?" — which means
it depends on the exit rules and the execution costs, not just on future price.
Two consequences drive this module's design:

**The take-profit barrier includes the round-trip cost.** A move that clears the
barrier but not the fees is a loss, and labelling it a win teaches the model to
find trades that lose money. Coinbase charges per contract, so the cost differs
by an order of magnitude across the universe — roughly 1bp on DOGE against 50bp
on ETH — and it therefore has to be resolved per symbol rather than passed as one
number. `barrier_spec_for` does that from the loaded fee schedule.

**There is one implementation.** The previous module had two — one iterating the
OHLCV frame, one indexed by the feature frame — plus an
`assert_label_path_consistency` helper whose existence was an admission that they
could disagree. Two implementations of the same decision is the defect, not the
absence of a test for it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import fee_floor, get_contract_spec
from core.profiles import CoinProfile

logger = logging.getLogger(__name__)

# Label encoding. Neutral bars — no directional consensus — are NaN and dropped
# before training rather than encoded as a class: "no opinion" is not an outcome.
WIN = 1.0
LOSS = 0.0

# Volatility estimate for barrier width. 48 bars is steadier than 24 without
# lagging a regime change badly, and it is shifted one bar so the barrier at t
# is set from information available at t.
VOL_LOOKBACK_BARS = 48


@dataclass(frozen=True)
class BarrierSpec:
    """Barrier geometry for one instrument.

    `round_trip_cost` is a fraction of notional covering both sides — fees plus
    slippage. It widens the take-profit barrier so a label can only be a win if
    the move covered the cost of taking it.
    """

    horizon_bars: int
    tp_mult: float
    sl_mult: float
    round_trip_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError(f'horizon_bars must be >= 1, got {self.horizon_bars}')
        if self.tp_mult <= 0 or self.sl_mult <= 0:
            raise ValueError('tp_mult and sl_mult must be positive')

    @property
    def cost_bps(self) -> float:
        return self.round_trip_cost * 10_000


def round_trip_cost(
    symbol: str,
    price: float,
    config: Config,
    *,
    contracts: int = 1,
) -> float:
    """Round-trip execution cost as a fraction of notional.

    Because the commission is per contract and notional per contract is fixed,
    this is size-invariant — one contract and a hundred cost the same fraction.
    """
    spec = get_contract_spec(symbol)
    notional = spec.notional(contracts, price)
    if notional <= 0:
        return 0.0

    floor = fee_floor(symbol, config) * contracts
    fee_per_side = max(config.fee_pct_per_side, floor / notional)
    slippage = config.slippage_bps / 10_000.0 if config.apply_slippage else 0.0
    return 2.0 * (fee_per_side + slippage)


def barrier_spec_for(
    symbol: str,
    *,
    profile: Optional[CoinProfile] = None,
    config: Optional[Config] = None,
    reference_price: float | None = None,
) -> BarrierSpec:
    """Build a spec from the profile's exits and the venue's real fee schedule.

    The horizon is the execution hold, not a separate label horizon: a label must
    span at least as long as a position can stay open, or the model is trained on
    an outcome the backtest never waits for.
    """
    config = config or Config()
    price = reference_price if reference_price and reference_price > 0 else 1.0

    return BarrierSpec(
        horizon_bars=config.label_horizon_hours(profile),
        tp_mult=config.resolve('vol_mult_tp', profile),
        sl_mult=config.resolve('vol_mult_sl', profile),
        round_trip_cost=round_trip_cost(symbol, price, config),
    )


def momentum_direction(
    ohlcv: pd.DataFrame,
    *,
    score_threshold: int = 2,
) -> pd.Series:
    """Directional intent per bar: 1 long, -1 short, 0 no consensus.

    Three components vote — 24h return, 72h return, and price against its 50-bar
    mean. `score_threshold` of 2 needs all three to agree (the score is odd, so
    2 rounds up to 3); 1 accepts two of three, which produces more labelled bars
    in a sideways market at the cost of a weaker directional prior.
    """
    close = ohlcv['close']
    votes = (
        np.where(close.pct_change(24) > 0, 1, -1)
        + np.where(close.pct_change(72) > 0, 1, -1)
        + np.where(close > close.rolling(50).mean(), 1, -1)
    )
    direction = pd.Series(0.0, index=ohlcv.index, dtype=float)
    direction[votes >= score_threshold] = 1.0
    direction[votes <= -score_threshold] = -1.0
    return direction


def barrier_prices(
    entry: float,
    volatility: float,
    direction: int,
    spec: BarrierSpec,
) -> tuple[float, float]:
    """Take-profit and stop-loss prices for one entry.

    The cost widens the take-profit only. A stop at `sl_mult` volatilities is a
    gross move whose net loss is larger once fees are paid; pushing the stop out
    to compensate would be labelling losses as survivable. Erring this way makes
    labels harder to win, which is the safe direction.
    """
    tp_move = spec.tp_mult * volatility + spec.round_trip_cost
    sl_move = spec.sl_mult * volatility

    if direction == 1:
        return entry * (1.0 + tp_move), entry * (1.0 - sl_move)
    return entry * (1.0 - tp_move), entry * (1.0 + sl_move)


def _first_touch(
    highs: np.ndarray,
    lows: np.ndarray,
    direction: int,
    tp_price: float,
    sl_price: float,
) -> float:
    """Which barrier the path reached first, over one forward window.

    Ties — both barriers touched within the same bar — resolve to a loss. Bar
    data cannot say which came first, and assuming the profitable order is how a
    backtest quietly inflates its win rate.
    """
    if direction == 1:
        tp_hits = highs >= tp_price
        sl_hits = lows <= sl_price
    else:
        tp_hits = lows <= tp_price
        sl_hits = highs >= sl_price

    tp_at = int(np.argmax(tp_hits)) if tp_hits.any() else -1
    sl_at = int(np.argmax(sl_hits)) if sl_hits.any() else -1

    if tp_at == -1 and sl_at == -1:
        return LOSS          # timed out without resolving: not a win
    if sl_at == -1:
        return WIN
    if tp_at == -1:
        return LOSS
    return WIN if tp_at < sl_at else LOSS


def triple_barrier_labels(
    ohlcv: pd.DataFrame,
    spec: BarrierSpec,
    *,
    direction: Optional[pd.Series] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.Series:
    """Binary labels over `index` (default: every bar).

    NaN means unlabelled, for one of three reasons, all of which must stay out of
    training rather than defaulting to a class: no directional consensus, no
    usable volatility estimate, or not enough forward bars to resolve the
    barriers.
    """
    frame = ohlcv.sort_index()
    target_index = pd.DatetimeIndex(index) if index is not None else frame.index
    labels = pd.Series(np.nan, index=target_index, dtype=float)

    if direction is None:
        direction = momentum_direction(frame)

    volatility = (
        frame['close'].pct_change()
        .rolling(VOL_LOOKBACK_BARS).std()
        .shift(1)          # the barrier at t uses volatility known before t
        .ffill()
    )

    closes = frame['close'].to_numpy(dtype=float)
    highs = frame['high'].to_numpy(dtype=float)
    lows = frame['low'].to_numpy(dtype=float)
    vols = volatility.to_numpy(dtype=float)
    sides = direction.reindex(frame.index).fillna(0.0).to_numpy(dtype=float)

    positions = {timestamp: i for i, timestamp in enumerate(frame.index)}
    n = len(frame)
    horizon = spec.horizon_bars

    for timestamp in target_index:
        i = positions.get(timestamp)
        if i is None or i + horizon >= n:
            continue

        side = int(sides[i])
        volatility_i = vols[i]
        if side == 0 or not np.isfinite(volatility_i) or volatility_i <= 0:
            continue

        tp_price, sl_price = barrier_prices(closes[i], volatility_i, side, spec)
        window = slice(i + 1, i + 1 + horizon)
        labels.loc[timestamp] = _first_touch(
            highs[window], lows[window], side, tp_price, sl_price
        )

    return labels


@dataclass(frozen=True)
class LabelSummary:
    """What the labelling produced, and why rows were dropped."""

    total_bars: int
    labelled: int
    wins: int
    losses: int
    unlabelled: int

    @property
    def win_rate(self) -> float:
        return self.wins / self.labelled if self.labelled else 0.0

    @property
    def labelled_fraction(self) -> float:
        return self.labelled / self.total_bars if self.total_bars else 0.0

    def __str__(self) -> str:
        return (
            f"{self.labelled}/{self.total_bars} labelled "
            f"({self.labelled_fraction:.1%}), win rate {self.win_rate:.1%}"
        )


def summarise_labels(labels: pd.Series) -> LabelSummary:
    """Class balance and coverage.

    A win rate far from 50% usually means the barriers are asymmetric rather than
    that an edge was found, and a low labelled fraction means the direction
    filter is rejecting most bars — both worth seeing before training on them.
    """
    resolved = labels.dropna()
    wins = int((resolved == WIN).sum())
    return LabelSummary(
        total_bars=int(len(labels)),
        labelled=int(len(resolved)),
        wins=wins,
        losses=int(len(resolved) - wins),
        unlabelled=int(len(labels) - len(resolved)),
    )


def label_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    profiles: dict[str, CoinProfile],
    config: Optional[Config] = None,
    index_by_symbol: Optional[dict[str, pd.DatetimeIndex]] = None,
) -> pd.Series:
    """Labels for the whole universe, MultiIndexed by (event_time, symbol).

    Matches the shape of `core.features.build_panel`, so the pooled model can
    join features to labels on the index rather than by position.
    """
    config = config or Config()
    pieces: dict[str, pd.Series] = {}

    for symbol, bars in bars_by_symbol.items():
        profile = profiles.get(symbol)
        spec = barrier_spec_for(
            symbol,
            profile=profile,
            config=config,
            reference_price=float(bars['close'].iloc[-1]) if len(bars) else None,
        )
        direction = momentum_direction(
            bars, score_threshold=int(config.resolve('direction_score_threshold', profile))
            if profile is not None else 2,
        )
        labels = triple_barrier_labels(
            bars, spec, direction=direction,
            index=(index_by_symbol or {}).get(symbol),
        )
        if labels.notna().any():
            pieces[symbol] = labels

    if not pieces:
        return pd.Series(dtype=float)

    panel = pd.concat(pieces, names=['symbol', 'event_time'])
    return panel.reorder_levels(['event_time', 'symbol']).sort_index()
