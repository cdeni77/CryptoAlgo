"""Prediction targets: net return over a horizon, decomposed.

This replaces triple-barrier classification, and the reason is worth stating
because the old formulation is what the previous system was built on.

A binary "did take-profit come before stop-loss" target has three problems that
no amount of hygiene fixes. It discards magnitude, so a trade that wins by five
times the barrier and one that scrapes it are the same observation. It requires
the direction to be decided *before* labelling — the old pipeline used a
three-vote momentum rule, which discarded a third of all bars and fixed the side
on the rest, leaving the model only the question "accept this call or not". And
it cannot express carry at all: funding is a cash flow you collect for holding a
position, not a direction price will move.

That last point matters most on Coinbase perps, where funding settles hourly. At
2bp an hour a position collects roughly 48bp a day, against a round-trip cost of
about 5bp on the group-B contracts. The carry can clear its own cost in under
three hours. The question is not "which way will price go" but "does what I
collect exceed the risk I take to collect it" — and that is a magnitude question
about two separable components.

So the targets here are:

    price       simple return over the horizon, direction-free
    carry       funding a long position accrues over the horizon, direction-free
    cost        round-trip execution cost, known at decision time

and the tradeable quantities are built from them:

    net_long  =  price + carry - cost
    net_short = -price - carry - cost

Note that `net_long + net_short = -2 * cost`. At most one side can be positive,
and the cost is what both sides must clear — the hurdle is structural rather than
a threshold someone tuned.

Keeping price and carry apart is the diagnostic. Carry is published and
persistent, so it is genuinely predictable; price at an hourly horizon is barely
predictable at all. Knowing which component a strategy's edge comes from is the
difference between a result you can trust and a number you hope holds.
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

# Column names of the target frame. `net_long`/`net_short` are derived, kept
# alongside the components so a report can show where the return came from.
TARGET_COLUMNS = ('price', 'carry', 'cost', 'net_long', 'net_short', 'best_side', 'best_net')


@dataclass(frozen=True)
class TargetSpec:
    """How far ahead to look, and what it costs to get there."""

    horizon_bars: int
    round_trip_cost: float = 0.0

    def __post_init__(self) -> None:
        if self.horizon_bars < 1:
            raise ValueError(f'horizon_bars must be >= 1, got {self.horizon_bars}')

    @property
    def cost_bps(self) -> float:
        return self.round_trip_cost * 10_000


def round_trip_cost(symbol: str, price: float, config: Config, *, contracts: int = 1) -> float:
    """Round-trip execution cost as a fraction of notional.

    Size-invariant: a per-contract commission is a fixed fraction of notional,
    because notional per contract is fixed.
    """
    spec = get_contract_spec(symbol)
    notional = spec.notional(contracts, price)
    if notional <= 0:
        return 0.0
    fee_per_side = max(config.fee_pct_per_side, fee_floor(symbol, config) * contracts / notional)
    slippage = config.slippage_bps / 10_000.0 if config.apply_slippage else 0.0
    return 2.0 * (fee_per_side + slippage)


def target_spec_for(
    symbol: str,
    *,
    profile: Optional[CoinProfile] = None,
    config: Optional[Config] = None,
    reference_price: float | None = None,
) -> TargetSpec:
    """Horizon from the profile's hold period, cost from the venue's schedule."""
    config = config or Config()
    price = reference_price if reference_price and reference_price > 0 else 1.0
    return TargetSpec(
        horizon_bars=config.label_horizon_hours(profile),
        round_trip_cost=round_trip_cost(symbol, price, config),
    )


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def price_return(close: pd.Series, horizon_bars: int) -> pd.Series:
    """Simple return over the horizon, from t to t+h.

    Simple rather than log so that subtracting a cost — itself a fraction of
    notional — is exact rather than approximate.
    """
    forward = close.shift(-int(horizon_bars))
    return (forward / close.replace(0, np.nan)) - 1.0


def carry_return(
    funding: Optional[pd.Series],
    index: pd.DatetimeIndex,
    horizon_bars: int,
) -> pd.Series:
    """Funding a *long* position accrues over (t, t+h].

    Sign convention: a positive funding rate means longs pay shorts, so a long's
    carry is the negative of the summed rate. A short's carry is its negation,
    which is why only one series is needed.

    This is a forward sum and therefore not known at t. It is also the most
    predictable thing in the whole system — funding is published and strongly
    persistent — which is precisely why it is worth predicting separately from
    price.
    """
    if funding is None or funding.empty:
        return pd.Series(0.0, index=index)

    rates = funding.sort_index().reindex(index).ffill().fillna(0.0)
    horizon = int(horizon_bars)

    # Forward-looking sum over (t, t+h]: reverse, roll, reverse back.
    forward_sum = rates.iloc[::-1].rolling(horizon, min_periods=horizon).sum().iloc[::-1]
    forward_sum = forward_sum.shift(-1)          # exclude the bar itself
    return -forward_sum


def build_targets(
    bars: pd.DataFrame,
    spec: TargetSpec,
    *,
    funding: Optional[pd.DataFrame] = None,
    index: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Target frame for one instrument.

    Rows without a full forward window are NaN throughout: there is no outcome
    to record, and imputing one would put fabricated observations at exactly the
    recent edge of the sample.
    """
    frame = bars.sort_index()
    target_index = pd.DatetimeIndex(index) if index is not None else frame.index

    rates = None
    if funding is not None and not funding.empty and 'rate' in funding.columns:
        rates = funding['rate']

    price = price_return(frame['close'], spec.horizon_bars)
    carry = carry_return(rates, frame.index, spec.horizon_bars)

    out = pd.DataFrame(index=frame.index)
    out['price'] = price
    out['carry'] = carry
    out['cost'] = spec.round_trip_cost
    out['net_long'] = out['price'] + out['carry'] - out['cost']
    out['net_short'] = -out['price'] - out['carry'] - out['cost']

    # The side worth taking, and what it returns. Zero when neither clears cost.
    long_better = out['net_long'] >= out['net_short']
    out['best_side'] = np.where(long_better, 1.0, -1.0)
    out['best_net'] = np.where(long_better, out['net_long'], out['net_short'])
    stand_aside = out['best_net'] <= 0
    out.loc[stand_aside, 'best_side'] = 0.0

    # Unresolvable rows carry nothing.
    resolvable = out['price'].notna()
    out.loc[~resolvable, :] = np.nan

    return out.reindex(target_index)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def build_target_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    *,
    profiles: Optional[dict[str, CoinProfile]] = None,
    funding_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    index_by_symbol: Optional[dict[str, pd.DatetimeIndex]] = None,
    horizon_bars: Optional[int] = None,
) -> pd.DataFrame:
    """Targets for the universe, MultiIndexed by (event_time, symbol).

    Matches `core.features.build_panel`, so features and targets join on the
    index rather than by position.
    """
    config = config or Config()
    profiles = profiles or {}
    funding_by_symbol = funding_by_symbol or {}
    pieces: dict[str, pd.DataFrame] = {}

    for symbol, bars in bars_by_symbol.items():
        if bars.empty:
            continue
        spec = target_spec_for(
            symbol,
            profile=profiles.get(symbol),
            config=config,
            reference_price=float(bars['close'].iloc[-1]),
        )
        if horizon_bars is not None:
            spec = TargetSpec(horizon_bars=horizon_bars, round_trip_cost=spec.round_trip_cost)

        targets = build_targets(
            bars, spec,
            funding=funding_by_symbol.get(symbol),
            index=(index_by_symbol or {}).get(symbol),
        )
        if targets['price'].notna().any():
            pieces[symbol] = targets

    if not pieces:
        return pd.DataFrame(columns=list(TARGET_COLUMNS))

    panel = pd.concat(pieces, names=['symbol', 'event_time'])
    return panel.reorder_levels(['event_time', 'symbol']).sort_index()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetSummary:
    """What the targets look like, and how much of them is carry.

    `carry_share` is the headline. If it is high, the strategy is a carry
    harvester and should be evaluated as one; if it is near zero, the system is
    making a directional bet and the honest expectation is much lower.

    `hindsight_profitable` is deliberately named. It counts rows where the
    realised better side beat the cost — which is almost all of them, because a
    96-hour price move usually exceeds a few basis points in magnitude. It says
    nothing about opportunity. Knowing the sign in advance is the hard part, and
    that is what the model is measured on.
    """

    rows: int
    resolved: int
    hindsight_profitable: int
    mean_price_bps: float
    mean_carry_bps: float
    mean_cost_bps: float
    carry_share: float
    long_fraction: float

    def __str__(self) -> str:
        return (
            f"{self.resolved} resolved rows | "
            f"price {self.mean_price_bps:+.1f}bp, carry {self.mean_carry_bps:+.1f}bp, "
            f"cost {self.mean_cost_bps:.1f}bp | carry share {self.carry_share:.1%} | "
            f"{self.hindsight_profitable / self.resolved:.1%} beat cost in hindsight"
        )


def summarise_targets(targets: pd.DataFrame) -> TargetSummary:
    """Magnitudes and the carry/price split."""
    resolved = targets.dropna(subset=['price'])
    if resolved.empty:
        return TargetSummary(len(targets), 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    price = resolved['price'].abs().mean()
    carry = resolved['carry'].abs().mean()
    denominator = price + carry
    beat_cost = resolved[resolved['best_side'] != 0]

    return TargetSummary(
        rows=int(len(targets)),
        resolved=int(len(resolved)),
        hindsight_profitable=int(len(beat_cost)),
        mean_price_bps=float(resolved['price'].mean() * 10_000),
        mean_carry_bps=float(resolved['carry'].mean() * 10_000),
        mean_cost_bps=float(resolved['cost'].mean() * 10_000),
        carry_share=float(carry / denominator) if denominator > 0 else 0.0,
        long_fraction=float((beat_cost['best_side'] > 0).mean()) if len(beat_cost) else 0.0,
    )
