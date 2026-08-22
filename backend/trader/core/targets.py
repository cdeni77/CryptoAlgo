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
from core.costs import per_contract_fee, get_contract_spec
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
    # Additive, not a max(): the venue bills a percentage of notional *and* a
    # per-contract commission. See `core.costs.per_contract_fee`.
    fee_per_side = config.fee_pct_per_side + per_contract_fee(symbol, config) * contracts / notional
    slippage = config.slippage_bps / 10_000.0 if config.apply_slippage else 0.0
    return 2.0 * (fee_per_side + slippage)


def round_trip_cost_series(
    symbol: str, close: pd.Series, config: Config, *, contracts: int = 1
) -> pd.Series:
    """Round-trip cost as a fraction of notional, per bar.

    A per-contract commission is a fixed number of dollars, so as a fraction of
    notional it moves inversely with price — for BTC ranging 30k to 100k that is
    a 3.3x swing in the cost of the same trade. Pricing every row off one
    reference price got this wrong twice: the cost was constant when it should
    vary, and the reference used was the *last* close in the loaded history, which
    put end-of-sample information into every training row's target and into the
    hurdle `decide()` compares against. `features.cost_features` was already
    per-bar, so the feature and the target disagreed.
    """
    spec = get_contract_spec(symbol)
    price = pd.to_numeric(close, errors='coerce')
    notional = price * spec.units * float(contracts)

    pct_fee = float(config.fee_pct_per_side)
    commission_dollars = per_contract_fee(symbol, config) * float(contracts)
    with np.errstate(divide='ignore', invalid='ignore'):
        commission_fraction = np.where(notional > 0, commission_dollars / notional, np.nan)

    fee_per_side = pct_fee + commission_fraction
    slippage = config.slippage_bps / 10_000.0 if config.apply_slippage else 0.0
    return pd.Series(2.0 * (fee_per_side + slippage), index=price.index)


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


def price_return(
    bars: pd.DataFrame | pd.Series,
    horizon_bars: int,
    *,
    entry: str = 'next_open',
) -> pd.Series:
    """Simple return over the horizon, from the first price a decision can fill at.

    Simple rather than log so that subtracting a cost — itself a fraction of
    notional — is exact rather than approximate.

    `entry='next_open'` measures `open(t+1+h) / open(t+1) - 1`. That is the
    honest target: a bar's `available_time` is the moment it closes, so a decision
    using bar `t` cannot be made until `t+1` has begun, and the earliest price it
    can transact at is `open(t+1)`.

    `entry='close'` measures `close(t+h) / close(t) - 1`, which is what this
    function used to do unconditionally. On a liquid instrument the two agree,
    because `close(t)` and `open(t+1)` are the same moment. On a thin one they do
    not, and the difference is not small:

    * `close(t)` is the *last trade* in bar `t`, which on a nano perp can be
      twenty minutes before the bar ends while spot keeps moving.
    * Measured on this repo's 399-day store, across 14 contracts and three
      walk-forward quarters, `basis_z_168h` scored IC **-0.50** against the
      `close(t) -> open(t+1)` gap alone. The gap *is* the perp's stale print
      catching up.
    * So a model trained on the close-to-close target learns to forecast that
      catch-up. Its IC looks strong and none of it is reachable: the same
      cross_venue+trend model scored **+0.114** close-to-close and **+0.002**
      open-to-open at a 1h horizon. Ninety-eight percent of the apparent edge was
      a price that no longer existed.

    That gap is why every backtest in this repo lost money while the reported IC
    looked healthy. The simulation always entered at the next open — correctly —
    and the target it was scored against never did.

    Keep `entry='close'` only for reproducing an old artifact. Nothing should
    train on it.
    """
    horizon = int(horizon_bars)
    if isinstance(bars, pd.Series):
        # Legacy call shape: a bare close series can only support the close mode.
        if entry != 'close':
            raise TypeError(
                "price_return needs the bars frame (for 'open') unless "
                "entry='close'; pass the DataFrame"
            )
        close = bars
        return (close.shift(-horizon) / close.replace(0, np.nan)) - 1.0

    if entry == 'close':
        close = bars['close']
        return (close.shift(-horizon) / close.replace(0, np.nan)) - 1.0

    if entry != 'next_open':
        raise ValueError(f"entry must be 'next_open' or 'close', got {entry!r}")

    opens = bars['open']
    fill = opens.shift(-1)
    exit_price = opens.shift(-(1 + horizon))
    return (exit_price / fill.replace(0, np.nan)) - 1.0


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
    cost: Optional[pd.Series] = None,
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

    price = price_return(frame, spec.horizon_bars)
    carry = carry_return(rates, frame.index, spec.horizon_bars)

    out = pd.DataFrame(index=frame.index)
    out['price'] = price
    out['carry'] = carry
    # Per bar when supplied, so the cost of a trade reflects the price at the
    # bar it would open at rather than one reference price for the whole sample.
    out['cost'] = (
        pd.to_numeric(cost, errors='coerce').reindex(frame.index)
        if cost is not None else spec.round_trip_cost
    )
    out['net_long'] = out['price'] + out['carry'] - out['cost']
    out['net_short'] = -out['price'] - out['carry'] - out['cost']

    # The side worth taking, and what it returns. Zero when neither clears cost.
    long_better = out['net_long'] >= out['net_short']
    out['best_side'] = np.where(long_better, 1.0, -1.0)
    out['best_net'] = np.where(long_better, out['net_long'], out['net_short'])
    stand_aside = out['best_net'] <= 0
    out.loc[stand_aside, 'best_side'] = 0.0

    # Unresolvable rows carry nothing. `cost` is in the mask because a NaN cost
    # with a resolvable price left `best_net` NaN while `best_side` fell through
    # to -1 (NaN comparisons are False both ways), so the row looked like a
    # short with an unknown outcome.
    resolvable = out['price'].notna() & out['cost'].notna()
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
            # Only the horizon is taken from the spec now; the cost is per bar.
            reference_price=float(bars['close'].iloc[0]),
        )
        cost = round_trip_cost_series(symbol, bars['close'], config)
        if horizon_bars is not None:
            spec = TargetSpec(horizon_bars=horizon_bars, round_trip_cost=spec.round_trip_cost)

        targets = build_targets(
            bars, spec,
            funding=funding_by_symbol.get(symbol),
            index=(index_by_symbol or {}).get(symbol),
            cost=cost,
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
