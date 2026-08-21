"""Execution simulation: fills, funding accrual, and liquidation.

The gap between a backtest and a live account is almost never the model. It is
this layer, and the previous system modelled it as a flat 2bp slippage constant,
which gets four things wrong at once:

**Fills are not free at any size.** Crossing the spread costs more when your
order is a larger share of the bar's volume. A constant slippage figure is right
for exactly one order size and optimistic for everything above it, so a strategy
appears to scale when it does not.

**Signals do not fill at the price that produced them.** A decision made from a
bar's close fills at the next bar's open at the earliest. Filling at the
decision price is a lookahead of one bar, and at hourly frequency that bar is
the whole move.

**Funding is not a rounding error.** Coinbase settles hourly. At 4x leverage a
position paying 2bp an hour bleeds roughly 8bp of equity an hour, and over a
96-hour hold that dominates most of the price moves being forecast.

**Positions can be liquidated intrabar.** A backtest that marks to the close
never sees the low that wiped the account. At 4x, a 25% adverse move is fatal
and it does not need to be there at the close.

Every one of those makes a backtest look better than reality, which is why they
are modelled here rather than assumed away.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import fee_floor, get_contract_spec

logger = logging.getLogger(__name__)

# Fraction of a bar's volume above which our own order starts moving the price.
# Below it, crossing the spread is the whole cost; above it, impact grows with
# the square root of participation — the standard concave form, because each
# additional unit walks further up a book that is refilling.
PARTICIPATION_FLOOR = 0.01

# Impact in basis points at 100% participation, used to scale the square-root
# law. Deliberately conservative for a venue as thin as CDE.
IMPACT_AT_FULL_PARTICIPATION_BPS = 250.0

# Maintenance margin as a fraction of notional. Coinbase's tiers vary by
# contract and size; this is a single conservative stand-in until the real
# schedule is loaded, and it errs toward liquidating early.
MAINTENANCE_MARGIN_FRACTION = 0.05


class ExitReason(str, Enum):
    """Why a position closed. Reported so a loss can be attributed."""

    TAKE_PROFIT = 'take_profit'
    STOP_LOSS = 'stop_loss'
    HORIZON = 'horizon'
    LIQUIDATION = 'liquidation'
    SIGNAL_FLIP = 'signal_flip'
    END_OF_DATA = 'end_of_data'


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------


def participation_rate(contracts: int, price: float, bar_volume: float, symbol: str) -> float:
    """Our order as a fraction of the bar's traded volume.

    Volume arrives in base units, so it converts to contracts through the
    contract size. A participation rate above about 10% is a warning that the
    backtest is describing a market that would have noticed you.
    """
    spec = get_contract_spec(symbol)
    if bar_volume <= 0 or spec.units <= 0:
        return 1.0
    bar_contracts = float(bar_volume) / spec.units
    if bar_contracts <= 0:
        return 1.0
    return float(min(max(contracts / bar_contracts, 0.0), 1.0))


def slippage_bps(
    contracts: int,
    price: float,
    bar_volume: float,
    symbol: str,
    *,
    spread_bps: float,
    book_depth_bps: Optional[float] = None,
) -> float:
    """Cost of crossing, in basis points, as a function of participation.

    Half the spread is unavoidable. Above `PARTICIPATION_FLOOR` an impact term
    grows as the square root of participation, which is the standard concave
    form: each extra unit walks further up a book that refills behind it.

    `book_depth_bps`, when persisted order-book snapshots are available, replaces
    the spread assumption with a measurement. Until then this is an assumption
    that should be stress-tested rather than trusted — see the cost-stress gate.
    """
    crossing = (book_depth_bps if book_depth_bps is not None else spread_bps) / 2.0
    rate = participation_rate(contracts, price, bar_volume, symbol)
    if rate <= PARTICIPATION_FLOOR:
        return float(crossing)
    impact = IMPACT_AT_FULL_PARTICIPATION_BPS * np.sqrt(rate)
    return float(crossing + impact)


def fill_price(
    reference_price: float,
    direction: int,
    slippage: float,
) -> float:
    """Price actually paid, moved against us by the slippage in basis points."""
    adjustment = 1.0 + direction * slippage / 10_000.0
    return float(reference_price * adjustment)


# ---------------------------------------------------------------------------
# Position
# ---------------------------------------------------------------------------


@dataclass
class Position:
    """An open position and everything accrued against it."""

    symbol: str
    direction: int
    contracts: int
    entry_price: float
    entry_time: pd.Timestamp
    entry_fee: float
    margin: float
    take_profit: Optional[float] = None
    stop_loss: Optional[float] = None
    hold_until: Optional[pd.Timestamp] = None
    funding_paid: float = 0.0
    bars_held: int = 0

    @property
    def units(self) -> float:
        return get_contract_spec(self.symbol).units * self.contracts

    def notional(self, price: float) -> float:
        return self.units * float(price)

    def unrealised(self, price: float) -> float:
        """Mark-to-market before fees and funding."""
        return self.units * (float(price) - self.entry_price) * self.direction

    def equity(self, price: float) -> float:
        """What the position is worth to the account right now."""
        return self.margin + self.unrealised(price) - self.funding_paid - self.entry_fee

    @property
    def available_margin(self) -> float:
        """Margin still backing the position after fees and accrued funding."""
        return self.margin - self.funding_paid - self.entry_fee

    @property
    def under_margined(self) -> bool:
        """True when maintenance already exceeds what is posted.

        A position in this state should never have been opened, and its solved
        liquidation price is meaningless — it comes out on the wrong side of
        entry, because the algebra is asking what price would restore adequate
        margin rather than what price would destroy it.
        """
        required = MAINTENANCE_MARGIN_FRACTION * self.notional(self.entry_price)
        return self.available_margin < required

    def liquidation_price(self) -> float:
        """Price at which posted margin no longer covers maintenance.

        Solved from  available + units * (P - entry) * direction
                     = maintenance * units * P,
        which gives different denominators per side:

            long   P = (units * entry - available) / (units * (1 - maintenance))
            short  P = (units * entry + available) / (units * (1 + maintenance))

        The short case previously used (direction + maintenance) as the
        denominator, which liquidated shorts about ten percent later than the
        real level — understating risk, which is the dangerous direction to be
        wrong in.

        Returns 0.0 when the position is already under-margined, so callers see
        "no meaningful level" rather than a number on the wrong side of entry.
        """
        units = self.units
        if units <= 0 or self.under_margined:
            return 0.0

        available = self.available_margin
        maintenance = MAINTENANCE_MARGIN_FRACTION
        if self.direction == 1:
            price = (units * self.entry_price - available) / (units * (1.0 - maintenance))
        else:
            price = (units * self.entry_price + available) / (units * (1.0 + maintenance))
        return float(max(price, 0.0))


# ---------------------------------------------------------------------------
# Funding
# ---------------------------------------------------------------------------


def accrue_funding(
    position: Position,
    funding_rate: float,
    mark_price: float,
) -> float:
    """Funding charged for one settlement, in account currency.

    Positive rate means longs pay. Charged on notional at the mark, not on
    margin, which is what makes it bite at leverage: a 4x position pays funding
    on four times the equity backing it.
    """
    if not np.isfinite(funding_rate):
        return 0.0
    payment = position.notional(mark_price) * float(funding_rate) * position.direction
    position.funding_paid += payment
    return float(payment)


# ---------------------------------------------------------------------------
# Exits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BarOutcome:
    """What happened to a position during one bar."""

    exited: bool
    reason: Optional[ExitReason]
    exit_price: float
    liquidated: bool = False


def barrier_prices(
    entry: float,
    volatility: float,
    direction: int,
    *,
    tp_mult: float,
    sl_mult: float,
) -> tuple[float, float]:
    """Take-profit and stop-loss prices as volatility multiples.

    Risk management, not a prediction target. The forecast decides whether to
    enter and how large; these bound how wrong one trade can go.
    """
    tp_move = tp_mult * volatility
    sl_move = sl_mult * volatility
    if direction == 1:
        return entry * (1.0 + tp_move), entry * (1.0 - sl_move)
    return entry * (1.0 - tp_move), entry * (1.0 + sl_move)


def resolve_bar(
    position: Position,
    bar: pd.Series,
    timestamp: pd.Timestamp,
) -> BarOutcome:
    """Whether the position survives this bar, and at what price it left.

    Order of checks is the conservative one, and it matters:

    1. Liquidation first. If the bar's extreme took the account out, nothing
       else in the bar happened.
    2. Then stop-loss before take-profit. Bar data cannot order two touches
       inside one bar, and assuming the profitable order is how a backtest
       inflates its results.
    3. Then the holding horizon.
    """
    high, low = float(bar['high']), float(bar['low'])
    adverse = low if position.direction == 1 else high

    if position.under_margined:
        # Maintenance already exceeds posted margin, so the position is gone at
        # this bar's close whatever the range did.
        return BarOutcome(True, ExitReason.LIQUIDATION, float(bar['close']), liquidated=True)

    liquidation = position.liquidation_price()
    if liquidation > 0:
        breached = adverse <= liquidation if position.direction == 1 else adverse >= liquidation
        if breached:
            return BarOutcome(True, ExitReason.LIQUIDATION, liquidation, liquidated=True)

    if position.stop_loss is not None:
        hit = low <= position.stop_loss if position.direction == 1 else high >= position.stop_loss
        if hit:
            return BarOutcome(True, ExitReason.STOP_LOSS, position.stop_loss)

    if position.take_profit is not None:
        hit = high >= position.take_profit if position.direction == 1 else low <= position.take_profit
        if hit:
            return BarOutcome(True, ExitReason.TAKE_PROFIT, position.take_profit)

    if position.hold_until is not None and timestamp >= position.hold_until:
        return BarOutcome(True, ExitReason.HORIZON, float(bar['close']))

    return BarOutcome(False, None, float(bar['close']))


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def fractional_kelly(
    expected_return: float,
    sigma: float,
    *,
    fraction: float = 0.25,
    cap: float = 0.25,
) -> float:
    """Fraction of equity to risk, from a forecast and its uncertainty.

    Full Kelly is mu / sigma-squared, and it is the wrong answer whenever mu is
    estimated rather than known — which it always is here. Estimation error in mu
    makes full Kelly systematically overbet, and the loss function is brutally
    asymmetric: overbetting by 2x loses more growth than underbetting by 2x.
    A quarter of Kelly is the usual compromise, and the cap bounds the damage
    when the dispersion head underestimates risk.
    """
    if sigma <= 1e-9 or expected_return <= 0:
        return 0.0
    kelly = expected_return / (sigma ** 2)
    return float(min(max(kelly * fraction, 0.0), cap))


def size_from_forecast(
    *,
    equity: float,
    price: float,
    symbol: str,
    expected_return: float,
    sigma: float,
    config: Config,
    kelly_fraction: float = 0.25,
    max_fraction: float = 0.25,
) -> int:
    """Contracts to trade, from the forecast, its risk, and the account.

    Returns zero when the position rounds below one contract — which for a small
    account on an expensive contract is a real constraint rather than an edge
    case, and one the previous fixed-fraction sizing hid.
    """
    spec = get_contract_spec(symbol)
    notional_per_contract = spec.units * float(price)
    if notional_per_contract <= 0 or equity <= 0:
        return 0

    fraction = fractional_kelly(
        expected_return, sigma, fraction=kelly_fraction, cap=max_fraction
    )
    if fraction <= 0:
        return 0

    target_notional = equity * fraction * float(config.leverage)
    return int(max(target_notional // notional_per_contract, 0))


def entry_cost(contracts: int, price: float, symbol: str, config: Config) -> float:
    """Commission for one side, in account currency."""
    spec = get_contract_spec(symbol)
    percentage = spec.notional(contracts, price) * config.fee_pct_per_side
    floor = contracts * fee_floor(symbol, config)
    return float(max(percentage, floor))


# ---------------------------------------------------------------------------
# Open and close
# ---------------------------------------------------------------------------


@dataclass
class Fill:
    """A completed transaction, with its costs separated for attribution."""

    symbol: str
    timestamp: pd.Timestamp
    direction: int
    contracts: int
    reference_price: float
    fill_price: float
    fee: float
    slippage_bps: float
    participation: float
    notional: float


def open_position(
    *,
    symbol: str,
    direction: int,
    contracts: int,
    bar: pd.Series,
    timestamp: pd.Timestamp,
    config: Config,
    volatility: float,
    tp_mult: float,
    sl_mult: float,
    hold_bars: int,
    spread_bps: float = 4.0,
    book_depth_bps: Optional[float] = None,
) -> tuple[Position, Fill]:
    """Enter at this bar's open, paying to cross.

    The reference price is the bar's open rather than the previous close, because
    a decision taken from the previous close cannot fill before this bar starts.
    """
    reference = float(bar['open'])
    slip = slippage_bps(
        contracts, reference, float(bar.get('volume', 0.0)), symbol,
        spread_bps=spread_bps, book_depth_bps=book_depth_bps,
    )
    price = fill_price(reference, direction, slip)
    fee = entry_cost(contracts, price, symbol, config)
    spec = get_contract_spec(symbol)
    notional = spec.notional(contracts, price)

    take_profit, stop_loss = barrier_prices(
        price, volatility, direction, tp_mult=tp_mult, sl_mult=sl_mult
    )

    position = Position(
        symbol=symbol,
        direction=direction,
        contracts=contracts,
        entry_price=price,
        entry_time=timestamp,
        entry_fee=fee,
        margin=notional / max(float(config.leverage), 1e-9),
        take_profit=take_profit,
        stop_loss=stop_loss,
        hold_until=timestamp + pd.Timedelta(hours=int(hold_bars)),
    )
    fill = Fill(
        symbol=symbol, timestamp=timestamp, direction=direction, contracts=contracts,
        reference_price=reference, fill_price=price, fee=fee, slippage_bps=slip,
        participation=participation_rate(contracts, reference, float(bar.get('volume', 0.0)), symbol),
        notional=notional,
    )
    return position, fill


@dataclass
class ClosedTrade:
    """A round trip, decomposed so a result can be attributed.

    `price_pnl`, `funding_pnl` and `fees` sum to `net_pnl`. That decomposition
    is the point: a strategy whose gross price PnL is positive and whose net is
    negative has a cost problem, not a model problem, and the two need different
    fixes.
    """

    symbol: str
    direction: int
    contracts: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    exit_reason: ExitReason
    price_pnl: float
    funding_pnl: float
    fees: float
    net_pnl: float
    notional: float
    bars_held: int
    max_participation: float = 0.0

    @property
    def net_return(self) -> float:
        return self.net_pnl / self.notional if self.notional > 0 else 0.0

    @property
    def liquidated(self) -> bool:
        return self.exit_reason is ExitReason.LIQUIDATION


def close_position(
    position: Position,
    *,
    bar: pd.Series,
    timestamp: pd.Timestamp,
    exit_price: float,
    reason: ExitReason,
    config: Config,
    spread_bps: float = 4.0,
    book_depth_bps: Optional[float] = None,
    entry_participation: float = 0.0,
) -> tuple[ClosedTrade, Fill]:
    """Exit, paying to cross again.

    A liquidation is not a graceful exit: it fills at the liquidation price with
    the same crossing cost, and the resulting loss is the whole margin plus
    whatever the crossing took.
    """
    slip = slippage_bps(
        position.contracts, exit_price, float(bar.get('volume', 0.0)), position.symbol,
        spread_bps=spread_bps, book_depth_bps=book_depth_bps,
    )
    # Exiting means trading the opposite way, so slippage works against us again.
    filled = fill_price(exit_price, -position.direction, slip)
    exit_fee = entry_cost(position.contracts, filled, position.symbol, config)

    price_pnl = position.units * (filled - position.entry_price) * position.direction
    funding_pnl = -position.funding_paid
    fees = position.entry_fee + exit_fee
    net = price_pnl + funding_pnl - fees

    trade = ClosedTrade(
        symbol=position.symbol,
        direction=position.direction,
        contracts=position.contracts,
        entry_time=position.entry_time,
        exit_time=timestamp,
        entry_price=position.entry_price,
        exit_price=filled,
        exit_reason=reason,
        price_pnl=float(price_pnl),
        funding_pnl=float(funding_pnl),
        fees=float(fees),
        net_pnl=float(net),
        notional=float(position.notional(position.entry_price)),
        bars_held=int(position.bars_held),
        max_participation=float(max(
            entry_participation,
            participation_rate(position.contracts, filled,
                               float(bar.get('volume', 0.0)), position.symbol),
        )),
    )
    fill = Fill(
        symbol=position.symbol, timestamp=timestamp, direction=-position.direction,
        contracts=position.contracts, reference_price=exit_price, fill_price=filled,
        fee=exit_fee, slippage_bps=slip,
        participation=trade.max_participation,
        notional=position.notional(filled),
    )
    return trade, fill
