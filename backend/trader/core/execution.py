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
from core.costs import per_contract_fee, get_contract_spec

logger = logging.getLogger(__name__)

# Fraction of a bar's volume above which our own order starts moving the price.
# Below it, crossing the spread is the whole cost; above it, impact grows with
# the square root of participation — the standard concave form, because each
# additional unit walks further up a book that is refilling.
PARTICIPATION_FLOOR = 0.01

# Window and quantile for the pessimistic liquidity estimate used to size
# entries. One day of hourly bars at the lower quartile: thin enough to respect
# a quiet market, long enough not to be one outlier bar.
LIQUIDITY_WINDOW_BARS = 24
LIQUIDITY_QUANTILE = 0.25

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


def liquidity_floor(
    volume: pd.Series,
    *,
    window: int = LIQUIDITY_WINDOW_BARS,
    quantile: float = LIQUIDITY_QUANTILE,
) -> pd.Series:
    """Pessimistic liquidity estimate: a trailing low quantile of volume.

    Sizing against the *deciding* bar's volume is the mistake that lets a
    backtest claim capacity it does not have. Entry size is a control, but the
    exit is not: the barrier fires when it fires, into whatever bar happens to
    be there, and that bar is often much thinner than the one the decision saw.
    A run that capped entries at 10% of the deciding bar was still dumping 47%
    of the exit bar.

    So the cap is applied against liquidity that is usually available rather
    than liquidity that happened to be available once — the same reasoning a
    capacity study uses. The value at t is knowable at t: the window closes on
    the previous bar.
    """
    return volume.rolling(window, min_periods=max(window // 4, 2)).quantile(quantile).shift(1)


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


# Most a single position may lose if it runs to its stop, as a fraction of
# equity. This is the binding constraint in practice, not Kelly: Kelly assumes
# the expected return is known, and a forecast of 10bp against a 3% volatility
# implies betting over 100% of capital, which is what it says and not what
# anyone should do with an estimate.
MAX_RISK_PER_TRADE = 0.01

# Ceiling on notional exposure per position as a fraction of equity, before
# leverage. Bounds the damage when the dispersion head understates risk.
MAX_POSITION_FRACTION = 0.05


def fractional_kelly(
    expected_return: float,
    sigma: float,
    *,
    fraction: float = 0.25,
    cap: float = MAX_POSITION_FRACTION,
) -> float:
    """Fraction of equity to commit, from a forecast and its uncertainty.

    Full Kelly is mu / sigma-squared, and it is the wrong answer whenever mu is
    estimated rather than known — which it always is here. Estimation error makes
    full Kelly systematically overbet, and the loss function is asymmetric:
    overbetting by 2x costs more growth than underbetting by 2x does.

    Note how readily this saturates. A 10bp forecast against 3% volatility gives
    a raw Kelly of 1.1 — commit 110% of capital — so the cap is doing the real
    work almost always, and `risk_budget_fraction` is what actually sizes the
    position.
    """
    if sigma <= 1e-9 or expected_return <= 0:
        return 0.0
    kelly = expected_return / (sigma ** 2)
    return float(min(max(kelly * fraction, 0.0), cap))


def risk_budget_fraction(
    sigma: float,
    *,
    stop_multiple: float,
    max_risk: float = MAX_RISK_PER_TRADE,
    leverage: float = 1.0,
) -> float:
    """Notional fraction whose stop-out costs at most `max_risk` of equity.

    A stop sits `stop_multiple` volatilities away, so a position of notional N
    loses about N * stop_multiple * sigma when it triggers. Solving for N bounds
    the loss regardless of whether the forecast was any good — which is the point,
    because the forecast usually is not.

    This is what keeps a run of bad forecasts survivable. Sizing on Kelly alone,
    a strategy trading noise lost 92% of the account over 110 trades.

    `leverage` divides the budget because the caller multiplies the fraction it
    returns by `config.leverage` when it converts to notional. Without the
    division the bound scaled with leverage: at the compose default of 4x, a
    declared 1% `MAX_RISK_PER_TRADE` allowed a 4% loss at the stop. The parameter
    was already declared here and referenced nowhere, which is what it was for.
    """
    loss_per_unit = max(stop_multiple, 1e-9) * max(sigma, 1e-9)
    if loss_per_unit <= 0:
        return 0.0
    return float(max(max_risk / loss_per_unit / max(leverage, 1e-9), 0.0))


def size_from_forecast(
    *,
    equity: float,
    price: float,
    symbol: str,
    expected_return: float,
    sigma: float,
    config: Config,
    stop_multiple: float = 3.0,
    stop_sigma: Optional[float] = None,
    kelly_fraction: float = 0.25,
    max_fraction: float = MAX_POSITION_FRACTION,
    max_risk: float = MAX_RISK_PER_TRADE,
) -> int:
    """Contracts to trade: the smaller of what Kelly wants and what risk allows.

    Two independent limits, and the tighter one wins:

    * Kelly, quartered and capped, sizes on conviction. This wants `sigma` — the
      dispersion head's expected error on the **horizon** return, which is the
      scale the forecast lives on.
    * The risk budget sizes on consequence — a stop-out must not cost more than
      `max_risk` of equity, whatever the forecast claimed. This wants the
      volatility the stop is actually placed with, which is the **per-bar**
      realised vol `barrier_prices` receives.

    `stop_sigma` is that per-bar figure. Passing the horizon sigma to both, as
    this did, compares quantities that differ by about sqrt(horizon) — a factor of
    ~10 at a 96-bar hold — so `MAX_RISK_PER_TRADE` bounded roughly a tenth of what
    it claimed, and the docstring's "two independent limits" was comparing two
    different units. The direction was conservative, which is why it never
    surfaced as a loss.

    Returns zero when the result rounds below one contract, which for a small
    account on an expensive contract is a real constraint rather than an edge
    case.
    """
    spec = get_contract_spec(symbol)
    notional_per_contract = spec.units * float(price)
    if notional_per_contract <= 0 or equity <= 0:
        return 0

    conviction = fractional_kelly(
        expected_return, sigma, fraction=kelly_fraction, cap=max_fraction
    )
    if conviction <= 0:
        return 0

    budget = risk_budget_fraction(
        sigma if stop_sigma is None else stop_sigma,
        stop_multiple=stop_multiple, max_risk=max_risk,
        leverage=float(config.leverage),
    )
    fraction = min(conviction, budget)

    target_notional = equity * fraction * float(config.leverage)
    return int(max(target_notional // notional_per_contract, 0))


def entry_cost(contracts: int, price: float, symbol: str, config: Config) -> float:
    """Commission for one side, in account currency.

    Percentage *plus* per-contract, which is what the venue's order ticket
    charges. It used to be `max()` of the two; `core.costs.per_contract_fee`
    carries the three tickets that ruled that out.
    """
    spec = get_contract_spec(symbol)
    percentage = spec.notional(contracts, price) * config.fee_pct_per_side
    commission = contracts * per_contract_fee(symbol, config)
    return float(percentage + commission)


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


def max_contracts_at_participation(
    price: float,
    bar_volume: float,
    symbol: str,
    limit: float,
) -> int:
    """Largest order that stays within `limit` of the bar's volume."""
    spec = get_contract_spec(symbol)
    if bar_volume <= 0 or spec.units <= 0:
        return 0
    return int(max((float(bar_volume) / spec.units) * limit, 0.0))


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
    participation_limit: Optional[float] = None,
    liquidity: Optional[float] = None,
) -> tuple[Optional[Position], Optional[Fill]]:
    """Enter at this bar's open, paying to cross.

    The reference price is the bar's open rather than the previous close, because
    a decision taken from the previous close cannot fill before this bar starts.

    That gap is also why `participation_limit` is re-applied here. The decision
    checked participation against the *deciding* bar's volume; the fill happens
    on the next bar, which may be far thinner. Without a second check an order
    passed the 10% gate and then took 100% of the bar it actually traded in.
    Oversized orders are trimmed rather than dropped, which is what a broker
    would do.

    `liquidity` tightens what the limit is measured against. Callers with a
    `liquidity_floor` should pass it: the size has to be one the position can
    *leave* through, and the exit bar is not this one. It is a floor, not a
    substitute — the limit binds against whichever is smaller, so the order both
    stays exitable and never exceeds its share of the bar it actually trades in.
    Slippage still uses the real volume, because the real fill is real.
    """
    if participation_limit is not None:
        bar_volume = float(bar.get('volume', 0.0))
        reference_volume = (
            bar_volume if liquidity is None else min(bar_volume, float(liquidity))
        )
        allowed = max_contracts_at_participation(
            float(bar['open']), reference_volume, symbol, participation_limit
        )
        contracts = min(contracts, allowed)
        if contracts < 1:
            return None, None

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
    # Kept apart on purpose. The entry rate is a control — we chose the size, so
    # a breach there is a bug. The exit rate is a consequence — the barrier fires
    # into whatever bar is there — so a breach there is a capacity finding.
    entry_participation: float = 0.0
    exit_participation: float = 0.0

    @property
    def max_participation(self) -> float:
        return max(self.entry_participation, self.exit_participation)

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
        entry_participation=float(entry_participation),
        exit_participation=participation_rate(
            position.contracts, filled, float(bar.get('volume', 0.0)), position.symbol
        ),
    )
    fill = Fill(
        symbol=position.symbol, timestamp=timestamp, direction=-position.direction,
        contracts=position.contracts, reference_price=exit_price, fill_price=filled,
        fee=exit_fee, slippage_bps=slip,
        participation=trade.exit_participation,
        notional=position.notional(filled),
    )
    return trade, fill
