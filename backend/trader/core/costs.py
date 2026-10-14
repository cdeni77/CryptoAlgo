"""What a Kalshi binary costs to trade: the fee, the tick and the break-even price.

Sizing and the running account live in `core/book.py`. Kalshi charges, per order:

    fee = ceil(fee_rate * contracts * price * (1 - price) * 10_000) / 10_000

in dollars, with `price` in dollars (0..1) and `fee_rate` 0.07. The ceiling is
to a hundredth of a cent, per order, so splitting an order never costs less.
Settlement is free, so a held-to-expiry binary pays one fee.
The `p(1-p)` term makes the fee largest at 50c: a confident bet is a cheap bet.

The taker rate is verified against real fills. The maker rate is modelled, not
verified, so the half-spread stays a separate parameter that a stress run can move.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from core.config import Config, DEFAULT_CONFIG

Numeric = Union[float, np.ndarray]

# A contract settles at $1 or $0. Every price here is in dollars, not cents.
CONTRACT_PAYOUT = 1.0

# The venue's `price_level_structure` is `tapered_deci_cent`: a tenth of a cent
# below 0.10 and above 0.90, and a full cent between.
PRICE_TICKS: tuple[tuple[float, float, float], ...] = (
    (0.0000, 0.1000, 0.0010),
    (0.1000, 0.9000, 0.0100),
    (0.9000, 1.0000, 0.0010),
)
TICK = 0.01          # the middle band's tick, kept for the fee-rounding maths
# The venue's maker rate, applied with `Config.maker_fee_multiplier`.
MAKER_FEE_COEFFICIENT = 0.0175
MIN_PRICE = 0.0010
MAX_PRICE = 0.9990


def tick_at(price: Numeric) -> Numeric:
    """The venue's tick size at a price."""
    values = np.asarray(price, dtype=float)
    out = np.full(values.shape, PRICE_TICKS[1][2], dtype=float)
    for low, high, step in PRICE_TICKS:
        out = np.where((values >= low) & (values < high), step, out)
    return float(out) if np.ndim(price) == 0 else out


def round_to_tick(price: Numeric) -> Numeric:
    """Snap to a price the venue accepts, on the tapered tick grid."""
    values = np.asarray(price, dtype=float)
    step = np.asarray(tick_at(values), dtype=float)
    snapped = np.round(values / step) * step
    snapped = np.clip(np.round(snapped, 4), MIN_PRICE, MAX_PRICE)
    return float(snapped) if np.ndim(price) == 0 else snapped


# Fees round up to a hundredth of a cent; this matches real `/portfolio/fills` tickets.
FEE_GRANULARITY_PER_DOLLAR = 10_000.0


def _ceil_fee(dollars: Numeric) -> Numeric:
    """Round a fee up to the venue's granularity — a hundredth of a cent."""
    scaled = np.asarray(dollars, dtype=float) * FEE_GRANULARITY_PER_DOLLAR
    # Round first, so float error cannot push an exact multiple up a step.
    rounded = np.ceil(np.round(scaled, 6)) / FEE_GRANULARITY_PER_DOLLAR
    return float(rounded) if np.isscalar(dollars) or np.ndim(dollars) == 0 else rounded


def trade_fee(
    contracts: Numeric,
    price: Numeric,
    config: Config = DEFAULT_CONFIG,
    *,
    maker: Optional[bool] = None,
) -> Numeric:
    """Total dollar fee for an order of `contracts` at `price`, rounded up per order."""
    is_maker = config.assume_maker if maker is None else maker
    contracts_arr = np.asarray(contracts, dtype=float)
    price_arr = np.asarray(price, dtype=float)
    rate = (config.maker_fee_multiplier * MAKER_FEE_COEFFICIENT if is_maker
            else config.fee_rate)
    return _ceil_fee(rate * contracts_arr * price_arr * (1.0 - price_arr))


def fee_per_contract(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """The unrounded per-contract fee, for deriving break-even thresholds.

    Not what an order is charged: `trade_fee` rounds up per order.
    """
    price_arr = np.asarray(price, dtype=float)
    rate = (config.maker_fee_multiplier * MAKER_FEE_COEFFICIENT
            if config.assume_maker else config.fee_rate)
    return rate * price_arr * (1.0 - price_arr)


def effective_price(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """All-in cost of a contract: quoted price, plus the half-spread, plus the fee.

    On the probability scale, so it compares directly to a forecast.
    """
    crossed = np.asarray(price, dtype=float) + config.half_spread_cents / 100.0
    crossed = np.clip(crossed, TICK, 1.0 - TICK)
    return crossed + fee_per_contract(crossed, config)


def break_even_probability(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """The true probability at which buying at `price` breaks even.

    Equal to `effective_price`: a $1 binary breaks even when the win probability
    equals the all-in cost.
    """
    return effective_price(price, config)


def required_edge_pp(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """Edge over the quoted price, in probability points, needed to break even."""
    return (np.asarray(effective_price(price, config), dtype=float)
            - np.asarray(price, dtype=float)) * 100.0


def expected_value_per_contract(
    probability: Numeric,
    price: Numeric,
    config: Config = DEFAULT_CONFIG,
) -> Numeric:
    """Expected dollars per contract, net of the half-spread and the fee.

    `probability` is the forecast for the side being bought.
    """
    cost = np.asarray(effective_price(price, config), dtype=float)
    return np.asarray(probability, dtype=float) * CONTRACT_PAYOUT - cost


@dataclass(frozen=True)
class FeeSchedule:
    """A rendered view of the schedule, for reports and for the API."""

    fee_rate: float
    maker_fee_multiplier: float
    half_spread_cents: float
    version: str
    verified_against_ticket: bool = False

    @classmethod
    def of(cls, config: Config = DEFAULT_CONFIG) -> 'FeeSchedule':
        return cls(
            fee_rate=config.fee_rate,
            maker_fee_multiplier=config.maker_fee_multiplier,
            half_spread_cents=config.half_spread_cents,
            version=config.fee_config_version,
        )

    def table(self, prices: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)) -> list[dict]:
        config = Config(
            fee_rate=self.fee_rate,
            maker_fee_multiplier=self.maker_fee_multiplier,
            half_spread_cents=self.half_spread_cents,
        )
        rows = []
        for price in prices:
            fee = float(fee_per_contract(price, config))
            rows.append({
                'price': price,
                'fee_per_contract': fee,
                'fee_share_of_stake': fee / price,
                'required_edge_pp': float(required_edge_pp(price, config)),
                'break_even_probability': float(break_even_probability(price, config)),
            })
        return rows


def unaffordable_price_band(
    max_required_edge_pp: float,
    config: Config = DEFAULT_CONFIG,
    *,
    step: float = 0.01,
) -> tuple[float, float]:
    """The contiguous price range where fee and half-spread alone demand more
    than `max_required_edge_pp` of edge over the quote.

    The threshold is not `config.min_edge_pp`: break-even edge is over the quote,
    while `min_edge_pp` is the surplus over break-even.
    Reports the unaffordable middle, because the affordable set is two disjoint tails.
    `Config.min_traded_price` / `max_traded_price` bound where the model trades.
    """
    grid = np.arange(TICK, 1.0, step)
    expensive = grid[required_edge_pp(grid, config) > max_required_edge_pp]
    if expensive.size == 0:
        return (math.nan, math.nan)
    return (float(expensive.min()), float(expensive.max()))
