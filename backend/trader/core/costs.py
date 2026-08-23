"""What a Kalshi binary costs to trade.

This module is the single source of truth for the money *inputs*. Sizing and
the running account are deliberately elsewhere — `core/book.py` — because the
last version of this repo mixed them and the fee model could not be corrected
without touching position sizing.

**The fee is a function of price, and that shape is the whole strategy.**
Kalshi charges, per order:

    fee = ceil(fee_rate * contracts * price * (1 - price) * 100) / 100

in dollars, with `price` the contract price in dollars (0..1) and `fee_rate`
0.07. Settlement is free, so a held-to-expiry binary pays *one* fee, not a
round trip. The `p(1-p)` term means the fee is maximal at 50c and falls toward
either extreme:

| price | fee/contract | as a share of the stake |
|------:|-------------:|------------------------:|
|   50c |      $0.0175 |                   3.50% |
|   70c |      $0.0147 |                   2.10% |
|   85c |      $0.0089 |                   1.05% |
|   90c |      $0.0063 |                   0.70% |
|   95c |      $0.0033 |                   0.35% |

A perpetual future charges a fixed toll regardless of conviction, so a
confident forecast and a marginal one pay the same. Here they do not: a
confident bet is a cheap bet. That is why the barrier framing and this venue
fit together — the barrier's confident predictions are the large-displacement,
late-in-the-window ones, which is exactly where `p(1-p)` is small.

**The ceiling matters at a $100 account.** One contract at 50c owes $0.0175 and
is charged $0.02 — 14% more than the formula, because the rounding is per
order, not per contract. At two contracts it is $0.035 -> $0.04. Anything
that reasons about fees in percentage terms without the ceiling understates
the cost of the smallest orders, which are the only orders a $100 account
places.

**The half-spread is larger than the fee above 83c, and it is an assumption.**
No Kalshi order ticket has been read against this module. The taker formula
above is the published schedule; the maker rate is modelled as a flat
per-contract charge and is *unverified*. The last venue this repo priced was
wrong in both shape and magnitude for a day, and it was settled by reading
three real tickets rather than by reasoning — so treat every number here as
provisional until a filled order confirms it, and keep the half-spread
reported separately so a stress run can move it alone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from core.config import Config, DEFAULT_CONFIG

Numeric = Union[float, np.ndarray]

# A contract settles at $1 or $0. Every price here is in dollars, not cents,
# because mixing the two is the classic factor-of-100 error and the venue's own
# UI uses cents.
CONTRACT_PAYOUT = 1.0

# The venue's `price_level_structure` is `tapered_deci_cent`: a tenth of a cent
# in the tails and a full cent through the middle. Read off a live market's own
# `price_ranges`, and confirmed by its order book, whose levels step 0.0010 below
# 0.10 and 0.0100 above it.
#
# `TICK` was a flat 0.01 and that was wrong in the direction that matters. It
# also invalidated the stated reason for excluding low prices — "below 10c a
# one-cent tick is a 10% relative price error" — when the tick there is a tenth
# of that.
PRICE_TICKS: tuple[tuple[float, float, float], ...] = (
    (0.0000, 0.1000, 0.0010),
    (0.1000, 0.9000, 0.0100),
    (0.9000, 1.0000, 0.0010),
)
TICK = 0.01          # the middle band's tick, kept for the fee-rounding maths
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
    """Snap to a price the venue will actually accept.

    Tapered, so a 4c quote rounds to a tenth of a cent and a 40c quote to a
    cent. Rounding everything to a cent silently moved every tail price by up to
    half a cent, which at 2c is a 25% relative error on the thing being traded.
    """
    values = np.asarray(price, dtype=float)
    step = np.asarray(tick_at(values), dtype=float)
    snapped = np.round(values / step) * step
    snapped = np.clip(np.round(snapped, 4), MIN_PRICE, MAX_PRICE)
    return float(snapped) if np.ndim(price) == 0 else snapped


def _ceil_cents(dollars: Numeric) -> Numeric:
    """Round up to the next whole cent."""
    scaled = np.asarray(dollars, dtype=float) * 100.0
    # Guard the float representation: 0.07 * 1 * 0.25 * 100 is 1.7499999...,
    # and ceiling that gives 2 correctly, but an exact 2.0 must not become 3.
    rounded = np.ceil(np.round(scaled, 9)) / 100.0
    return float(rounded) if np.isscalar(dollars) or np.ndim(dollars) == 0 else rounded


def trade_fee(
    contracts: Numeric,
    price: Numeric,
    config: Config = DEFAULT_CONFIG,
    *,
    maker: Optional[bool] = None,
) -> Numeric:
    """Total dollar fee for an order of `contracts` at `price`.

    Rounded up to the next cent *per order*, which is how the venue charges and
    why a one-contract order pays a disproportionate rate.
    """
    is_maker = config.assume_maker if maker is None else maker
    contracts_arr = np.asarray(contracts, dtype=float)
    price_arr = np.asarray(price, dtype=float)
    if is_maker:
        raw = config.maker_fee_rate * contracts_arr
    else:
        raw = config.fee_rate * contracts_arr * price_arr * (1.0 - price_arr)
    return _ceil_cents(raw)


def fee_per_contract(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """The unrounded per-contract fee, for reasoning about the schedule.

    Not what an order is charged — `trade_fee` is, and it rounds up per order.
    This is the continuous function used to derive break-even thresholds, where
    the ceiling would make the answer depend on order size.
    """
    price_arr = np.asarray(price, dtype=float)
    if config.assume_maker:
        return np.full_like(price_arr, config.maker_fee_rate)
    return config.fee_rate * price_arr * (1.0 - price_arr)


def effective_price(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """What a contract really costs: quoted price, plus the half-spread crossed,
    plus the fee.

    This is the number a forecast has to beat. Expressed on the probability
    scale, because a binary's price *is* a probability and the comparison is
    then dimensionless — no basis points, no notional.
    """
    crossed = np.asarray(price, dtype=float) + config.half_spread_cents / 100.0
    crossed = np.clip(crossed, TICK, 1.0 - TICK)
    return crossed + fee_per_contract(crossed, config)


def break_even_probability(price: Numeric, config: Config = DEFAULT_CONFIG) -> Numeric:
    """The true probability at which buying at `price` breaks even.

    Identical to `effective_price` — a binary paying $1 breaks even when the
    win probability equals the all-in cost. Named separately because the two
    readings are used in different arguments and conflating them is how a
    required-edge table ends up off by a fee.
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

    `probability` is the forecast for the side being bought. Positive means the
    trade is worth taking before any sizing or risk consideration; sizing then
    decides how much, and the exposure gates decide whether at all.
    """
    cost = np.asarray(effective_price(price, config), dtype=float)
    return np.asarray(probability, dtype=float) * CONTRACT_PAYOUT - cost


@dataclass(frozen=True)
class FeeSchedule:
    """A rendered view of the schedule, for reports and for the API."""

    fee_rate: float
    maker_fee_rate: float
    half_spread_cents: float
    version: str
    verified_against_ticket: bool = False

    @classmethod
    def of(cls, config: Config = DEFAULT_CONFIG) -> 'FeeSchedule':
        return cls(
            fee_rate=config.fee_rate,
            maker_fee_rate=config.maker_fee_rate,
            half_spread_cents=config.half_spread_cents,
            version=config.fee_config_version,
        )

    def table(self, prices: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)) -> list[dict]:
        config = Config(
            fee_rate=self.fee_rate,
            maker_fee_rate=self.maker_fee_rate,
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
    """The contiguous price range where the fee and half-spread alone demand
    more than `max_required_edge_pp` of edge over the quote.

    The threshold is an argument rather than `config.min_edge_pp` because the
    two are different quantities and conflating them costs a fee: break-even
    edge is measured *over the quoted price*, while `min_edge_pp` is the
    surplus demanded *over break-even*.

    Reported as the *unaffordable* middle rather than the affordable ends,
    because `p(1-p)` makes the affordable set two disjoint tails and the
    min/max of a disjoint set reads as "everything is affordable", which is the
    opposite of what the schedule says. What the band means: inside it, no
    forecast this project has ever produced could pay for the trade; outside
    it, the venue is not the binding constraint.

    This says where the venue makes a trade affordable at all, which is a
    different question from where the model has measured skill.
    `Config.min_traded_price` / `max_traded_price` enforce the second.
    """
    grid = np.arange(TICK, 1.0, step)
    expensive = grid[required_edge_pp(grid, config) > max_required_edge_pp]
    if expensive.size == 0:
        return (math.nan, math.nan)
    return (float(expensive.min()), float(expensive.max()))
