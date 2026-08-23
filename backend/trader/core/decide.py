"""The only place a trade is chosen.

The backtest, the live signal writer and the paper engine all call `decide`.
That is not tidiness — it is the reason they cannot drift. The previous
incarnation of this project had per-family strategy classes, and the backtest
and the live path disagreed about entry price for months without anything
failing.

**Abstention is the default action, and every refusal is named.** `decide`
returns a `Decision` whose `reason` is one of a fixed set, so a run reports a
rejection histogram rather than a trade count. On the perp system the single
most informative number ever produced was `edge_below_cost rejected 61,750 of
75,545` — it said the forecast did not cover the fee, which no Sharpe ratio
said. The equivalent here is `edge_below_gate`, and it is expected to dominate.

**The counterfactual price is the baseline.** This system does not use Kalshi
data to forecast, and it has no Kalshi price history to backtest against, so
the market price is assumed to be the calibrated barrier baseline rounded to
the nearest cent. That is the conservative choice and it is the right null: the
market is at least as smart as the arithmetic a clock and a volatility estimate
can do. If the model cannot beat that after fees, there is no trade — and if it
can, the live path will find out whether the real quote is better or worse.
Assuming a *worse* market would be assuming an edge.

**Sizing is fractional Kelly on the account, then floored to whole contracts.**
At a $100 bankroll one contract at 90c is 0.9% of the account, so the integer
floor is a gate rather than a rounding detail — and the fee's per-order ceiling
means a one-contract order pays a higher rate than the schedule implies, which
can flip a marginal trade negative. Both are checked after rounding, not
before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.costs import TICK, effective_price, expected_value_per_contract, trade_fee


class Side(str, Enum):
    UP = 'up'
    DOWN = 'down'


class Reason(str, Enum):
    """Why no trade. Ordered by where the gate sits in the funnel."""

    TRADED = 'traded'
    NOT_FINITE = 'not_finite'                  # missing probability or price
    PRICE_OUT_OF_BAND = 'price_out_of_band'    # tick and spread dominate here
    DISAGREEMENT_IMPLAUSIBLE = 'disagreement_implausible'  # too far from the quote
    EDGE_BELOW_GATE = 'edge_below_gate'        # the forecast does not pay
    BELOW_MIN_CONTRACTS = 'below_min_contracts'  # Kelly stake under one contract
    FEE_CEILING = 'fee_ceiling'                # negative EV once the order fee rounds up
    WINDOW_EXPOSURE = 'window_exposure'        # too much already at risk this window
    POSITION_LIMIT = 'position_limit'          # too many correlated legs this window
    ALREADY_ENTERED = 'already_entered'        # one entry per window
    BANKROLL_FLOOR = 'bankroll_floor'          # stop-trading floor breached


@dataclass(frozen=True)
class Decision:
    """What to do about one (symbol, window, offset) row."""

    symbol: str
    window_open: pd.Timestamp
    settle_time: pd.Timestamp
    offset: int
    reason: Reason
    side: Optional[Side] = None
    price: float = float('nan')          # quoted, in dollars
    effective_cost: float = float('nan')  # price + half-spread + fee, per contract
    model_probability: float = float('nan')   # for the side taken
    baseline_probability: float = float('nan')
    edge: float = float('nan')           # model probability minus effective cost
    contracts: int = 0
    stake: float = 0.0
    fee: float = 0.0
    kelly_fraction: float = float('nan')

    @property
    def traded(self) -> bool:
        return self.reason is Reason.TRADED and self.contracts > 0

    @property
    def edge_pp(self) -> float:
        return self.edge * 100.0

    def describe(self) -> str:
        if not self.traded:
            return (f'{self.symbol} {self.window_open:%Y-%m-%d %H:%M} +{self.offset}m: '
                    f'no trade ({self.reason.value})')
        return (f'{self.symbol} {self.window_open:%Y-%m-%d %H:%M} +{self.offset}m: '
                f'{self.side.value} {self.contracts} @ {self.price:.2f} '
                f'(q={self.model_probability:.4f}, cost={self.effective_cost:.4f}, '
                f'edge={self.edge_pp:+.2f}pp, stake=${self.stake:.2f})')


@dataclass
class WindowExposure:
    """What is already at risk in the window being decided.

    Passed in rather than looked up, so `decide` stays a pure function of its
    arguments — which is what lets the backtest and the live path share it.
    """

    stake: float = 0.0
    positions: int = 0
    symbols_entered: frozenset[str] = frozenset()

    def with_(self, decision: Decision) -> 'WindowExposure':
        if not decision.traded:
            return self
        return WindowExposure(
            stake=self.stake + decision.stake,
            positions=self.positions + 1,
            symbols_entered=self.symbols_entered | {decision.symbol},
        )


def round_to_tick_array(price: np.ndarray) -> np.ndarray:
    """Kalshi quotes in whole cents, and the rounding is a real friction."""
    return np.clip(np.round(np.asarray(price, dtype=float) / TICK) * TICK,
                   TICK, 1.0 - TICK)


def round_to_tick(price: float) -> float:
    return float(round_to_tick_array(np.asarray(price)))


def kelly_fraction_for(probability: float, cost: float) -> float:
    """Full-Kelly fraction of the bankroll for a binary paying $1.

    `f* = (q - c) / (1 - c)` where `c` is the all-in cost per contract. Derived
    from net odds `b = (1 - c)/c`: staking `f` buys `f/c` contracts, so a win
    multiplies the stake by `1/c`. Note the denominator is `1 - c`, not `1 - q`
    — the classic slip, and it under-sizes cheap contracts and over-sizes dear
    ones.
    """
    if not (0.0 < cost < 1.0):
        return 0.0
    return max(0.0, (probability - cost) / (1.0 - cost))


def price_and_edge(
    q_up: np.ndarray,
    market_up: np.ndarray,
    config: Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Choose the better side, and price it. Vectorised, and the only copy.

    Returns `(is_up, price, probability, cost, edge)`. `decide` calls this for
    one row and `stateless_screen` calls it for millions; there is deliberately
    no second implementation of this arithmetic, because the last time this repo
    had the backtest and the live path price a trade separately they disagreed
    about the entry for months.

    Buying "down" at `1 - p` is the same trade as selling "up" at `p`, so both
    sides are always evaluated. The edge here is a disagreement about
    sigma_remaining and that points both ways: a smaller sigma than the market
    assumes makes the probability more extreme than the quote, so buy the
    favourite; a larger sigma makes the favourite overpriced, so buy the
    longshot.
    """
    q_up = np.asarray(q_up, dtype=float)
    price_up = round_to_tick_array(np.asarray(market_up, dtype=float))
    price_down = round_to_tick_array(1.0 - price_up)
    cost_up = np.asarray(effective_price(price_up, config), dtype=float)
    cost_down = np.asarray(effective_price(price_down, config), dtype=float)
    edge_up = q_up - cost_up
    edge_down = (1.0 - q_up) - cost_down
    is_up = edge_up >= edge_down
    return (
        is_up,
        np.where(is_up, price_up, price_down),
        np.where(is_up, q_up, 1.0 - q_up),
        np.where(is_up, cost_up, cost_down),
        np.where(is_up, edge_up, edge_down),
    )


def stateless_screen(
    rows: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the gates that do not depend on account state, vectorised.

    Four of the ten refusal reasons are pure functions of the row —
    `not_finite`, `price_out_of_band`, `disagreement_implausible` and
    `edge_below_gate` — and on real data they account for almost every row. The
    remaining six need the bankroll and the window's existing exposure, so they
    stay in the scalar loop.

    This is an optimisation and nothing else: the survivors are re-decided by
    `decide` itself, so the arithmetic that admits a trade is the same
    arithmetic in both paths and this function can only ever *narrow* what
    reaches it.
    """
    if rows.empty:
        return rows, pd.Series(dtype=int)
    q = rows['model_probability'].to_numpy(dtype=float)
    m = rows['baseline_probability'].to_numpy(dtype=float)
    counts: dict[str, int] = {}

    finite = np.isfinite(q) & np.isfinite(m)
    counts[Reason.NOT_FINITE.value] = int((~finite).sum())

    _, price, probability, _, edge = price_and_edge(np.where(finite, q, 0.5), m, config)
    in_band = (price >= config.min_traded_price) & (price <= config.max_traded_price)
    plausible = np.abs(probability - price) * 100.0 <= config.max_disagreement_pp
    clears = edge >= config.min_edge_pp / 100.0

    stage_band = finite & ~in_band
    stage_disagree = finite & in_band & ~plausible
    stage_edge = finite & in_band & plausible & ~clears
    counts[Reason.PRICE_OUT_OF_BAND.value] = int(stage_band.sum())
    counts[Reason.DISAGREEMENT_IMPLAUSIBLE.value] = int(stage_disagree.sum())
    counts[Reason.EDGE_BELOW_GATE.value] = int(stage_edge.sum())

    survivors = finite & in_band & plausible & clears
    return rows.loc[survivors], pd.Series(counts, dtype=int)


def decide(
    row: pd.Series | dict,
    config: Config = DEFAULT_CONFIG,
    *,
    bankroll: float,
    exposure: Optional[WindowExposure] = None,
) -> Decision:
    """Choose a side, a size, or nothing, for one scored row.

    `row` needs `symbol`, `window_open`, `settle_time`, `offset`,
    `model_probability` and `baseline_probability`.
    """
    exposure = exposure or WindowExposure()
    get = row.get if hasattr(row, 'get') else row.__getitem__
    symbol = get('symbol')
    base = Decision(
        symbol=symbol,
        window_open=get('window_open'),
        settle_time=get('settle_time'),
        offset=int(get('offset')),
        reason=Reason.NOT_FINITE,
        model_probability=float(get('model_probability')),
        baseline_probability=float(get('baseline_probability')),
    )

    def refuse(reason: Reason, **extra) -> Decision:
        from dataclasses import replace
        return replace(base, reason=reason, **extra)

    q_up = base.model_probability
    p_market = base.baseline_probability
    if not (np.isfinite(q_up) and np.isfinite(p_market)):
        return base

    if bankroll < config.starting_bankroll * config.ruin_floor_fraction:
        return refuse(Reason.BANKROLL_FLOOR)
    if symbol in exposure.symbols_entered:
        return refuse(Reason.ALREADY_ENTERED)
    if exposure.positions >= config.max_positions_per_window:
        return refuse(Reason.POSITION_LIMIT)

    is_up_a, price_a, probability_a, cost_a, edge_a = price_and_edge(
        np.array([q_up]), np.array([p_market]), config)
    side = Side.UP if bool(is_up_a[0]) else Side.DOWN
    price = float(price_a[0])
    probability = float(probability_a[0])
    cost = float(cost_a[0])
    edge = float(edge_a[0])

    common = dict(side=side, price=price, effective_cost=cost,
                  model_probability=probability, edge=edge)

    if not (config.min_traded_price <= price <= config.max_traded_price):
        return refuse(Reason.PRICE_OUT_OF_BAND, **common)
    if abs(probability - price) * 100.0 > config.max_disagreement_pp:
        return refuse(Reason.DISAGREEMENT_IMPLAUSIBLE, **common)
    if edge < config.min_edge_pp / 100.0:
        return refuse(Reason.EDGE_BELOW_GATE, **common)

    kelly = kelly_fraction_for(probability, cost)
    # The base the fractions apply to. Compounding is opt-in: sizing off the
    # starting bankroll makes the equity curve additive, so its slope is the
    # per-trade edge rather than an exponential of it.
    sizing_base = bankroll if config.compound else config.starting_bankroll
    stake_target = min(
        config.kelly_fraction * kelly * sizing_base,
        config.max_stake_fraction * sizing_base,
    )
    if config.max_stake_dollars is not None:
        stake_target = min(stake_target, config.max_stake_dollars)
    # Never stake more than is actually there, whatever the fractions say.
    stake_target = min(stake_target, bankroll)
    room = config.max_window_exposure_fraction * sizing_base - exposure.stake
    if room <= 0:
        return refuse(Reason.WINDOW_EXPOSURE, kelly_fraction=kelly, **common)
    stake_target = min(stake_target, room)

    contracts = int(math.floor(stake_target / cost))
    if contracts < config.min_contracts:
        return refuse(Reason.BELOW_MIN_CONTRACTS, kelly_fraction=kelly, **common)

    # The order fee rounds up to a whole cent *per order*, so the realised cost
    # of a small order exceeds the schedule. Re-check expected value against
    # what will actually be charged rather than against the continuous formula.
    fee = float(trade_fee(contracts, price, config))
    outlay = contracts * (price + config.half_spread_cents / 100.0) + fee
    realised_cost = outlay / contracts
    if probability - realised_cost <= 0:
        return refuse(Reason.FEE_CEILING, kelly_fraction=kelly, **common)

    from dataclasses import replace
    return replace(
        base, reason=Reason.TRADED, contracts=contracts, stake=outlay, fee=fee,
        kelly_fraction=kelly, **common,
    )


def decide_window(
    rows: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    *,
    bankroll: float,
) -> list[Decision]:
    """Walk one window's rows in offset order and take the first that clears.

    Offset order, not best-of: at offset 3 you cannot know what offset 12 will
    look like, so choosing the best offset in hindsight is not a strategy that
    can be run. `scripts/evaluate.py` reports edge per offset separately, which
    is how the offset set gets narrowed on evidence instead.
    """
    exposure = WindowExposure()
    decisions: list[Decision] = []
    entered = 0
    candidates, _ = stateless_screen(rows, config)
    for _, row in candidates.sort_values(['offset', 'symbol']).iterrows():
        # `max_entries_per_window` is per (symbol, window) and is enforced by
        # ALREADY_ENTERED; this cap is on distinct correlated legs in the same
        # fifteen minutes, which is the exposure that matters at $100.
        if entered >= config.max_positions_per_window:
            break
        decision = decide(row, config, bankroll=bankroll, exposure=exposure)
        decisions.append(decision)
        if decision.traded:
            exposure = exposure.with_(decision)
            entered += 1
    return decisions


def rejection_histogram(decisions: list[Decision]) -> pd.Series:
    """Counts by reason, in funnel order. The most informative single output."""
    # Keyed on the enum's *value*, not the member: `Reason` subclasses `str`, so
    # a dict keyed on members already indexes by string and re-mapping the index
    # afterwards fails on the strings it produced.
    counts = pd.Series({r.value: 0 for r in Reason}, dtype=int)
    for decision in decisions:
        counts[decision.reason.value] += 1
    return counts
