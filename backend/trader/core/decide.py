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
from core.costs import (
    MAX_PRICE, MIN_PRICE, TICK, effective_price, expected_value_per_contract,
    fee_per_contract, round_to_tick as _round_to_tick, trade_fee,
)


class Side(str, Enum):
    UP = 'up'
    DOWN = 'down'


class Reason(str, Enum):
    """Why no trade. Ordered by where the gate sits in the funnel."""

    TRADED = 'traded'
    NOT_FINITE = 'not_finite'                  # missing probability or price
    PROBABILITY_INVALID = 'probability_invalid'  # outside [0, 1] after scoring
    NO_QUOTE = 'no_quote'                      # live, and the book was not readable
    PRICE_OUT_OF_BAND = 'price_out_of_band'    # tick and spread dominate here
    DISAGREEMENT_IMPLAUSIBLE = 'disagreement_implausible'  # too far from the quote
    EDGE_BELOW_GATE = 'edge_below_gate'        # the forecast does not pay
    BELOW_MIN_CONTRACTS = 'below_min_contracts'  # Kelly stake under one contract
    FEE_CEILING = 'fee_ceiling'                # negative EV once the order fee rounds up
    WINDOW_EXPOSURE = 'window_exposure'        # too much already at risk this window
    POSITION_LIMIT = 'position_limit'          # too many correlated legs this window
    ALREADY_ENTERED = 'already_entered'        # one entry per window
    OFFSET_NOT_TRADED = 'offset_not_traded'    # scored, but not an entry offset
    BANKROLL_FLOOR = 'bankroll_floor'          # stop-trading floor breached
    HALTED = 'halted'                          # a circuit breaker is latched


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
    # 'quote' when the venue's own ask priced this, 'baseline' when the
    # calibrated barrier stood in for a market that was not observed.
    price_source: str = 'baseline'
    market_ticker: Optional[str] = None

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


def _optional(get, key: str) -> Optional[float]:
    """A finite float from the row, or None. A NaN is an absent value here."""
    try:
        value = get(key)
    except (KeyError, IndexError):
        return None
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _optional_str(get, key: str) -> Optional[str]:
    try:
        value = get(key)
    except (KeyError, IndexError):
        return None
    return None if value is None else str(value)


def round_to_tick_array(price: np.ndarray) -> np.ndarray:
    """Snap to the venue's tick ladder — a tenth of a cent in the tails.

    Delegates to `core.costs.round_to_tick` so there is one ladder. This used to
    round everything to a whole cent, which moved every tail price by up to half
    a cent: at 2c that is a 25% relative error on the price being traded.
    """
    return np.asarray(_round_to_tick(np.asarray(price, dtype=float)), dtype=float)


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
    *,
    ask_up: Optional[np.ndarray] = None,
    ask_down: Optional[np.ndarray] = None,
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
    if ask_up is not None and ask_down is not None:
        # A real book. The ask already includes the spread, so adding an assumed
        # half-spread on top would charge for crossing it twice.
        price_up = round_to_tick_array(np.asarray(ask_up, dtype=float))
        price_down = round_to_tick_array(np.asarray(ask_down, dtype=float))
        cost_up = price_up + np.asarray(fee_per_contract(price_up, config), dtype=float)
        cost_down = price_down + np.asarray(fee_per_contract(price_down, config), dtype=float)
    else:
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

    # If the frame carries a real book, screen against it — otherwise the
    # vectorised path and the scalar path would disagree about which rows can
    # trade, and the histogram would not add up to the decisions.
    has_book = 'ask_up' in rows.columns and 'ask_down' in rows.columns
    _, price, probability, _, edge = price_and_edge(
        np.where(finite, q, 0.5), m, config,
        ask_up=rows['ask_up'].to_numpy(dtype=float) if has_book else None,
        ask_down=rows['ask_down'].to_numpy(dtype=float) if has_book else None,
    )
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
    require_quote: bool = False,
    halted: bool = False,
) -> Decision:
    """Choose a side, a size, or nothing, for one scored row.

    `row` needs `symbol`, `window_open`, `settle_time`, `offset`,
    `model_probability` and `baseline_probability`.

    `require_quote` is for the live path. Without it, a row carrying no
    `ask_up`/`ask_down` falls back to the backtest's counterfactual price — the
    calibrated baseline — and can still return TRADED. That is right in a
    backtest, which has no market to ask, and wrong live: it priced against our
    own forecast, and the caller then booked a position for an order it could not
    send. Measured with an unresolved market: 0 orders placed, a 5-contract
    position written, $3.10 debited. Measured with a one-sided book: a claimed
    +8.04pp when the truth against the real 0.90 ask was -11.10pp.
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
    # A probability outside [0, 1] is not a confident forecast, it is a broken
    # one. `clip_prob` guards the scoring path, but nothing guarded `decide`, and
    # Kelly does not object: q=1.02 against a 0.90 baseline sized at 1.225x
    # bankroll and q=1.20 at 7.128x. Only `max_stake_fraction` stood between that
    # and the order.
    if not (0.0 <= q_up <= 1.0 and 0.0 <= p_market <= 1.0):
        return refuse(Reason.PROBABILITY_INVALID)

    # `nan < x` is False, so a NaN bankroll used to sail past the floor and get
    # sized. It reaches here whenever the venue's balance could not be parsed —
    # `balance()` returns 0.0 on a missing field and a `"nan"` string parses.
    if not np.isfinite(bankroll) or (
            bankroll < config.starting_bankroll * config.ruin_floor_fraction):
        return refuse(Reason.BANKROLL_FLOOR)
    # A latched circuit breaker refuses the entry and nothing else.
    #
    # It used to short-circuit the whole cycle: `run_cycle` set `offset = None` on
    # a halt, which returns before `score_live`, so a halted account wrote no
    # predictions and recorded no quotes. The breaker exists to stop risking
    # money, and stopping the measurement was collateral damage nobody chose —
    # and the expensive kind, because the market benchmark needs windows and a
    # halt froze the count. On a $100 account with a 15% daily-loss breaker
    # sitting almost exactly on the expected burn rate, that meant the recording
    # run could not reach its own sample-size target.
    #
    # Refusing here instead keeps the row honest: it is scored, priced against
    # the real book, written with `reason='halted'`, and settled like any other,
    # so `market_benchmark` and the calibration history keep accruing while no
    # money moves.
    if halted:
        return refuse(Reason.HALTED)
    # Scored but not tradeable. Every offset is still measured — that sample is
    # what the forecast tests read — but only `entry_offsets` may open a position.
    # See `Config.entry_offsets`: the earliest offset that cleared was taking 90%
    # of windows at 0.040c per contract, against 3.304c at +12m alone.
    tradeable = config.entry_offsets
    if tradeable is not None and int(base.offset) not in tuple(tradeable):
        return refuse(Reason.OFFSET_NOT_TRADED)
    if symbol in exposure.symbols_entered:
        return refuse(Reason.ALREADY_ENTERED)
    if exposure.positions >= config.max_positions_per_window:
        return refuse(Reason.POSITION_LIMIT)

    ask_up = _optional(get, 'ask_up')
    ask_down = _optional(get, 'ask_down')
    has_book = ask_up is not None and ask_down is not None
    if require_quote and not has_book:
        return refuse(Reason.NO_QUOTE)
    is_up_a, price_a, probability_a, cost_a, edge_a = price_and_edge(
        np.array([q_up]), np.array([p_market]), config,
        ask_up=np.array([ask_up]) if has_book else None,
        ask_down=np.array([ask_down]) if has_book else None,
    )
    side = Side.UP if bool(is_up_a[0]) else Side.DOWN
    price = float(price_a[0])
    probability = float(probability_a[0])
    cost = float(cost_a[0])
    edge = float(edge_a[0])

    # `model_probability` is the probability of the side actually taken, so the
    # baseline has to be too. It was not: a DOWN trade stored P(down) beside
    # P(up), and their difference — which reads like the disagreement being
    # traded — was meaningless for every DOWN row. With model 0.28 and baseline
    # 0.40 the stored pair was (0.72, 0.40), a gap of 0.32 against a real
    # disagreement of 0.12.
    common = dict(side=side, price=price, effective_cost=cost,
                  model_probability=probability,
                  baseline_probability=p_market if side is Side.UP else 1.0 - p_market,
                  edge=edge,
                  price_source='quote' if has_book else 'baseline',
                  market_ticker=_optional_str(get, 'market_ticker'))

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
    # A measured depth beats the standing guess. `max_stake_dollars` exists
    # because nobody had read the book; when a row carries what is actually
    # resting at the touch, that is the real cap.
    measured_depth = _optional(get, f'depth_{side.value}')
    if measured_depth is not None and measured_depth > 0:
        # Only a fraction of it: see `Config.depth_fraction`. The quote is read
        # seconds before the order lands and a quarter of the time the touch has
        # halved by then.
        stake_target = min(stake_target, measured_depth * config.depth_fraction)
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
    # The crossing cost belongs only to the counterfactual price. When `price` is
    # a real ask it already includes the spread, which `price_and_edge` gets
    # right — but this line added the half-spread unconditionally, so live
    # recorded a stake $0.005/contract above the cash actually paid ($4.98 booked
    # against $4.94 charged). That debited a phantom loss on every trade and
    # manufactured exactly the balance drift the operator is told to read as an
    # unrecorded fill. It also made the live EV re-check 0.5c/contract stricter
    # than the backtest's, so the two paths abstained differently.
    crossing = 0.0 if has_book else config.half_spread_cents / 100.0
    outlay = contracts * (price + crossing) + fee
    realised_cost = outlay / contracts
    # `> min_edge_pp`, not `> 0`. Clearing the continuous gate and then landing at
    # break-even once the per-order fee rounds up is not a trade worth taking:
    # measured at 0.90 with one contract, a +0.505pp continuous edge is +0.107pp
    # after rounding, and it was accepted.
    if probability - realised_cost <= config.min_edge_pp / 100.0:
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
        # One entry per (symbol, window) is enforced by ALREADY_ENTERED below
        # (a hardcoded set-membership check, not a config field — it is an
        # invariant, not a policy). `max_positions_per_window` is a different
        # cap: on distinct correlated legs in the same fifteen minutes, which
        # is the exposure that matters at $100.
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
