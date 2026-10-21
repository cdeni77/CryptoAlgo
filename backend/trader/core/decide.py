"""The only place a trade is chosen.

The backtest, the live signal writer and the paper engine all call `decide`,
so they cannot drift apart.

Abstention is the default. Every refusal carries a named `Reason`, so a run
reports a rejection histogram, not only a trade count.

Without a quote, the counterfactual price is the calibrated barrier baseline,
rounded to the tick. Assuming a worse market would be assuming an edge.

Sizing is fractional Kelly on the account, floored to whole contracts. The floor
is a gate, and the per-order fee rounding can flip a marginal trade negative,
so both are checked after rounding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional

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
    NO_PEER_BOOK = 'no_peer_book'              # cross_venue's peer was unavailable
    PRICE_OUT_OF_BAND = 'price_out_of_band'    # tick and spread dominate here
    DISAGREEMENT_IMPLAUSIBLE = 'disagreement_implausible'  # too far from the quote
    EDGE_BELOW_GATE = 'edge_below_gate'        # the forecast does not pay
    EDGE_UNCERTAIN = 'edge_uncertain'          # clears only at the ensemble's mean
    MEMBERS_DISAGREE = 'members_disagree'      # a member does not beat the cost
    BELOW_MIN_CONTRACTS = 'below_min_contracts'  # Kelly stake under one contract
    FEE_CEILING = 'fee_ceiling'                # negative EV once the order fee rounds up
    WINDOW_EXPOSURE = 'window_exposure'        # too much already at risk this window
    POSITION_LIMIT = 'position_limit'          # too many correlated legs this window
    ALREADY_ENTERED = 'already_entered'        # entries per (symbol, window) used up
    OPPOSITE_SIDE = 'opposite_side'            # window already held on the other side
    OFFSET_NOT_TRADED = 'offset_not_traded'    # scored, but not an entry offset
    BANKROLL_FLOOR = 'bankroll_floor'          # stop-trading floor breached
    HALTED = 'halted'                          # a circuit breaker is latched
    MODEL_NOT_PINNED = 'model_not_pinned'      # not the model pinned for trading


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
    # calibrated barrier stood in for an unobserved market.
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

    Passed in, not looked up, so `decide` stays a pure function of its arguments.
    """

    stake: float = 0.0
    positions: int = 0
    symbols_entered: frozenset[str] = frozenset()
    #: How many entries each symbol already holds in this window. Needed once
    #: `max_entries_per_symbol_window` can exceed 1, because a set answers
    #: "has it entered" and not "how many times". Left empty by callers that
    #: predate it; the check below then falls back to the set, so the default
    #: one-entry rule behaves exactly as it always did.
    entry_counts: Mapping[str, int] = field(default_factory=dict)
    #: The side each symbol is already held on in this window. **Once more than
    #: one entry per (symbol, window) is allowed, nothing else stops two
    #: offsets taking OPPOSITE sides** -- and that is not a hedge, it is a
    #: guaranteed loss: measured live 2026-10-20, three such markets settled
    #: with revenue $0.00 and -$6.29, the whole stake, because buying the other
    #: side closes the first at a loss rather than holding both to settlement.
    #: `already_entered` used to prevent this as a side effect of forbidding
    #: any second entry at all; with that relaxed the constraint has to be
    #: stated explicitly.
    sides: Mapping[str, str] = field(default_factory=dict)

    def side_for(self, symbol: str) -> Optional[str]:
        """The side this symbol is already held on, or None."""
        value = self.sides.get(symbol)
        return str(value) if value is not None else None

    def entries_for(self, symbol: str) -> int:
        """Entries already taken for `symbol` in this window.

        Falls back to membership of `symbols_entered` so an exposure built the
        old way still reports 1 rather than 0 — reporting 0 would re-open a
        window that is already held, which is the double-entry this guards.
        """
        if symbol in self.entry_counts:
            return int(self.entry_counts[symbol])
        return 1 if symbol in self.symbols_entered else 0

    def with_(self, decision: Decision) -> 'WindowExposure':
        if not decision.traded:
            return self
        counts = dict(self.entry_counts)
        counts[decision.symbol] = self.entries_for(decision.symbol) + 1
        sides = dict(self.sides)
        if decision.side is not None:
            # First side wins and is never overwritten, so a later opposite
            # decision is refused against the side actually held.
            sides.setdefault(decision.symbol, str(decision.side.value))
        return WindowExposure(
            stake=self.stake + decision.stake,
            positions=self.positions + 1,
            symbols_entered=self.symbols_entered | {decision.symbol},
            entry_counts=counts,
            sides=sides,
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
    """Snap to the venue's tapered tick ladder, via `core.costs.round_to_tick`."""
    return np.asarray(_round_to_tick(np.asarray(price, dtype=float)), dtype=float)


def round_to_tick(price: float) -> float:
    return float(round_to_tick_array(np.asarray(price)))


def kelly_fraction_for(probability: float, cost: float) -> float:
    """Full-Kelly fraction of the bankroll for a binary paying $1.

    `f* = (q - c) / (1 - c)`, where `c` is the all-in cost per contract.
    Staking `f` buys `f/c` contracts, so a win multiplies the stake by `1/c`.
    The denominator is `1 - c`, not `1 - q`.
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
    one row and `stateless_screen` for many; there is no second implementation.

    Both sides are always evaluated: buying DOWN at `1 - p` is selling UP at `p`.
    A sigma disagreement points both ways, toward the favourite or the longshot.
    """
    q_up = np.asarray(q_up, dtype=float)
    if ask_up is not None and ask_down is not None:
        # A real book. The ask already includes the spread, so no assumed
        # half-spread is added.
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


def as_maker_prices(frame):
    """The price a RESTING order pays, in the columns `decide()` reads.

    **`price_and_edge` takes the price to PAY, and that depends on how we
    enter.** A taker lifts the ask; a maker rests at the bid. Handing decide()
    the ask while resting at the bid understates every edge by a full spread
    and refuses trades that are profitable — the mirror of the defect that had
    the live path adding a half-spread on top of a real ask.

    `scripts/live.py` does not call this: it has the book and takes the true
    touch from `Quote.bid_for`. A backtest has only the summarised asks, so the
    bids are derived — `yes_bid = 1 - ask_down`, `no_bid = 1 - ask_up` — which
    is exact whenever both asks came from the same snapshot, and they do.

    Worth the arithmetic: measured 2026-10-18 on 54,100 live-recorded rows, the
    SAME model with no retraining went -$77.25 (-0.108pp/ct) as a taker to
    +$2,445.76 (+1.608pp/ct) priced this way. The fee is only half of it — the
    venue charges makers nothing, and on a measured 1.000c spread buying the
    bid is worth as much again.

    **It does not model whether the order FILLS**, which is the whole risk.
    This used to add that nothing had ever been observed to rest; that stopped
    being true on 2026-10-18, and 512 of 668 fills since (77%) are makers. The
    untested half is now narrower but sharper: a resting order fills only when
    somebody chooses to cross to it, so the filled subset is adversely selected
    relative to the orders placed. See `Config.maker_entry`.
    """
    if frame is None or not len(frame):
        return frame
    if 'ask_up' not in frame.columns or 'ask_down' not in frame.columns:
        return frame
    out = frame.copy()
    up, down = out['ask_up'], out['ask_down']
    out['ask_up'], out['ask_down'] = 1.0 - down, 1.0 - up
    return out


def outside_dead_band(price, config) -> np.ndarray:
    """True where `price` is tradeable, i.e. NOT inside `exclude_price_band`.

    The cost is a parabola and the signal is not: `0.07*p(1-p)` is 1.75pp of
    stake at a coin flip against 0.33pp at 95c, so the middle of the book is the
    most expensive place to trade — and measured on 54,100 live-recorded rows
    graded on the venue's own settlement, 35c-65c is also where the model's
    disagreement was worth -0.021pp. Maximum toll, zero signal.

    All-True when the band is unset, which is the default, so this is inert
    unless asked for.
    """
    band = getattr(config, 'exclude_price_band', None)
    price = np.asarray(price, dtype=float)
    if band is None:
        return np.ones(np.shape(price), dtype=bool)
    lo, hi = float(band[0]), float(band[1])
    return ~((price >= lo) & (price <= hi))


def ensemble_gates(
    is_up: np.ndarray,
    probability: np.ndarray,
    cost: np.ndarray,
    sd: Optional[np.ndarray],
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    config: Config = DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray]:
    """`(uncertain, disagree)` masks for the two ensemble gates. Vectorised.

    A row without the member columns (missing or NaN) passes both gates.
    """
    n = np.shape(probability)[0]
    uncertain = np.zeros(n, dtype=bool)
    disagree = np.zeros(n, dtype=bool)
    z = float(getattr(config, 'edge_lower_bound_z', 0.0) or 0.0)
    if z > 0 and sd is not None:
        sd = np.asarray(sd, dtype=float)
        lower = probability - z * sd - cost
        uncertain = np.isfinite(sd) & (lower < config.min_edge_pp / 100.0)
    if getattr(config, 'require_member_agreement', False) \
            and lo is not None and hi is not None:
        lo = np.asarray(lo, dtype=float)
        hi = np.asarray(hi, dtype=float)
        worst = np.where(is_up, lo, 1.0 - hi)
        disagree = np.isfinite(worst) & (worst - cost <= 0.0)
    return uncertain, disagree


def _column_or_none(rows: pd.DataFrame, name: str) -> Optional[np.ndarray]:
    return rows[name].to_numpy(dtype=float) if name in rows.columns else None


def stateless_screen(
    rows: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply the gates that do not depend on account state, vectorised.

    `not_finite`, `price_out_of_band`, `disagreement_implausible`, `edge_below_gate`
    and the ensemble gates are pure functions of the row. The rest need the
    bankroll and exposure, so they stay in `decide`.
    An optimisation only: survivors are re-decided by `decide`, so this can only
    narrow what reaches it.
    """
    if rows.empty:
        return rows, pd.Series(dtype=int)
    q = rows['model_probability'].to_numpy(dtype=float)
    m = rows['baseline_probability'].to_numpy(dtype=float)
    counts: dict[str, int] = {}

    finite = np.isfinite(q) & np.isfinite(m)
    counts[Reason.NOT_FINITE.value] = int((~finite).sum())

    # Screen against a real book when the frame carries one, so the vectorised
    # and scalar paths agree.
    has_book = 'ask_up' in rows.columns and 'ask_down' in rows.columns
    is_up, price, probability, cost, edge = price_and_edge(
        np.where(finite, q, 0.5), m, config,
        ask_up=rows['ask_up'].to_numpy(dtype=float) if has_book else None,
        ask_down=rows['ask_down'].to_numpy(dtype=float) if has_book else None,
    )
    in_band = ((price >= config.min_traded_price)
               & (price <= config.max_traded_price)
               & outside_dead_band(price, config))
    plausible = np.abs(probability - price) * 100.0 <= config.max_disagreement_pp
    clears = edge >= config.min_edge_pp / 100.0
    uncertain, disagree = ensemble_gates(
        is_up, probability, cost,
        _column_or_none(rows, 'model_probability_sd'),
        _column_or_none(rows, 'model_probability_min'),
        _column_or_none(rows, 'model_probability_max'), config)

    stage_band = finite & ~in_band
    stage_disagree = finite & in_band & ~plausible
    stage_edge = finite & in_band & plausible & ~clears
    passed_edge = finite & in_band & plausible & clears
    stage_uncertain = passed_edge & uncertain
    stage_members = passed_edge & ~uncertain & disagree
    counts[Reason.PRICE_OUT_OF_BAND.value] = int(stage_band.sum())
    counts[Reason.DISAGREEMENT_IMPLAUSIBLE.value] = int(stage_disagree.sum())
    counts[Reason.EDGE_BELOW_GATE.value] = int(stage_edge.sum())
    counts[Reason.EDGE_UNCERTAIN.value] = int(stage_uncertain.sum())
    counts[Reason.MEMBERS_DISAGREE.value] = int(stage_members.sum())

    survivors = passed_edge & ~uncertain & ~disagree
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

    `require_quote` is for the live path: it refuses a row without
    `ask_up`/`ask_down`, instead of pricing it against our own baseline.
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
        # Set before any refusal, so a refused row with a real book reads 'quote'.
        price_source=('quote' if _optional(get, 'ask_up') is not None
                      and _optional(get, 'ask_down') is not None else 'baseline'),
    )

    def refuse(reason: Reason, **extra) -> Decision:
        from dataclasses import replace
        return replace(base, reason=reason, **extra)

    q_up = base.model_probability
    p_market = base.baseline_probability
    if not (np.isfinite(q_up) and np.isfinite(p_market)):
        return base
    # A probability outside [0, 1] is broken, not confident. Kelly would size it
    # above the bankroll.
    if not (0.0 <= q_up <= 1.0 and 0.0 <= p_market <= 1.0):
        return refuse(Reason.PROBABILITY_INVALID)

    # A NaN bankroll is refused: `nan < x` is False, so test finiteness explicitly.
    if not np.isfinite(bankroll) or (
            bankroll < config.starting_bankroll * config.ruin_floor_fraction):
        return refuse(Reason.BANKROLL_FLOOR)
    # A latched circuit breaker refuses the entry only. The row is still scored,
    # priced, written with `reason='halted'` and settled, so measurement continues
    # while no money moves.
    if halted:
        return refuse(Reason.HALTED)
    pinned = getattr(config, 'require_model_version', None)
    if pinned and _optional_str(get, 'model_version') != pinned:
        return refuse(Reason.MODEL_NOT_PINNED)
    # Every offset is scored and measured, but only `entry_offsets` may open a
    # position. See `Config.entry_offsets`.
    tradeable = config.entry_offsets
    if tradeable is not None and int(base.offset) not in tuple(tradeable):
        return refuse(Reason.OFFSET_NOT_TRADED)
    # `>=`, counted, not `in`: with `max_entries_per_symbol_window` at its
    # default of 1 this is exactly the old membership test.
    if exposure.entries_for(symbol) >= config.max_entries_per_symbol_window:
        return refuse(Reason.ALREADY_ENTERED)
    if exposure.positions >= config.max_positions_per_window:
        return refuse(Reason.POSITION_LIMIT)

    ask_up = _optional(get, 'ask_up')
    ask_down = _optional(get, 'ask_down')
    has_book = ask_up is not None and ask_down is not None
    if require_quote and not has_book:
        return refuse(Reason.NO_QUOTE)
    # The peer book, when the configuration requires it. `cross_venue` skill
    # depends on a contemporaneous peer book, and training already requires it
    # under `--complete-cases`; refusing here makes live match.
    # A row without the column at all is not refused: absent feature, not absent book.
    if getattr(config, 'require_peer_book', False):
        # The raw value, not `_optional`: NaN-to-None would hide whether the
        # column is absent (not refused) or the book is missing (refused).
        try:
            raw = get('venue_prob_gap')
        except (KeyError, IndexError):
            raw = None
        if raw is not None:
            try:
                present = np.isfinite(float(raw))
            except (TypeError, ValueError):
                present = False
            if not present:
                return refuse(Reason.NO_PEER_BOOK)
    is_up_a, price_a, probability_a, cost_a, edge_a = price_and_edge(
        np.array([q_up]), np.array([p_market]), config,
        ask_up=np.array([ask_up]) if has_book else None,
        ask_down=np.array([ask_down]) if has_book else None,
    )
    side = Side.UP if bool(is_up_a[0]) else Side.DOWN
    # **Never both sides of one market.** Checked here rather than beside
    # ALREADY_ENTERED because the side is not known until now. Buying the
    # opposite side does not hedge -- the venue nets it against the open
    # position, so the pair settles at revenue $0.00 and the entire premium on
    # both legs is lost. Refusing is also the honest reading: the model has
    # changed its mind, and with hold-to-settle there is no exit to take.
    held = exposure.side_for(symbol)
    if held is not None and held != str(side.value):
        return refuse(Reason.OPPOSITE_SIDE)
    price = float(price_a[0])
    probability = float(probability_a[0])
    cost = float(cost_a[0])
    edge = float(edge_a[0])

    # `model_probability` is for the side taken, so the baseline is too.
    common = dict(side=side, price=price, effective_cost=cost,
                  model_probability=probability,
                  baseline_probability=p_market if side is Side.UP else 1.0 - p_market,
                  edge=edge,
                  price_source='quote' if has_book else 'baseline',
                  market_ticker=_optional_str(get, 'market_ticker'))

    if not (config.min_traded_price <= price <= config.max_traded_price):
        return refuse(Reason.PRICE_OUT_OF_BAND, **common)
    if not bool(outside_dead_band(np.array([price]), config)[0]):
        return refuse(Reason.PRICE_OUT_OF_BAND, **common)
    if abs(probability - price) * 100.0 > config.max_disagreement_pp:
        return refuse(Reason.DISAGREEMENT_IMPLAUSIBLE, **common)
    if edge < config.min_edge_pp / 100.0:
        return refuse(Reason.EDGE_BELOW_GATE, **common)
    member = {}
    for k in ('sd', 'min', 'max'):
        value = _optional(get, f'model_probability_{k}')
        member[k] = np.array([np.nan if value is None else value])
    uncertain, disagree = ensemble_gates(
        is_up_a, probability_a, cost_a, member['sd'], member['min'],
        member['max'], config)
    if bool(uncertain[0]):
        return refuse(Reason.EDGE_UNCERTAIN, **common)
    if bool(disagree[0]):
        return refuse(Reason.MEMBERS_DISAGREE, **common)

    kelly = kelly_fraction_for(probability, cost)
    # The sizing base. Compounding is opt-in: sizing off the starting bankroll
    # keeps the equity curve additive, so its slope is the per-trade edge.
    sizing_base = bankroll if config.compound else config.starting_bankroll
    stake_target = min(
        config.kelly_fraction * kelly * sizing_base,
        config.max_stake_fraction * sizing_base,
    )
    if config.max_stake_dollars is not None:
        stake_target = min(stake_target, config.max_stake_dollars)
    # A measured depth at the touch replaces the `max_stake_dollars` guess.
    measured_depth = _optional(get, f'depth_{side.value}')
    if measured_depth is not None and measured_depth > 0:
        # Only a fraction of it (`Config.depth_fraction`): the touch can shrink
        # between the quote and the order.
        stake_target = min(stake_target, measured_depth * config.depth_fraction)
    # Never stake more than the bankroll holds.
    stake_target = min(stake_target, bankroll)
    room = config.max_window_exposure_fraction * sizing_base - exposure.stake
    if room <= 0:
        return refuse(Reason.WINDOW_EXPOSURE, kelly_fraction=kelly, **common)
    stake_target = min(stake_target, room)

    contracts = int(math.floor(stake_target / cost))
    if contracts < config.min_contracts:
        return refuse(Reason.BELOW_MIN_CONTRACTS, kelly_fraction=kelly, **common)

    # The fee rounds up per order, so re-check expected value against what will
    # actually be charged.
    fee = float(trade_fee(contracts, price, config))
    # The crossing cost applies only to the counterfactual price: a real ask
    # already includes the spread.
    crossing = 0.0 if has_book else config.half_spread_cents / 100.0
    outlay = contracts * (price + crossing) + fee
    realised_cost = outlay / contracts
    # `<= min_edge_pp`, not `<= 0`: a trade that clears the continuous gate and
    # lands near break-even after fee rounding is refused.
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

    Offset order, not best-of: at offset 3 you cannot know offset 12, so a
    hindsight best offset cannot be run.
    """
    exposure = WindowExposure()
    decisions: list[Decision] = []
    entered = 0
    candidates, _ = stateless_screen(rows, config)
    for _, row in candidates.sort_values(['offset', 'symbol']).iterrows():
        # One entry per (symbol, window) is an invariant, enforced by
        # ALREADY_ENTERED. `max_positions_per_window` caps distinct correlated
        # legs in the same window.
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
    # Keyed on the enum's value: `Reason` subclasses `str`, so member keys
    # would already index by string.
    counts = pd.Series({r.value: 0 for r in Reason}, dtype=int)
    for decision in decisions:
        counts[decision.reason.value] += 1
    return counts
