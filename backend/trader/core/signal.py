"""The decision. One implementation, called by backtest, simulation and live.

The previous system had three copies of this logic — `run_backtest` (676 lines),
`run_signals` (329) and `run_inference` (300) — each independently deciding
thresholds, calibration, regime and momentum. Two of them were a copy-paste pair
that had drifted. That is why the backtest and the live path disagreed: they were
different programs computing the same thing.

Here there is one `decide()`. The backtest calls it, the simulator calls it, the
live signal writer calls it. They cannot disagree, because there is nothing to
disagree with.

The decision is a chain of gates, and every rejection is named. When a strategy
stops trading, the counter says which gate closed rather than leaving anyone to
guess — that diagnostic is worth more than any single gate.

Gate order is deliberate: cheapest and most decisive first, so the expensive
checks only run on candidates that could still trade.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config
from core.costs import get_contract_spec
from core.execution import size_from_forecast
from core.profiles import CoinProfile

logger = logging.getLogger(__name__)


# Below this many overlapping bars a pairwise correlation describes noise.
MIN_CORRELATION_OBSERVATIONS = 24


class Gate(str, Enum):
    """Why a candidate did not become a trade.

    Ordered as evaluated. `EDGE_BELOW_COST` dominating the counts means the
    forecast is not clearing the hurdle, which is a different problem from
    `RISK_UNAVAILABLE` dominating — that means the dispersion head is failing.
    """

    NO_FORECAST = 'no_forecast'
    STALE_FEATURES = 'stale_features'
    VOLATILITY_REGIME = 'volatility_regime'
    RISK_UNAVAILABLE = 'risk_unavailable'
    EDGE_BELOW_COST = 'edge_below_cost'
    EDGE_TO_RISK = 'edge_to_risk'
    CONVICTION = 'conviction'
    COOLDOWN = 'cooldown'
    POSITION_LIMIT = 'position_limit'
    CORRELATION_LIMIT = 'correlation_limit'
    SIZE_BELOW_ONE_CONTRACT = 'size_below_one_contract'
    PARTICIPATION_LIMIT = 'participation_limit'


# Refuse to trade more than this share of a bar's volume. Above it the backtest
# is describing a market that would have moved away from the order.
MAX_PARTICIPATION = 0.10


@dataclass(frozen=True)
class Decision:
    """What to do, and why.

    A rejection carries as much information as an entry: `gate` names the reason
    and the forecast components are still populated, so a rejected candidate can
    be inspected rather than merely counted.
    """

    symbol: str
    timestamp: pd.Timestamp
    side: int = 0
    contracts: int = 0
    gate: Optional[Gate] = None
    expected_net: float = 0.0
    expected_price: float = 0.0
    expected_carry: float = 0.0
    cost: float = 0.0
    sigma: float = 0.0
    edge_to_risk: float = 0.0
    volatility: float = 0.0
    price: float = 0.0
    notional: float = 0.0
    participation: float = 0.0
    # Highest absolute correlation with an already-accepted position, or None
    # when there was not enough overlapping history to measure it.
    max_correlation: Optional[float] = None
    # The liquidity the size was measured against, carried forward so the fill
    # can re-apply the same cap against the same number rather than against the
    # fill bar, which the decision never saw.
    sizing_liquidity: float = 0.0

    @property
    def tradeable(self) -> bool:
        return self.side != 0 and self.contracts > 0

    @property
    def carry_share(self) -> float:
        """How much of the expected edge is carry rather than price.

        The per-decision version of the diagnostic that matters most: a book of
        trades whose edge is mostly carry is a different strategy, with different
        risks, from one betting on direction.
        """
        total = abs(self.expected_price) + abs(self.expected_carry)
        return float(abs(self.expected_carry) / total) if total > 0 else 0.0

    def as_row(self) -> dict[str, Any]:
        """Flat record for the signals table and the search ledger."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'side': self.side,
            'contracts': self.contracts,
            'gate': self.gate.value if self.gate else None,
            'expected_net_bps': self.expected_net * 10_000,
            'expected_price_bps': self.expected_price * 10_000,
            'expected_carry_bps': self.expected_carry * 10_000,
            'cost_bps': self.cost * 10_000,
            'sigma_bps': self.sigma * 10_000,
            'edge_to_risk': self.edge_to_risk,
            'carry_share': self.carry_share,
            'price': self.price,
            'notional': self.notional,
            'participation': self.participation,
            'sizing_liquidity': self.sizing_liquidity,
        }


@dataclass
class DecisionContext:
    """Everything `decide` needs that is not the forecast itself.

    Passing state explicitly rather than reading it from a live database is what
    lets the backtest and the live path run the identical function.
    """

    equity: float
    volatility: float
    bar_volume: float
    price: float
    open_positions: int = 0
    bars_since_exit: Optional[int] = None
    max_positions: int = 5


@dataclass
class GateCounter:
    """Tally of why candidates were rejected."""

    counts: Counter = field(default_factory=Counter)
    accepted: int = 0
    evaluated: int = 0

    def record(self, decision: Decision) -> Decision:
        self.evaluated += 1
        if decision.tradeable:
            self.accepted += 1
        elif decision.gate is not None:
            self.counts[decision.gate.value] += 1
        return decision

    def summary(self, top: int = 6) -> dict[str, Any]:
        return {
            'evaluated': self.evaluated,
            'accepted': self.accepted,
            'acceptance_rate': self.accepted / self.evaluated if self.evaluated else 0.0,
            'top_gates': dict(self.counts.most_common(top)),
        }

    def __str__(self) -> str:
        if not self.evaluated:
            return 'no candidates evaluated'
        ranked = ', '.join(f'{name}={count}' for name, count in self.counts.most_common(4))
        return (
            f'{self.accepted}/{self.evaluated} accepted '
            f'({self.accepted / self.evaluated:.2%}) | {ranked}'
        )


def decide(
    *,
    symbol: str,
    timestamp: pd.Timestamp,
    forecast: pd.Series,
    context: DecisionContext,
    config: Config,
    profile: Optional[CoinProfile] = None,
    counter: Optional[GateCounter] = None,
) -> Decision:
    """Turn one forecast into a position, or explain why not.

    `forecast` is a row from `ForecastModel.predict`: expected price, expected
    carry, cost, sigma, side and expected_net. Everything else about the world
    arrives through `context`.
    """
    def field_value(name: str, default: float = 0.0) -> float:
        """Read one forecast field.

        Written out rather than using `x or default`, because zero is falsy: a
        forecast with zero expected carry — entirely normal — would otherwise
        read as missing and be rejected as NO_FORECAST.
        """
        value = forecast.get(name, default)
        if value is None:
            return default
        value = float(value)
        return value if np.isfinite(value) else default

    def result(**kwargs: Any) -> Decision:
        decision = Decision(
            symbol=symbol, timestamp=timestamp,
            expected_price=field_value('price'),
            expected_carry=field_value('carry'),
            cost=field_value('cost'),
            sigma=field_value('sigma'),
            expected_net=field_value('expected_net'),
            edge_to_risk=field_value('edge_to_risk'),
            volatility=float(context.volatility),
            price=float(context.price),
            sizing_liquidity=float(context.bar_volume),
            **kwargs,
        )
        return counter.record(decision) if counter else decision

    # -- the forecast must exist and be finite ------------------------------
    required = ('price', 'carry', 'cost', 'sigma', 'expected_net')
    missing = [
        name for name in required
        if name not in forecast.index
        or forecast[name] is None
        or not np.isfinite(float(forecast[name]))
    ]
    if missing:
        return result(gate=Gate.NO_FORECAST)

    # -- volatility regime: refuse markets the model was not fitted for -----
    # Too quiet and the cost hurdle cannot be cleared; too wild and the risk
    # estimate is extrapolating past anything in the training sample.
    min_vol = float(config.resolve('min_vol_24h', profile))
    max_vol = float(config.resolve('max_vol_24h', profile))
    if not np.isfinite(context.volatility) or not (min_vol <= context.volatility <= max_vol):
        return result(gate=Gate.VOLATILITY_REGIME)

    # -- a position needs a risk estimate to be sized ----------------------
    sigma = float(forecast['sigma'])
    if sigma <= 1e-9:
        return result(gate=Gate.RISK_UNAVAILABLE)

    # -- the edge must clear the round trip --------------------------------
    side = int(forecast.get('side', 0) or 0)
    expected_net = float(forecast['expected_net'])
    if side == 0 or expected_net <= 0:
        return result(gate=Gate.EDGE_BELOW_COST)

    # -- and it must be large relative to its own uncertainty --------------
    edge_to_risk = expected_net / sigma
    if edge_to_risk < float(config.resolve('min_edge_to_risk', profile)):
        return result(gate=Gate.EDGE_TO_RISK, side=side)

    # -- conviction floor, scaled to what the trade costs -----------------
    # The forecast has to clear the round trip by a margin, not merely exceed it,
    # because the cost is known and the forecast is not. Relative to cost rather
    # than absolute: the round trip ranges from ~5bp on the group-B contracts to
    # ~54bp on ETH, so any single basis-point floor is trivial for one and
    # unreachable for the other.
    cost = float(forecast['cost'])
    if expected_net < float(config.min_edge_over_cost) * cost:
        return result(gate=Gate.CONVICTION, side=side)

    # -- portfolio constraints --------------------------------------------
    cooldown_bars = int(float(config.resolve('cooldown_hours', profile)))
    if context.bars_since_exit is not None and context.bars_since_exit < cooldown_bars:
        return result(gate=Gate.COOLDOWN, side=side)

    if context.open_positions >= context.max_positions:
        return result(gate=Gate.POSITION_LIMIT, side=side)

    # -- sizing ------------------------------------------------------------
    contracts = size_from_forecast(
        equity=context.equity,
        price=context.price,
        symbol=symbol,
        expected_return=expected_net,
        sigma=sigma,
        config=config,
        # The risk budget needs to know where the stop will sit, or it cannot
        # bound what a stop-out costs — and the stop is placed with the per-bar
        # realised volatility, not the horizon-scale forecast dispersion.
        stop_multiple=float(config.resolve('vol_mult_sl', profile)),
        stop_sigma=float(context.volatility),
    )
    if contracts < 1:
        return result(gate=Gate.SIZE_BELOW_ONE_CONTRACT, side=side)

    # -- would the order move the market? ---------------------------------
    from core.execution import participation_rate

    participation = participation_rate(contracts, context.price, context.bar_volume, symbol)
    if participation > MAX_PARTICIPATION:
        return result(gate=Gate.PARTICIPATION_LIMIT, side=side,
                      participation=participation)

    spec = get_contract_spec(symbol)
    return result(
        side=side,
        contracts=contracts,
        notional=spec.notional(contracts, context.price),
        participation=participation,
    )


def _max_correlation(
    returns: pd.DataFrame, symbol: str, against: Sequence[str]
) -> Optional[float]:
    """Highest absolute correlation between `symbol` and any already-accepted one.

    None when there is not enough overlapping history to say — which must not be
    read as "uncorrelated": the caller admits the position, because refusing to
    trade on a missing measurement would halt the book on a newly listed
    instrument.
    """
    if symbol not in returns.columns:
        return None
    peers = [s for s in against if s in returns.columns and s != symbol]
    if not peers:
        return None

    target = returns[symbol]
    worst: Optional[float] = None
    for peer in peers:
        pair = pd.concat([target, returns[peer]], axis=1).dropna()
        if len(pair) < MIN_CORRELATION_OBSERVATIONS:
            continue
        value = float(pair.iloc[:, 0].corr(pair.iloc[:, 1]))
        if not np.isfinite(value):
            continue
        worst = abs(value) if worst is None else max(worst, abs(value))
    return worst


def decide_panel(
    forecasts: pd.DataFrame,
    *,
    contexts: dict[str, DecisionContext],
    config: Config,
    profiles: Optional[dict[str, CoinProfile]] = None,
    counter: Optional[GateCounter] = None,
    returns: Optional[pd.DataFrame] = None,
) -> list[Decision]:
    """Decide for every instrument at one timestamp, best edge first.

    Ranking by edge-to-risk before applying the position limit means the limit
    binds on the weakest candidates rather than on whichever instrument happened
    to be first alphabetically.
    """
    profiles = profiles or {}
    ranked = forecasts.sort_values('edge_to_risk', ascending=False)
    decisions: list[Decision] = []
    taken = 0
    accepted: list[str] = []

    limit = float(config.max_portfolio_correlation)
    for (timestamp, symbol), row in ranked.iterrows():
        context = contexts.get(symbol)
        if context is None:
            continue

        # A count limit is not a diversification limit. On a crypto panel where
        # cross-correlation runs 0.7-0.9, five "diversified" positions are one bet
        # at five times the size — and `max_portfolio_correlation` was declared,
        # parsed from `--max-correlation`, and read by nothing. Candidates arrive
        # best-edge-first, so rejecting the later one keeps the stronger of a
        # correlated pair.
        if accepted and returns is not None and 0.0 < limit < 1.0:
            worst = _max_correlation(returns, symbol, accepted)
            if worst is not None and worst > limit:
                # Carry the forecast components like every other rejection, so a
                # capped candidate can be inspected rather than merely counted.
                rejected = Decision(
                    symbol=symbol, timestamp=timestamp, side=int(row.get('side', 0) or 0),
                    contracts=0, gate=Gate.CORRELATION_LIMIT,
                    expected_net=float(row.get('expected_net', 0.0) or 0.0),
                    sigma=float(row.get('sigma', 0.0) or 0.0),
                    edge_to_risk=float(row.get('edge_to_risk', 0.0) or 0.0),
                    cost=float(row.get('cost', 0.0) or 0.0),
                    max_correlation=worst,
                )
                if counter is not None:
                    counter.record(rejected)
                decisions.append(rejected)
                continue
        adjusted = DecisionContext(
            equity=context.equity,
            volatility=context.volatility,
            bar_volume=context.bar_volume,
            price=context.price,
            open_positions=context.open_positions + taken,
            bars_since_exit=context.bars_since_exit,
            max_positions=context.max_positions,
        )
        decision = decide(
            symbol=symbol, timestamp=timestamp, forecast=row, context=adjusted,
            config=config, profile=profiles.get(symbol), counter=counter,
        )
        decisions.append(decision)
        if decision.tradeable:
            taken += 1
            accepted.append(symbol)

    return decisions
