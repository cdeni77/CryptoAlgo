"""Backtest: an event loop over `decide()`.

The only thing this file does is walk bars forward, ask `core.signal.decide` what
to do, and hand the answer to `core.execution`. It contains no thresholds, no
gates and no sizing of its own, which is the property that matters: the live
signal writer calls the same `decide()` with the same forecasts, so the two
cannot drift apart. The previous system's backtest and live path were separate
676-line and 329-line implementations, and that is why they disagreed.

Bar timing is the other thing this gets right. At each bar the loop:

1. Accrues funding on open positions at that bar's settlement.
2. Resolves exits against the bar's own high and low, checking liquidation
   before the stop and the stop before the take-profit.
3. Marks equity to the close.
4. Decides using features and forecasts as of that close — and enters at the
   *next* bar's open.

Deciding from a close and filling at the same close is a one-bar lookahead, and
at hourly frequency that bar is the whole move.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from core.config import Config
from core.execution import (
    ClosedTrade,
    ExitReason,
    Fill,
    Position,
    accrue_funding,
    close_position,
    open_position,
    resolve_bar,
)
from core.metrics import DrawdownProfile, drawdown_profile, sharpe_ratio
from core.model import ForecastModel
from core.profiles import CoinProfile
from core.signal import Decision, DecisionContext, GateCounter, decide_panel

logger = logging.getLogger(__name__)

HOURS_PER_YEAR = 24 * 365

# Volatility estimate used for barrier widths and the regime gate. Matches the
# window the features use, so the backtest and the model see the same regime.
VOL_WINDOW_BARS = 24


@dataclass
class BacktestResult:
    """Everything a run produced, decomposed for attribution.

    `price_pnl`, `funding_pnl` and `fees` sum to the net. That split is what
    distinguishes a model problem from a cost problem: gross price PnL positive
    and net negative is the second, and no amount of retraining fixes it.
    """

    trades: list[ClosedTrade] = field(default_factory=list)
    fills: list[Fill] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    decisions: list[Decision] = field(default_factory=list)
    gates: GateCounter = field(default_factory=GateCounter)
    initial_equity: float = 0.0

    # -- aggregates ---------------------------------------------------------

    @property
    def net_pnl(self) -> float:
        return float(sum(t.net_pnl for t in self.trades))

    @property
    def price_pnl(self) -> float:
        return float(sum(t.price_pnl for t in self.trades))

    @property
    def funding_pnl(self) -> float:
        return float(sum(t.funding_pnl for t in self.trades))

    @property
    def fees(self) -> float:
        return float(sum(t.fees for t in self.trades))

    @property
    def final_equity(self) -> float:
        return float(self.equity_curve.iloc[-1]) if len(self.equity_curve) else self.initial_equity

    @property
    def n_trades(self) -> int:
        return len(self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return float(np.mean([t.net_pnl > 0 for t in self.trades]))

    @property
    def liquidations(self) -> int:
        return sum(1 for t in self.trades if t.liquidated)

    @property
    def carry_contribution(self) -> float:
        """Share of gross profit that came from funding rather than price.

        Near 1 means a carry harvester. Near 0 means a directional bet. The two
        deserve different scrutiny, and a single PnL number hides which you have.
        """
        gross = abs(self.price_pnl) + abs(self.funding_pnl)
        return float(abs(self.funding_pnl) / gross) if gross > 0 else 0.0

    @property
    def max_participation(self) -> float:
        """Largest share of a bar's volume any single order took."""
        return max((t.max_participation for t in self.trades), default=0.0)

    # -- risk ---------------------------------------------------------------

    def returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna() if len(self.equity_curve) > 1 else pd.Series(dtype=float)

    @property
    def sharpe(self) -> float:
        return sharpe_ratio(self.returns(), periods_per_year=HOURS_PER_YEAR)

    @property
    def drawdown(self) -> DrawdownProfile:
        return drawdown_profile(self.equity_curve, periods_per_year=HOURS_PER_YEAR)

    def trades_frame(self) -> pd.DataFrame:
        """Trades as a frame, for the ledger and for bootstrap resampling."""
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([{
            'symbol': t.symbol, 'direction': t.direction, 'contracts': t.contracts,
            'entry_time': t.entry_time, 'exit_time': t.exit_time,
            'entry_price': t.entry_price, 'exit_price': t.exit_price,
            'exit_reason': t.exit_reason.value,
            'price_pnl': t.price_pnl, 'funding_pnl': t.funding_pnl, 'fees': t.fees,
            'net_pnl': t.net_pnl, 'net_return': t.net_return,
            'notional': t.notional, 'bars_held': t.bars_held,
            'max_participation': t.max_participation,
        } for t in self.trades])

    def summary(self) -> dict[str, Any]:
        drawdown = self.drawdown
        return {
            'trades': self.n_trades,
            'net_pnl': round(self.net_pnl, 2),
            'price_pnl': round(self.price_pnl, 2),
            'funding_pnl': round(self.funding_pnl, 2),
            'fees': round(self.fees, 2),
            'carry_contribution': round(self.carry_contribution, 3),
            'return_pct': round(
                (self.final_equity / self.initial_equity - 1) * 100, 2
            ) if self.initial_equity else 0.0,
            'sharpe': round(self.sharpe, 3),
            'max_drawdown': round(drawdown.max_drawdown, 4),
            'time_to_recovery': drawdown.time_to_recovery,
            'win_rate': round(self.win_rate, 4),
            'liquidations': self.liquidations,
            'max_participation': round(self.max_participation, 4),
            'gates': self.gates.summary(),
        }

    def __str__(self) -> str:
        return (
            f"{self.n_trades} trades | net {self.net_pnl:+,.0f} "
            f"(price {self.price_pnl:+,.0f}, funding {self.funding_pnl:+,.0f}, "
            f"fees {self.fees:,.0f}) | Sharpe {self.sharpe:+.2f} | "
            f"maxDD {self.drawdown.max_drawdown:.1%} | "
            f"carry {self.carry_contribution:.0%} | liq {self.liquidations}"
        )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _realised_volatility(close: pd.Series) -> pd.Series:
    """Trailing volatility, shifted so the value at t is knowable at t."""
    return close.pct_change().rolling(VOL_WINDOW_BARS).std().shift(1)


def run_backtest(
    *,
    forecasts: pd.DataFrame,
    bars_by_symbol: dict[str, pd.DataFrame],
    funding_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    profiles: Optional[dict[str, CoinProfile]] = None,
    initial_equity: float = 100_000.0,
    spread_bps: float = 4.0,
) -> BacktestResult:
    """Walk the panel forward, deciding at each close and filling at the next open.

    `forecasts` is the output of `ForecastModel.predict`, MultiIndexed by
    (event_time, symbol). Bars and funding are per instrument.
    """
    config = config or Config()
    profiles = profiles or {}
    funding_by_symbol = funding_by_symbol or {}

    timestamps = pd.DatetimeIndex(
        forecasts.index.get_level_values('event_time').unique()
    ).sort_values()
    if timestamps.empty:
        return BacktestResult(initial_equity=initial_equity)

    volatility = {s: _realised_volatility(b['close']) for s, b in bars_by_symbol.items()}
    funding = {
        s: f['rate'].reindex(bars_by_symbol[s].index).ffill()
        for s, f in funding_by_symbol.items() if s in bars_by_symbol
    }

    equity = float(initial_equity)
    open_positions: dict[str, Position] = {}
    entry_participation: dict[str, float] = {}
    last_exit_bar: dict[str, int] = {}
    pending: list[Decision] = []

    result = BacktestResult(initial_equity=initial_equity)
    curve: dict[pd.Timestamp, float] = {}

    for bar_number, timestamp in enumerate(timestamps):
        # --- 1. fill what was decided at the previous close ---------------
        for decision in pending:
            bars = bars_by_symbol.get(decision.symbol)
            if bars is None or timestamp not in bars.index:
                continue
            if decision.symbol in open_positions:
                continue
            vol = volatility[decision.symbol].get(timestamp, np.nan)
            if not np.isfinite(vol) or vol <= 0:
                continue
            position, fill = open_position(
                symbol=decision.symbol,
                direction=decision.side,
                contracts=decision.contracts,
                bar=bars.loc[timestamp],
                timestamp=timestamp,
                config=config,
                volatility=float(vol),
                tp_mult=float(config.resolve('vol_mult_tp', profiles.get(decision.symbol))),
                sl_mult=float(config.resolve('vol_mult_sl', profiles.get(decision.symbol))),
                hold_bars=config.label_horizon_hours(profiles.get(decision.symbol)),
                spread_bps=spread_bps,
            )
            open_positions[decision.symbol] = position
            entry_participation[decision.symbol] = fill.participation
            equity -= fill.fee
            result.fills.append(fill)
        pending = []

        # --- 2. funding, then exits, on every open position ---------------
        for symbol in list(open_positions):
            position = open_positions[symbol]
            bars = bars_by_symbol[symbol]
            if timestamp not in bars.index:
                continue
            bar = bars.loc[timestamp]
            position.bars_held += 1

            rate = funding.get(symbol, pd.Series(dtype=float)).get(timestamp, 0.0)
            equity -= accrue_funding(position, float(rate or 0.0), float(bar['close']))

            outcome = resolve_bar(position, bar, timestamp)
            if outcome.exited:
                trade, fill = close_position(
                    position, bar=bar, timestamp=timestamp,
                    exit_price=outcome.exit_price, reason=outcome.reason,
                    config=config, spread_bps=spread_bps,
                    entry_participation=entry_participation.get(symbol, 0.0),
                )
                # Funding was charged to equity as it accrued, so only the price
                # move and the exit fee land here — charging it twice would
                # double-count the largest cost in the system.
                equity += trade.price_pnl - fill.fee
                result.trades.append(trade)
                result.fills.append(fill)
                open_positions.pop(symbol)
                entry_participation.pop(symbol, None)
                last_exit_bar[symbol] = bar_number

        # --- 3. mark to market -------------------------------------------
        unrealised = 0.0
        for symbol, position in open_positions.items():
            bars = bars_by_symbol[symbol]
            if timestamp in bars.index:
                unrealised += position.unrealised(float(bars.loc[timestamp, 'close']))
        curve[timestamp] = equity + unrealised

        if equity <= 0:
            logger.warning('account wiped out at %s', timestamp)
            break

        # --- 4. decide for the next bar ----------------------------------
        try:
            slice_ = forecasts.xs(timestamp, level='event_time', drop_level=False)
        except KeyError:
            continue

        contexts: dict[str, DecisionContext] = {}
        for symbol in slice_.index.get_level_values('symbol'):
            bars = bars_by_symbol.get(symbol)
            if bars is None or timestamp not in bars.index or symbol in open_positions:
                continue
            bar = bars.loc[timestamp]
            since_exit = (
                bar_number - last_exit_bar[symbol] if symbol in last_exit_bar else None
            )
            contexts[symbol] = DecisionContext(
                equity=equity,
                volatility=float(volatility[symbol].get(timestamp, np.nan)),
                bar_volume=float(bar.get('volume', 0.0)),
                price=float(bar['close']),
                open_positions=len(open_positions),
                bars_since_exit=since_exit,
                max_positions=config.max_positions,
            )

        if contexts:
            decisions = decide_panel(
                slice_, contexts=contexts, config=config,
                profiles=profiles, counter=result.gates,
            )
            result.decisions.extend(decisions)
            pending = [d for d in decisions if d.tradeable]

    # --- close anything still open ---------------------------------------
    final = timestamps[-1]
    for symbol, position in list(open_positions.items()):
        bars = bars_by_symbol[symbol]
        if final not in bars.index:
            continue
        bar = bars.loc[final]
        trade, fill = close_position(
            position, bar=bar, timestamp=final, exit_price=float(bar['close']),
            reason=ExitReason.END_OF_DATA, config=config, spread_bps=spread_bps,
            entry_participation=entry_participation.get(symbol, 0.0),
        )
        equity += trade.price_pnl - fill.fee
        result.trades.append(trade)
        result.fills.append(fill)
        curve[final] = equity

    result.equity_curve = pd.Series(curve).sort_index()
    return result


def backtest_from_model(
    model: ForecastModel,
    features: pd.DataFrame,
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    costs: pd.Series,
    funding_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    profiles: Optional[dict[str, CoinProfile]] = None,
    initial_equity: float = 100_000.0,
    spread_bps: float = 4.0,
    allow_in_sample: bool = False,
) -> BacktestResult:
    """Score a feature panel with one model, then backtest the result.

    Refuses in-sample rows unless explicitly told otherwise, because trading a
    model's own training window is not a backtest — it is a measurement of how
    well the model memorised. On driftless random walks that produced a mean
    price PnL of +95,000 with a t-statistic of +7 across six seeds.

    For anything that matters, use `walk_forward_backtest`.
    """
    model.assert_compatible(features)

    overlap = model.in_sample_rows(features)
    if overlap and not allow_in_sample:
        raise ValueError(
            f'{overlap} of {len(features)} rows fall inside the training window '
            f'(ending {model.train_end}). Backtesting them measures memorisation, '
            f'not skill. Use walk_forward_backtest, or pass allow_in_sample=True '
            f'if you specifically want the in-sample number.'
        )

    aligned_cost = costs.reindex(features.index).ffill().fillna(0.0)
    forecasts = model.predict(features, cost=aligned_cost.to_numpy())
    return run_backtest(
        forecasts=forecasts,
        bars_by_symbol=bars_by_symbol,
        funding_by_symbol=funding_by_symbol,
        config=config,
        profiles=profiles,
        initial_equity=initial_equity,
        spread_bps=spread_bps,
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardForecasts:
    """Out-of-sample forecasts, and the models that produced them."""

    forecasts: pd.DataFrame
    models: list[ForecastModel] = field(default_factory=list)
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)

    @property
    def coverage(self) -> int:
        return len(self.forecasts)

    def summary(self) -> dict[str, Any]:
        return {
            'periods': len(self.periods),
            'rows': self.coverage,
            'first': str(self.periods[0][0]) if self.periods else None,
            'last': str(self.periods[-1][1]) if self.periods else None,
            'mean_effective_observations': round(
                float(np.mean([m.effective_observations for m in self.models])), 1
            ) if self.models else 0.0,
        }


def generate_walk_forward_forecasts(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    config: Optional[Config] = None,
    profiles: Optional[dict[str, CoinProfile]] = None,
    n_periods: int = 6,
    min_train_fraction: float = 0.35,
) -> WalkForwardForecasts:
    """Retrain periodically and forecast only forward.

    For each period the model is fitted on everything before it, minus one label
    horizon so no training outcome resolves inside the period being forecast.
    Every returned row is therefore a forecast the model could have made at the
    time, which is the only kind a backtest may trade.

    This is also what the simulation layer consumes: CPCV paths, the bootstrap
    and the synthetic panels all need out-of-sample forecasts, not in-sample ones.
    """
    from core.model import align_panel, train_forecast_model

    config = config or Config()
    profiles = profiles or {}

    x, y = align_panel(features, targets)
    if x.empty:
        return WalkForwardForecasts(pd.DataFrame())

    times = pd.DatetimeIndex(x.index.get_level_values('event_time'))
    unique = times.unique().sort_values()
    horizon = config.label_horizon_hours()

    start = int(len(unique) * min_train_fraction)
    if start >= len(unique) - n_periods:
        return WalkForwardForecasts(pd.DataFrame())

    edges = np.linspace(start, len(unique), n_periods + 1).astype(int)
    pieces: list[pd.DataFrame] = []
    models: list[ForecastModel] = []
    periods: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for begin, end in zip(edges[:-1], edges[1:]):
        if end <= begin:
            continue
        period_start, period_end = unique[begin], unique[end - 1]

        # Purge one horizon: a training row entered just before `period_start`
        # resolves inside the period we are about to forecast.
        train_cutoff = period_start - pd.Timedelta(hours=horizon)
        train_mask = times < train_cutoff
        if train_mask.sum() < 500:
            continue

        model = train_forecast_model(
            x[train_mask], y[train_mask], config=config,
            data_as_of=str(train_cutoff),
        )
        if model is None:
            continue

        period_mask = (times >= period_start) & (times <= period_end)
        period_features = x[period_mask]
        cost = y.loc[period_features.index, 'cost'].to_numpy()
        pieces.append(model.predict(period_features, cost=cost))
        models.append(model)
        periods.append((period_start, period_end))

    if not pieces:
        return WalkForwardForecasts(pd.DataFrame())

    return WalkForwardForecasts(
        forecasts=pd.concat(pieces).sort_index(), models=models, periods=periods
    )


def walk_forward_backtest(
    features: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    bars_by_symbol: dict[str, pd.DataFrame],
    funding_by_symbol: Optional[dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    profiles: Optional[dict[str, CoinProfile]] = None,
    n_periods: int = 6,
    initial_equity: float = 100_000.0,
    spread_bps: float = 4.0,
) -> tuple[BacktestResult, WalkForwardForecasts]:
    """The only honest backtest: retrain forward, trade only what was forecastable.

    Returns the result and the forecast set, because the forecasts are worth
    keeping — the gates and the bootstrap both operate on them.
    """
    generated = generate_walk_forward_forecasts(
        features, targets, config=config, profiles=profiles, n_periods=n_periods,
    )
    if generated.forecasts.empty:
        return BacktestResult(initial_equity=initial_equity), generated

    result = run_backtest(
        forecasts=generated.forecasts,
        bars_by_symbol=bars_by_symbol,
        funding_by_symbol=funding_by_symbol,
        config=config,
        profiles=profiles,
        initial_equity=initial_equity,
        spread_bps=spread_bps,
    )
    return result, generated
