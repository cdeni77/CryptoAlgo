"""The account: what was staked, what settled, what the equity did.

Kept separate from `core/decide.py` on purpose. The previous incarnation of
this repo mixed sizing into the cost model and the fee schedule could not be
corrected without touching position sizing — so here the decision is a pure
function and this module only records its consequences.

**Binaries make the accounting genuinely simple, and that is a feature.** A
position is bought once and settles once. There is no funding accrual, no
mark-to-market, no margin, no liquidation, and no exit fee, because settlement
is free and the loss is capped at the stake from the instant of entry. The
whole account is: subtract the outlay at entry, add $1 per winning contract at
settlement.

**Annualisation is per trade, not per window.** A strategy that trades 3% of
available windows and is scored as though it traded all of them reports a
Sharpe ratio inflated by the reciprocal of its duty cycle. That mistake was
made in this project before — 2.28 became 1.19 once the 27% duty cycle was
accounted for — so `trades_per_year` is measured from realised trade count over
realised elapsed time and nothing here multiplies by a constant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG
from core.decide import Decision, Side

logger = logging.getLogger(__name__)

SECONDS_PER_YEAR = 365.25 * 24 * 3600


@dataclass(frozen=True)
class Position:
    symbol: str
    window_open: pd.Timestamp
    settle_time: pd.Timestamp
    offset: int
    side: Side
    contracts: int
    price: float
    outlay: float          # everything paid: contracts x (price + half spread) + fee
    fee: float
    model_probability: float
    baseline_probability: float
    edge: float

    @classmethod
    def of(cls, decision: Decision) -> 'Position':
        if not decision.traded:
            raise ValueError('cannot open a position from a refusal')
        return cls(
            symbol=decision.symbol, window_open=decision.window_open,
            settle_time=decision.settle_time, offset=decision.offset,
            side=decision.side, contracts=decision.contracts, price=decision.price,
            outlay=decision.stake, fee=decision.fee,
            model_probability=decision.model_probability,
            baseline_probability=decision.baseline_probability,
            edge=decision.edge,
        )

    def payout(self, settled_up: bool) -> float:
        """Dollars received at settlement: $1 per contract on the winning side."""
        won = settled_up if self.side is Side.UP else not settled_up
        return float(self.contracts) if won else 0.0


@dataclass(frozen=True)
class Settlement:
    position: Position
    settled_up: bool
    payout: float
    pnl: float
    bankroll_after: float

    @property
    def won(self) -> bool:
        return self.payout > 0.0

    @property
    def return_on_stake(self) -> float:
        return self.pnl / self.position.outlay if self.position.outlay else float('nan')


@dataclass
class Book:
    """A running account. Enter, settle, read the curve."""

    config: Config = DEFAULT_CONFIG
    bankroll: float = float('nan')
    open_positions: list[Position] = field(default_factory=list)
    settlements: list[Settlement] = field(default_factory=list)
    refusals: list[Decision] = field(default_factory=list)
    equity_points: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    halted_at: Optional[pd.Timestamp] = None
    # Refusals that `core.decide.stateless_screen` handled vectorised, so they
    # never became Decision objects. Carried here so the rejection histogram is
    # complete rather than only counting what reached the scalar loop.
    stateless_rejections: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))

    def __post_init__(self) -> None:
        if not np.isfinite(self.bankroll):
            self.bankroll = self.config.starting_bankroll

    # ---- mutation --------------------------------------------------------
    def record(self, decision: Decision) -> Optional[Position]:
        if not decision.traded:
            self.refusals.append(decision)
            return None
        if decision.stake > self.bankroll:
            logger.warning('%s: stake $%.2f exceeds bankroll $%.2f, refused',
                           decision.symbol, decision.stake, self.bankroll)
            self.refusals.append(decision)
            return None
        position = Position.of(decision)
        self.bankroll -= position.outlay
        self.open_positions.append(position)
        return position

    def settle(self, outcomes: dict[tuple[str, pd.Timestamp], bool],
               *, at: Optional[pd.Timestamp] = None) -> list[Settlement]:
        """Settle every open position whose outcome is known.

        `outcomes` is keyed on (symbol, window_open) rather than window alone:
        the three symbols settle at the same instant on different outcomes, and
        keying on the timestamp alone would settle all three against whichever
        one was looked up.
        """
        settled: list[Settlement] = []
        remaining: list[Position] = []
        for position in self.open_positions:
            key = (position.symbol, position.window_open)
            if key not in outcomes:
                remaining.append(position)
                continue
            settled_up = bool(outcomes[key])
            payout = position.payout(settled_up)
            self.bankroll += payout
            record = Settlement(
                position=position, settled_up=settled_up, payout=payout,
                pnl=payout - position.outlay, bankroll_after=self.bankroll,
            )
            settled.append(record)
            self.settlements.append(record)
            self.equity_points.append((position.settle_time, self.bankroll))
        self.open_positions = remaining
        floor = self.config.starting_bankroll * self.config.ruin_floor_fraction
        if self.halted_at is None and self.bankroll < floor and settled:
            self.halted_at = settled[-1].position.settle_time
            logger.warning('bankroll $%.2f fell below the $%.2f floor at %s — halted',
                           self.bankroll, floor, self.halted_at)
        return settled

    # ---- reading ---------------------------------------------------------
    @property
    def equity(self) -> float:
        """Bankroll plus the cost basis of anything still open.

        Deliberately *not* marked to a model probability. Marking an open binary
        at our own forecast books the edge we believe in as profit we have not
        received, which is how a losing system reports a rising equity curve.
        """
        return self.bankroll + sum(p.outlay for p in self.open_positions)

    def equity_curve(self) -> pd.Series:
        if not self.equity_points:
            return pd.Series(dtype=float)
        frame = pd.DataFrame(self.equity_points, columns=['time', 'equity'])
        return frame.groupby('time')['equity'].last().sort_index()

    def trades(self) -> pd.DataFrame:
        if not self.settlements:
            return pd.DataFrame(columns=[
                'symbol', 'window_open', 'settle_time', 'offset', 'side', 'contracts',
                'price', 'outlay', 'fee', 'model_probability', 'baseline_probability',
                'edge', 'settled_up', 'payout', 'pnl', 'return_on_stake', 'bankroll_after'])
        rows = []
        for s in self.settlements:
            p = s.position
            rows.append({
                'symbol': p.symbol, 'window_open': p.window_open,
                'settle_time': p.settle_time, 'offset': p.offset, 'side': p.side.value,
                'contracts': p.contracts, 'price': p.price, 'outlay': p.outlay,
                'fee': p.fee, 'model_probability': p.model_probability,
                'baseline_probability': p.baseline_probability, 'edge': p.edge,
                'settled_up': s.settled_up, 'payout': s.payout, 'pnl': s.pnl,
                'return_on_stake': s.return_on_stake, 'bankroll_after': s.bankroll_after,
            })
        return pd.DataFrame(rows)


@dataclass(frozen=True)
class BookStats:
    """What the account did, with the duty cycle in the annualisation."""

    n_trades: int
    n_windows_available: int
    coverage: float
    starting_bankroll: float
    ending_equity: float
    total_return: float
    total_pnl: float
    total_fees: float
    win_rate: float
    mean_edge_pp: float
    realised_edge_pp: float
    mean_return_on_stake: float
    sd_return_on_stake: float
    trades_per_year: float
    sharpe: float
    max_drawdown: float
    halted: bool
    # The old per-trade figure, kept for continuity because it is what every
    # historical report contains. `sharpe` is now the account's.
    sharpe_per_trade: float = float('nan')

    @property
    def growth_multiple(self) -> float:
        """Ending equity as a multiple of the start.

        Reported alongside the percentage because a compounding run produces
        percentages with seventeen digits, and `x2.4` is legible where
        `+140.3%` and `+2.2e17%` are not comparable at a glance.
        """
        return (self.ending_equity / self.starting_bankroll
                if self.starting_bankroll else float('nan'))

    @property
    def fees_share_of_gross(self) -> float:
        gross = self.total_pnl + self.total_fees
        return self.total_fees / gross if gross else float('nan')

    def summary(self) -> str:
        return (
            f'{self.n_trades:,} trades of {self.n_windows_available:,} windows '
            f'({self.coverage:.2%}) | ${self.starting_bankroll:.2f} -> '
            f'${self.ending_equity:.2f} ({self.total_return:+.2%})\n'
            f'  win rate {self.win_rate:.2%} | edge predicted {self.mean_edge_pp:+.2f}pp '
            f'realised {self.realised_edge_pp:+.2f}pp | fees ${self.total_fees:.2f}\n'
            f'  Sharpe {self.sharpe:+.2f} on {self.trades_per_year:,.0f} trades/yr | '
            f'maxDD {self.max_drawdown:.2%}'
            + ('  | HALTED at the bankroll floor' if self.halted else '')
        )


def summarise(book: Book, *, windows_available: int) -> BookStats:
    """Reduce a book to comparable numbers.

    `realised_edge_pp` is the honest counterpart of the predicted edge: the
    actual win rate minus the mean effective cost paid, in probability points.
    Predicted edge is what the model claimed; realised is what happened. A large
    gap between them is the winner's curse, and it is the number to look at
    before any Sharpe ratio.
    """
    trades = book.trades()
    n = len(trades)
    equity = book.equity_curve()
    ending = book.equity if n == 0 else float(equity.iloc[-1]) if len(equity) else book.equity
    if n == 0:
        return BookStats(
            n_trades=0, n_windows_available=windows_available, coverage=0.0,
            starting_bankroll=book.config.starting_bankroll, ending_equity=ending,
            total_return=ending / book.config.starting_bankroll - 1.0,
            total_pnl=0.0, total_fees=0.0, win_rate=float('nan'),
            mean_edge_pp=float('nan'), realised_edge_pp=float('nan'),
            mean_return_on_stake=float('nan'), sd_return_on_stake=float('nan'),
            trades_per_year=0.0, sharpe=float('nan'), max_drawdown=0.0,
            halted=book.halted_at is not None,
        )

    returns = trades['return_on_stake'].to_numpy(dtype=float)
    span_seconds = max(
        (trades['settle_time'].max() - trades['window_open'].min()).total_seconds(), 1.0)
    trades_per_year = n * SECONDS_PER_YEAR / span_seconds
    sd = float(np.std(returns, ddof=1)) if n > 1 else float('nan')
    # The old number: the mean of per-trade *ratios*, annualised by how often
    # trades happened to cluster. Kept because every historical report contains
    # it, and reported separately so the two can be compared.
    sharpe_per_trade = (float(np.mean(returns)) / sd * np.sqrt(trades_per_year)
                        ) if sd and sd > 0 else float('nan')

    # The account's Sharpe: dollars per calendar day, zero days included.
    #
    # `sharpe_per_trade` is not the portfolio's risk-adjusted return and can
    # carry the opposite sign. It equal-weights `pnl / outlay`, while the account
    # experiences dollars — and stakes range from cents to the cap, so the two
    # diverge. Measured on constructed books: 900 small wins plus 50 large losses
    # gives +35.53 per-trade against -37.48 for the account (which lost $214),
    # and a size-skewed profitable book gives -116.51 per-trade against +68.81
    # for the account. A gate reading the first is not policing the second, which
    # matters most for `sharpe_implausible`, whose whole job is to notice a
    # number that cannot be real.
    #
    # Annualising on calendar time rather than on the traded span also stops a
    # sparse strategy inflating its own frequency: four trades inside one hour of
    # a five-year evaluation were being annualised as 35,064 trades a year.
    # Idle days belong in the denominator — with `compound=False` the capital is
    # committed to the strategy whether or not it fires.
    daily = (trades.set_index('settle_time')['pnl'].sort_index()
             .resample('1D').sum())
    if len(daily) > 1:
        calendar = pd.date_range(daily.index.min(), daily.index.max(),
                                 freq='1D', tz=daily.index.tz)
        daily = daily.reindex(calendar, fill_value=0.0)
    daily_values = daily.to_numpy(dtype=float)
    daily_sd = float(np.std(daily_values, ddof=1)) if len(daily_values) > 1 else float('nan')
    sharpe = (float(np.mean(daily_values)) / daily_sd * np.sqrt(365.25)
              ) if daily_sd and daily_sd > 0 else float('nan')
    running_max = equity.cummax()
    drawdown = float(((running_max - equity) / running_max).max()) if len(equity) else 0.0

    effective_cost = trades['outlay'] / trades['contracts']
    return BookStats(
        n_trades=n, n_windows_available=windows_available,
        coverage=n / windows_available if windows_available else float('nan'),
        starting_bankroll=book.config.starting_bankroll, ending_equity=ending,
        total_return=ending / book.config.starting_bankroll - 1.0,
        total_pnl=float(trades['pnl'].sum()), total_fees=float(trades['fee'].sum()),
        win_rate=float((trades['payout'] > 0).mean()),
        mean_edge_pp=float(trades['edge'].mean() * 100.0),
        realised_edge_pp=float(((trades['payout'] > 0).mean() - effective_cost.mean()) * 100.0),
        mean_return_on_stake=float(np.mean(returns)), sd_return_on_stake=sd,
        trades_per_year=trades_per_year, sharpe=sharpe,
        sharpe_per_trade=sharpe_per_trade, max_drawdown=drawdown,
        halted=book.halted_at is not None,
    )
