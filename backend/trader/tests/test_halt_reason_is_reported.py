"""The book must say WHICH breaker halted it.

Four conditions latch the halt -- ruin floor, drawdown, consecutive losses and
the daily loss limit -- and the summary named the ruin floor for all of them.
A run that ended on the daily limit while the book was UP 51% read as "HALTED
at the bankroll floor", which sends the reader to look at a bankroll that was
never near its floor.
"""

from __future__ import annotations

import pandas as pd

from core.book import Book, Side, summarise
from core.config import Config


def _config(**kw) -> Config:
    base = dict(starting_bankroll=500.0, max_daily_loss_fraction=0.25,
                max_consecutive_losses=25, max_drawdown_fraction=0.35,
                ruin_floor_fraction=0.50)
    base.update(kw)
    return Config(**base)


def _lose(book: Book, when: pd.Timestamp, amount: float) -> None:
    """Book one settled loss of `amount` at `when`."""
    from core.book import Position, Settlement

    position = Position(
        symbol='BTC-USD', window_open=when - pd.Timedelta(minutes=15),
        settle_time=when, offset=3, side=Side.UP, contracts=1,
        price=0.5, outlay=amount, fee=0.0, edge=0.01,
        model_probability=0.6, baseline_probability=0.5,
    )
    book.bankroll -= amount
    record = Settlement(position=position, settled_up=False, payout=0.0,
                        pnl=-amount, bankroll_after=book.bankroll)
    book.settlements.append(record)
    book.equity_points.append((when, book.bankroll))
    book.halted_at = None
    reason = book._halt_reason([record])
    if reason is not None and book.halted_at is None:
        book.halted_at = when
        book.halted_reason = reason


def test_the_daily_limit_is_named_rather_than_the_ruin_floor():
    """A book well above its floor must not blame the floor."""
    book = Book(config=_config())
    day = pd.Timestamp('2026-02-03 12:00:00+00:00')
    # One day's realised loss past 25% of $500, on a bankroll nowhere near
    # the $250 ruin floor.
    _lose(book, day, 126.0)

    assert book.halted_at is not None
    assert 'daily loss limit' in (book.halted_reason or '')
    assert book.bankroll > _config().starting_bankroll * 0.50

    stats = summarise(book, windows_available=1000)
    assert stats.halted is True
    assert stats.halted_reason == book.halted_reason
    assert 'bankroll floor' not in stats.summary()
    assert 'daily loss limit' in stats.summary()
