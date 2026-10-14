"""Tests that the backtest halts on the same conditions as the live account.

Live latches a halt on the ruin floor, a daily realised loss, drawdown below the
starting bankroll, and a run of consecutive losses. The backtest must apply all four.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.book import Book
from core.config import Config


class _Pos:
    def __init__(self, settle_time):
        self.settle_time = settle_time


class _Settlement:
    def __init__(self, pnl, settle_time):
        self.pnl = pnl
        self.position = _Pos(settle_time)


def _book(**over):
    cfg = Config(**{'starting_bankroll': 1000.0, **over})
    return Book(config=cfg)


def _settle(book, pnls, day='2026-07-01'):
    made = []
    for i, pnl in enumerate(pnls):
        t = pd.Timestamp(f'{day}T12:00Z') + pd.Timedelta(minutes=15 * i)
        rec = _Settlement(pnl, t)
        book.settlements.append(rec)
        book.bankroll += pnl
        book.equity_points.append((t, book.bankroll))
        made.append(rec)
    return made


def test_a_run_of_losses_halts():
    book = _book(max_consecutive_losses=5)
    made = _settle(book, [-1.0] * 5)
    assert book._halt_reason(made) is not None
    assert 'consecutive' in book._halt_reason(made)


def test_a_shorter_run_does_not():
    book = _book(max_consecutive_losses=5)
    made = _settle(book, [-1.0] * 4)
    assert book._halt_reason(made) is None


def test_a_win_resets_the_run():
    book = _book(max_consecutive_losses=3)
    made = _settle(book, [-1.0, -1.0, +1.0, -1.0])
    assert book._halt_reason(made) is None


def test_the_daily_loss_limit_halts():
    book = _book(max_daily_loss_fraction=0.10, max_consecutive_losses=0)
    made = _settle(book, [-60.0, -50.0])  # -110 on a 1000 bankroll
    assert 'daily loss' in (book._halt_reason(made) or '')


def test_losses_on_a_different_day_do_not_add_up():
    book = _book(max_daily_loss_fraction=0.10, max_consecutive_losses=0)
    _settle(book, [-60.0], day='2026-07-01')
    made = _settle(book, [-50.0], day='2026-07-02')
    assert book._halt_reason(made) is None


def test_drawdown_only_counts_below_the_starting_bankroll():
    """Drawdown from a peak above the starting bankroll does not halt, as in live."""
    book = _book(max_drawdown_fraction=0.20, max_consecutive_losses=0,
                 max_daily_loss_fraction=0.0)
    made = _settle(book, [+500.0, -200.0])
    assert book.bankroll > book.config.starting_bankroll
    assert book._halt_reason(made) is None


def test_drawdown_below_the_start_halts():
    book = _book(max_drawdown_fraction=0.20, max_consecutive_losses=0,
                 max_daily_loss_fraction=0.0)
    made = _settle(book, [+100.0, -400.0])  # peak 1100 -> 700, 36%
    assert 'drawdown' in (book._halt_reason(made) or '')


def test_the_ruin_floor_still_halts():
    book = _book(max_consecutive_losses=0, max_daily_loss_fraction=0.0,
                 max_drawdown_fraction=0.0)
    made = _settle(book, [-999.0])
    assert 'floor' in (book._halt_reason(made) or '')


def test_a_healthy_book_is_not_halted():
    book = _book()
    made = _settle(book, [+5.0, -2.0, +7.0])
    assert book._halt_reason(made) is None
