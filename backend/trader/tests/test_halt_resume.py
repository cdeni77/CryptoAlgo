"""The backtest models the operator, not just the breaker.

Live, a circuit breaker latches and a person clears it by hand the next day.
The walk-forward has no person, so `halted_at` latched and eight months of
money measurement vanished: the book stopped on the first bad day of a crash
and never resumed, while the forecast metrics went on describing the whole
range.

`halt_resume_hours` simulates the operator. Off by default, so live and every
existing backtest are bit-identical; the retrain sets it. What matters is not
that a halt happened -- it is how OFTEN, which is what `halt_events` carries.
"""

from __future__ import annotations

import pandas as pd

from core.book import Book, Position, Settlement, Side, summarise
from core.config import Config


def _config(**kw) -> Config:
    base = dict(starting_bankroll=1000.0, max_daily_loss_fraction=0.25,
                max_consecutive_losses=25, max_drawdown_fraction=0.35,
                ruin_floor_fraction=0.50)
    base.update(kw)
    return Config(**base)


def _settle(book: Book, when: pd.Timestamp, pnl: float) -> None:
    """Book one settled trade of `pnl` at `when`, through the real path."""
    position = Position(
        symbol='BTC-USD', window_open=when - pd.Timedelta(minutes=15),
        settle_time=when, offset=3, side=Side.UP, contracts=1,
        price=0.5, outlay=abs(pnl), fee=0.0, edge=0.01,
        model_probability=0.6, baseline_probability=0.5,
    )
    book.open_positions.append(position)
    book.settle({('BTC-USD', position.window_open): pnl > 0}, at=when)


def _blow_the_day(book: Book, day: str) -> None:
    """Lose past the 25% daily limit on `day`."""
    for i in range(3):
        _settle(book, pd.Timestamp(f'{day} {12 + i}:00:00+00:00'), -100.0)


def test_latching_is_still_the_default():
    """Existing backtests and the live mirror must not move."""
    book = Book(config=_config())
    _blow_the_day(book, '2026-02-03')
    assert book.halted_at is not None
    assert book.is_halted(pd.Timestamp('2026-03-01 00:00:00+00:00')) is True


def test_the_book_resumes_after_the_cooling_off_period():
    book = Book(config=_config(halt_resume_hours=24.0))
    _blow_the_day(book, '2026-02-03')
    assert book.halted_at is not None

    # Still shut inside the window.
    assert book.is_halted(pd.Timestamp('2026-02-04 10:00:00+00:00')) is True
    # Open again once it has elapsed.
    assert book.is_halted(pd.Timestamp('2026-02-05 12:00:00+00:00')) is False


def test_every_halt_is_counted_not_just_the_first():
    """`halted` says it happened once; the COUNT says whether it is a pattern."""
    book = Book(config=_config(halt_resume_hours=24.0))
    _blow_the_day(book, '2026-02-03')
    # Past the cooling-off, and a second bad day.
    book.settle({}, at=pd.Timestamp('2026-02-06 00:00:00+00:00'))
    _blow_the_day(book, '2026-02-06')

    assert len(book.halt_events) == 2
    assert all('daily loss limit' in reason for _, reason in book.halt_events)

    stats = summarise(book, windows_available=1000)
    assert stats.n_halts == 2
    assert 'HALTED 2x' in stats.summary()


def test_a_resumed_streak_does_not_immediately_re_halt():
    """A cleared halt forgives the streak, or the book latches again at once.

    The consecutive-loss counter walks back through every settlement. Without a
    baseline at the resume point, the losses that caused the halt are still
    there and the next loss re-trips it for ever.
    """
    config = _config(halt_resume_hours=24.0, max_consecutive_losses=3,
                     max_daily_loss_fraction=0.90)
    book = Book(config=config)
    for i in range(3):
        _settle(book, pd.Timestamp(f'2026-02-03 1{i}:00:00+00:00'), -5.0)
    assert len(book.halt_events) == 1

    # One more loss, well past the cooling-off.
    _settle(book, pd.Timestamp('2026-02-05 12:00:00+00:00'), -5.0)
    assert len(book.halt_events) == 1, 'the pre-halt streak must not count again'
