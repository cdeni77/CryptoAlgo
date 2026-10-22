"""A resting fill is booked the cycle it is seen, not at the next decision.

`livetrade/ledger.py` notices an unbooked venue position on EVERY cycle and
logs it. Booking only ran in the decision path, so a fill at +3 sat unbooked
until +6 while the log repeated "an order was filled and not booked" once a
minute, the dashboard showed a position short, and `exposure`,
`max_positions_per_window` and `max_window_exposure_fraction` were all
computed without it -- a position no risk limit binds on.
"""

from __future__ import annotations

import inspect

from scripts import live


def test_the_between_decisions_branch_books_fills():
    src = inspect.getsource(live.run_cycle)
    branch = src[src.index('if offset is None:'):]
    assert 'book_resting_fills' in branch, (
        'a cycle that notices an unbooked fill must book it')


def test_it_books_before_deciding_too():
    """The decision path keeps its own call: a fill must be booked before the
    cancel sweep can decide the order is gone."""
    src = inspect.getsource(live.run_cycle)
    assert src.count('book_resting_fills') >= 2


def test_booking_precedes_the_cancel_sweep():
    """An order that FILLED must be booked, not written off as cancelled."""
    src = inspect.getsource(live.run_cycle)
    assert src.index('book_resting_fills') < src.index('cancel_stale_resting')


def test_it_stays_out_of_the_decision_latency_path():
    """The between-decisions call must be inside the `offset is None` branch.
    Anything added between the book read and `decided_at` is charged to
    `quote_age_seconds` and judged against a 5-second bound."""
    src = inspect.getsource(live.run_cycle)
    branch_at = src.index('if offset is None:')
    first_call = src.index('book_resting_fills')
    assert first_call > branch_at, (
        'the extra call must come after the between-decisions guard, not '
        'before it in the shared path')
