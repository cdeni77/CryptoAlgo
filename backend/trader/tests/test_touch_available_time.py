"""The decision's book is stamped when it was USED, not when it was filed.

`_record_touch` runs after the orders, deliberately: writing the archive must
not delay a fill. `decision_age` bounds a `live_touch` row by
`available_time - event_time` when the row carries no age of its own, so the
write instant would put order-placement latency into that bound.

The bound is a fallback. A row that carries `quote_age_seconds` is judged on
that column and ignores `available_time` entirely.
"""

from __future__ import annotations

import inspect

import pandas as pd
import pytest

from core.quotes import decision_age
from livetrade import quotes as lq

WINDOW = pd.Timestamp('2026-10-22 13:00', tz='UTC')
EVENT = WINDOW + pd.Timedelta(minutes=3)
DECIDED = EVENT + pd.Timedelta(seconds=3.0)      # the decision instant
WRITTEN = EVENT + pd.Timedelta(seconds=5.4)      # after the orders


def age_for(available, quote_age=None):
    frame = pd.DataFrame([{
        'source': 'live_touch', 'event_time': EVENT,
        'available_time': available, 'quote_age_seconds': quote_age}])
    return float(decision_age(frame, 'stale', 3.0).iloc[0])


def test_the_row_is_stamped_with_the_decision_not_the_write():
    src = inspect.getsource(lq._record_touch)                 # noqa: SLF001
    assert 'pd.Timestamp(decided_at)' in src
    assert "'available_time': pd.Timestamp.now(tz='UTC')," not in src


def test_it_still_falls_back_when_no_decision_time_is_given():
    src = inspect.getsource(lq._record_touch)                 # noqa: SLF001
    assert 'else pd.Timestamp.now(tz=' in src


def test_the_write_stamp_inflates_the_fallback_bound():
    """What the fix prevents, on the rows that use the bound."""
    assert age_for(WRITTEN) == pytest.approx(5.4)
    assert age_for(DECIDED) == pytest.approx(3.0)
    assert age_for(DECIDED) < age_for(WRITTEN)


def test_a_row_with_its_own_age_ignores_available_time():
    """The bound is a FALLBACK. `_effective_age` prefers the column, so the
    stamp cannot move the age of a row that recorded one."""
    assert age_for(WRITTEN, quote_age=0.4) == pytest.approx(0.4)
    assert age_for(DECIDED, quote_age=0.4) == pytest.approx(0.4)


def test_the_quote_age_column_is_the_books_own_age_and_is_unchanged():
    class Q:
        book_time = (EVENT + pd.Timedelta(seconds=2.6)).timestamp()

    assert lq.quote_age_seconds(Q(), DECIDED) == pytest.approx(0.4, abs=1e-6)
