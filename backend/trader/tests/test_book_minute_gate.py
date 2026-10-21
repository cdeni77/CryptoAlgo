"""The sampler must not wait for a decision before reading the book.

`sample_once` reads an in-process dict and makes no venue call, so gating it
behind `gate.idle()` buys nothing. It costs the minute: a decision cycle runs
far longer than `BOOK_SAMPLE_MAX_AGE_SECONDS`, so the sample lands stale and
every row for that minute is refused.

The refused minutes are the DECISION minutes (3, 6, 9, 12), and every
`book_flow` feature looks back one or three minutes onto exactly those.
"""

from __future__ import annotations

import datetime as dt
import inspect

import pandas as pd

from livetrade import features as lf
from scripts import record_book_minutes as rbm

UTC = dt.timezone.utc
OPEN = dt.datetime(2026, 10, 21, 21, 0, tzinfo=UTC)
LEVELS = [(0.40, 100.0)], [(0.58, 120.0)]


def _record(offset, age=0.2):
    yes, no = LEVELS
    return lf.record_book_minute('BTC-USD', window_open=OPEN, offset=offset,
                                 yes_levels=yes, no_levels=no, age_seconds=age)


def _scored(offset):
    return pd.DataFrame({'symbol': ['BTC-USD'], 'window_open': [pd.Timestamp(OPEN)],
                         'offset': [offset]})


def setup_function(_):
    lf._BOOK_HISTORY.clear()                                  # noqa: SLF001


def test_a_sample_later_than_the_freshness_bound_is_refused():
    assert _record(3, age=0.2) is True
    lf._BOOK_HISTORY.clear()                                  # noqa: SLF001
    assert _record(3, age=lf.BOOK_SAMPLE_MAX_AGE_SECONDS + 1.0) is False


def test_losing_the_decision_minute_kills_book_flow_three_minutes_later():
    """Offset 6 differences against offset 3 — a decision minute."""
    for off in (4, 5, 6):
        assert _record(off)
    # offset 3 is missing: the gate held the sampler past the bound
    row = lf.book_flow_row(_scored(6))
    assert row['mid_change_3'].isna().all()


def test_the_same_window_scores_when_the_decision_minute_survives():
    for off in (3, 4, 5, 6):
        assert _record(off)
    row = lf.book_flow_row(_scored(6))
    assert row['mid_change_3'].notna().any()


def test_the_sampler_does_not_wait_for_a_decision_before_reading():
    """`sample_once` touches no venue, so the gate only costs freshness."""
    src = inspect.getsource(rbm.run)
    before = src.split('sample_once')[0]
    assert 'gate.idle()' not in before, (
        'the gate is awaited before sampling; a decision cycle outlasts '
        'BOOK_SAMPLE_MAX_AGE_SECONDS and the minute is refused')
