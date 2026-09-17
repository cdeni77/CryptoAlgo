"""The ladder recorder must hold its phase and see the whole window.

Two defects, both measured on the real store.

**Phase drift.** `asyncio.sleep(interval)` at the END of a cycle makes the real
period `interval + work`, so the sampling phase walks through the minute. Over
2026-09-15 it swept 21.0s -> 41.3s, p5/p50/p95 of 5.3/32.9/57.5s — essentially
uniform rather than pinned where `run_live.align_to_phase` put it at startup.
That voids the phase table in `run_live.COMPONENTS` within the first hour, and
whole minutes go unsampled: 979 two-minute steps and 40 longer ones over seven
days.

**`status: 'open'`.** `record_stream._trading` documents that `status` lags the
market by up to forty seconds and was changed away from it for that reason; the
ladder recorder still asked for it. Measured in 10s buckets inside the first
minute of a window: 8, 73, 170, 323, 332, 324 rows against ~340/bucket in
minute 1 — the first twenty seconds were ~95% and ~78% missing, at the offset
where the edge is largest.
"""

from __future__ import annotations

import inspect
from datetime import datetime, timezone

import pytest

import scripts.record_ladder as rl
from scripts.record_ladder import seconds_to_next_mark


def _at(second, micro=0):
    return datetime(2026, 9, 16, 12, 30, second, micro, tzinfo=timezone.utc)


@pytest.mark.parametrize('now_s,expected', [
    (0, 25.0),    # start of the minute -> wait to :25
    (10, 15.0),   # mid -> wait to :25
    (24, 1.0 + 60.0),  # within a second of the mark -> take the NEXT one
    (30, 55.0),   # past it -> next minute's :25
    (59, 26.0),
])
def test_it_sleeps_to_the_next_mark(now_s, expected):
    got = seconds_to_next_mark(_at(now_s), 60.0, 25.0)
    assert got == pytest.approx(expected, abs=0.01)


def test_the_phase_is_restored_after_a_slow_cycle():
    """The whole point: an overrun must not shift every later sample. A cycle
    that finishes at :40 waits 45s and lands back on :25."""
    assert seconds_to_next_mark(_at(40), 60.0, 25.0) == pytest.approx(45.0)


def test_it_never_returns_a_spin():
    for second in range(60):
        assert seconds_to_next_mark(_at(second), 60.0, 25.0) > 1.0


def test_a_phase_beyond_the_period_wraps():
    assert seconds_to_next_mark(_at(0), 60.0, 85.0) == pytest.approx(25.0)


def test_the_recorder_does_not_ask_for_status_open():
    src = inspect.getsource(rl.run)
    assert "'status': 'open'" not in src, (
        "`status` lags the market by up to 40s, so the replacement market is "
        "invisible for the first part of every window — see "
        "record_stream._trading, which was changed away from it for this reason")
    assert 'min_close_ts' in src, 'closing time is the honest filter'


def test_the_loop_sleeps_through_the_helper():
    """A seam test: the bug was an unaligned `sleep(interval)` in the loop, and
    the helper is new — testing it alone proves nothing about the caller."""
    src = inspect.getsource(rl.run)
    assert 'seconds_to_next_mark(' in src
    assert 'asyncio.sleep(args.interval)' not in src
