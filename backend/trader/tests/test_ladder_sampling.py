"""Tests that the ladder recorder keeps its sampling phase and sees the whole window.

Sleeping a fixed interval after each cycle makes the period `interval + work`,
so the phase drifts. The recorder sleeps to the next mark instead. It filters
markets by close time, not `status: 'open'`, which lags the market by up to 40s.
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
    (0, 25.0),  # start of the minute -> wait to :25
    (10, 15.0),  # mid -> wait to :25
    (24, 1.0 + 60.0),  # within a second of the mark -> take the next one
    (30, 55.0),  # past it -> next minute's :25
    (59, 26.0),
])
def test_it_sleeps_to_the_next_mark(now_s, expected):
    got = seconds_to_next_mark(_at(now_s), 60.0, 25.0)
    assert got == pytest.approx(expected, abs=0.01)


def test_the_phase_is_restored_after_a_slow_cycle():
    """A slow cycle that ends at :40 waits 45s and returns to :25."""
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
    """The loop sleeps through `seconds_to_next_mark`, not a fixed `sleep(interval)`."""
    src = inspect.getsource(rl.run)
    assert 'seconds_to_next_mark(' in src
    assert 'asyncio.sleep(args.interval)' not in src
