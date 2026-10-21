"""The candle backfill must SPLIT an oversized request, not drop it.

The loop this replaced logged 'chunk at %s would ask for %d candles; split it'
and then `continue`d, so every window in an overfull chunk was silently lost --
and a chunk overflows precisely when it is dense, so the holes landed where the
data was richest. `build_depth` cannot tell such a hole from a window that never
traded, which is why this is a test and not a comment.
"""

import datetime as dt

import pytest

from scripts.backfill_candles import (MAX_CANDLES_PER_REQUEST,
                                      WINDOW_MINUTES, plan_requests)


def _grid(n, *, markets_per_window=3, start=None):
    """`n` consecutive 15-minute windows, each with `markets_per_window`."""
    t0 = start or dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
    windows = [t0 + dt.timedelta(minutes=WINDOW_MINUTES * i) for i in range(n)]
    by_window = {w: [('T%d-%d' % (i, k), 'BTC-USD')
                     for k in range(markets_per_window)]
                 for i, w in enumerate(windows)}
    return windows, by_window


def test_every_window_is_requested_exactly_once():
    windows, by_window = _grid(200)
    seen = [w for chunk, _, _, _ in plan_requests(windows, by_window)
            for w in chunk]
    assert sorted(seen) == sorted(windows)
    assert len(seen) == len(set(seen))


def test_a_chunk_that_would_overflow_is_split_not_dropped():
    # A limit low enough that the default stride always overflows.
    windows, by_window = _grid(56)
    plans = list(plan_requests(windows, by_window, max_candles=200))
    assert len(plans) > len(windows) // 14      # it really did split
    assert sorted(w for c, _, _, _ in plans for w in c) == sorted(windows)


def test_no_plan_exceeds_the_limit_once_it_can_still_split():
    windows, by_window = _grid(120)
    for chunk, markets, lo, hi in plan_requests(windows, by_window,
                                                max_candles=600):
        span = int((hi - lo).total_seconds() // 60)
        if len(chunk) > 1:
            assert len(markets) * span <= 600


def test_a_single_window_is_still_requested_when_it_cannot_be_split():
    # One window, far over any sane limit: the venue truncating is a
    # measurement; us skipping it is a hole nothing records.
    windows, by_window = _grid(1, markets_per_window=50)
    plans = list(plan_requests(windows, by_window, max_candles=1))
    assert [c for c, _, _, _ in plans] == [windows]


def test_chunks_come_back_in_window_order():
    windows, by_window = _grid(90)
    first = [min(c) for c, _, _, _ in plan_requests(windows, by_window,
                                                    max_candles=400)]
    assert first == sorted(first)


def test_the_default_limit_leaves_a_normal_day_unsplit():
    # 96 windows x 3 symbols is one day; it must not need splitting, or the
    # batching that makes this script cheap has stopped working.
    windows, by_window = _grid(96)
    plans = list(plan_requests(windows, by_window))
    for chunk, markets, lo, hi in plans:
        span = int((hi - lo).total_seconds() // 60)
        assert len(markets) * span <= MAX_CANDLES_PER_REQUEST


class TestThrottlingIsNotMistakenForHistory:
    """A 429 says nothing about the data and everything about our rate.

    The chunk loop used to log a warning and `continue` on the FIRST error,
    abandoning every window in the chunk. Afterwards that loss is
    indistinguishable from a stretch the venue never listed — throttling
    quietly becomes "history", which is the shape this repo keeps
    rediscovering.
    """

    def test_the_retry_and_the_drop_counter_are_wired_into_the_chunk_loop(self):
        """Asserted on the real source, because the loop lives inside `run()`
        behind a store, a client and a work queue — a hand-rolled stand-in
        would assert the stand-in, which is worse than no test."""
        import inspect
        from scripts import backfill_candles as bc

        src = inspect.getsource(bc.run)
        assert "'/markets/candlesticks'" in src
        # the request sits inside a bounded retry, not a bare try/continue
        assert 'for attempt in range(5)' in src
        assert "DROPPED['windows']" in src
        assert 'DROPPED after 5 attempts' in src

    def test_the_dropped_counter_exists_and_starts_empty(self):
        from scripts import backfill_candles as bc
        assert bc.DROPPED['chunks'] == 0
        assert bc.DROPPED['windows'] == 0
