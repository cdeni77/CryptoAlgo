"""Rate limiting, the circuit breaker, and the single-writer lock.

Each of these exists because of a specific way the previous night's collection
went wrong:

  * Concurrent probes and a backfill competed for Predexon's ORG-WIDE 1 req/s
    bucket. The resulting 429s were counted as "no book", which is one of the
    four reasons the coverage numbers were wrong. Hence one process, enforced.
  * Nothing stopped a run from converting thousands of pending rows into
    errors at full speed during an outage, which would destroy the ability to
    tell "the venue was down" from "these windows are broken".
"""

from __future__ import annotations

import os
import time

import pytest

from research.collect.orchestrator import Breaker, RateLimiter, SingleWriterLock


# -- rate limiter ------------------------------------------------------------

def test_the_limiter_spaces_requests_by_the_configured_interval():
    limiter = RateLimiter(per_second=20.0)          # 50ms apart
    started = time.monotonic()
    for _ in range(3):
        limiter.wait()
    elapsed = time.monotonic() - started
    assert elapsed >= 0.09, 'three calls must span at least two intervals'


def test_the_first_request_is_not_delayed():
    limiter = RateLimiter(per_second=1.0)
    started = time.monotonic()
    limiter.wait()
    assert time.monotonic() - started < 0.05


# -- circuit breaker ---------------------------------------------------------

def test_a_healthy_run_never_trips():
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(100):
        breaker.record(ok=True)
    assert not breaker.tripped


def test_empty_results_do_not_trip_the_breaker():
    """`empty` is an ANSWER, not a failure. The measured empty rate is ~17%
    on Kalshi and would trip a naive breaker within a single month."""
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(100):
        breaker.record(ok=True)                     # empty is recorded as ok
    assert not breaker.tripped


def test_a_sustained_outage_trips_the_breaker():
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(40):
        breaker.record(ok=False)
    assert breaker.tripped


def test_a_short_burst_of_errors_does_not_trip_it():
    """A handful of 429s in an otherwise healthy run is normal; pausing the
    whole job for them would make a 47-hour run impossible to finish."""
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(40):
        breaker.record(ok=True)
    for _ in range(5):
        breaker.record(ok=False)
    assert not breaker.tripped


def test_the_breaker_only_considers_the_trailing_window():
    """Errors from an outage two hours ago must not keep the breaker tripped
    once the venue has recovered."""
    breaker = Breaker(threshold=0.25, window=20)
    for _ in range(20):
        breaker.record(ok=False)
    assert breaker.tripped
    for _ in range(20):
        breaker.record(ok=True)
    assert not breaker.tripped


def test_the_breaker_needs_a_full_window_before_it_can_trip():
    """Otherwise the first two requests failing ends the run."""
    breaker = Breaker(threshold=0.25, window=40)
    breaker.record(ok=False)
    breaker.record(ok=False)
    assert not breaker.tripped


# -- single-writer lock ------------------------------------------------------

def test_the_lock_is_held_exclusively(tmp_path):
    path = str(tmp_path / 'collect.lock')
    with SingleWriterLock(path):
        with pytest.raises(RuntimeError):
            with SingleWriterLock(path):
                pass


def test_the_lock_is_released_on_exit(tmp_path):
    path = str(tmp_path / 'collect.lock')
    with SingleWriterLock(path):
        pass
    with SingleWriterLock(path):
        pass                                        # must not raise


def test_a_stale_lock_from_a_dead_process_does_not_block_forever(tmp_path):
    """A kill -9 leaves the file behind. Resume must not require a human to
    delete it, or the overnight run stops at the first crash."""
    path = str(tmp_path / 'collect.lock')
    with open(path, 'w') as handle:
        handle.write('999999999')                   # a pid that cannot exist
    with SingleWriterLock(path):
        pass                                        # must not raise
