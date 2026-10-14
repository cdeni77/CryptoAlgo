"""Tests for the collection rate limiter, circuit breaker and single-writer lock.

Predexon's rate limit is org-wide, so only one process may collect. The breaker pauses a
run during an outage, so an outage does not turn pending windows into errors.
"""

from __future__ import annotations

import os
import time

import pytest

from research.collect.orchestrator import Breaker, RateLimiter, SingleWriterLock


# -- rate limiter ------------------------------------------------------------

def test_the_limiter_spaces_requests_by_the_configured_interval():
    limiter = RateLimiter(per_second=20.0)  # 50ms apart
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
    """`empty` is an answer, not a failure, so it never trips the breaker."""
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(100):
        breaker.record(ok=True)  # empty is recorded as ok
    assert not breaker.tripped


def test_a_sustained_outage_trips_the_breaker():
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(40):
        breaker.record(ok=False)
    assert breaker.tripped


def test_a_short_burst_of_errors_does_not_trip_it():
    """A short burst of errors in a healthy run does not trip the breaker."""
    breaker = Breaker(threshold=0.25, window=40)
    for _ in range(40):
        breaker.record(ok=True)
    for _ in range(5):
        breaker.record(ok=False)
    assert not breaker.tripped


def test_the_breaker_only_considers_the_trailing_window():
    """Only the trailing window counts, so the breaker resets after the venue recovers."""
    breaker = Breaker(threshold=0.25, window=20)
    for _ in range(20):
        breaker.record(ok=False)
    assert breaker.tripped
    for _ in range(20):
        breaker.record(ok=True)
    assert not breaker.tripped


def test_the_breaker_needs_a_full_window_before_it_can_trip():
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
        pass  # must not raise


def test_a_stale_lock_from_a_dead_process_does_not_block_forever(tmp_path):
    """A lock file left by a dead process does not block a new run."""
    path = str(tmp_path / 'collect.lock')
    with open(path, 'w') as handle:
        handle.write('999999999')  # a pid that cannot exist
    with SingleWriterLock(path):
        pass  # must not raise


def test_the_limiter_is_safe_to_share_between_threads():
    """The limiter never gives two threads the same slot."""
    import threading
    limiter = RateLimiter(per_second=25.0)  # 40ms apart
    stamps, lock = [], threading.Lock()

    def worker():
        for _ in range(4):
            slot = limiter.wait()
            with lock:
                stamps.append(slot)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stamps.sort()
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    assert len(stamps) == 16
    # Checks the reserved slots, not wake-up times: a loaded machine can wake two
    # threads together.
    assert min(gaps) >= 0.040 - 1e-9, f'limiter double-granted a slot: {min(gaps):.4f}s'
