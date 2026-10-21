"""Rate limiting, failure containment, and the single-writer guarantee.

Predexon's 1 req/s limit is one bucket for the whole organisation, so
`SingleWriterLock` allows one collector at a time. `Breaker` stops a run during
a venue outage, so pending rows are not converted to `error`.
"""
from __future__ import annotations

import errno
import os
import threading
import time
from collections import deque


class RateLimiter:
    """A single token bucket, spacing calls to at most `per_second`.

    Thread-safe, because the limit is on requests, not connections, and
    fetches are transfer-bound. A slot is reserved under the lock; the sleep
    happens outside it, so callers do not serialise.
    """

    def __init__(self, per_second: float = 1.0):
        self.interval = 1.0 / float(per_second)
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> float:
        """Sleep until this caller's slot and return the slot's start time."""
        with self._lock:
            start = max(time.monotonic(), self._next_at)
            self._next_at = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        return start


class Breaker:
    """Trips when the trailing window is mostly failures.

    Only request failures count. An `empty` result is an answer, not a failure.
    It cannot trip before the window is full.
    """

    def __init__(self, threshold: float = 0.25, window: int = 40):
        self.threshold = threshold
        self.window = window
        self._recent: deque[bool] = deque(maxlen=window)

    def record(self, *, ok: bool) -> None:
        self._recent.append(bool(ok))

    @property
    def failure_rate(self) -> float:
        if not self._recent:
            return 0.0
        return sum(1 for good in self._recent if not good) / len(self._recent)

    @property
    def tripped(self) -> bool:
        if len(self._recent) < self.window:
            return False
        return self.failure_rate > self.threshold


class SingleWriterLock:
    """A pid lockfile, so two collectors cannot share one org-wide bucket.

    A dead owner's lock is taken over, so a crashed run can restart unattended.
    """

    def __init__(self, path: str):
        self.path = path
        self._held = False

    def _owner_alive(self) -> bool:
        try:
            with open(self.path) as handle:
                pid = int((handle.read() or '0').strip() or 0)
        except (OSError, ValueError):
            return False
        if pid <= 0:
            return False
        if pid == os.getpid():
            # Our own live lock. Not re-entrant: two loops in one process
            # share the bucket as badly as two processes.
            return True
        try:
            os.kill(pid, 0)
        except OSError as exc:
            return exc.errno == errno.EPERM       # alive, just not ours
        return True

    def __enter__(self):
        if os.path.exists(self.path) and self._owner_alive():
            raise RuntimeError(
                f'another collector holds {self.path}; the Predexon bucket is '
                f'org-wide and only one may run')
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.path, 'w') as handle:
            handle.write(str(os.getpid()))
        self._held = True
        return self

    def __exit__(self, *exc):
        if self._held:
            try:
                os.remove(self.path)
            except OSError:
                pass
            self._held = False
        return False
