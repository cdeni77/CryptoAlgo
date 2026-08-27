"""Rate limiting, failure containment, and the single-writer guarantee.

These three are the parts of collection that have nothing to do with either
venue's API, and everything to do with not repeating the previous night:

  * Predexon's 1 req/s is an ORG-WIDE bucket. Running probes alongside a
    backfill made them throttle each other, and the resulting 429s were
    recorded as "no book" — one of four reasons the coverage numbers came out
    wrong. `SingleWriterLock` makes "one collector at a time" enforced rather
    than remembered.
  * A venue outage, left alone, marches through thousands of pending rows
    converting them to `error` at full speed. Afterwards nothing distinguishes
    "the venue was down for two hours" from "these particular windows are
    broken". `Breaker` stops the run instead.
"""

from __future__ import annotations

import errno
import os
import threading
import time
from collections import deque


class RateLimiter:
    """A single token bucket, spacing calls to at most `per_second`.

    Thread-safe, because the limit is on REQUESTS and not on connections.
    Fetches here are transfer-bound — a window with a large book takes 1.5-5.4s
    of which one request is issued, i.e. as little as 0.19 req/s against a
    1 req/s budget — so several fetches can run at once and still stay inside
    the limit, provided no two of them are handed the same slot.

    The slot is reserved under the lock and the sleeping happens outside it.
    Holding the lock while sleeping would serialise the callers and give back
    exactly the concurrency this exists to allow.
    """

    def __init__(self, per_second: float = 1.0):
        self.interval = 1.0 / float(per_second)
        self._next_at = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            start = max(time.monotonic(), self._next_at)
            self._next_at = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class Breaker:
    """Trips when the trailing window is mostly failures.

    Only genuine request FAILURES count against it. An `empty` result is an
    answer — the measured empty rate is ~17% on Kalshi — and counting those
    would trip the breaker inside the first month of a healthy run, which is
    the same empty-vs-error conflation this whole design exists to prevent.

    Requires a full window before it can trip, so two unlucky requests at
    startup cannot end a 47-hour job.
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

    A stale lock left by a killed process must not block a restart: the
    overnight run has to survive a crash without a human deleting a file, so
    the pid is checked and a dead owner's lock is taken over.
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
            # Our own live lock. Not re-entrant on purpose: acquiring twice
            # means two collection loops in one process, which shares the
            # org-wide bucket just as badly as two processes would.
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
