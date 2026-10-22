"""Per-component liveness, because `supervise` cannot see a hang.

**The container healthcheck proved a TCP connect to Postgres and nothing
consumed it.** It never touched the loop, the venue, the stream or the model;
nothing `depends_on` it and there is no autoheal, so under
`restart: unless-stopped` even a FAILING check changed nothing. The process
could be wedged, abstaining on every window, or scoring all-NaN features while
`docker compose ps` read `Up (healthy)` — verified, five hours of it.

`supervise` restarts a component that raises or returns, and handles both
correctly. What it cannot distinguish is a coroutine legitimately awaiting from
one that will await forever: the documented websocket deadlock, and the
store-sync subprocess that had no timeout until today. Both presented as a
healthy container with one log line missing.

So liveness is signalled from INSIDE each loop, at the point where the loop has
demonstrably done its job, and the healthcheck asserts the files are recent.
A stamp written anywhere else — at startup, by the supervisor, by a timer —
would prove only that the process exists, which is what the old check proved.

Deliberately dependency-free and failure-tolerant: a heartbeat that can break
the thing it measures is worse than none, so every write is best-effort.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Where the stamps live. A directory, one file per component.
#: Under `data/`, not `logs/`. The compose file mounts `/app/logs` as a named
#: volume the container writes as root, so the host user cannot write there and
#: a host-side check could never read a stamp it had just written. `data/` is
#: the bind mount both sides already share.
ROOT = Path(os.getenv('HEARTBEAT_ROOT')
            or Path(__file__).resolve().parents[1] / 'data' / 'alive')

#: How stale a component's stamp may be before it is considered dead, in
#: seconds. Generous multiples of each component's own period: the point is to
#: catch a wedge, not to page on one slow cycle.
MAX_AGE: dict[str, float] = {
    'trade': 300.0,        # a 60s cycle; five missed is not a blip
    'stream': 180.0,       # continuous, flushes often
    'pm_stream': 300.0,
    'spot_stream': 180.0,  # heartbeats every second; stamps every ~5s
    'index': 180.0,        # stamps every ~5s while a proxy is being produced
    # The venue's own settlement index, ~1 frame/s/symbol. Ninety seconds of
    # silence is ninety missed prints on three sockets, which is the socket
    # gone rather than a quiet market -- the index does not stop printing.
    'index_stream': 90.0,
    'ladder': 300.0,       # 60s cadence
    # The public tape. A 60s poll, and it stamps at the END of each cycle
    # whether or not the cycle produced a row -- a 15-minute market that never
    # printed in a minute is normal, so silence is not evidence of a wedge and
    # must not be the thing this measures.
    #
    # **Until the trader is restarted this reads STALE, and that is correct.**
    # This directory is bind-mounted into the live container, so the file is
    # current while the running process read its component list hours ago and
    # is not stamping this. A declared component that is not running is a true
    # unhealthy, not a false alarm -- the fix is the restart, not a quieter
    # check.
    'trades': 300.0,
    'pm_ladder': 300.0,
    'implied_vol': 600.0,  # slower, and tolerant of venue gaps
    # Reads the in-process stream cache every ~5s, so 120s of silence is
    # twenty-four missed passes — that is the loop stopped, not a slow one.
    # It marks only while something is OPEN, and touches each pass regardless
    # so an idle book is not mistaken for a dead task.
    'marks': 120.0,
    # The per-minute book sampler that feeds `book_flow`. A 60s period, so
    # three missed minutes is a real stall -- and a stall here is not loud:
    # `book_flow` simply scores NaN and the live model silently stops being
    # the one the gates measured, which is the failure this component was
    # added to end.
    'book_minutes': 200.0,
    'store_sync': 7200.0,  # hourly, so two missed passes
}


def touch(name: str) -> None:
    """Record that `name` completed a cycle. Best effort, never raises."""
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        (ROOT / name).write_text(str(time.time()))
    except OSError as exc:                                # noqa: BLE001
        logger.debug('could not write the %s heartbeat: %s', name, exc)


def age(name: str, *, now: Optional[float] = None) -> Optional[float]:
    """Seconds since `name` last completed a cycle, or None if never seen."""
    path = ROOT / name
    try:
        stamp = float(path.read_text().strip())
    except (OSError, ValueError):
        return None
    return max(0.0, (time.time() if now is None else now) - stamp)


def stale(names, *, now: Optional[float] = None) -> list[tuple[str, Optional[float], float]]:
    """`(name, age, limit)` for every component that is late or never seen.

    A component that has NEVER written is reported, not excused: "not started
    yet" and "died before its first cycle" look identical from outside, and the
    second is the one worth catching.
    """
    late = []
    for name in names:
        limit = MAX_AGE.get(name, 600.0)
        seen = age(name, now=now)
        if seen is None or seen > limit:
            late.append((name, seen, limit))
    return late
