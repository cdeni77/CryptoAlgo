"""Per-component liveness stamps, because `supervise` cannot see a hang.

Each loop writes its stamp from inside, at the point where it has done its job.
The healthcheck asserts the stamps are recent. Every write is best effort.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

#: Directory of stamps, one file per component.
#: Under `data/`, not `logs/`: only `data/` is a bind mount that both the
#: container and the host can write.
ROOT = Path(os.getenv('HEARTBEAT_ROOT')
            or Path(__file__).resolve().parents[1] / 'data' / 'alive')

#: Maximum stamp age per component, in seconds.
#: Generous multiples of each period: the goal is to catch a wedge, not one slow cycle.
MAX_AGE: dict[str, float] = {
    'trade': 300.0,        # 60s cycle
    'stream': 180.0,       # continuous, flushes often
    'pm_stream': 300.0,
    'spot_stream': 180.0,  # stamps every ~5s
    'index': 180.0,        # stamps every ~5s while a proxy is being produced
    'ladder': 300.0,       # 60s cadence
    'pm_ladder': 300.0,
    'implied_vol': 600.0,  # slower, and tolerant of venue gaps
    'store_sync': 7200.0,  # hourly; two missed passes
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

    A component that never wrote is reported: "not started" and "died before
    its first cycle" look the same from outside.
    """
    late = []
    for name in names:
        limit = MAX_AGE.get(name, 600.0)
        seen = age(name, now=now)
        if seen is None or seen > limit:
            late.append((name, seen, limit))
    return late
