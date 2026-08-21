"""Epoch <-> datetime conversion, in one place, for naive UTC.

Every timestamp in the store is naive UTC. That convention is fine, but it makes
the two stdlib calls you reach for silently wrong, because both consult the local
zone when the datetime carries no tzinfo:

    datetime.fromtimestamp(ms / 1000)        # decodes into LOCAL time
    naive_utc_datetime.timestamp()           # encodes as if it were LOCAL time

`docker-compose.yml` sets `TZ=America/New_York`, so both were off by 4-5 hours
depending on the season, and the offset *changes* at a DST boundary, which puts a
duplicated hour and a missing hour in the middle of a long history.

Two things made this hard to see. The Coinbase candle path already passed
`tz=timezone.utc` and was correct, so right and wrong timestamps landed in the
same tables from different sources. And the validator treats a naive timestamp as
UTC and only rejects times in the *future*, so a negative offset just made bars
look older than they were.

The encode direction had a second, worse consequence: the funding backfill window
is `(last_seen, now)`, and shifting it forward by the offset pushed the whole
request into the future once the history was current, so funding silently stopped
advancing after the initial backfill.

Use these four functions rather than the stdlib ones anywhere an exchange epoch
meets a stored datetime.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

__all__ = [
    'ensure_naive_utc',
    'epoch_seconds_to_naive_utc',
    'epoch_millis_to_naive_utc',
    'naive_utc_to_epoch_seconds',
    'naive_utc_to_epoch_millis',
    'utc_now',
]


def ensure_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Drop the tzinfo, converting to UTC first if there is one."""
    if dt is None:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def utc_now() -> datetime:
    """Now, as naive UTC. The non-deprecated spelling of `datetime.utcnow()`."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def epoch_seconds_to_naive_utc(seconds: float) -> datetime:
    """An exchange epoch in seconds, as naive UTC."""
    return datetime.fromtimestamp(float(seconds), tz=timezone.utc).replace(tzinfo=None)


def epoch_millis_to_naive_utc(millis: float) -> datetime:
    """An exchange epoch in milliseconds, as naive UTC."""
    return epoch_seconds_to_naive_utc(float(millis) / 1000.0)


def naive_utc_to_epoch_seconds(dt: datetime) -> int:
    """A naive-UTC datetime as a whole-second epoch.

    `dt.timestamp()` would read the local zone. Attaching UTC first is what makes
    the round trip with `epoch_seconds_to_naive_utc` exact.
    """
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.timestamp())


def naive_utc_to_epoch_millis(dt: datetime) -> int:
    """A naive-UTC datetime as a millisecond epoch."""
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.timestamp() * 1000)
