"""Epoch <-> naive-UTC datetime conversion.

Every timestamp in the store is naive UTC. The stdlib calls read the local zone
when a datetime has no tzinfo, and the containers set a non-UTC `TZ`:

    datetime.fromtimestamp(ms / 1000)        # decodes into LOCAL time
    naive_utc_datetime.timestamp()           # encodes as if it were LOCAL time

Use these functions wherever an exchange epoch meets a stored datetime.
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

    Attaches UTC first, because `dt.timestamp()` reads the local zone.
    """
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.timestamp())


def naive_utc_to_epoch_millis(dt: datetime) -> int:
    """A naive-UTC datetime as a millisecond epoch."""
    aware = dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
    return int(aware.timestamp() * 1000)
