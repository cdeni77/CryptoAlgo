"""Tests for `data_collection.timeutil`, the one conversion between epochs and stored
naive-UTC datetimes.

Tests run under a non-UTC `TZ`, since under `TZ=UTC` the stdlib mistakes do not show.
"""

from __future__ import annotations

import importlib
import os
import re
import time
from datetime import datetime, timedelta, timezone

import pytest

from data_collection.timeutil import (
    ensure_naive_utc,
    epoch_millis_to_naive_utc,
    epoch_seconds_to_naive_utc,
    naive_utc_to_epoch_millis,
    naive_utc_to_epoch_seconds,
    utc_now,
)

# 2026-01-01T00:00:00Z, and a summer instant so the DST offset differs.
WINTER_EPOCH = 1_767_225_600
SUMMER_EPOCH = 1_782_864_000  # 2026-07-01T00:00:00Z


@pytest.fixture
def eastern_tz():
    """Runs the test with `TZ=America/New_York`, then restores the previous zone."""
    previous = os.environ.get('TZ')
    os.environ['TZ'] = 'America/New_York'
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop('TZ', None)
        else:
            os.environ['TZ'] = previous
        time.tzset()


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


def test_seconds_decode_as_utc_not_local(eastern_tz):
    assert epoch_seconds_to_naive_utc(WINTER_EPOCH) == datetime(2026, 1, 1, 0, 0)
    # What the stdlib call gives under the local zone.
    assert datetime.fromtimestamp(WINTER_EPOCH) == datetime(2025, 12, 31, 19, 0)


def test_millis_decode_as_utc_not_local(eastern_tz):
    assert epoch_millis_to_naive_utc(WINTER_EPOCH * 1000) == datetime(2026, 1, 1, 0, 0)


def test_the_offset_does_not_move_across_a_dst_boundary(eastern_tz):
    winter = epoch_seconds_to_naive_utc(WINTER_EPOCH)
    summer = epoch_seconds_to_naive_utc(SUMMER_EPOCH)

    assert winter == datetime(2026, 1, 1, 0, 0)
    assert summer == datetime(2026, 7, 1, 0, 0)
    # Exactly the real elapsed time, with no daylight-saving artefact.
    assert (summer - winter).total_seconds() == SUMMER_EPOCH - WINTER_EPOCH


def test_decoded_values_carry_no_tzinfo(eastern_tz):
    """Decoded values are naive, matching the store's convention."""
    assert epoch_seconds_to_naive_utc(WINTER_EPOCH).tzinfo is None
    assert epoch_millis_to_naive_utc(WINTER_EPOCH * 1000).tzinfo is None


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def test_seconds_encode_from_naive_utc(eastern_tz):
    assert naive_utc_to_epoch_seconds(datetime(2026, 1, 1, 0, 0)) == WINTER_EPOCH
    # The stdlib call adds 18,000 seconds of local offset.
    assert datetime(2026, 1, 1, 0, 0).timestamp() == WINTER_EPOCH + 18_000


def test_millis_encode_from_naive_utc(eastern_tz):
    assert naive_utc_to_epoch_millis(datetime(2026, 1, 1, 0, 0)) == WINTER_EPOCH * 1000


def test_an_aware_datetime_encodes_to_the_same_instant(eastern_tz):
    naive = datetime(2026, 1, 1, 0, 0)
    aware = naive.replace(tzinfo=timezone.utc)

    assert naive_utc_to_epoch_seconds(aware) == naive_utc_to_epoch_seconds(naive)


@pytest.mark.parametrize('epoch', [WINTER_EPOCH, SUMMER_EPOCH, 0, 1_000_000_000])
def test_the_round_trip_is_exact(eastern_tz, epoch):
    assert naive_utc_to_epoch_seconds(epoch_seconds_to_naive_utc(epoch)) == epoch


def test_an_incremental_window_does_not_land_in_the_future(eastern_tz):
    """An incremental window `(now - 1h, now)` encodes to the past, not the future."""
    now = utc_now()
    start, end = now - timedelta(hours=1), now
    real_now = int(datetime.now(timezone.utc).timestamp())

    assert naive_utc_to_epoch_seconds(start) <= real_now + 1
    assert naive_utc_to_epoch_seconds(end) <= real_now + 1

    # What the stdlib call does instead.
    assert start.timestamp() > real_now + 3_600


# ---------------------------------------------------------------------------
# ensure_naive_utc and utc_now
# ---------------------------------------------------------------------------


def test_ensure_naive_utc_converts_before_stripping(eastern_tz):
    eastern = datetime(2026, 1, 1, 0, 0, tzinfo=timezone(timedelta(hours=-5)))

    assert ensure_naive_utc(eastern) == datetime(2026, 1, 1, 5, 0)


def test_ensure_naive_utc_passes_through_naive_and_none(eastern_tz):
    assert ensure_naive_utc(datetime(2026, 1, 1)) == datetime(2026, 1, 1)
    assert ensure_naive_utc(None) is None


def test_utc_now_tracks_utc_not_the_local_zone(eastern_tz):
    delta = abs((utc_now() - datetime.now(timezone.utc).replace(tzinfo=None)).total_seconds())

    assert delta < 5, 'utc_now() is reading the local zone'


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------


def test_no_connector_decodes_an_epoch_without_a_zone(eastern_tz):
    """No connector decodes an epoch without a real timezone; there must be one conversion path."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in [
        *(root / 'data_collection').glob('*.py'),
        root / 'scripts' / 'scrape.py',
    ]:
        if path.name == 'timeutil.py':
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split('#')[0]
            if 'utcfromtimestamp' in code:
                offenders.append(f'{path.name}:{number} utcfromtimestamp')
            elif 'fromtimestamp(' in code and not re.search(r'tz\s*=\s*(?!None)\S', code):
                # `tz=None` contains 'tz=' but reads the local zone, so a substring
                # check is not enough.
                offenders.append(f'{path.name}:{number} fromtimestamp without a real tz')
            elif '.timestamp()' in code and 'naive_utc_to_epoch' not in code:
                offenders.append(f'{path.name}:{number} .timestamp() on a naive value')

    assert not offenders, (
        'these read the local zone; use data_collection.timeutil instead:\n  '
        + '\n  '.join(offenders)
    )


def test_every_module_that_uses_a_helper_imports_it(eastern_tz):
    """Every module that calls a timeutil helper imports it.

    A missing import fails only at call time, and callers that catch every exception
    would hide it. The check is derived from the source, not a fixed list.
    """
    from pathlib import Path

    helpers = importlib.import_module('data_collection.timeutil')
    root = Path(__file__).resolve().parents[1]

    problems = []
    for path in [
        *(root / 'data_collection').glob('*.py'),
        *(root / 'scripts').glob('*.py'),
    ]:
        if path.name in ('timeutil.py', '__init__.py'):
            continue
        text = path.read_text()
        used = [name for name in helpers.__all__ if f'{name}(' in text]
        if not used:
            continue

        package = 'data_collection' if path.parent.name == 'data_collection' else 'scripts'
        module = importlib.import_module(f'{package}.{path.stem}')
        for name in used:
            if getattr(module, name, None) is None:
                problems.append(f'{path.parent.name}/{path.name} calls {name}() without importing it')

    assert not problems, '\n'.join(problems)


def test_the_in_memory_queue_can_publish(eastern_tz):
    """`InMemoryQueue.publish` works; it calls a timeutil helper, and its callers
    swallow exceptions.
    """
    import asyncio

    from data_collection.queue import InMemoryQueue

    queue = InMemoryQueue()
    asyncio.run(queue.publish('test-channel', {'value': 1}))
