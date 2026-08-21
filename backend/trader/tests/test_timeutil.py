"""Exchange epochs meet stored datetimes in exactly one place.

Every timestamp in the store is naive UTC, which makes both stdlib calls you
reach for silently wrong: `datetime.fromtimestamp(x)` decodes into the *local*
zone, and `naive.timestamp()` encodes as if the value were *local*. Compose sets
`TZ=America/New_York`, so both were off by 4-5 hours, and the offset changes at a
DST boundary — putting a duplicated hour and a missing hour inside a long
history.

Two things hid it. The Coinbase candle path already passed `tz=timezone.utc` and
was correct, so right and wrong timestamps arrived in the same tables from
different sources. And the validator treats naive as UTC and only rejects times
in the *future*, so a negative offset just made bars look older.

The encode direction was worse than an offset: the funding backfill window is
`(last_seen, now)`, and shifting it forward pushed the whole request into the
future once the history was current, so funding stopped advancing after the
initial backfill and logged only "No funding rates found".

Every test here runs under a deliberately non-UTC `TZ`, because under `TZ=UTC`
the broken and correct implementations are indistinguishable.
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
    """Run the body with the container's configured zone, then restore."""
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
    # What the bug produced, for the record.
    assert datetime.fromtimestamp(WINTER_EPOCH) == datetime(2025, 12, 31, 19, 0)


def test_millis_decode_as_utc_not_local(eastern_tz):
    assert epoch_millis_to_naive_utc(WINTER_EPOCH * 1000) == datetime(2026, 1, 1, 0, 0)


def test_the_offset_does_not_move_across_a_dst_boundary(eastern_tz):
    """A varying offset is what put duplicate and missing hours mid-history."""
    winter = epoch_seconds_to_naive_utc(WINTER_EPOCH)
    summer = epoch_seconds_to_naive_utc(SUMMER_EPOCH)

    assert winter == datetime(2026, 1, 1, 0, 0)
    assert summer == datetime(2026, 7, 1, 0, 0)
    # Exactly the real elapsed time, with no daylight-saving artefact.
    assert (summer - winter).total_seconds() == SUMMER_EPOCH - WINTER_EPOCH


def test_decoded_values_carry_no_tzinfo(eastern_tz):
    """The store's convention is naive; a tz-aware value would compare wrongly."""
    assert epoch_seconds_to_naive_utc(WINTER_EPOCH).tzinfo is None
    assert epoch_millis_to_naive_utc(WINTER_EPOCH * 1000).tzinfo is None


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def test_seconds_encode_from_naive_utc(eastern_tz):
    assert naive_utc_to_epoch_seconds(datetime(2026, 1, 1, 0, 0)) == WINTER_EPOCH
    # The bug: 18,000 seconds of local-zone offset added to the request window.
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
    """The concrete failure: funding stopped advancing after the first backfill.

    With the window at `(now - 1h, now)`, adding the local offset to both bounds
    pushed the entire request past the present, so the exchange had nothing to
    return and the log said only "No funding rates found".
    """
    now = utc_now()
    start, end = now - timedelta(hours=1), now
    real_now = int(datetime.now(timezone.utc).timestamp())

    assert naive_utc_to_epoch_seconds(start) <= real_now + 1
    assert naive_utc_to_epoch_seconds(end) <= real_now + 1

    # And what the bug did instead.
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
    """A new raw `fromtimestamp` would reintroduce the bug in one line.

    Two implementations of this conversion is how right and wrong timestamps came
    to sit in the same table, so the property worth pinning is that there is
    exactly one.
    """
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in [
        *(root / 'data_collection').glob('*.py'),
        root / 'scripts' / 'run_pipeline.py',
    ]:
        if path.name == 'timeutil.py':
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split('#')[0]
            if 'utcfromtimestamp' in code:
                offenders.append(f'{path.name}:{number} utcfromtimestamp')
            elif 'fromtimestamp(' in code and not re.search(r'tz\s*=\s*(?!None)\S', code):
                # `tz=None` contains 'tz=' and reads the local zone — it *is* the
                # bug, and a substring check for 'tz=' let it through.
                offenders.append(f'{path.name}:{number} fromtimestamp without a real tz')
            elif '.timestamp()' in code and 'naive_utc_to_epoch' not in code:
                offenders.append(f'{path.name}:{number} .timestamp() on a naive value')

    assert not offenders, (
        'these read the local zone; use data_collection.timeutil instead:\n  '
        + '\n  '.join(offenders)
    )


def test_every_module_that_uses_a_helper_imports_it(eastern_tz):
    """A name used without an import is a NameError at call time, not import time.

    The narrower version of this test hardcoded three modules, and the sweep that
    replaced `datetime.utcnow()` with `utc_now()` touched a fourth —
    `data_collection/queue.py` — without adding the import. Nothing failed on
    import; `InMemoryQueue.publish` raised `NameError` on every call, and the
    callers wrap it in `except Exception`, so real-time collection logged an error
    per tick and collected nothing.

    So the property is derived from the source rather than a list: any module
    naming one of these helpers must be able to resolve it.
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
    """The concrete call the missing import broke.

    `publish` stamps a message with the current time, so it exercises the helper
    that was unresolvable. Its callers in `pipeline` swallow every exception, so
    only calling it proves anything.
    """
    import asyncio

    from data_collection.queue import InMemoryQueue

    queue = InMemoryQueue()
    asyncio.run(queue.publish('test-channel', {'value': 1}))
