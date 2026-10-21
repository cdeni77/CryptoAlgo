"""The freshness measurement, published for a process that cannot make it.

`core.freshness` can only be run where pyarrow and the store are — the API has
neither. So the document written here is the only thing standing between a dead
feed and a dashboard that renders it as healthy, and what matters about it is
the shape: absolute timestamps the reader ages against its own clock, and
`None` rather than `False` wherever nothing was measured.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

from core import collection_status

NOW = datetime(2026, 10, 21, 12, 0, tzinfo=timezone.utc)


def _write(root, dataset, newest, *, column='event_time', rows=3):
    path = root / dataset / 'venue=kalshi' / 'symbol=BTC-USD'
    path.mkdir(parents=True, exist_ok=True)
    stamps = [newest - timedelta(minutes=i) for i in range(rows)][::-1]
    pd.DataFrame({column: pd.to_datetime(stamps, utc=True),
                  'value': range(rows)}).to_parquet(path / 'data.parquet',
                                                    index=False)


def test_it_publishes_an_absolute_instant_not_a_frozen_age(tmp_path):
    """An age written into a file is wrong by however long the file sits
    there. The reader must be able to recompute it."""
    _write(tmp_path, 'venue_ladder', NOW - timedelta(minutes=2))
    document = collection_status.snapshot(str(tmp_path),
                                          datasets=['venue_ladder'], now=NOW)

    row = document['datasets'][0]
    assert row['newest_event_time'].startswith('2026-10-21T11:58')
    assert row['age_seconds'] == 120.0
    assert row['stale'] is False


def test_a_dead_recorder_is_marked_stale(tmp_path):
    _write(tmp_path, 'venue_ladder', NOW - timedelta(days=31))
    document = collection_status.snapshot(str(tmp_path),
                                          datasets=['venue_ladder'], now=NOW)

    assert document['datasets'][0]['stale'] is True
    assert collection_status.late(document) == [
        ('venue_ladder', 31 * 86400.0, 900.0)]


def test_a_dataset_with_no_rows_is_null_rather_than_fresh(tmp_path):
    document = collection_status.snapshot(str(tmp_path),
                                          datasets=['venue_ladder'], now=NOW)
    row = document['datasets'][0]

    assert row['newest_event_time'] is None
    assert row['age_seconds'] is None
    # Not False. Nothing was measured, so there is no verdict to publish.
    assert row['stale'] is None
    assert row['reason']
    assert collection_status.late(document) == [('venue_ladder', None, 900.0)]


def test_the_component_limits_travel_but_the_ages_do_not(tmp_path):
    """A heartbeat age would be wrong seconds after it was written, so only the
    LIMITS are published and the stamps are read live by the reader."""
    document = collection_status.snapshot(str(tmp_path), datasets=[], now=NOW)

    from core import heartbeat
    assert document['component_limits'] == dict(heartbeat.MAX_AGE)
    assert 'component_ages' not in document


def test_the_document_is_replaced_atomically(tmp_path):
    """A reader polling this must never catch a half-written file and report
    every dataset missing."""
    target = tmp_path / 'status' / 'collection.json'
    _write(tmp_path, 'venue_ladder', NOW - timedelta(minutes=2))
    document = collection_status.snapshot(str(tmp_path),
                                          datasets=['venue_ladder'], now=NOW)

    assert collection_status.write(document, target) == target
    assert json.loads(target.read_text())['schema'] == collection_status.SCHEMA
    # No temp file left behind to be mistaken for the document.
    assert [p.name for p in target.parent.iterdir()] == ['collection.json']


def test_a_write_failure_never_raises(tmp_path):
    """A status writer that can take down the loop it reports on is worse than
    no status writer."""
    assert collection_status.write({'schema': 1},
                                   tmp_path / 'x' / '\0bad' / 'c.json') is None
