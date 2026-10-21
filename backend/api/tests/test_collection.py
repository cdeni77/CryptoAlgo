"""Data-collection health: what it serves, and what it refuses to assume.

The failure being designed against is not a 500. It is a dead feed that renders
as a healthy one — `venue_ladder` wrote nothing for thirty-one days while every
component heartbeat stayed green, and the only evidence was a log line nobody
reads. So the assertions below are mostly about the ABSENT cases: no status
document, no stamp file, no published limit. Each must come back null with a
reason, because the one unacceptable answer is "fine".
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone


def _document(**overrides):
    now = datetime.now(timezone.utc)
    document = {
        'schema': 1,
        'generated_at': now.isoformat(),
        'store_root': '/app/data/research',
        'datasets': [
            {'dataset': 'venue_ladder',
             'newest_event_time': (now - timedelta(seconds=120)).isoformat(),
             'age_seconds': 120.0, 'limit_seconds': 900.0, 'stale': False,
             'reason': None},
            {'dataset': 'venue_trades',
             'newest_event_time': (now - timedelta(days=31)).isoformat(),
             'age_seconds': 2_678_400.0, 'limit_seconds': 900.0, 'stale': True,
             'reason': None},
            {'dataset': 'minute_bars', 'newest_event_time': None,
             'age_seconds': None, 'limit_seconds': 10_800.0, 'stale': None,
             'reason': 'no rows found in the store for this dataset'},
        ],
        'component_limits': {'trade': 300.0, 'ladder': 300.0},
        'heartbeat_root': '/app/data/alive',
    }
    document.update(overrides)
    return document


def _publish(tmp_path, monkeypatch, document):
    path = tmp_path / 'collection.json'
    path.write_text(json.dumps(document))
    monkeypatch.setenv('COLLECTION_STATUS_PATH', str(path))
    return path


def _stamps(tmp_path, monkeypatch, **ages):
    directory = tmp_path / 'alive'
    directory.mkdir(exist_ok=True)
    for name, age in ages.items():
        (directory / name).write_text(str(time.time() - age))
    monkeypatch.setenv('HEARTBEAT_ROOT', str(directory))
    return directory


# --------------------------------------------------------------- freshness --

def test_an_unpublished_measurement_is_not_a_healthy_one(client, tmp_path,
                                                         monkeypatch):
    """No status document must never render as "everything is current"."""
    monkeypatch.setenv('COLLECTION_STATUS_PATH', str(tmp_path / 'absent.json'))
    body = client.get('/collection/freshness').json()

    assert body['available'] is False
    assert body['datasets'] == []
    assert 'check_freshness' in body['reason']
    # Not a zero and not a verdict.
    assert body['measurement_age_seconds'] is None
    assert body['measurement_stale'] is None


def test_a_document_this_api_cannot_parse_is_refused(client, tmp_path,
                                                     monkeypatch):
    path = tmp_path / 'collection.json'
    path.write_text('{ this is not json')
    monkeypatch.setenv('COLLECTION_STATUS_PATH', str(path))

    body = client.get('/collection/freshness').json()
    assert body['available'] is False
    assert 'JSON' in body['reason']


def test_an_unknown_schema_is_refused_rather_than_part_read(client, tmp_path,
                                                            monkeypatch):
    """Half of a shape we do not know reads as "no expectation", which is the
    one direction a health surface must not be wrong in."""
    _publish(tmp_path, monkeypatch, _document(schema=99))

    body = client.get('/collection/freshness').json()
    assert body['available'] is False
    assert '99' in body['reason']


def test_each_dataset_carries_its_age_its_limit_and_a_verdict(client, tmp_path,
                                                              monkeypatch):
    _publish(tmp_path, monkeypatch, _document())
    body = client.get('/collection/freshness').json()

    assert body['available'] is True
    rows = {r['dataset']: r for r in body['datasets']}
    assert set(rows) == {'venue_ladder', 'venue_trades', 'minute_bars'}

    current = rows['venue_ladder']
    assert current['stale'] is False
    assert 60 < current['age_seconds'] < 900
    assert current['limit_seconds'] == 900.0

    dead = rows['venue_trades']
    assert dead['stale'] is True
    assert dead['age_seconds'] > 900


def test_the_age_is_recomputed_against_this_clock(client, tmp_path, monkeypatch):
    """The document's own `age_seconds` was true when it was written and is not
    now. Reading it back would make an hour-old measurement report an hour-old
    age forever."""
    document = _document()
    document['datasets'][0]['age_seconds'] = 1.0       # a lie, deliberately
    document['datasets'][0]['newest_event_time'] = (
        datetime.now(timezone.utc) - timedelta(seconds=4000)).isoformat()
    _publish(tmp_path, monkeypatch, document)

    row = next(r for r in client.get('/collection/freshness').json()['datasets']
               if r['dataset'] == 'venue_ladder')
    assert row['age_seconds'] > 3900
    assert row['stale'] is True


def test_a_dataset_with_no_rows_is_null_with_a_reason(client, tmp_path,
                                                      monkeypatch):
    _publish(tmp_path, monkeypatch, _document())
    row = next(r for r in client.get('/collection/freshness').json()['datasets']
               if r['dataset'] == 'minute_bars')

    assert row['age_seconds'] is None
    assert row['newest_event_time'] is None
    # Not False. Nothing was measured, so there is no verdict.
    assert row['stale'] is None
    assert row['reason']


def test_an_unreadable_timestamp_does_not_borrow_the_no_rows_reason(
        client, tmp_path, monkeypatch):
    """"The feed is dead" and "this process cannot parse what the feed wrote"
    are different findings, and the second is a bug here rather than there."""
    document = _document()
    document['datasets'][0]['newest_event_time'] = 'last Tuesday'
    _publish(tmp_path, monkeypatch, document)

    row = next(r for r in client.get('/collection/freshness').json()['datasets']
               if r['dataset'] == 'venue_ladder')
    assert row['age_seconds'] is None
    assert row['stale'] is None
    assert 'last Tuesday' in row['reason']


def test_a_dataset_with_no_cadence_gets_no_verdict(client, tmp_path, monkeypatch):
    """`limit: null` means nothing writes it continuously. Reporting that as
    fresh would be a claim nobody measured."""
    document = _document()
    document['datasets'].append(
        {'dataset': 'venue_quotes',
         'newest_event_time': (datetime.now(timezone.utc)
                               - timedelta(days=9)).isoformat(),
         'age_seconds': 777_600.0, 'limit_seconds': None, 'stale': None,
         'reason': None})
    _publish(tmp_path, monkeypatch, document)

    row = next(r for r in client.get('/collection/freshness').json()['datasets']
               if r['dataset'] == 'venue_quotes')
    assert row['age_seconds'] > 0
    assert row['limit_seconds'] is None
    assert row['stale'] is None
    assert row['reason']


def test_an_old_measurement_says_so(client, tmp_path, monkeypatch):
    """Rows may have arrived since it was taken, so every age below it is an
    upper bound rather than a reading."""
    stale_at = datetime.now(timezone.utc) - timedelta(hours=9)
    _publish(tmp_path, monkeypatch, _document(generated_at=stale_at.isoformat()))

    body = client.get('/collection/freshness').json()
    assert body['available'] is True
    assert body['measurement_stale'] is True
    assert body['measurement_age_seconds'] > 8 * 3600


# ---------------------------------------------------------------- liveness --

def test_each_component_carries_the_seconds_since_its_last_cycle(
        client, tmp_path, monkeypatch):
    _publish(tmp_path, monkeypatch, _document())
    _stamps(tmp_path, monkeypatch, trade=30.0, ladder=4000.0)

    body = client.get('/collection/liveness').json()
    assert body['available'] is True
    rows = {r['component']: r for r in body['components']}

    assert rows['trade']['stale'] is False
    assert 0 <= rows['trade']['age_seconds'] < 300
    assert rows['ladder']['stale'] is True
    assert rows['ladder']['age_seconds'] > 300


def test_a_component_that_never_stamped_is_reported_not_excused(
        client, tmp_path, monkeypatch):
    """"Not started yet" and "died before its first cycle" look identical from
    outside, and the second is the one worth catching."""
    _publish(tmp_path, monkeypatch, _document())
    _stamps(tmp_path, monkeypatch, trade=30.0)          # no `ladder` stamp

    rows = {r['component']: r
            for r in client.get('/collection/liveness').json()['components']}
    assert 'ladder' in rows
    assert rows['ladder']['age_seconds'] is None
    assert rows['ladder']['stale'] is None
    assert rows['ladder']['reason']


def test_an_age_without_a_published_limit_keeps_the_age_and_drops_the_verdict(
        client, tmp_path, monkeypatch):
    """The age is a measurement; "is that too old" needs a number the trader
    publishes. Without it, serve the first and refuse the second."""
    monkeypatch.setenv('COLLECTION_STATUS_PATH', str(tmp_path / 'absent.json'))
    _stamps(tmp_path, monkeypatch, trade=30.0)

    body = client.get('/collection/liveness').json()
    row = next(r for r in body['components'] if r['component'] == 'trade')
    assert row['age_seconds'] is not None
    assert row['limit_seconds'] is None
    assert row['stale'] is None
    assert row['reason']
    assert body['reason']                      # why the limits are missing


def test_an_unreadable_heartbeat_directory_is_not_healthy(client, tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv('HEARTBEAT_ROOT', str(tmp_path / 'nowhere'))
    monkeypatch.setenv('COLLECTION_STATUS_PATH', str(tmp_path / 'absent.json'))

    body = client.get('/collection/liveness').json()
    assert body['available'] is False
    assert body['components'] == []
    assert body['reason']


# ------------------------------------------------------------------- both --

def test_the_two_questions_are_served_separately(client, tmp_path, monkeypatch):
    """Freshness and liveness are never merged into one verdict: a loop that
    cycles while writing nothing is green on one and red on the other, and that
    pair IS the finding."""
    _publish(tmp_path, monkeypatch, _document())
    _stamps(tmp_path, monkeypatch, trade=30.0, ladder=30.0)

    body = client.get('/collection').json()
    assert set(body) == {'freshness', 'liveness'}
    assert body['liveness']['components']
    # Every component cycling, one dataset dead for 31 days. That is the shape
    # of the failure this surface exists for.
    assert all(r['stale'] is False for r in body['liveness']['components'])
    assert any(r['stale'] is True for r in body['freshness']['datasets'])
