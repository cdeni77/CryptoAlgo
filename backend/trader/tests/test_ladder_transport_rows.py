"""Both samplers write the same minute, and the rows must be distinguishable."""
from __future__ import annotations

import json

import pandas as pd

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from scripts.record_ladder import ws_row

NOW = pd.Timestamp('2026-08-26 12:00:25', tz='UTC')
OPEN = pd.Timestamp('2026-08-26 12:00', tz='UTC')


def _cache(received=99.5, now=100.0):
    cache = BookCache(now=lambda: now)
    cache.apply(BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                          received=received, seq=1, yes=[(0.3, 10.0)],
                          no=[(0.69, 4.0)]))
    return cache


def test_a_ws_row_carries_its_transport_and_its_age():
    row = ws_row(_cache(), ticker='K', symbol='BTC-USD', now=NOW,
                 open_time=OPEN, minute=0.417)
    assert row['transport'] == 'ws'
    assert row['book_age_ms'] == 500.0
    assert row['yes_total'] == 10.0 and row['no_total'] == 4.0
    assert json.loads(row['yes_levels']) == [[0.3, 10.0]]


def test_no_book_means_no_row_rather_than_an_empty_ladder():
    assert ws_row(BookCache(now=lambda: 0.0), ticker='K', symbol='BTC-USD',
                  now=NOW, open_time=OPEN, minute=0.417) is None, (
        'an empty ladder would record a dead subscription as a quiet market')


def test_a_stale_book_is_still_archived_with_its_age():
    """The archive fails honest; the trading path fails closed."""
    row = ws_row(_cache(received=0.0, now=100.0), ticker='K', symbol='BTC-USD',
                 now=NOW, open_time=OPEN, minute=0.417)
    assert row is not None and row['book_age_ms'] == 100000.0


def test_the_row_is_schema_compatible_with_the_rest_row(tmp_path):
    from core.datastore import ResearchStore
    row = ws_row(_cache(), ticker='K', symbol='BTC-USD', now=NOW,
                 open_time=OPEN, minute=0.417)
    store = ResearchStore(tmp_path)
    assert store.write('venue_ladder', pd.DataFrame([row])) == 1


def test_available_time_is_when_the_book_was_knowable_not_when_we_looked():
    """A cache read happens after the REST round trip, so the read instant is
    later than the frame that produced the book. Stamping the read instant made
    the two rows look simultaneous when they are ~100-150ms apart — and that
    skew alone accounts for the whole live agreement shortfall."""
    read_at = pd.Timestamp('2026-08-26 12:00:25.400', tz='UTC')
    row = ws_row(_cache(received=99.0, now=100.0), ticker='K', symbol='BTC-USD',
                 now=NOW, open_time=OPEN, minute=0.417, read_at=read_at)
    # age is 1.0s, so the frame landed a second before the read.
    assert row['available_time'] == read_at - pd.Timedelta(seconds=1.0)
    assert row['event_time'] == NOW.floor('min'), 'still pairs with the REST row'


def test_available_time_never_precedes_event_time():
    """The store rejects a row published before the event it describes."""
    row = ws_row(_cache(received=0.0, now=100.0), ticker='K', symbol='BTC-USD',
                 now=NOW, open_time=OPEN, minute=0.417, read_at=NOW)
    assert row['available_time'] >= row['event_time']
