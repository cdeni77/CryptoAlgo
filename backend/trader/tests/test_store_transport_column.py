"""Tests for the `transport` column on `venue_ladder`.

Older files without the column must still read, and the column keeps both transports'
rows.
"""
from __future__ import annotations

import pandas as pd

from core.datastore import ResearchStore, event_key


def _row(**over):
    base = dict(venue='kalshi', symbol='BTC-USD',
                event_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                available_time=pd.Timestamp('2026-08-26 12:00:05', tz='UTC'),
                quality='valid', market_ticker='KXBTC15M-26AUG2612-15',
                window_open=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                minute_into_window=0.0, yes_levels='[]', no_levels='[]',
                yes_total=0.0, no_total=0.0)
    base.update(over)
    return base


def test_transport_is_part_of_the_event_key():
    assert 'transport' in event_key('venue_ladder')
    assert 'transport' in event_key('pm_ladder')


def test_two_transports_for_one_minute_both_survive_a_read(tmp_path):
    store = ResearchStore(tmp_path)
    store.write('venue_ladder', pd.DataFrame([
        _row(transport='rest', book_age_ms=0.0),
        _row(transport='ws', book_age_ms=120.0,
             available_time=pd.Timestamp('2026-08-26 12:00:06', tz='UTC')),
    ]))
    got = store.read('venue_ladder')
    assert sorted(got['transport']) == ['rest', 'ws'], (
        'the WS row and the REST row are independent observations, not a '
        'revision and its predecessor')


def test_a_revision_within_one_transport_still_collapses(tmp_path):
    store = ResearchStore(tmp_path)
    store.write('venue_ladder', pd.DataFrame([
        _row(transport='ws', book_age_ms=1.0, yes_total=1.0),
        _row(transport='ws', book_age_ms=2.0, yes_total=2.0,
             available_time=pd.Timestamp('2026-08-26 12:00:09', tz='UTC')),
    ]))
    got = store.read('venue_ladder')
    assert len(got) == 1 and got.iloc[0]['yes_total'] == 2.0


def test_a_partition_written_before_the_column_existed_still_reads(tmp_path):
    """A partition without the `transport` column still reads."""
    store = ResearchStore(tmp_path)
    prepared = store._prepare('venue_ladder', pd.DataFrame([_row()]))  # noqa: SLF001
    legacy = prepared.drop(columns=['transport', 'book_age_ms'])
    part = tmp_path / 'venue_ladder' / 'venue=kalshi' / 'symbol=BTC-USD' / 'month=2026-08'
    part.mkdir(parents=True)
    legacy.to_parquet(part / 'data.parquet', index=False)

    store.write('venue_ladder', pd.DataFrame([
        _row(transport='ws', book_age_ms=5.0, minute_into_window=1.0,
             event_time=pd.Timestamp('2026-08-26 12:01', tz='UTC'),
             available_time=pd.Timestamp('2026-08-26 12:01:05', tz='UTC'))]))
    got = store.read('venue_ladder')
    assert len(got) == 2
    assert got['transport'].isna().sum() == 1
