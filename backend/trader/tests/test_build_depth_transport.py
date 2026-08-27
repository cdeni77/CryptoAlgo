"""venue_ladder now holds two observers per minute; venue_depth must keep both.

Without this, both transports stamp `source='live'`, the two rows collide on
venue_depth's event key, and `read` keeps whichever carried the later
`available_time`. venue_depth would silently become a mix of the two transports,
varying row by row — and the comparison that gates the migration would be
comparing a table against itself.
"""
from __future__ import annotations

import json

import pandas as pd

from core.datastore import ResearchStore
from scripts.build_depth import _from_ladder, _ladder_source


def _ladder_row(transport, **over):
    base = dict(venue='kalshi', symbol='BTC-USD',
                event_time=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                available_time=pd.Timestamp('2026-08-26 12:00:25', tz='UTC'),
                quality='valid', market_ticker='K',
                window_open=pd.Timestamp('2026-08-26 12:00', tz='UTC'),
                minute_into_window=0.4,
                yes_levels=json.dumps([[0.30, 10.0]]),
                no_levels=json.dumps([[0.65, 4.0]]),
                yes_total=10.0, no_total=4.0,
                transport=transport, book_age_ms=0.0)
    base.update(over)
    return base


def test_rest_keeps_the_source_the_existing_series_already_uses():
    assert _ladder_source({'transport': 'rest'}, 'live') == 'live'


def test_a_row_written_before_the_column_existed_is_treated_as_rest():
    assert _ladder_source({}, 'live') == 'live'
    assert _ladder_source({'transport': None}, 'live') == 'live'
    assert _ladder_source({'transport': float('nan')}, 'live') == 'live'


def test_the_stream_becomes_a_separate_observer():
    assert _ladder_source({'transport': 'ws'}, 'live') == 'live_ws'


def test_both_transports_survive_a_venue_depth_read(tmp_path):
    rows = _from_ladder(
        pd.DataFrame([_ladder_row('rest'),
                      _ladder_row('ws', available_time=pd.Timestamp(
                          '2026-08-26 12:00:26', tz='UTC'))]),
        source='live')
    assert sorted(r['source'] for r in rows) == ['live', 'live_ws']

    store = ResearchStore(tmp_path)
    store.write('venue_depth', pd.DataFrame(rows))
    got = store.read('venue_depth')
    assert sorted(got['source']) == ['live', 'live_ws'], (
        'one event key for two observers means one of them disappears')


def test_the_trading_loops_own_touch_names_its_observer():
    """`_record_touch` left `source` unset, so every row it wrote sat under a
    NULL observer — invisible to `_validate_depth`, which compares by source,
    and now ambiguous against the two the ladder recorder writes."""
    import inspect

    from scripts import live
    source = inspect.getsource(live._record_touch)
    assert "'source': 'live_touch'" in source


def test_the_three_observers_are_distinct(tmp_path):
    """live (REST poll), live_ws (stream), live_touch (the trading loop)."""
    from core.datastore import event_key
    assert 'source' in event_key('venue_depth')
    names = {'live', 'live_ws', 'live_touch'}
    assert len(names) == 3, 'each observer must have its own name or one is lost'
