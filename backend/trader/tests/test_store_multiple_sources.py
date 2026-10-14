"""Tests that two observers of one `venue_depth` minute are both kept.

`source` is part of the event key here, so a recording and a backfill coexist; revisions
still collapse within one source.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.datastore import ResearchStore

W = pd.Timestamp('2026-08-26 00:46', tz='UTC')


def row(source: str, available: pd.Timestamp, bid: float) -> dict:
    return {
        'venue': 'kalshi', 'symbol': 'ETH-USD', 'event_time': W,
        'available_time': available, 'quality': 'valid',
        'market_ticker': 'KXETH15M-26AUG252100-00',
        'window_open': W - pd.Timedelta(minutes=1), 'offset_minutes': 1,
        'yes_bid': bid, 'yes_ask': bid + 0.01, 'yes_bid_size': 10.0,
        'yes_ask_size': 10.0, 'depth_bid_1c': 10.0, 'depth_bid_5c': 10.0,
        'depth_ask_1c': 10.0, 'depth_ask_5c': 10.0, 'depth_bid_total': 10.0,
        'depth_ask_total': 10.0, 'levels_bid': 1.0, 'levels_ask': 1.0,
        'seq': float('nan'), 'gaps': 0.0, 'source': source,
        'quote_age_seconds': 0.0,
    }


@pytest.fixture()
def store(tmp_path) -> ResearchStore:
    return ResearchStore(str(tmp_path / 'store'))


def test_live_and_backfill_of_the_same_minute_both_survive(store):
    store.write('venue_depth', pd.DataFrame([
        row('backfill', W, 0.40),
        row('live', W + pd.Timedelta(seconds=3), 0.41),
    ]))
    got = store.read('venue_depth')
    assert sorted(got['source']) == ['backfill', 'live']


def test_they_survive_across_separate_writes(store):
    store.write('venue_depth', pd.DataFrame([row('live', W + pd.Timedelta(seconds=3), 0.41)]))
    store.write('venue_depth', pd.DataFrame([row('backfill', W, 0.40)]))
    got = store.read('venue_depth')
    assert sorted(got['source']) == ['backfill', 'live']
    assert set(got['yes_bid'].round(2)) == {0.40, 0.41}


def test_a_revision_within_one_source_still_supersedes(store):
    """Within one source, the later revision still wins."""
    store.write('venue_depth', pd.DataFrame([row('backfill', W, 0.40)]))
    store.write('venue_depth', pd.DataFrame([
        row('backfill', W + pd.Timedelta(minutes=5), 0.44)]))
    got = store.read('venue_depth')
    assert len(got) == 1
    assert got['yes_bid'].iloc[0] == pytest.approx(0.44)


def test_a_dataset_without_a_source_column_is_unaffected(store):
    """A dataset without a `source` column keeps one row per event."""
    base = {'venue': 'coinbase_spot', 'symbol': 'BTC-USD', 'event_time': W,
            'quality': 'valid', 'open': 1.0, 'high': 1.0, 'low': 1.0,
            'close': 1.0, 'volume': 1.0}
    store.write('minute_bars', pd.DataFrame([
        {**base, 'available_time': W, 'close': 1.0},
        {**base, 'available_time': W + pd.Timedelta(minutes=1), 'close': 2.0},
    ]))
    got = store.read('minute_bars')
    assert len(got) == 1
    assert got['close'].iloc[0] == pytest.approx(2.0)
