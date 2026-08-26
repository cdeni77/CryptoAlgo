"""Two instruments observing one event are not two revisions of it.

The research store is a point-in-time revision store: `read` keeps exactly one
row per `(venue, symbol, event_time)` — the one with the latest
`available_time` — so a corrected figure supersedes the one it corrects. That is
right for a revised series like funding or open interest.

It is wrong for `venue_depth`, which holds the SAME minute of the SAME book
observed two ways: recorded live while the market was open, and reconstructed
afterwards from Predexon. Those are independent measurements, not a correction
and its predecessor, and the whole point of holding both is to compare them.

Under the plain key the live row always won, because its `available_time` is the
poll instant while the backfill's is the minute mark itself. Measured: 58
(symbol, window) pairs overlapped and `_validate_depth.py` still reported zero
comparable rows — the backfill row existed on disk and was invisible to every
read. Silently, with no error, on exactly the rows the comparison needed.

So `source` joins the event key for this dataset: same event, two observers.
Revisions still collapse WITHIN an observer.
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
    """The overwrite happened on the second write, not only within one frame."""
    store.write('venue_depth', pd.DataFrame([row('live', W + pd.Timedelta(seconds=3), 0.41)]))
    store.write('venue_depth', pd.DataFrame([row('backfill', W, 0.40)]))
    got = store.read('venue_depth')
    assert sorted(got['source']) == ['backfill', 'live']
    assert set(got['yes_bid'].round(2)) == {0.40, 0.41}


def test_a_revision_within_one_source_still_supersedes(store):
    """Two backfill reads of the same minute: the later one wins, as before."""
    store.write('venue_depth', pd.DataFrame([row('backfill', W, 0.40)]))
    store.write('venue_depth', pd.DataFrame([
        row('backfill', W + pd.Timedelta(minutes=5), 0.44)]))
    got = store.read('venue_depth')
    assert len(got) == 1
    assert got['yes_bid'].iloc[0] == pytest.approx(0.44)


def test_a_dataset_without_a_source_column_is_unaffected(store):
    """`bars` and the rest keep the plain one-row-per-event contract."""
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
