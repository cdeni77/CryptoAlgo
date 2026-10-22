"""Parsing and folding the live CF Benchmarks index.

Both traps here are silent ones, captured from the real channel 2026-10-22:
`msg.data` is a JSON STRING, and `avg_60s_data` needs sixty seconds of
connection before it is the settlement quantity rather than an average of a
different interval.
"""

from __future__ import annotations

import json

import pytest

from core.index_bars import FULL_MINUTE_PRINTS, IndexBars, minute_of
from data_collection.stream.cfbenchmarks import (
    FULL_WINDOW, parse_value_frame, subscribe_params)

MINUTE = 1792643040          # a minute boundary: 1792643040 % 60 == 0


def _frame(index_id='BRTI', value='82831.30', at_ms=None,
           avg='82808.05', window=60, end_ms=None, data_as_object=False):
    at_ms = at_ms if at_ms is not None else (MINUTE + 30) * 1000
    end_ms = end_ms if end_ms is not None else (MINUTE + 60) * 1000
    inner = {'type': 'value', 'time': at_ms, 'id': index_id, 'value': value}
    return {'type': 'cfbenchmarks_value', 'sid': 1, 'seq': 2,
            'msg': {'index_id': index_id, 'received_at': at_ms,
                    'data': inner if data_as_object else json.dumps(inner),
                    'avg_60s_data': {'value': avg, 'window_size': window,
                                     'window_start_ts_ms': end_ms - 60000,
                                     'window_end_ts_exclusive': end_ms}},
            'sending_ts_ms': at_ms + 5}


class TestParsing:
    def test_the_nested_data_string_is_decoded(self):
        """`msg['data']['value']` raises; `.get('data', {}).get(...)` returns
        None. Either way the value is silently lost."""
        got = parse_value_frame(_frame())
        assert got is not None
        assert got.symbol == 'BTC-USD'
        assert got.value == pytest.approx(82831.30)

    def test_an_object_payload_also_works(self):
        """The venue has changed an encoding before."""
        got = parse_value_frame(_frame(data_as_object=True))
        assert got is not None and got.value == pytest.approx(82831.30)

    def test_every_traded_index_maps_to_its_symbol(self):
        for index_id, symbol in (('BRTI', 'BTC-USD'),
                                 ('ETHUSD_RTI', 'ETH-USD'),
                                 ('SOLUSD_RTI', 'SOL-USD')):
            assert parse_value_frame(_frame(index_id)).symbol == symbol

    def test_other_frames_are_ignored_rather_than_guessed_at(self):
        assert parse_value_frame({'type': 'subscribed', 'sid': 1}) is None
        assert parse_value_frame({'type': 'orderbook_delta', 'msg': {}}) is None
        assert parse_value_frame(_frame(index_id='UNKNOWN_RTI')) is None
        assert parse_value_frame({}) is None

    def test_we_subscribe_with_index_ids_not_market_tickers(self):
        params = subscribe_params(['BTC-USD', 'ETH-USD'])
        assert params['channels'] == ['cfbenchmarks_value']
        assert params['index_ids'] == ['BRTI', 'ETHUSD_RTI']
        assert 'market_tickers' not in params


class TestTheWarmUpWindow:
    """`window_size` climbs 0 -> 60 over the first minute of a connection."""

    def test_a_full_window_settles(self):
        assert parse_value_frame(_frame(window=FULL_WINDOW)).settles

    def test_a_partial_window_does_not(self):
        for size in (0, 8, 32, 59):
            assert not parse_value_frame(_frame(window=size)).settles, size

    def test_an_empty_window_looks_exactly_like_the_spot_value(self):
        """Which is what makes it dangerous -- it reads as correct."""
        got = parse_value_frame(_frame(value='82833.38', avg='82833.38000000',
                                       window=0))
        assert got.value == pytest.approx(got.avg_60s)
        assert not got.settles


class TestFolding:
    def test_the_bar_is_keyed_on_its_START_minute(self):
        """A bar stamped at its end shifts every window by a minute and
        nothing raises."""
        bars = IndexBars()
        bars.add(parse_value_frame(_frame(at_ms=(MINUTE + 30) * 1000)))
        assert bars.complete('BTC-USD', now=MINUTE + 120) == [MINUTE]

    def test_the_venue_average_becomes_the_minute_mean(self):
        bars = IndexBars()
        # a window ending at MINUTE+60 covers exactly [MINUTE, MINUTE+60)
        bars.add(parse_value_frame(_frame(avg='82808.05', window=60,
                                          end_ms=(MINUTE + 60) * 1000)))
        bar = bars.bar('BTC-USD', MINUTE)
        assert bar['mean_price'] == pytest.approx(82808.05)
        assert bar['mean_source'] == 'venue_avg_60s'

    def test_a_partial_average_is_refused_and_the_bar_says_so(self):
        """A partial average is a different interval, not a worse estimate."""
        bars = IndexBars()
        bars.add(parse_value_frame(_frame(value='100.0', avg='999.0',
                                          window=8)))
        bar = bars.bar('BTC-USD', MINUTE)
        assert bar['mean_price'] == pytest.approx(100.0)
        assert bar['mean_source'] == 'prints'

    def test_ohlc_comes_from_the_prints(self):
        bars = IndexBars()
        for i, v in enumerate(('100', '103', '99', '101')):
            bars.add(parse_value_frame(
                _frame(value=v, at_ms=(MINUTE + i) * 1000, window=0)))
        bar = bars.bar('BTC-USD', MINUTE)
        assert (bar['open'], bar['high'], bar['low'], bar['close']) == \
            (100.0, 103.0, 99.0, 101.0)

    def test_a_minute_still_running_has_no_bar(self):
        bars = IndexBars()
        bars.add(parse_value_frame(_frame(at_ms=(MINUTE + 10) * 1000)))
        assert bars.complete('BTC-USD', now=MINUTE + 30) == []

    def test_a_thin_minute_is_marked_suspicious_not_dropped(self):
        bars = IndexBars()
        bars.add(parse_value_frame(_frame(window=0)))
        assert bars.bar('BTC-USD', MINUTE)['complete'] is False
        assert FULL_MINUTE_PRINTS <= 60

    def test_draining_forgets_what_it_yielded(self):
        bars = IndexBars()
        bars.add(parse_value_frame(_frame()))
        assert len(list(bars.drain('BTC-USD', now=MINUTE + 120))) == 1
        assert list(bars.drain('BTC-USD', now=MINUTE + 120)) == []


class TestStoreRows:
    def test_available_time_is_never_before_the_minute_it_describes(self):
        """The bar for minute m is only knowable once m has ended."""
        import pandas as pd

        from core.index_bars import rows_for_store
        bars = IndexBars()
        bars.add(parse_value_frame(_frame()))
        rows = rows_for_store(bars.drain('BTC-USD', now=MINUTE + 120),
                              available_time=pd.Timestamp(MINUTE, unit='s',
                                                          tz='UTC'))
        assert rows[0]['available_time'] >= rows[0]['event_time']
        assert rows[0]['venue'] == 'cfbenchmarks'
        assert rows[0]['mean_price'] == pytest.approx(82808.05)


class TestWiring:
    def test_it_is_a_run_live_component_with_a_heartbeat(self):
        """A recorder nothing supervises is a recorder that dies quietly."""
        import inspect

        from core.heartbeat import MAX_AGE
        from scripts import run_live

        assert 'index_stream' in MAX_AGE
        assert MAX_AGE['index_stream'] <= 180.0, 'too tolerant for ~1 frame/s'
        src = inspect.getsource(run_live)
        assert "Component('index_stream'" in src
        assert "'index_stream': index_stream" in src

    def test_it_writes_the_same_table_the_backfill_writes(self):
        """One continuous series, switched with --venue, not a second path."""
        from scripts.backfill_index import VENUE
        from core.index_bars import rows_for_store
        assert VENUE == 'cfbenchmarks'
        rows = rows_for_store([{'symbol': 'BTC-USD', 'minute': MINUTE,
                                'open': 1.0, 'high': 2.0, 'low': 0.5,
                                'close': 1.5, 'mean_price': 1.25,
                                'mean_source': 'venue_avg_60s',
                                'n_prints': 60, 'complete': True}],
                              available_time=__import__('pandas').Timestamp(
                                  MINUTE + 60, unit='s', tz='UTC'))
        assert rows[0]['venue'] == VENUE
        assert rows[0]['quality'] == 'valid'

    def test_the_index_mapping_matches_the_backfill(self):
        """Two mappings that can disagree will."""
        from data_collection.stream.cfbenchmarks import INDEX_BY_SYMBOL as live
        from scripts.backfill_index import INDEX_BY_SYMBOL as back
        assert live == back
