"""Tests for loading the consolidated collection into the research store.

`venue_depth` holds one row per minute 0..15 per window, not one per snapshot.
Every minute is kept because the offset grid is itself under test. Loading
streams, because one partition is too large to merge in memory.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from research.collect.load_store import (
    BACKFILL_SOURCE, summarise_window, to_depth_rows,
)

UTC = dt.timezone.utc
W0 = dt.datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _snaps(rows):
    """Rows are (offset_seconds, best_bid_cents, best_ask_cents)."""
    return pd.DataFrame([{
        'venue': 'kalshi', 'symbol': 'BTC-USD', 'market_id': 'KX-1',
        'window_open': pd.Timestamp(W0), 'ts': int((W0.timestamp() + o) * 1000),
        'event_time': pd.Timestamp(W0 + dt.timedelta(seconds=o)),
        'offset_seconds': float(o),
        'best_bid': b, 'best_ask': a, 'bid_at_touch': 10.0, 'ask_at_touch': 20.0,
        'bid_1c': 30.0, 'ask_1c': 40.0, 'bid_5c': 50.0, 'ask_5c': 60.0,
        'bid_levels': 7.0, 'ask_levels': 8.0, 'bid_vol': 100.0, 'ask_vol': 200.0,
    } for o, b, a in rows])


def test_a_minute_takes_the_last_snapshot_at_or_before_it():
    """Each minute takes the last snapshot at or before its mark (minute 1 is t+60s)."""
    got = summarise_window(_snaps([(50, 40, 42), (110, 44, 46), (130, 48, 50)]))
    row = got[got['offset_minutes'] == 1].iloc[0]
    assert row['yes_bid'] == pytest.approx(0.40), 'must take t+50s, not t+110s'
    # Minute 2's mark is t+120s, so it takes t+110s, not t+130s.
    assert got[got['offset_minutes'] == 2].iloc[0]['yes_bid'] == pytest.approx(0.44)


def test_cents_become_dollars():
    """Prices convert from cents to dollars."""
    got = summarise_window(_snaps([(0, 63, 64)]))
    assert got['yes_bid'].iloc[0] == pytest.approx(0.63)
    assert got['yes_ask'].iloc[0] == pytest.approx(0.64)


def test_sizes_and_depths_are_not_rescaled():
    """Only prices are in cents; sizes are contracts and pass through unchanged."""
    got = summarise_window(_snaps([(0, 63, 64)]))
    r = got.iloc[0]
    assert r['yes_bid_size'] == 10.0 and r['yes_ask_size'] == 20.0
    assert r['depth_bid_1c'] == 30.0 and r['depth_ask_5c'] == 60.0
    assert r['depth_bid_total'] == 100.0 and r['depth_ask_total'] == 200.0


def test_a_minute_with_no_snapshot_yet_is_omitted_not_carried():
    """A minute with no snapshot at or before its mark gets no row."""
    got = summarise_window(_snaps([(90, 44, 46)]))
    assert 0 not in set(got['offset_minutes'])
    assert 1 not in set(got['offset_minutes'])
    assert 2 in set(got['offset_minutes'])


def test_every_minute_of_the_window_is_kept_not_just_the_decision_offsets():
    """Every minute of the window is kept, not just the decision offsets."""
    snaps = _snaps([(s, 40, 42) for s in range(0, 16 * 60, 30)])
    got = summarise_window(snaps)
    assert set(got['offset_minutes']) >= set(range(0, 16))


def test_nothing_past_the_window_close_is_emitted():
    """A snapshot past +15m belongs to the next window and is not emitted."""
    got = summarise_window(_snaps([(0, 40, 42), (16 * 60 + 5, 90, 92)]))
    assert max(got['offset_minutes']) <= 15


def test_the_source_marks_it_as_backfill():
    """Rows carry `source=BACKFILL_SOURCE`, so they coexist with recorded rows under the
    event key.
    """
    got = summarise_window(_snaps([(0, 40, 42)]))
    assert got['source'].iloc[0] == BACKFILL_SOURCE
    assert BACKFILL_SOURCE, 'must be a non-empty marker, not NaN'


def test_the_frame_carries_the_columns_the_store_expects():
    got = summarise_window(_snaps([(0, 40, 42)]))
    for column in ('venue', 'symbol', 'event_time', 'available_time',
                   'market_ticker', 'window_open', 'offset_minutes',
                   'yes_bid', 'yes_ask', 'source'):
        assert column in got.columns, column


def test_a_one_sided_book_keeps_the_side_it_has():
    """A one-sided book keeps the side it has."""
    got = summarise_window(_snaps([(0, 40, float('nan'))]))
    assert got['yes_bid'].iloc[0] == pytest.approx(0.40)
    assert pd.isna(got['yes_ask'].iloc[0])


def test_an_empty_partition_yields_no_rows_rather_than_raising():
    assert to_depth_rows(_snaps([]).iloc[0:0]).empty


# --- the store's schema, not the collection's ------------------------------
#
# A frame with wrong column names does not raise: the columns land as absent
# and readers see nulls. So the loader's output is checked against the schema.

def test_settlements_use_the_store_column_names():
    from research.collect.load_store import settlement_rows
    got = settlement_rows([{'symbol': 'BTC-USD', 'market_id': 'KX-1',
                            'window_open': '2026-07-01T12:00:00+00:00',
                            'result': 'yes', 'expiration_value': 80383.64,
                            'close_time': '2026-07-01T12:15:00+00:00'}])
    assert 'market_ticker' in got.columns and 'market_id' not in got.columns
    assert got['market_ticker'].iloc[0] == 'KX-1'


def test_settled_up_is_derived_from_the_venues_own_result():
    """`settled_up` is derived here from the venue's `result`."""
    from research.collect.load_store import settlement_rows
    base = {'symbol': 'BTC-USD', 'market_id': 'k',
            'window_open': '2026-07-01T12:00:00+00:00',
            'close_time': '2026-07-01T12:15:00+00:00'}
    assert settlement_rows([{**base, 'result': 'yes'}])['settled_up'].iloc[0]
    assert not settlement_rows([{**base, 'result': 'no'}])['settled_up'].iloc[0]


def test_a_settlement_price_is_not_forced_into_a_column_that_does_not_exist():
    """`venue_settlements` has no price column, so the loader does not add one."""
    from research.collect.load_store import settlement_rows
    from core.datastore import SCHEMAS
    got = settlement_rows([{'symbol': 'BTC-USD', 'market_id': 'k',
                            'window_open': '2026-07-01T12:00:00+00:00',
                            'result': 'yes', 'expiration_value': 80383.64,
                            'close_time': '2026-07-01T12:15:00+00:00'}])
    declared = set(getattr(SCHEMAS['venue_settlements'], 'columns',
                           SCHEMAS['venue_settlements']))
    assert set(got.columns) <= declared, set(got.columns) - declared


def test_implied_vol_rows_carry_close_time():
    from research.collect.load_store import implied_vol_rows
    got = implied_vol_rows([{'symbol': 'BTC-USD', 'event_ticker': 'KXBTCD-1',
                             'event_time': '2026-07-01T12:00:00+00:00',
                             'close_time': '2026-07-01T12:30:00+00:00',
                             'minutes_to_close': 30.0,
                             'implied_sigma_per_min': 0.0006,
                             'implied_spot': 80000.0, 'atm_strike': 80000.0,
                             'n_strikes': 9, 'r2': 0.98}])
    assert pd.notna(got['close_time'].iloc[0])
    from core.datastore import SCHEMAS
    declared = set(getattr(SCHEMAS['venue_implied_vol'], 'columns',
                           SCHEMAS['venue_implied_vol']))
    assert set(got.columns) <= declared, set(got.columns) - declared


def test_depth_rows_stay_inside_the_declared_schema_too():
    from core.datastore import SCHEMAS
    got = summarise_window(_snaps([(0, 40, 42)]))
    declared = set(getattr(SCHEMAS['venue_depth'], 'columns',
                           SCHEMAS['venue_depth']))
    assert set(got.columns) <= declared, set(got.columns) - declared


# --- the DuckDB path --------------------------------------------------------
#
# `summarise_window` (pandas) and `summarise_partition` (DuckDB) compute the same
# thing. DuckDB's `//` on a DOUBLE divides and CAST rounds, unlike pandas floor,
# so the production DuckDB path has its own tests against a real Parquet file.

def _write_parquet(tmp_path, rows):
    frame = _snaps(rows)
    path = tmp_path / 'data.parquet'
    frame.to_parquet(path, index=False)
    return path


def test_a_book_persists_until_it_changes(tmp_path):
    """A book serves every following minute until it changes, taken at or before each mark.
    """
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(
        tmp_path, [(0, 40, 42), (59, 41, 43), (900, 44, 46), (959, 48, 50)]))
    by_minute = got.set_index('offset_minutes')
    assert set(by_minute.index) == set(range(0, 16))
    assert by_minute.loc[0, 'yes_bid'] == pytest.approx(0.40)
    assert by_minute.loc[1, 'yes_bid'] == pytest.approx(0.41), 't+59s serves minute 1'
    assert by_minute.loc[14, 'yes_bid'] == pytest.approx(0.41), 'still nothing newer'
    assert by_minute.loc[15, 'yes_bid'] == pytest.approx(0.44), 't+900s, not t+959s'


def test_no_row_is_built_from_a_book_that_did_not_exist_yet(tmp_path):
    """A quote from after a mark never informs that mark's row."""
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(tmp_path, [(190, 44, 46)]))
    assert (got['quote_age_seconds'] >= 0).all(), 'negative age = future book'
    assert got['offset_minutes'].min() == 4, 't+3m10s cannot serve minute 3'


def test_the_duckdb_path_emits_nothing_past_the_close(tmp_path):
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(
        tmp_path, [(0, 40, 42), (959, 48, 50)]))
    assert got['offset_minutes'].max() == 15, 'minute 16 is the next window'


def test_the_two_implementations_agree(tmp_path):
    """The pandas and DuckDB paths return the same offsets for the same input."""
    from research.collect.load_store import summarise_partition
    rows = [(s, 40 + s % 7, 42 + s % 7) for s in range(0, 16 * 60, 17)]
    duck = summarise_partition(_write_parquet(tmp_path, rows))
    pand = summarise_window(_snaps(rows))
    assert sorted(duck['offset_minutes']) == sorted(pand['offset_minutes'])


def test_the_duckdb_path_converts_cents_to_dollars_too(tmp_path):
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(tmp_path, [(0, 63, 64)]))
    assert got['yes_bid'].iloc[0] == pytest.approx(0.63)


# --- provenance and staleness ----------------------------------------------
#
# `source` must be the canonical 'backfill', which `_validate_depth` filters on.
# `quote_age_seconds` must be carried: Predexon serves book changes, so a quiet
# book carries forward, and the age shows how stale each row is.

def test_the_source_is_the_canonical_backfill_value():
    from research.collect.load_store import BACKFILL_SOURCE
    assert BACKFILL_SOURCE == 'backfill', (
        "must match scripts/build_depth.py:241 and what _validate_depth filters on")


def test_quote_age_says_how_stale_the_minute_is(tmp_path):
    """A snapshot at t+3m10s serves minute 4 with an age of 50s."""
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(tmp_path, [(190, 44, 46)]))
    row = got[got['offset_minutes'] == 4].iloc[0]
    assert row['quote_age_seconds'] == pytest.approx(50.0), row['quote_age_seconds']


def test_a_fresh_quote_has_near_zero_age(tmp_path):
    from research.collect.load_store import summarise_partition
    got = summarise_partition(_write_parquet(tmp_path, [(300, 44, 46)]))
    row = got[got['offset_minutes'] == 5].iloc[0]
    assert row['quote_age_seconds'] == pytest.approx(0.0)


# --- the venue's label must cover the tested period ------------------------
#
# Kalshi's own API purges older markets. Predexon's catalog `result` fills the
# earlier months. Kalshi's own rows win where both exist, since they also carry
# `expiration_value`.

def test_catalog_results_become_settlements():
    from research.collect.load_store import catalog_settlement_rows
    got = catalog_settlement_rows([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'market_id': 'k1',
         'window_open': '2026-02-01T12:00:00+00:00', 'result': 'yes'},
        {'venue': 'kalshi', 'symbol': 'ETH-USD', 'market_id': 'k2',
         'window_open': '2026-02-01T12:15:00+00:00', 'result': 'no'},
    ])
    assert list(got['settled_up']) == [True, False]
    assert list(got['market_ticker']) == ['k1', 'k2']


def test_a_catalog_row_without_a_result_is_skipped():
    """A catalog row with no result is skipped, not stored blank."""
    from research.collect.load_store import catalog_settlement_rows
    got = catalog_settlement_rows([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'market_id': 'k1',
         'window_open': '2026-02-01T12:00:00+00:00', 'result': ''},
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'market_id': 'k2',
         'window_open': '2026-02-01T12:15:00+00:00', 'result': None},
    ])
    assert got.empty


def test_only_kalshi_rows_are_taken_from_the_catalog():
    """Only rows labelled kalshi are taken from the catalog."""
    from research.collect.load_store import catalog_settlement_rows
    got = catalog_settlement_rows([
        {'venue': 'polymarket', 'symbol': 'BTC-USD', 'market_id': 'p1',
         'window_open': '2026-02-01T12:00:00+00:00', 'result': 'yes'}])
    assert got.empty
