"""`venue_candles` — the rest of the candle payload, which was being discarded.

`backfill_candles` lands the candle's CLOSING bid and ask in `venue_depth`,
which is right: that is a book summary at a minute. Everything else the venue
sends was thrown away — the traded-price OHLC, volume, open interest, and how
far the quote travelled inside the minute.

`price_low` is the one that matters. It is the cheapest PRINT inside the
minute, so a `price_low` at or below a resting bid is evidence that bid would
have been hit. That is the question the maker thesis turns on, and this is the
cheap approximation of it that reaches every market the endpoint serves.
"""

import datetime as dt

import pandas as pd
import pytest

from core.datastore import SCHEMAS, event_key
from scripts.backfill_candles import candle_rows_for_market, rows_for_market

# Read off the live wire 2026-10-21, not from the documentation -- this repo
# has been bitten three times by field names the docs got wrong.
WIRE = {
    "end_period_ts": 1786665240,
    "open_interest_fp": "12077.70",
    "volume_fp": "2729.04",
    "price": {"close_dollars": "0.0540", "high_dollars": "0.1200",
              "low_dollars": "0.0510", "mean_dollars": "0.0785",
              "open_dollars": "0.1000", "previous_dollars": "0.1100"},
    "yes_ask": {"close_dollars": "0.0640", "high_dollars": "0.1200",
                "low_dollars": "0.0570", "open_dollars": "0.1100"},
    "yes_bid": {"close_dollars": "0.0530", "high_dollars": "0.1100",
                "low_dollars": "0.0510", "open_dollars": "0.1000"},
}
TICKER = 'KXETH15M-26AUG132000-00'
NOW = dt.datetime(2026, 10, 21, tzinfo=dt.timezone.utc)


def _row():
    rows = candle_rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    assert len(rows) == 1
    return rows[0]


def test_every_field_the_venue_sends_is_stored():
    row = _row()
    missing = [c for c in SCHEMAS['venue_candles']
               if c not in ('venue', 'symbol', 'quality') and row.get(c) is None]
    assert not missing, 'discarded again: %s' % missing


def test_no_column_outside_the_schema():
    assert not set(_row()) - set(SCHEMAS['venue_candles'])


def test_the_traded_price_is_not_the_quote():
    # The whole point of keeping `price_*`: it is where trades PRINTED, which
    # is a different object from the bid and the ask. Conflating them would
    # make `price_low` useless for the fill question.
    row = _row()
    assert row['price_low'] == pytest.approx(0.0510)
    assert row['price_close'] == pytest.approx(0.0540)
    assert row['yes_bid_close'] == pytest.approx(0.0530)
    assert row['yes_ask_close'] == pytest.approx(0.0640)
    assert row['price_close'] != row['yes_bid_close']


def test_volume_and_open_interest_are_fixed_point_floats():
    row = _row()
    assert row['volume'] == pytest.approx(2729.04)
    assert row['open_interest'] == pytest.approx(12077.70)


def test_available_time_is_the_fetch_not_the_mark():
    # Open interest is RESTATED by the venue. A row claiming to have been
    # knowable at the mark would be a lookahead the store exists to refuse.
    row = _row()
    assert row['available_time'] == NOW
    assert row['event_time'] < row['available_time']


def test_the_depth_row_still_carries_only_the_closing_quote():
    # `venue_depth` must not drift into holding candle internals; the two
    # writers describe different concepts and are read by different code.
    depth = rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    assert len(depth) == 1
    assert depth[0]['yes_bid'] == pytest.approx(0.0530)
    assert depth[0]['source'] == 'candles'
    assert set(depth[0]) <= set(SCHEMAS['venue_depth'])


def test_the_event_key_separates_the_three_symbols_in_one_window():
    # Three symbols share a window_open, so the ticker must be in the key or
    # `_merge` destroys two of every three rows at WRITE time.
    key = event_key('venue_candles')
    assert 'market_ticker' in key and 'offset_minutes' in key


def test_a_candle_with_nothing_in_it_is_not_stored():
    empty = {'end_period_ts': WIRE['end_period_ts']}
    assert candle_rows_for_market(TICKER, 'ETH-USD', [empty], fetched_at=NOW) == []


def test_offsets_outside_the_window_are_refused():
    base = 1786665240 - 9 * 60          # the window open
    for off in (0, 16, 20):
        c = dict(WIRE, end_period_ts=base + off * 60)
        assert candle_rows_for_market(TICKER, 'ETH-USD', [c], fetched_at=NOW) == []


def test_a_frame_writes_through_the_store_schema(tmp_path):
    from core.datastore import ResearchStore
    store = ResearchStore(root=tmp_path)
    n = store.write('venue_candles', pd.DataFrame(
        candle_rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)))
    assert n == 1
    back = store.read('venue_candles', min_quality=None)
    assert len(back) == 1
    assert float(back.iloc[0]['open_interest']) == pytest.approx(12077.70)


def test_flush_writes_both_tables(tmp_path):
    """`_flush` is where the two datasets are actually written, and it was the
    one path with no test: extracting it to module scope lost access to the
    `pd` that `run()` imports locally, and it died with `NameError: name 'pd'
    is not defined` only after the tape backfill had finished and the chained
    run started. Everything else passed."""
    from core.datastore import ResearchStore
    from scripts.backfill_candles import _flush, rows_for_market

    store = ResearchStore(root=tmp_path)
    depth = rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    candles = candle_rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    assert _flush(store, depth, candles, False) == len(depth)
    assert len(store.read('venue_depth', min_quality=None)) == len(depth)
    assert len(store.read('venue_candles', min_quality=None)) == len(candles)


def test_flush_writes_nothing_on_a_dry_run(tmp_path):
    from core.datastore import ResearchStore
    from scripts.backfill_candles import _flush, rows_for_market

    store = ResearchStore(root=tmp_path)
    depth = rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    candles = candle_rows_for_market(TICKER, 'ETH-USD', [WIRE], fetched_at=NOW)
    assert _flush(store, depth, candles, True) == len(depth)
    assert not list(tmp_path.rglob('*.parquet'))


def test_flush_tolerates_an_empty_side(tmp_path):
    from core.datastore import ResearchStore
    from scripts.backfill_candles import _flush

    store = ResearchStore(root=tmp_path)
    assert _flush(store, [], [], False) == 0
