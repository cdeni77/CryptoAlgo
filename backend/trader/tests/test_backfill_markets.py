"""Constructed tickers must round-trip, or every window shifts silently.

`/markets` lists only ~69 days back, but the per-ticker endpoints serve far
older markets — the limit was the LISTING. That makes the ticker an index we
can construct, and makes `ticker_for` the inverse of
`scripts.backfill_candles.window_from_ticker`.

The failure mode if it is wrong is the Polymarket-slug trap again: the ticker
names a real market, the venue returns a real strike, and it belongs to a
different window. Nothing raises.
"""

import datetime as dt

import pytest

from scripts.backfill_candles import WINDOW_MINUTES, window_from_ticker
from scripts.backfill_markets import SERIES, row_for, ticker_for


def test_the_ticker_round_trips_through_the_parser():
    # Across a DST boundary in New York, which is where a local-time ticker
    # is most likely to be wrong.
    for day in ('2026-03-07', '2026-03-09', '2026-06-01', '2026-11-02'):
        base = dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc)
        for step in range(0, 96):
            opened = base + dt.timedelta(minutes=15 * step)
            close = opened + dt.timedelta(minutes=WINDOW_MINUTES)
            tk = ticker_for('KXBTC15M', close)
            back = window_from_ticker(tk)
            assert back is not None, tk
            assert back[0] == opened, (tk, back[0], opened)
            assert back[1] == close, (tk, back[1], close)


def test_the_suffix_is_the_close_minute():
    close = dt.datetime(2026, 10, 20, 22, 45, tzinfo=dt.timezone.utc)
    tk = ticker_for('KXBTC15M', close)
    assert tk.endswith('-45'), tk
    assert tk.startswith('KXBTC15M-'), tk


def test_every_series_builds():
    close = dt.datetime(2026, 5, 1, 12, 30, tzinfo=dt.timezone.utc)
    for series in SERIES:
        assert window_from_ticker(ticker_for(series, close)) is not None


def test_a_market_with_nothing_usable_is_not_stored():
    now = dt.datetime(2026, 10, 21, tzinfo=dt.timezone.utc)
    assert row_for({'ticker': 'X'}, 'BTC-USD', now, now, now) is None
    assert row_for({'ticker': 'X', 'floor_strike': ''}, 'BTC-USD', now, now, now) is None


def test_a_strike_without_a_settlement_is_still_stored():
    # Markets before ~2026-01-14 carry `floor_strike` but an empty
    # `expiration_value`; the strike alone is worth having.
    now = dt.datetime(2026, 10, 21, tzinfo=dt.timezone.utc)
    r = row_for({'ticker': 'X', 'floor_strike': '90857.52', 'expiration_value': '',
                 'result': 'yes'}, 'BTC-USD', now, now, now)
    assert r['floor_strike'] == pytest.approx(90857.52)
    assert r['expiration_value'] is None
    assert r['result'] == 'yes'


def test_the_row_is_keyed_on_the_window_open():
    from core.datastore import SCHEMAS
    opened = dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone.utc)
    close = opened + dt.timedelta(minutes=15)
    r = row_for({'ticker': 'X', 'floor_strike': '1'}, 'BTC-USD', opened, close, close)
    assert r['event_time'] == opened and r['window_open'] == opened
    assert r['close_time'] == close
    assert set(r) <= set(SCHEMAS['venue_strikes'])
