"""Tests for parsing Kalshi's own settlement records, including `expiration_value`.

The settlement price lets the label check measure our Coinbase proxy's bias in basis
points.
"""

from __future__ import annotations

import datetime as dt

from research.collect.kalshi_settlements import SERIES, parse_market, wanted_series

UTC = dt.timezone.utc


def _market(ticker='KXBTC15M-26AUG271045-45', **over):
    m = {'ticker': ticker, 'status': 'finalized', 'result': 'yes',
         'expiration_value': 80383.64,
         'open_time': '2026-08-27T14:30:00Z',
         'close_time': '2026-08-27T14:45:00Z'}
    m.update(over)
    return m


def test_a_settled_market_becomes_a_record_with_its_settlement_price():
    row = parse_market(_market())
    assert row['market_id'] == 'KXBTC15M-26AUG271045-45'
    assert row['symbol'] == 'BTC-USD'
    assert row['result'] == 'yes'
    assert row['expiration_value'] == 80383.64
    assert row['status'] == 'finalized'


def test_the_window_is_decoded_and_cross_checked_against_the_venues_times():
    """The record carries the window open in UTC, checked against the venue's own open
    and close times.
    """
    row = parse_market(_market())
    assert row['window_open'] == '2026-08-27T14:30:00+00:00'


def test_a_market_whose_stated_times_contradict_its_ticker_is_rejected():
    assert parse_market(_market(open_time='2026-08-27T15:30:00Z',
                                close_time='2026-08-27T15:45:00Z')) is None


def test_a_market_with_no_stated_times_is_rejected_not_assumed_correct():
    assert parse_market(_market(open_time=None, close_time=None)) is None


def test_a_malformed_ticker_is_skipped():
    assert parse_market(_market(ticker='KXBTC15M-garbage')) is None


def test_a_market_outside_our_three_series_is_skipped():
    assert parse_market(_market(ticker='KXDOGE15M-26AUG271045-45')) is None


def test_a_settlement_price_of_zero_is_kept_not_treated_as_missing():
    """A settlement price of zero is kept: it is a measurement, not a missing value."""
    row = parse_market(_market(expiration_value=0.0))
    assert row is not None and row['expiration_value'] == 0.0


def test_a_market_with_no_settlement_price_still_yields_its_result():
    row = parse_market(_market(expiration_value=None))
    assert row is not None and row['result'] == 'yes'
    assert row['expiration_value'] is None


def test_an_unsettled_market_is_skipped_rather_than_stored_as_blank():
    assert parse_market(_market(status='active', result='')) is None


def test_the_three_series_we_trade_are_the_ones_walked():
    assert wanted_series() == list(SERIES)
    assert set(SERIES.values()) == {'BTC-USD', 'ETH-USD', 'SOL-USD'}
