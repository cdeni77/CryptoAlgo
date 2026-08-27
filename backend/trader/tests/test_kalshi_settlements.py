"""The venue's own settlement, including the PRICE it settled at.

Two gaps this fills, both measured:

  * Predexon is missing `result` on 6,828 Kalshi markets that TRADED — and a
    market with $2.9M of volume unambiguously settled, so a blank result there
    is a provider gap rather than a void market. Kalshi's own API has them:
    KXBTC15M-26AUG062245-45 reads `status: finalized, result: yes` there and
    blank in our Predexon catalog.

  * `expiration_value` — the price the market actually settled at (80383.64 on
    a real BTC window). Nothing else in this project has it. Today
    `_validate_label.py` can only compare our Coinbase-derived UP/DOWN against
    the venue's UP/DOWN and report 97% agreement. With settlement PRICES the
    same check becomes numeric: how many basis points does our one-minute
    Coinbase mean differ from CF Benchmarks' BRTI? That turns a binary
    agreement rate into a measured bias.

Kalshi's own API is used rather than Predexon because it is free, unmetered,
handled six back-to-back requests with no throttling, and sits in a DIFFERENT
rate bucket — so this runs alongside the Predexon collection instead of
competing with its 1 req/s.

Its limit is retention, not rate: markets older than roughly two months are
purged (a ticker we hold in our own store 404s there), so this recovers the
recent months and cannot reach January.
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
    """The ticker encodes the CLOSE in Eastern; the record must carry the
    window OPEN in UTC, and it is only trusted when the venue's own open_time
    and close_time agree with the decode."""
    row = parse_market(_market())
    assert row['window_open'] == '2026-08-27T14:30:00+00:00'


def test_a_market_whose_stated_times_contradict_its_ticker_is_rejected():
    """The fifteen-minute-shift failure mode, caught rather than stored."""
    assert parse_market(_market(open_time='2026-08-27T15:30:00Z',
                                close_time='2026-08-27T15:45:00Z')) is None


def test_a_market_with_no_stated_times_is_rejected_not_assumed_correct():
    assert parse_market(_market(open_time=None, close_time=None)) is None


def test_a_malformed_ticker_is_skipped():
    assert parse_market(_market(ticker='KXBTC15M-garbage')) is None


def test_a_market_outside_our_three_series_is_skipped():
    assert parse_market(_market(ticker='KXDOGE15M-26AUG271045-45')) is None


def test_a_settlement_price_of_zero_is_kept_not_treated_as_missing():
    """Zero is a measurement. `_price` in the live client maps a zero QUOTE to
    None because an empty level means nothing is there — but a settlement
    value is an observation, and dropping a real zero would be the same class
    of error in the opposite direction."""
    row = parse_market(_market(expiration_value=0.0))
    assert row is not None and row['expiration_value'] == 0.0


def test_a_market_with_no_settlement_price_still_yields_its_result():
    """The result alone already fills a Predexon gap; the price is a bonus."""
    row = parse_market(_market(expiration_value=None))
    assert row is not None and row['result'] == 'yes'
    assert row['expiration_value'] is None


def test_an_unsettled_market_is_skipped_rather_than_stored_as_blank():
    """Storing a blank result would recreate exactly the ambiguity this is
    meant to remove."""
    assert parse_market(_market(status='active', result='')) is None


def test_the_three_series_we_trade_are_the_ones_walked():
    assert wanted_series() == list(SERIES)
    assert set(SERIES.values()) == {'BTC-USD', 'ETH-USD', 'SOL-USD'}
