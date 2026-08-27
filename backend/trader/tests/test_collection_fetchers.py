"""Venue drivers: parse a window, pack a book, and never guess.

Two classes of bug these guard against, both of which have already happened
here:

  * A window derived from a venue identifier that is plausible but wrong. The
    Polymarket slug was read as the window CLOSE when it is the OPEN, shifting
    every window fifteen minutes. Nothing raised: every window was a valid
    window and every book was a real book. It surfaced only as a settlement
    agreement rate of 49.85% against Kalshi's 96.98%. So the parse is checked
    against the venue's own stated times, and a disagreement is an error.

  * A request whose shape silently returns nothing. Omitting the
    start_time/end_time pair on the Kalshi orderbook endpoint returns an empty
    result for a window that certainly has a book — which is exactly how
    BOOK_COVERAGE_START came to sit five months late.
"""

from __future__ import annotations

import datetime as dt

import pytest

from research.collect.fetchers import (
    kalshi_window_open, pm_window_open, pack_kalshi, pack_pm, verify_window,
)

UTC = dt.timezone.utc


# -- Kalshi ticker -> window open -------------------------------------------

def test_a_kalshi_ticker_decodes_to_the_window_it_names():
    """Validated against a ticker held in our own settlement store:
    KXBTC15M-26JAN061730-30 was recorded against window_open 2026-01-06
    17:15 ET. The ticker encodes the CLOSE, in Eastern."""
    assert kalshi_window_open('KXBTC15M-26JAN061730-30') == \
        dt.datetime(2026, 1, 6, 22, 15, tzinfo=UTC)


def test_a_kalshi_ticker_in_summer_uses_eastern_daylight_time():
    """EDT, not a fixed UTC offset. KXBTC15M-26AUG262145-45 was served by the
    venue with close_time 2026-08-27T01:45Z, so its open is 01:30Z."""
    assert kalshi_window_open('KXBTC15M-26AUG262145-45') == \
        dt.datetime(2026, 8, 27, 1, 30, tzinfo=UTC)


def test_a_malformed_kalshi_ticker_raises_rather_than_guessing():
    with pytest.raises(ValueError):
        kalshi_window_open('KXBTC15M-nonsense')


# -- Polymarket slug -> window open ------------------------------------------

def test_a_polymarket_slug_stamp_is_the_window_open_not_its_close():
    """The bug that shifted every Polymarket window by fifteen minutes."""
    ts = int(dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC).timestamp())
    assert pm_window_open(f'btc-updown-15m-{ts}') == \
        dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC)


def test_a_polymarket_slug_without_a_stamp_raises():
    with pytest.raises(ValueError):
        pm_window_open('btc-updown-15m-notanumber')


# -- the cross-check ---------------------------------------------------------

def test_a_window_agreeing_with_the_venues_own_times_verifies():
    opened = dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC)
    assert verify_window(opened, venue_open=opened,
                         venue_close=opened + dt.timedelta(minutes=15)) is None


def test_a_fifteen_minute_shift_is_caught_rather_than_accepted():
    """This is the exact error that ran undetected for weeks."""
    opened = dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC)
    problem = verify_window(opened,
                            venue_open=opened + dt.timedelta(minutes=15),
                            venue_close=opened + dt.timedelta(minutes=30))
    assert problem is not None and 'open' in problem.lower()


def test_a_window_of_the_wrong_length_is_caught():
    opened = dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC)
    problem = verify_window(opened, venue_open=opened,
                            venue_close=opened + dt.timedelta(minutes=60))
    assert problem is not None


def test_missing_venue_times_are_not_treated_as_agreement():
    """A venue that returns no times cannot corroborate anything; saying
    'verified' there would be the same false confidence in a new place."""
    opened = dt.datetime(2026, 1, 8, 14, 0, tzinfo=UTC)
    assert verify_window(opened, venue_open=None, venue_close=None) is not None


# -- packing -----------------------------------------------------------------

def test_a_kalshi_snapshot_packs_touch_depth_and_totals():
    snap = {'timestamp': 1767268800000,
            'yes_bids': [{'price': 44, 'size': 25}, {'price': 43, 'size': 99},
                         {'price': 39, 'size': 30}],
            'yes_asks': [{'price': 46, 'size': 99}, {'price': 47, 'size': 40}]}
    row = pack_kalshi(snap)
    assert row[1] == 44 and row[2] == 46          # best bid / best ask
    assert row[3] == 25 and row[4] == 99          # size at each touch
    assert row[11] == 154 and row[12] == 139      # total resting each side


def test_a_one_sided_kalshi_book_packs_without_inventing_the_other_side():
    row = pack_kalshi({'timestamp': 1, 'yes_bids': [], 'yes_asks': [
        {'price': 99, 'size': 5}]})
    assert row[1] is None and row[2] == 99
    assert row[3] == 0


def test_an_empty_kalshi_snapshot_is_packable_and_marks_no_touch():
    row = pack_kalshi({'timestamp': 1, 'yes_bids': [], 'yes_asks': []})
    assert row[1] is None and row[2] is None and row[11] == 0


def test_polymarket_prices_are_converted_to_cents_to_match_kalshi():
    """Polymarket serves dollars, Kalshi integer cents. A schema that agrees
    on column names while disagreeing on units is worse than one that
    disagrees openly."""
    snap = {'timestamp': 1767268800000,
            'bids': [{'price': '0.44', 'size': '25'}],
            'asks': [{'price': '0.46', 'size': '99'}]}
    row = pack_pm(snap)
    assert row[1] == 44 and row[2] == 46


def test_polymarket_depth_within_a_cent_uses_the_same_definition_as_kalshi():
    snap = {'timestamp': 1,
            'bids': [{'price': '0.44', 'size': '10'}, {'price': '0.43', 'size': '5'},
                     {'price': '0.30', 'size': '99'}],
            'asks': []}
    row = pack_pm(snap)
    assert row[5] == 15, 'within 1c of the touch is 44c and 43c, not 30c'
