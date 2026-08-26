"""A Polymarket slug's trailing timestamp is the window OPEN, not its close.

This was assumed backwards, and the assumption is invisible: every window is a
valid window, the books are real, the settlements are real, and everything is
simply shifted by fifteen minutes. It surfaced only because the venue's own
settlement could be scored against our Coinbase label — Kalshi agreed 96.98% of
the time and Polymarket agreed 49.85%, which is a coin flip and therefore an
alignment error rather than a data-quality one. Kalshi against Polymarket agreed
on exactly 50.0% of 118 shared windows, confirming it sat in the mapping and not
in either venue.

The venue states it three ways at once, and all three agree:

    slug     btc-updown-15m-1787707800   -> 2026-08-26 01:30 UTC
    title    "Bitcoin Up or Down - August 25, 9:30PM-9:45PM ET"
    end_time 2026-08-26T01:45:00Z        =  slug timestamp + 15 minutes

So `end_time - slug_ts == window_minutes` is the invariant, and it is what makes
the live recorder ask for the CURRENT window rather than the one after it.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from core.config import DEFAULT_CONFIG
from scripts.record_pm_ladder import slug_for, window_of

WINDOW = DEFAULT_CONFIG.window_minutes


def test_the_slug_timestamp_is_the_window_open():
    assert window_of('btc-updown-15m-1787707800') == pd.Timestamp(
        '2026-08-26 01:30', tz='UTC')


def test_end_time_is_the_slug_timestamp_plus_one_window():
    """The venue's own `end_time`, which is how this was settled."""
    opened = window_of('btc-updown-15m-1787707800')
    assert opened + pd.Timedelta(minutes=WINDOW) == pd.Timestamp(
        '2026-08-26 01:45', tz='UTC')


def test_the_live_recorder_asks_for_the_window_it_is_inside():
    """Mid-window, the slug must name THIS window, not the next one.

    Building it from `ceil` returned a market that exists and trades — the venue
    lists the next one early — so the book came back healthy and was stamped with
    the previous window's `window_open`. Nothing raised.
    """
    now = dt.datetime(2026, 8, 26, 1, 37, 12, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787707800'


def test_exactly_on_the_boundary_the_new_window_has_started():
    now = dt.datetime(2026, 8, 26, 1, 30, 0, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787707800'


def test_a_second_before_the_boundary_is_still_the_old_window():
    now = dt.datetime(2026, 8, 26, 1, 29, 59, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787706900'
