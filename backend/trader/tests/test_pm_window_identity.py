"""Tests that a Polymarket slug's trailing timestamp is the window open, not its close.

The venue states it three ways that agree: the slug timestamp, the title, and
`end_time`, which equals the slug timestamp plus 15 minutes. The live recorder
must ask for the window it is inside.
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
    """The venue's `end_time` is the slug timestamp plus one window."""
    opened = window_of('btc-updown-15m-1787707800')
    assert opened + pd.Timedelta(minutes=WINDOW) == pd.Timestamp(
        '2026-08-26 01:45', tz='UTC')


def test_the_live_recorder_asks_for_the_window_it_is_inside():
    """Mid-window the slug names this window, not the next one.

    The venue lists the next window early, so a slug built with `ceil` still returns
    a healthy book, for the wrong window.
    """
    now = dt.datetime(2026, 8, 26, 1, 37, 12, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787707800'


def test_exactly_on_the_boundary_the_new_window_has_started():
    now = dt.datetime(2026, 8, 26, 1, 30, 0, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787707800'


def test_a_second_before_the_boundary_is_still_the_old_window():
    now = dt.datetime(2026, 8, 26, 1, 29, 59, tzinfo=dt.timezone.utc)
    assert slug_for('btc', now) == 'btc-updown-15m-1787706900'
