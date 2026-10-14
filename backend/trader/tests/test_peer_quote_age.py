"""Tests that the peer venue's quote gets the same staleness filter as the traded venue.

`venue_prob_gap` is the difference of the two venues' prices. If only one side
were age-filtered, a timing difference could look like skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.quotes import attach_quotes


def _row(venue, age, bid=0.50, ask=0.52):
    return {
        'venue': venue, 'symbol': 'BTC-USD',
        'window_open': pd.Timestamp('2026-09-03 12:00', tz='UTC'),
        'offset_minutes': 12,
        'event_time': pd.Timestamp('2026-09-03 12:12', tz='UTC'),
        'available_time': pd.Timestamp('2026-09-03 12:12', tz='UTC'),
        'quote_age_seconds': age,
        'yes_bid': bid, 'yes_ask': ask,
        'yes_bid_size': 100.0, 'yes_ask_size': 100.0,
        'depth_bid_1c': 100.0, 'depth_ask_1c': 100.0,
        'depth_bid_5c': 100.0, 'depth_ask_5c': 100.0,
        'depth_bid_total': 100.0, 'depth_ask_total': 100.0,
    }


def _windows():
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'window_open': [pd.Timestamp('2026-09-03 12:00', tz='UTC')],
        'offset': [12]})


def test_a_stale_peer_quote_is_refused_like_a_stale_traded_one():
    depth = pd.DataFrame([_row('kalshi', 2.0),
                          _row('polymarket', 600.0, bid=0.20, ask=0.22)])
    out = attach_quotes(_windows(), depth, max_age_seconds=30.0)
    assert np.isnan(out['pm_market_probability'].iloc[0]), (
        'a 600s-old peer quote reached the gap; the traded side would have '
        'been refused at 30s')


def test_a_fresh_peer_quote_still_attaches():
    depth = pd.DataFrame([_row('kalshi', 2.0),
                          _row('polymarket', 3.0, bid=0.20, ask=0.22)])
    out = attach_quotes(_windows(), depth, max_age_seconds=30.0)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.21)


def test_a_peer_row_with_no_age_recorded_is_kept():
    """A peer row with no recorded age is kept; a missing stamp is not evidence of
    staleness.
    """
    peer = _row('polymarket', np.nan, bid=0.20, ask=0.22)
    out = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 2.0), peer]),
                        max_age_seconds=30.0)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.21)


import pytest  # noqa: E402
