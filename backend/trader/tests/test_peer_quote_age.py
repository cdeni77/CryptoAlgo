"""The other venue had no staleness filter, so the gap could span two instants.

`attach_quotes` filters the traded venue at `max_age_seconds` (30) — a quote
older than that is NaN'd, because a book quoted twenty minutes ago is not a
price this decision could have taken. The PEER venue block computes
`pm_market_probability` with no age test at all: measured, Polymarket rows reach
899 seconds of staleness in the store.

`venue_prob_gap` is a DIFFERENCE of the two, and a difference is exactly where a
timing asymmetry turns into apparent skill. That matters more here than
anywhere: `cross_venue` is the only load-bearing group in the model — dropping
it takes skill from +0.00282 to -0.00015 — while scoring +0.000030 ALONE, below
the clock control. Useless alone and essential combined is what an interaction
looks like, and also what a leak looks like, and the pattern cannot tell them
apart.

Measured on the 431,198 rows the model actually sees, Polymarket is 4.0 seconds
STALER on average, so the leak is not the mean case. But it is fresher by more
than ten seconds on 6% of rows, and nothing bounds it.

Filtering both sides to the same tolerance is what makes the group's
contribution interpretable either way.
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
    """Absence of a stamp is not evidence of staleness — the backfilled
    fixtures carry rows without one, and dropping them would silently discard
    the sample rather than filter it."""
    peer = _row('polymarket', np.nan, bid=0.20, ask=0.22)
    out = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 2.0), peer]),
                        max_age_seconds=30.0)
    assert out['pm_market_probability'].iloc[0] == pytest.approx(0.21)


import pytest  # noqa: E402
