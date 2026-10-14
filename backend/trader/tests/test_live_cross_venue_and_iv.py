"""Tests for the live `cross_venue` and `implied_vol` feature rows.

Both groups are in the fitted artifact, so live must attach them. A missing
column scores NaN with a learned default direction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.live import cross_venue_row, implied_vol_row


class _Quote:
    def __init__(self, bid=0.60, ask=0.62):
        self.yes_bid, self.yes_ask = bid, ask


def test_cross_venue_gap_is_the_difference_of_the_two_mids():
    """Kalshi mid 0.61, Polymarket mid 0.57, so the gap is +0.04."""
    row = cross_venue_row({'best_bid': 56.0, 'best_ask': 58.0}, _Quote(0.60, 0.62))
    assert row['venue_prob_gap'] == pytest.approx(0.04, abs=1e-9)
    assert row['pm_available'] == 1.0


def test_a_missing_polymarket_book_is_absent_not_agreement():
    """A missing Polymarket book gives NaN, not zero (which would read as agreement)."""
    row = cross_venue_row(None, _Quote())
    assert np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 0.0


def test_a_one_sided_polymarket_book_gives_no_mid():
    """A one-sided Polymarket book has no mid."""
    row = cross_venue_row({'best_bid': 56.0, 'best_ask': np.nan}, _Quote())
    assert np.isnan(row['venue_prob_gap'])


def test_implied_vol_carries_the_log_ratio_against_realised():
    fit = {'implied_sigma_per_min': 8.0, 'r2': 0.98, 'n_strikes': 12.0,
           'at': pd.Timestamp('2026-08-28 19:00', tz='UTC')}
    row = implied_vol_row(fit, sigma_per_min=4.0,
                          now=pd.Timestamp('2026-08-28 19:30', tz='UTC'))
    assert row['iv_minus_realised'] == pytest.approx(np.log(2.0))
    assert row['implied_sigma_per_min'] == 8.0
    assert row['iv_r2'] == 0.98


def test_staleness_is_a_feature_not_a_filter():
    """Staleness is carried as a feature; a stale fit is not dropped."""
    fit = {'implied_sigma_per_min': 8.0, 'r2': 0.9, 'n_strikes': 10.0,
           'at': pd.Timestamp('2026-08-28 16:30', tz='UTC')}
    row = implied_vol_row(fit, sigma_per_min=4.0,
                          now=pd.Timestamp('2026-08-28 19:30', tz='UTC'))
    assert row['iv_staleness_minutes'] == pytest.approx(180.0)
    assert not np.isnan(row['implied_sigma_per_min']), 'stale is carried, not dropped'


def test_a_fit_older_than_the_cap_is_refused():
    """A fit older than MAX_FIT_AGE_MINUTES (360) is refused."""
    fit = {'implied_sigma_per_min': 8.0, 'r2': 0.9, 'n_strikes': 10.0,
           'at': pd.Timestamp('2026-08-28 05:00', tz='UTC')}
    row = implied_vol_row(fit, sigma_per_min=4.0,
                          now=pd.Timestamp('2026-08-28 19:30', tz='UTC'))
    assert np.isnan(row['implied_sigma_per_min'])
    assert np.isnan(row['iv_minus_realised'])


def test_no_fit_at_all_is_all_nan_rather_than_an_exception():
    row = implied_vol_row(None, sigma_per_min=4.0,
                          now=pd.Timestamp('2026-08-28 19:30', tz='UTC'))
    assert all(np.isnan(v) for v in row.values())


def test_a_rest_fallback_peer_for_another_window_is_refused_not_raised():
    """A REST fallback peer for another window is refused without raising.

    The REST fallback stores the window as `window_open`; the socket uses `window`.
    """
    peer = {'best_bid': 56.0, 'best_ask': 58.0,
            'window_open': pd.Timestamp('2026-10-14 10:00', tz='UTC')}
    row = cross_venue_row(peer, _Quote(0.60, 0.62),
                          window_open=pd.Timestamp('2026-10-14 10:15', tz='UTC'))
    assert np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 0.0
