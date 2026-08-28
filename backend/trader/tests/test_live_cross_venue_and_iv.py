"""`_attach_book_features` filled only market_state and market_price.

`CROSS_VENUE` and `IMPLIED_VOL` were declared in FEATURE_GROUPS, fitted into the
promoted artifact, and never attached live — so nine of forty-nine features
scored NaN every cycle and LightGBM substituted a learned default.

Dropping the two groups instead is not an option and the measurement says so:
refitted without them, log_loss_skill goes +0.00307 -> -0.00023 and folds
positive 6/6 -> 3/6. They forecast nothing ALONE (the group-alone ablation put
both at or below the clock control) and are load-bearing in combination. Alone
is not the same question as leave-one-out.
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
    """A zeroed gap reads as two venues concurring, which is the opposite of
    what no book means."""
    row = cross_venue_row(None, _Quote())
    assert np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 0.0


def test_a_one_sided_polymarket_book_gives_no_mid():
    """A lone bid says the probability is at least something, which is not a
    probability."""
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
    """Coverage is ~15% of the timeline with a five-hour mean gap. A sigma
    forward-filled from three hours ago is a different claim from a fresh one,
    and the model has to be able to tell them apart."""
    fit = {'implied_sigma_per_min': 8.0, 'r2': 0.9, 'n_strikes': 10.0,
           'at': pd.Timestamp('2026-08-28 16:30', tz='UTC')}
    row = implied_vol_row(fit, sigma_per_min=4.0,
                          now=pd.Timestamp('2026-08-28 19:30', tz='UTC'))
    assert row['iv_staleness_minutes'] == pytest.approx(180.0)
    assert not np.isnan(row['implied_sigma_per_min']), 'stale is carried, not dropped'


def test_a_fit_older_than_the_cap_is_refused():
    """MAX_FIT_AGE_MINUTES is 360. Beyond that the sigma describes a different
    session and forward-filling it is a fabrication."""
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
