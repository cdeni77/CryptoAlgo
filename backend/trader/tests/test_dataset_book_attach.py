"""Attaching observed book data to the window panel, and where the ratio lives.

Quotes and ladder fits are OBSERVED, not fitted, so they attach when the panel
is built — unlike the volatility model, the seasonality factor and the
baseline's scale/tail, all three of which must be fitted inside the fold.

But `iv_minus_realised` divides the market's implied sigma by the baseline's
`sigma_per_min`, and that denominator IS fitted. So the observed columns attach
at build time and the ratio is computed in `build_features`, where the fitted
sigma exists. Computing it at build time would use a sigma fitted on the whole
sample — a leak that makes the baseline stronger and the model look weaker,
which is the direction nobody audits.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.book_features import attach_implied_vol
from core.quotes import attach_quotes


def _panel():
    return pd.DataFrame({
        'symbol': ['BTC-USD', 'BTC-USD'],
        'window_open': pd.to_datetime(['2026-07-01T12:00Z'] * 2),
        'offset': [3, 12],
        'decision_time': pd.to_datetime(['2026-07-01T12:03Z', '2026-07-01T12:12Z']),
        'baseline_probability': [0.40, 0.40],
    })


def test_the_observed_columns_survive_both_attachments():
    """Order must not matter: each attach adds columns and touches no others."""
    depth = pd.DataFrame([{
        'venue': 'kalshi', 'symbol': 'BTC-USD',
        'window_open': pd.Timestamp('2026-07-01T12:00Z'),
        'offset_minutes': 3, 'yes_bid': 0.44, 'yes_ask': 0.46,
        'quote_age_seconds': 2.0, 'source': 'backfill'}])
    fits = pd.DataFrame({
        'symbol': ['BTC-USD'],
        'event_time': pd.to_datetime(['2026-07-01T11:50Z']),
        'implied_sigma_per_min': [0.0006], 'r2': [0.97], 'n_strikes': [9]})

    table = attach_implied_vol(attach_quotes(_panel(), depth), fits)
    assert table['ask_up'].iloc[0] == pytest.approx(0.46)
    assert table['implied_sigma_per_min'].iloc[0] == pytest.approx(0.0006)
    assert len(table) == 2


def test_the_ratio_is_not_computed_before_a_fitted_sigma_exists():
    """`sigma_per_min` is absent on a freshly built panel — it is attached per
    fold. The ratio must be NaN then, not silently computed against nothing."""
    fits = pd.DataFrame({
        'symbol': ['BTC-USD'],
        'event_time': pd.to_datetime(['2026-07-01T11:50Z']),
        'implied_sigma_per_min': [0.0006], 'r2': [0.97], 'n_strikes': [9]})
    table = attach_implied_vol(_panel(), fits)
    assert 'sigma_per_min' not in _panel().columns
    assert table['iv_minus_realised'].isna().all()
    # ...but the OBSERVED part is present and usable
    assert table['implied_sigma_per_min'].iloc[0] == pytest.approx(0.0006)


def test_the_ratio_appears_once_the_fold_supplies_sigma():
    from core.book_features import implied_vol_features
    table = _panel().assign(sigma_per_min=[0.0003, 0.0003])
    fits = pd.DataFrame({'implied_sigma_per_min': [0.0006, 0.0006],
                         'r2': [0.97, 0.97], 'n_strikes': [9, 9],
                         'staleness_minutes': [10.0, 10.0]})
    got = implied_vol_features(table, fits)
    assert got['iv_minus_realised'].iloc[0] == pytest.approx(np.log(2.0))
