"""Tests for attaching observed book data to the window panel.

Quotes and fits attach at build time; `iv_minus_realised` needs the fitted
`sigma_per_min`, so it is computed per fold.
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
    """Each attach adds its own columns and touches no others."""
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
    """The ratio is NaN until a fold supplies `sigma_per_min`."""
    fits = pd.DataFrame({
        'symbol': ['BTC-USD'],
        'event_time': pd.to_datetime(['2026-07-01T11:50Z']),
        'implied_sigma_per_min': [0.0006], 'r2': [0.97], 'n_strikes': [9]})
    table = attach_implied_vol(_panel(), fits)
    assert 'sigma_per_min' not in _panel().columns
    assert table['iv_minus_realised'].isna().all()
    # The observed column is still present.
    assert table['implied_sigma_per_min'].iloc[0] == pytest.approx(0.0006)


def test_the_ratio_appears_once_the_fold_supplies_sigma():
    table = _panel().assign(sigma_per_min=[0.0003, 0.0003])
    fits = pd.DataFrame({
        'symbol': ['BTC-USD'],
        'event_time': pd.to_datetime(['2026-07-01T11:50Z']),
        'implied_sigma_per_min': [0.0006], 'r2': [0.97], 'n_strikes': [9]})
    got = attach_implied_vol(table, fits)
    assert got['iv_minus_realised'].iloc[0] == pytest.approx(np.log(2.0))
