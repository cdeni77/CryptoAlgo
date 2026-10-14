"""Tests that a decision uses only a book seen at or before the decision instant.

Minute-grid recorders store how long after the minute they sampled; the backfill
stores how long before. One `abs(age) <= cap` rule would let a book recorded
after the decision price it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.quotes import attach_quotes, decision_age

W = pd.Timestamp('2026-10-01T12:00Z')


def _windows(offset=12):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': W, 'offset': offset,
                          'baseline_probability': 0.5}])


def _row(venue, source, age, bid=0.44, ask=0.46, offset=12):
    return {'venue': venue, 'symbol': 'BTC-USD', 'window_open': W,
            'offset_minutes': offset, 'yes_bid': bid, 'yes_ask': ask,
            'quote_age_seconds': age, 'source': source,
            'event_time': W + pd.Timedelta(minutes=offset),
            'available_time': W + pd.Timedelta(minutes=offset)}


def test_a_recorder_row_seen_after_the_decision_does_not_price_it():
    got = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 'live', 25.0)]))
    assert np.isnan(got.loc[0, 'ask_up'])


def test_a_recorder_row_seen_before_the_decision_prices_it():
    got = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 'live_ws', 1.0)]))
    assert got.loc[0, 'ask_up'] == 0.46
    assert got.loc[0, 'quote_age_seconds'] == 2.0


def test_a_backfill_row_counts_back_from_the_decision():
    got = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 'backfill', 10.0)]))
    assert got.loc[0, 'ask_up'] == 0.46
    assert got.loc[0, 'quote_age_seconds'] == 13.0
    stale = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 'backfill', 28.0)]),
                          max_age_seconds=30.0)
    assert np.isnan(stale.loc[0, 'ask_up'])


def test_the_live_touch_row_is_the_decision_book():
    got = attach_quotes(_windows(), pd.DataFrame([_row('kalshi', 'live_touch', 0.4)]))
    assert got.loc[0, 'ask_up'] == 0.46


def test_a_peer_book_from_after_the_decision_is_not_a_feature():
    depth = pd.DataFrame([_row('kalshi', 'live_touch', 0.4),
                          _row('polymarket', 'live', 29.0, bid=0.50, ask=0.52)])
    got = attach_quotes(_windows(), depth)
    assert got.loc[0, 'ask_up'] == 0.46
    assert np.isnan(got.loc[0, 'pm_market_probability'])


def test_a_peer_book_from_before_the_decision_is_a_feature():
    depth = pd.DataFrame([_row('kalshi', 'live_touch', 0.4),
                          _row('polymarket', 'backfill', 2.0, bid=0.50, ask=0.52)])
    got = attach_quotes(_windows(), depth)
    assert got.loc[0, 'pm_market_probability'] == 0.51


def test_the_latency_moves_the_decision_instant():
    depth = pd.DataFrame([_row('kalshi', 'live', 5.0)])
    assert np.isnan(attach_quotes(_windows(), depth,
                                  decision_latency_seconds=3.0).loc[0, 'ask_up'])
    assert attach_quotes(_windows(), depth,
                         decision_latency_seconds=6.0).loc[0, 'ask_up'] == 0.46


def test_decision_age_has_one_sign_convention():
    book = pd.DataFrame([_row('kalshi', 'backfill', 4.0),
                         _row('kalshi', 'live', 4.0),
                         _row('kalshi', 'live_touch', 1.0)])
    age = decision_age(book, 'fresh', 3.0).tolist()
    assert age == [7.0, -1.0, 1.0]


def test_an_implied_vol_fit_captured_after_the_decision_is_not_used():
    """An implied-vol fit captured after the decision (by `available_time`) is not used.
    """
    from core.book_features import attach_implied_vol

    table = pd.DataFrame({'symbol': ['BTC-USD'],
                          'decision_time': [W + pd.Timedelta(minutes=12)],
                          'sigma_per_min': [0.001]})
    minute = W + pd.Timedelta(minutes=12)

    def fits(captured_after):
        return pd.DataFrame({'symbol': ['BTC-USD'], 'event_time': [minute],
                             'available_time': [minute + pd.Timedelta(
                                 seconds=captured_after)],
                             'implied_sigma_per_min': [0.002], 'r2': [0.9],
                             'n_strikes': [8]})

    assert np.isnan(attach_implied_vol(table, fits(40.0))
                    .loc[0, 'implied_sigma_per_min'])
    assert attach_implied_vol(table, fits(1.0)).loc[0, 'implied_sigma_per_min'] == 0.002
