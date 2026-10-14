"""Tests that the live implied-vol recorder covers BTC, ETH and SOL.

The venue serves `KXBTCD`, `KXETHD` and `KXSOLD` with the same shape, and the
backfill covers all three, so live must too.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts.record_implied_vol import LADDERS, fits_for


def _market(event, strike, bid, ask, close):
    return {'event_ticker': event, 'close_time': close,
            'floor_strike': strike,
            'yes_bid_dollars': f'{bid:.4f}', 'yes_ask_dollars': f'{ask:.4f}'}


def test_all_three_crypto_ladders_are_recorded():
    assert dict(LADDERS) == {'KXBTCD': 'BTC-USD',
                             'KXETHD': 'ETH-USD',
                             'KXSOLD': 'SOL-USD'}


def test_rows_carry_the_symbol_of_the_series_they_came_from():
    """Each row carries the symbol of the series it came from."""
    now = dt.datetime(2026, 8, 28, 19, 0, tzinfo=dt.timezone.utc)
    close = (now + dt.timedelta(minutes=120)).isoformat()
    # A monotone ladder around a 3,200 spot: higher strikes are cheaper.
    markets = [_market('KXETHD-X', s, p - 0.01, p + 0.01, close)
               for s, p in ((3100.0, 0.90), (3150.0, 0.75), (3200.0, 0.50),
                            (3250.0, 0.25), (3300.0, 0.10))]
    rows, latest = fits_for(markets, now=now, symbol='ETH-USD',
                            min_minutes=5.0, max_minutes=600.0, min_r2=0.0)
    assert rows, 'a well-formed ladder produced no fit'
    assert {r['symbol'] for r in rows} == {'ETH-USD'}
    assert latest is not None and latest['implied_sigma_per_min'] > 0


def test_a_ladder_outside_the_time_band_is_skipped():
    now = dt.datetime(2026, 8, 28, 19, 0, tzinfo=dt.timezone.utc)
    close = (now + dt.timedelta(minutes=2)).isoformat()
    markets = [_market('KXSOLD-X', s, p - 0.01, p + 0.01, close)
               for s, p in ((95.0, 0.90), (100.0, 0.50), (105.0, 0.10))]
    rows, latest = fits_for(markets, now=now, symbol='SOL-USD',
                            min_minutes=5.0, max_minutes=600.0, min_r2=0.0)
    assert rows == [] and latest is None
