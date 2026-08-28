"""The live implied-vol recorder was BTC-only; the venue is not.

`record_implied_vol.py` hardcoded `SERIES = 'KXBTCD'` / `SYMBOL = 'BTC-USD'`,
so ETH and SOL carried NaN for all five implied-vol features on every live
cycle. The venue serves `KXETHD` and `KXSOLD` with the same shape — 200 open
markets each, every one carrying a strike, `strike_type=greater` — and the
BACKFILL already covers all three (BTC 22,248 rows, ETH 5,286, SOL 3,110).

So live was the half that lagged: the model was fitted against ETH/SOL implied
vol that existed, and scored live against NaN.
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
    """The bug this replaces stamped every row 'BTC-USD' regardless."""
    now = dt.datetime(2026, 8, 28, 19, 0, tzinfo=dt.timezone.utc)
    close = (now + dt.timedelta(minutes=120)).isoformat()
    # A monotone ladder around a 3,200 spot: deeper strikes are cheaper.
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
