"""The label no longer needs a research key in the trading container.

`collect_settlements` read the venue's `result` through Predexon, and Predexon
needs `PREDEXON_API_KEY` — which meant putting a third-party research key
alongside the Kalshi trading credential just to keep the training label current.

But Kalshi serves the same field itself. Measured on the live account:

    GET /markets?series_ticker=KXBTC15M&status=settled&limit=200
      -> 200 markets, 200/200 with a non-empty `result`
         KXBTC15M-26SEP031130-30  status=finalized  result=yes
         spanning 2026-09-01 .. 2026-09-03, cursor present

So Predexon was only ever needed for HISTORY — Kalshi purges older markets,
which is why the 196-day / 62,097-row backfill went through it. Going forward
the ongoing collection needs no new credential at all, because the live
container is already authenticated to Kalshi.

This is not a cosmetic simplification. `venue_outcome` closes a measured 43%
label leak: training on our Coinbase label while pricing against BRTI-based
quotes let the model bet the index disagreement (win rate 72.77% where the
labels differ against 56.17% where they agree). A label path that depends on a
key nobody remembered to pass is a leak waiting to reopen — which is exactly
what happened between 2026-08-27 and 2026-09-03.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from scripts.collect_settlements import rows_from_kalshi_markets


def _market(ticker, result, close, status='finalized'):
    return {'ticker': ticker, 'status': status, 'result': result,
            'close_time': close, 'volume': 12.0, 'open_interest': 3.0,
            'last_price': 0.55}


def test_a_settled_market_becomes_a_settlement_row():
    rows = rows_from_kalshi_markets(
        [_market('KXBTC15M-26SEP031130-30', 'yes', '2026-09-03T15:30:00Z')],
        symbol='BTC-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert len(rows) == 1
    r = rows[0]
    assert r['venue'] == 'kalshi' and r['symbol'] == 'BTC-USD'
    assert r['result'] == 'yes' and r['settled_up'] is True
    # The window OPENS fifteen minutes before it closes.
    assert r['window_open'] == pd.Timestamp('2026-09-03 15:15', tz='UTC')


def test_a_no_result_is_settled_down():
    rows = rows_from_kalshi_markets(
        [_market('KXETH15M-26SEP031130-30', 'no', '2026-09-03T15:30:00Z')],
        symbol='ETH-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows[0]['settled_up'] is False


def test_a_market_without_a_result_is_skipped():
    """`status=settled` can return a market mid-finalisation. A missing result
    is not a 'no' — that would fabricate half the labels it touched."""
    rows = rows_from_kalshi_markets(
        [_market('KXBTC15M-X', '', '2026-09-03T15:30:00Z', status='closed'),
         _market('KXBTC15M-Y', None, '2026-09-03T15:30:00Z')],
        symbol='BTC-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows == []


def test_a_market_with_no_close_time_is_skipped():
    """The window is derived from the close, so without one there is no row to
    key on."""
    rows = rows_from_kalshi_markets(
        [_market('KXBTC15M-Z', 'yes', None)],
        symbol='BTC-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows == []
