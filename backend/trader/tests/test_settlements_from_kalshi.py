"""Tests for `rows_from_kalshi_markets`: settlement labels taken straight from Kalshi.

Ongoing collection needs only the trading credential; Predexon is for older history.
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
    # The window opens fifteen minutes before it closes.
    assert r['window_open'] == pd.Timestamp('2026-09-03 15:15', tz='UTC')


def test_a_no_result_is_settled_down():
    rows = rows_from_kalshi_markets(
        [_market('KXETH15M-26SEP031130-30', 'no', '2026-09-03T15:30:00Z')],
        symbol='ETH-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows[0]['settled_up'] is False


def test_a_market_without_a_result_is_skipped():
    """A market without a result is skipped; a missing result is not a 'no'."""
    rows = rows_from_kalshi_markets(
        [_market('KXBTC15M-X', '', '2026-09-03T15:30:00Z', status='closed'),
         _market('KXBTC15M-Y', None, '2026-09-03T15:30:00Z')],
        symbol='BTC-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows == []


def test_a_market_with_no_close_time_is_skipped():
    """A market with no close time is skipped, since the window is derived from it."""
    rows = rows_from_kalshi_markets(
        [_market('KXBTC15M-Z', 'yes', None)],
        symbol='BTC-USD', now=pd.Timestamp('2026-09-03 16:00', tz='UTC'))
    assert rows == []


# --- "nothing new" is success, not failure ---------------------------------
#
# A source that answered with nothing new is success. No source answering at all
# is a reachability failure. A false failure on every quiet cycle would train
# the reader to ignore a real one.

def _args(**kw):
    import argparse
    return argparse.Namespace(**{'venue': 'kalshi', 'source': 'kalshi',
                                 'full': False, **kw})


def test_no_new_settlements_from_a_reachable_venue_is_success(monkeypatch):
    import asyncio
    from scripts import collect_settlements as cs

    async def empty(now, store):
        return []

    monkeypatch.setattr(cs, 'kalshi_direct', empty)
    monkeypatch.setattr(cs, 'ResearchStore', lambda *a, **k: object())
    code = asyncio.run(cs.run(_args()))
    assert code == 0, 'the store being current is not a failure'


def test_no_source_answering_is_still_a_failure(monkeypatch):
    """No source answering still fails; with `--source predexon` and no key nothing is
    attempted.
    """
    import asyncio
    from scripts import collect_settlements as cs

    monkeypatch.setattr(cs, 'ResearchStore', lambda *a, **k: object())
    monkeypatch.delenv('PREDEXON_API_KEY', raising=False)
    code = asyncio.run(cs.run(_args(source='predexon')))
    assert code == 1, 'unreachable sources must still report failure'
