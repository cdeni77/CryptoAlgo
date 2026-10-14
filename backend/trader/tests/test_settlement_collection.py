"""Tests that the settlement collector reads the venue's real field names.

Kalshi serves `volume_fp`, `open_interest_fp` and `last_price_dollars`, with the bare
names null. Reading only the bare names stores zeros that look like real values.
"""

from __future__ import annotations

import pandas as pd

from scripts.collect_settlements import rows_from_kalshi_markets

NOW = pd.Timestamp('2026-09-16T20:00:00Z')


def _market(**over):
    m = {'ticker': 'KXBTC15M-26SEP161500-00', 'result': 'yes',
         'close_time': '2026-09-16T15:15:00Z',
         'settlement_time': '2026-09-16T15:15:05Z',
         'volume': None, 'volume_fp': '4200',
         'open_interest': None, 'open_interest_fp': '1500',
         'last_price': None, 'last_price_dollars': '0.6300'}
    m.update(over)
    return m


def test_the_fixed_point_fields_are_read():
    row = rows_from_kalshi_markets([_market()], symbol='BTC-USD', now=NOW)[0]
    assert row['volume'] == 4200.0
    assert row['open_interest'] == 1500.0
    assert row['last_price'] == 0.63


def test_the_plain_fields_still_work_when_the_venue_sends_them():
    """The bare field names still work when the venue sends them."""
    row = rows_from_kalshi_markets(
        [_market(volume=99, volume_fp=None,
                 open_interest=7, open_interest_fp=None,
                 last_price_dollars=None, last_price=None)],
        symbol='BTC-USD', now=NOW)[0]
    assert row['volume'] == 99.0
    assert row['open_interest'] == 7.0


def test_a_genuinely_absent_field_is_zero_not_a_crash():
    row = rows_from_kalshi_markets(
        [_market(volume=None, volume_fp=None)], symbol='BTC-USD', now=NOW)[0]
    assert row['volume'] == 0.0


def test_the_settled_flag_follows_the_result():
    up = rows_from_kalshi_markets([_market(result='yes')], symbol='BTC-USD', now=NOW)[0]
    down = rows_from_kalshi_markets([_market(result='no')], symbol='BTC-USD', now=NOW)[0]
    assert up['settled_up'] is True and down['settled_up'] is False
