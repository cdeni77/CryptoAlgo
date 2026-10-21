"""Tests for `KalshiClient.position_size` on real V2 position rows.

V2 sends `position_fp`, negative for a NO position, and no `position` field.
"""

from __future__ import annotations

import pytest

from data_collection.kalshi_client import KalshiClient

# From the venue, 5 NO contracts on BTC.
REAL_ROW = {
    'exchange_index': 0,
    'fees_paid_dollars': '0.085800',
    'last_updated_ts': '2026-08-24T13:49:13.549069Z',
    'market_exposure_dollars': '2.150000',
    'position_fp': '-5.00',
    'realized_pnl_dollars': '0.000000',
    'ticker': 'KXBTC15M-26AUG241000-00',
    'total_traded_dollars': '2.150000',
}


def test_a_real_no_position_is_seen_as_open():
    """A real NO position reads as -5, not 0."""
    assert KalshiClient.position_size(REAL_ROW) == pytest.approx(-5.0)
    assert KalshiClient.position_size(REAL_ROW) != 0, (
        'a position we watched fill read as flat'
    )


def test_the_field_the_old_documentation_names_is_simply_absent():
    """The real row has no `position` field."""
    assert 'position' not in REAL_ROW
    assert int(REAL_ROW.get('position') or 0) == 0, (
        'the old expression really did evaluate to zero on a real open position'
    )


def test_a_short_and_a_long_are_both_open():
    """The sign carries the side; `!= 0` is the open test, since `> 0` would drop every
    NO position.
    """
    long_yes = dict(REAL_ROW, position_fp='7.00')
    assert KalshiClient.position_size(long_yes) == pytest.approx(7.0)
    assert KalshiClient.position_size(REAL_ROW) < 0 < KalshiClient.position_size(long_yes)


def test_a_flat_market_is_flat():
    """A closed position's row reads zero."""
    assert KalshiClient.position_size(dict(REAL_ROW, position_fp='0.00')) == 0


@pytest.mark.parametrize('row', [
    {},  # nothing at all
    {'position': 5},  # the legacy integer encoding
    {'position': '5'},  # ... as a string
    {'position_fp': '5.00', 'position': 5},  # both, agreeing
])
def test_the_legacy_encoding_still_parses(row):
    """The legacy integer `position` field still parses."""
    expected = 0.0 if not row else 5.0
    assert KalshiClient.position_size(row) == pytest.approx(expected)


def test_garbage_reads_as_flat_rather_than_raising():
    """An unparseable value reads as flat instead of raising inside `reconcile_with_venue`."""
    assert KalshiClient.position_size({'position_fp': 'nonsense'}) == 0
    assert KalshiClient.position_size({'position_fp': None}) == 0
