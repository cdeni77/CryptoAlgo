"""`fee_cost` on a settlement is already dollars. `revenue` is integer cents.

Both arrive without a `_dollars` suffix, and `_money` falls back to multiplying
an unsuffixed field by a cent. That is right for `revenue` and wrong for
`fee_cost`, so the fee is stored a hundred times too small and
`pnl = revenue - cost - fee_cost` is inflated by the difference. Measured on the
live account: 365 settlements reported $0.28 of fees where the venue had charged
about $28, and realised P&L read $40.34 instead of roughly $12.62.

Every existing fixture for this used `fee_cost_dollars`, which the venue does not
serve on a settlement — invented rather than copied off the wire, so the suite
passed while the real path was wrong.

The proof that it is dollars is the published schedule. This row, read from the
live account:

    no_count_fp             5.00
    no_total_cost_dollars   4.515000
    revenue                 500          <- cents, $5.00
    fee_cost                0.030700

    ceil(0.07 * 5 * 0.903 * 0.097 * 10000) / 10000 = 0.0307

which is the fee in dollars, to the hundredth of a cent, exactly as
`core/costs.py` computes it. A fee is never a round number of cents on this
venue, so an unsuffixed `fee_cost` that is already fractional cannot be a
cents-encoded integer.
"""

from __future__ import annotations

import math

import pytest

from data_collection.kalshi_client import parse_settlement

# Copied verbatim from a live settlement, keys and all.
WIRE = {
    'ticker': 'KXBTC15M-26AUG260730-30',
    'event_ticker': 'KXBTC15M-26AUG260730',
    'market_result': 'no',
    'yes_count_fp': '0.00', 'no_count_fp': '5.00',
    'yes_total_cost_dollars': '0.000000',
    'no_total_cost_dollars': '4.515000',
    'revenue': 500,
    'fee_cost': 0.030700,
    'settled_time': '2026-08-26T11:30:07.006492Z',
}


def test_the_fee_is_read_as_dollars_not_cents():
    assert parse_settlement(WIRE).fee_cost == pytest.approx(0.0307, abs=1e-9)


def test_the_fee_matches_the_published_schedule():
    """0.07 * n * p * (1-p), to the hundredth of a cent. Same as core/costs."""
    price = 4.515 / 5.0
    expected = math.ceil(0.07 * 5 * price * (1 - price) * 10_000.0) / 10_000.0
    assert parse_settlement(WIRE).fee_cost == pytest.approx(expected, abs=1e-9)


def test_revenue_is_still_read_from_cents():
    """The same fallback is correct here — five contracts paying $1 each."""
    assert parse_settlement(WIRE).revenue == pytest.approx(5.00, abs=1e-9)


def test_the_pnl_charges_the_whole_fee():
    """5.00 - 4.515 - 0.0307. Understating the fee inflates every settled trade."""
    assert parse_settlement(WIRE).pnl == pytest.approx(0.4543, abs=1e-9)


def test_a_suffixed_fee_still_wins_where_the_venue_sends_one():
    served = dict(WIRE)
    served['fee_cost_dollars'] = '0.0400'
    assert parse_settlement(served).fee_cost == pytest.approx(0.04, abs=1e-9)


def test_a_zero_fee_stays_zero_rather_than_becoming_missing():
    served = dict(WIRE)
    served['fee_cost'] = 0.0
    assert parse_settlement(served).fee_cost == 0.0


def test_an_absent_fee_is_none_and_does_not_silently_become_free():
    served = {k: v for k, v in WIRE.items() if k != 'fee_cost'}
    assert parse_settlement(served).fee_cost is None
