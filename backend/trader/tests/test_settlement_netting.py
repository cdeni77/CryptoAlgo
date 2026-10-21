"""Revenue on a market where BOTH sides were held.

Kalshi nets offsetting contracts within one market and credits that cash when
the netting happens rather than at settlement, so the settlement row reports
`revenue: 0` for the netted part. Reading it straight understates the market's
P&L by exactly the netted amount -- measured 2026-10-20 as a $5.00 hole in one
day's realised P&L, which was the whole of that day's disagreement with the
venue's own cash.
"""
from __future__ import annotations

import pytest

from data_collection.kalshi_client import parse_settlement, settlement_payout


# The three real rows that exposed this, verbatim in the venue's own shape.
BOTH_SIDES = [
    # ticker, yes_count, no_count, value, market_result, owed
    ('KXETH15M-26OCT200715-15', '1.00', '4.00', 100, 'yes', 1.0),
    ('KXSOL15M-26OCT200745-45', '2.00', '2.00', 100, 'yes', 2.0),
    ('KXETH15M-26OCT200800-00', '2.00', '2.00', 0, 'no', 2.0),
]


def _row(ticker, yes_fp, no_fp, value, result, *, revenue=0):
    return {
        'ticker': ticker,
        'event_ticker': ticker.rsplit('-', 1)[0],
        'market_result': result,
        'yes_count_fp': yes_fp,
        'no_count_fp': no_fp,
        'yes_total_cost_dollars': '0.000000',
        'no_total_cost_dollars': '0.000000',
        'revenue': revenue,
        'value': value,
        'fee_cost': '0.000000',
        'settled_time': '2026-10-20T11:15:06.830436Z',
    }


@pytest.mark.parametrize('ticker,yes_fp,no_fp,value,result,owed', BOTH_SIDES)
def test_netted_market_pays_its_winning_contracts(
        ticker, yes_fp, no_fp, value, result, owed):
    """The venue says revenue 0; the winning contracts are still owed $1 each."""
    raw = _row(ticker, yes_fp, no_fp, value, result)
    assert raw['revenue'] == 0, 'the fixture must reproduce the venue’s zero'
    assert parse_settlement(raw).revenue == pytest.approx(owed)


def test_the_served_revenue_is_still_visible():
    """Overriding the field must not hide what the venue actually sent."""
    raw = _row(*BOTH_SIDES[0][:5])
    assert parse_settlement(raw).raw['revenue'] == 0


def test_an_ordinary_market_is_unchanged():
    """One side only: the computed payout must agree with the served revenue."""
    raw = _row('KXETH15M-26OCT201445-45', '0.00', '3.00', 0, 'no', revenue=300)
    assert parse_settlement(raw).revenue == pytest.approx(3.0)


def test_a_loser_still_settles_at_zero():
    """Zero is a measurement here, not a missing field."""
    raw = _row('KXBTC15M-26OCT200000-00', '0.00', '1.00', 100, 'yes')
    assert parse_settlement(raw).revenue == pytest.approx(0.0)


def test_an_unresolved_row_is_not_a_loss():
    """Nothing says which side won, so the payout is unknown, not zero."""
    raw = _row('KXBTC15M-26OCT200000-00', '1.00', '0.00', None, None)
    raw.pop('value')
    assert settlement_payout(raw) is None
