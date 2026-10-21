"""The cross-shard total, which is the figure the venue's own app shows.

`balance(exchange_index=...)` answers what an ORDER can draw on. The app shows
the whole account. Recording only the first made the dashboard read $482.46
against an app saying $483.57 while drift sat at exactly 0.00 -- two correct
numbers, one of them invisible, and the difference looked like a sync fault.
"""
from __future__ import annotations

import math

from core.pg_writer import _finite_or_none


class _Payload:
    """The shape `/portfolio/balance` actually returns, measured 2026-10-18."""

    REAL = {
        'balance': 48356,
        'balance_breakdown': [
            {'balance': '1.0955', 'exchange_index': 0},
            {'balance': '0.0100', 'exchange_index': 1},
            {'balance': '482.4606', 'exchange_index': 2},
            {'balance': '0.0000', 'exchange_index': 3},
        ],
        'balance_dollars': '483.5661',
        'portfolio_value': 0,
    }


def _breakdown(payload: dict) -> dict:
    """The parsing half of `KalshiClient.balance_breakdown`, without the socket."""
    shards: dict[int, float] = {}
    rows = payload.get('balance_breakdown')
    if isinstance(rows, list):
        for row in rows:
            try:
                shards[int(row.get('exchange_index', -1))] = float(row.get('balance'))
            except (TypeError, ValueError):
                pass
    total = float('nan')
    dollars = payload.get('balance_dollars')
    if dollars is not None:
        try:
            total = float(dollars)
        except (TypeError, ValueError):
            pass
    return {'total': total, 'shards': shards}


def test_the_total_is_the_app_figure_and_the_shard_is_what_an_order_can_spend():
    detail = _breakdown(_Payload.REAL)
    assert detail['total'] == 483.5661
    assert detail['shards'][2] == 482.4606
    # The gap the user spotted: cash that exists and cannot buy a crypto contract.
    assert round(detail['total'] - detail['shards'][2], 4) == 1.1055


def test_the_dollars_field_is_preferred_because_the_cent_field_truncates():
    """Two reasons to read `balance_dollars`, and the second was a surprise.

    Reading the integer `balance` as dollars is a 100x error -- the same trap
    already recorded for quotes, where reading only the cents field reported
    "no two-sided book" against a market quoting 0.19/0.20.

    And the cent field is not merely coarser, it TRUNCATES: measured, the venue
    served `balance: 48356` beside `balance_dollars: '483.5661'`, so the cent
    field is 0.61c light. Sub-cent cash is real here -- maker fills land on the
    deci-cent grid and fees round to a hundredth of a cent -- so the dollars
    field is the exact one, not a prettier rendering of the same number.
    """
    detail = _breakdown(_Payload.REAL)
    cents_as_dollars = _Payload.REAL['balance'] / 100.0
    assert detail['total'] == 483.5661
    assert cents_as_dollars == 483.56
    # Light, never heavy: truncation can only lose money, never invent it.
    assert 0 < detail['total'] - cents_as_dollars < 0.01


def test_a_missing_total_is_nan_rather_than_zero():
    """A zero total would render as a drained account and, worse, as a
    `stranded` of minus the whole balance."""
    detail = _breakdown({'balance_breakdown': [{'balance': '5.0', 'exchange_index': 2}]})
    assert math.isnan(detail['total'])
    assert detail['shards'][2] == 5.0


def test_nan_is_stored_as_null_not_as_a_number():
    """NaN in a DOUBLE PRECISION column poisons every AVG and MAX downstream,
    silently. NULL is the honest encoding of 'the venue did not say'."""
    assert _finite_or_none(float('nan')) is None
    assert _finite_or_none(float('inf')) is None
    assert _finite_or_none(None) is None
    assert _finite_or_none(483.5661) == 483.5661


def test_an_unparseable_shard_row_does_not_lose_the_other_shards():
    detail = _breakdown({
        'balance_dollars': '10.00',
        'balance_breakdown': [
            {'balance': 'not-a-number', 'exchange_index': 0},
            {'balance': '9.50', 'exchange_index': 2},
        ],
    })
    assert detail['shards'][2] == 9.50
    assert detail['total'] == 10.00
