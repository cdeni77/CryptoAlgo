"""A killed order took nothing, so it must not hold the window's entry slot.

Live, 2026-08-26 02:12. Two `immediate_or_cancel` orders found nothing resting
at their limit and the venue cancelled both — `fill_count_fp: "0.00"`,
`taker_fill_cost_dollars: "0.000000"`, no position anywhere. The very next cycle
logged:

    window 2026-08-26 02:00:00+00:00 already holds ['BTC-USD', 'SOL-USD']
    ($8.55); those symbols will refuse

$8.55 is exactly the two *decision* stakes, $3.82 + $4.73. Nothing was held at
all. `entries_for_window` counted every ticket whose status was not 'skipped',
and a kill is written as 'killed'.

Tickets are counted on purpose: one exists from the moment an order is sent, so
a crash between sending and booking still shows up, and 'unknown' — a POST that
failed in flight and may well have been accepted — must keep holding the slot.
But a kill is the one outcome the venue has affirmatively told us produced no
position, so it is the one that must release it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.pg_writer import PgWriter

WINDOW = datetime(2026, 8, 26, 2, 0, tzinfo=timezone.utc)


@pytest.fixture()
def writer(tmp_path) -> PgWriter:
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    w.ensure_account(100.0, mode='live')
    return w


def ticket(writer: PgWriter, symbol: str, *, status: str) -> int:
    ticket_id = writer.write_ticket(
        symbol=symbol, window_open=WINDOW,
        settle_time=WINDOW + timedelta(minutes=15), offset_minutes=12,
        side='down', contracts=5, limit_price=0.75, max_price=0.77, expected_cost=3.82,
        model_probability=0.80, edge=0.0368,
        market_ticker=f'KX{symbol[:3]}15M-26AUG252215-15', status='pending')
    writer.resolve_ticket(ticket_id, status=status)
    return ticket_id


def test_a_killed_order_releases_the_window_slot(writer):
    ticket(writer, 'BTC-USD', status='killed')
    symbols, stake, count = writer.entries_for_window(WINDOW)
    assert symbols == frozenset()
    assert stake == pytest.approx(0.0)
    assert count == 0


def test_an_in_flight_unknown_still_holds_the_slot(writer):
    """It may have been accepted. Re-entering could double the position."""
    ticket(writer, 'BTC-USD', status='unknown')
    symbols, _stake, count = writer.entries_for_window(WINDOW)
    assert symbols == frozenset({'BTC-USD'})
    assert count == 1


def test_a_filled_ticket_still_holds_the_slot(writer):
    ticket(writer, 'ETH-USD', status='filled')
    assert writer.entries_for_window(WINDOW)[2] == 1


def test_a_refused_order_still_releases_the_slot(writer):
    ticket(writer, 'SOL-USD', status='skipped')
    assert writer.entries_for_window(WINDOW)[2] == 0
