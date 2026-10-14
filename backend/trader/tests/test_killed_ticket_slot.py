"""Tests that a killed order releases its window entry slot.

'unknown' may have been accepted and keeps the slot; 'killed' confirms no position.
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
    """An in-flight 'unknown' may have been accepted, so it still holds the slot."""
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
