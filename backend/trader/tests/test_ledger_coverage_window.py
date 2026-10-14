"""Tests for `first_position_time`, the start of our own books.

`sync_venue` compares our realised P&L with the venue's. If our store starts
later than the venue's history, part of the gap is only the coverage difference.
`first_position_time` lets the comparison use the same period.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from core.pg_writer import PgWriter

W = datetime(2026, 8, 25, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def writer(tmp_path) -> PgWriter:
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    w.ensure_account(100.0, mode='live')
    return w


def position(writer: PgWriter, when: datetime) -> None:
    writer.open_position(
        symbol='BTC-USD', window_open=when,
        settle_time=when + timedelta(minutes=15), offset_minutes=12,
        side='up', contracts=5, price=0.40, outlay=2.05, fee=0.05,
        model_probability=0.55, baseline_probability=0.50, edge=0.03)


def test_an_empty_store_has_no_start(writer):
    """An empty store has no start date."""
    assert writer.first_position_time() is None


def test_it_reports_the_earliest_position(writer):
    position(writer, W)
    position(writer, W - timedelta(days=1))
    position(writer, W + timedelta(hours=3))
    assert writer.first_position_time() == W - timedelta(days=1)


def test_the_answer_is_timezone_aware_utc(writer):
    """The result is timezone-aware UTC, so it compares with the venue's aware times."""
    position(writer, W)
    got = writer.first_position_time()
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)
