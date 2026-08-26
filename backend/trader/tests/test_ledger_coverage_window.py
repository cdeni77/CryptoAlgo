"""Our books and the venue's cover different periods, and the totals say so.

`sync_venue` prints the venue's realised P&L beside ours and calls the
difference a gap. Measured the first time it ran against the live account: the
venue held 365 settlements spanning four days while our store had been wiped the
previous night for a clean experiment, so it held one. The gap was almost
entirely that.

The three causes the tool listed — a settlement our Coinbase proxy called
differently, a fee we mispriced, a fill nobody booked — are all real and none of
them was this one. Wiping the store between experiments is deliberate practice
here, so the mismatch recurs every time, and a number presented as error when it
is arithmetic sends someone hunting a bug that does not exist.

`first_position_time` is what makes the comparison honest: it says when our
books actually start.
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
    """Nothing to compare against, and it must not claim a date anyway."""
    assert writer.first_position_time() is None


def test_it_reports_the_earliest_position(writer):
    position(writer, W)
    position(writer, W - timedelta(days=1))
    position(writer, W + timedelta(hours=3))
    assert writer.first_position_time() == W - timedelta(days=1)


def test_the_answer_is_timezone_aware_utc(writer):
    """A naive datetime compared against the venue's aware one raises."""
    position(writer, W)
    got = writer.first_position_time()
    assert got.tzinfo is not None
    assert got.utcoffset() == timedelta(0)
