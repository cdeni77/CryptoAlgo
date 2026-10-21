"""A position is brought up to the venue's fill total, never down.

`book_resting_fills` books what has filled SO FAR and then resolves the ticket,
which drops it out of `open_tickets()` — so an order that keeps filling
afterwards leaves us holding fewer contracts than the venue does. Measured live
2026-10-20 on the 15:00 SOL window: three entries recorded 2, 1 and 0 contracts
against the venue's 6. That under-credits the payout at settlement, which
surfaced as a $1.00 balance drift.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.pg_writer import Account, PgWriter, Position

WINDOW = pd.Timestamp('2026-10-20 15:00', tz='UTC')


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')
    w.ensure_account(mode='paper', starting_bankroll=100.0)
    return w


def _open(w, *, offset=3, contracts=1, price=0.70):
    return w.open_position(
        symbol='SOL-USD', window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=offset,
        side='down', contracts=contracts, price=price, outlay=contracts * price,
        fee=0.0, model_probability=0.75, baseline_probability=0.70, edge=0.05)


def _size(w, pid):
    with w._session() as s:
        row = s.get(Position, pid)
        return int(row.contracts), float(row.outlay)


def test_a_position_grows_to_the_venues_total(writer):
    pid = _open(writer, contracts=1, price=0.70)
    assert writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=2, cost=1.40, fee=0.0) is True
    assert _size(writer, pid) == (2, 1.40)


def test_only_the_difference_is_debited(writer):
    """The bankroll already paid for the contracts we booked. Debiting the
    whole cost again would charge them twice."""
    pid = _open(writer, contracts=1, price=0.70)
    with writer._session() as s:
        before = float(s.query(Account).one().bankroll)
    writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=2, cost=1.40, fee=0.0)
    with writer._session() as s:
        after = float(s.query(Account).one().bankroll)
    assert before - after == pytest.approx(0.70, abs=1e-9)
    assert _size(writer, pid)[0] == 2


def test_a_position_is_never_shrunk(writer):
    """Shrinking would discard contracts the venue says we hold — the opposite
    of the error this exists to fix."""
    pid = _open(writer, contracts=3, price=0.70)
    assert writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=1, cost=0.70, fee=0.0) is False
    assert _size(writer, pid)[0] == 3


def test_an_equal_total_is_a_no_op(writer):
    pid = _open(writer, contracts=2, price=0.70)
    assert writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=2, cost=1.40, fee=0.0) is False
    assert _size(writer, pid)[0] == 2


def test_a_settled_position_is_never_touched(writer):
    """It already has a payout computed from its size; changing the size here
    would contradict that."""
    pid = _open(writer, contracts=1, price=0.70)
    writer.settle_position(pid, settled_up=False, fee=0.0)
    assert writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=5, cost=3.50, fee=0.0) is False
    assert _size(writer, pid)[0] == 1


def test_offsets_are_corrected_independently(writer):
    """Multi-entry puts several positions on one window; a fill on one order
    must not resize another offset's position."""
    a = _open(writer, offset=3, contracts=1)
    b = _open(writer, offset=6, contracts=1)
    writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=4, cost=2.80, fee=0.0)
    assert _size(writer, a)[0] == 4
    assert _size(writer, b)[0] == 1


def test_a_sub_contract_total_changes_nothing(writer):
    """Rounding 0.4 to 0 is how the zero-contract positions were created."""
    pid = _open(writer, contracts=1, price=0.70)
    assert writer.reconcile_position_size(
        symbol='SOL-USD', window_open=WINDOW, offset_minutes=3,
        contracts=0.4, cost=0.28, fee=0.0) is False
    assert _size(writer, pid)[0] == 1
