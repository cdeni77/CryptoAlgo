"""A position is keyed on the DECISION, not on the window.

`uq_position_window` was UNIQUE (symbol, window_open), which made one position
per window a database invariant. That was right while `already_entered` allowed
exactly one entry and wrong the moment `max_entries_per_symbol_window` could
exceed 1: entries 2..N could not be booked at all, so our books recorded 1
contract where the venue held 4 and `pnl_gap` drifted by the difference.

Widened to (symbol, window_open, offset_minutes). The protection it was written
for -- a duplicate order for the SAME decision surviving a restart -- is kept,
because a decision is (symbol, window, offset).
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.pg_writer import PgWriter, Position

WINDOW = pd.Timestamp('2026-10-20 12:00', tz='UTC')


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')
    w.ensure_account(mode='paper', starting_bankroll=100.0)
    return w


def _open(w, *, offset, symbol='BTC-USD', contracts=1, price=0.70):
    return w.open_position(
        symbol=symbol, window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=offset,
        side='up', contracts=contracts, price=price, outlay=contracts * price,
        fee=0.0, model_probability=0.75, baseline_probability=0.70, edge=0.05)


def test_two_offsets_in_one_window_both_book(writer):
    """The whole point. Before this, the second returned None and the venue's
    fill was never recorded on our side."""
    first = _open(writer, offset=3)
    second = _open(writer, offset=6)
    assert first is not None
    assert second is not None
    assert first != second


def test_all_four_offsets_can_be_held(writer):
    ids = [_open(writer, offset=o) for o in (3, 6, 9, 12)]
    assert all(i is not None for i in ids)
    assert len(set(ids)) == 4


def test_the_same_decision_twice_is_still_refused(writer):
    """The protection the constraint was written for: a duplicate order for one
    decision, surviving a process restart. Keyed on the offset, that is still
    exactly one position."""
    assert _open(writer, offset=3) is not None
    assert _open(writer, offset=3) is None


def test_other_symbols_are_unaffected(writer):
    assert _open(writer, offset=3, symbol='BTC-USD') is not None
    assert _open(writer, offset=3, symbol='ETH-USD') is not None


def test_the_stake_of_every_entry_is_recorded(writer):
    """`max_window_exposure_fraction` caps notional. If entries 2..N are not
    booked, the cap is computed against less than is actually at risk."""
    _open(writer, offset=3, contracts=2, price=0.50)
    _open(writer, offset=6, contracts=3, price=0.60)
    counts, stake, entries, sides = writer.entry_counts_for_window(
        WINDOW.to_pydatetime())
    assert entries == 2
    assert counts['BTC-USD'] == 2
    assert stake == pytest.approx(1.0 + 1.8)
    assert sides['BTC-USD'] == 'up'


def test_a_fresh_database_gets_the_new_constraint_from_the_orm(writer):
    """SQLite never runs the swap — `create_all` builds it from the ORM. If it
    did not, this file's first test would pass for the wrong reason."""
    names = {c.name for c in Position.__table__.constraints if c.name}
    assert 'uq_position_window_offset' in names
    assert 'uq_position_window' not in names


# --- the same widening on order_tickets --------------------------------------
#
# Missing this left a worse hole than the positions one. The order for the
# second offset was still PLACED and FILLED at the venue -- two distinct
# order_ids on KXETH15M-26OCT201000-00, 13:48:09 @ 0.680 and 13:51:05 @ 0.750 --
# but `write_ticket` returned the FIRST ticket's id instead of recording it. No
# ticket means no position, so our books showed 1 contract against the venue's
# 2 and `pnl_gap` moved on every window that took a second entry.

def _ticket(w, *, offset, symbol='BTC-USD', contracts=1):
    return w.write_ticket(
        symbol=symbol, window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=offset,
        side='up', contracts=contracts, limit_price=0.70, max_price=0.71,
        expected_cost=contracts * 0.70, model_probability=0.75, edge=0.05,
        status='new')


def test_two_offsets_get_two_tickets(writer):
    first = _ticket(writer, offset=3)
    second = _ticket(writer, offset=6)
    assert first != second, (
        'the second offset got the first offset\'s ticket id, so its order '
        'was placed at the venue and never recorded'
    )


def test_the_same_decision_returns_the_same_ticket(writer):
    """The idempotency this constraint was written for: one order per decision
    surviving a process restart."""
    assert _ticket(writer, offset=3) == _ticket(writer, offset=3)


def test_four_offsets_are_four_tickets(writer):
    ids = [_ticket(writer, offset=o) for o in (3, 6, 9, 12)]
    assert len(set(ids)) == 4


def test_a_fresh_database_gets_the_widened_ticket_constraint(writer):
    from core.pg_writer import OrderTicket
    names = {c.name for c in OrderTicket.__table__.constraints if c.name}
    assert 'uq_ticket_window_offset' in names
    assert 'uq_ticket_window' not in names
