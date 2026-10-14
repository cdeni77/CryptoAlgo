"""Tests that the live loop stores the venue ledger it reads during reconcile.

These drive `reconcile_with_venue` and check the stored rows, not the source text.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from core.pg_writer import PgWriter
from data_collection.kalshi_client import KalshiClient
from scripts.live import (
    adopt_venue_balance, persist_venue_ledger, reconcile_with_venue,
)

FILLS = [
    {'trade_id': 'f1', 'order_id': 'o1', 'ticker': 'KXBTC15M-A', 'side': 'no',
     'action': 'buy', 'count_fp': '5.00', 'no_price_dollars': '0.3000',
     'is_taker': True, 'created_time': '2026-08-25T10:03:00Z'},
]
SETTLEMENTS = [
    # `fee_cost` is unsuffixed and already in dollars; the venue serves no
    # `fee_cost_dollars` on a settlement.
    {'ticker': 'KXBTC15M-A', 'market_result': 'no', 'no_count_fp': '5.00',
     'no_total_cost_dollars': '1.5000', 'revenue_dollars': '5.0000',
     'fee_cost': 0.0700, 'settled_time': '2026-08-25T10:15:00Z'},
]


def _writer(tmp_path) -> PgWriter:
    writer = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    writer.ensure_account(100.0, mode='live')
    return writer


def _client() -> KalshiClient:
    """A client whose reconcile returns a known ledger without a network."""
    client = KalshiClient(key_id='k', private_key_pem='')

    async def reconcile(*, exchange_index=None):
        return {'balance': 103.43, 'positions': [], 'fills': FILLS,
                'settlements': SETTLEMENTS}

    client.reconcile = reconcile
    return client


def test_a_reconcile_stores_the_ledger_it_reads(tmp_path):
    """The cycle's own reconcile fills the venue tables, with no extra request."""
    writer = _writer(tmp_path)
    state = asyncio.run(reconcile_with_venue(writer, _client()))

    assert state.balance == pytest.approx(103.43)

    fills = writer.venue_fills()
    assert len(fills) == 1
    assert fills[0].trade_id == 'f1'
    # Priced from the NO side, not the YES book.
    assert fills[0].price == pytest.approx(0.30)

    settled = writer.venue_settlements()
    assert len(settled) == 1
    assert settled[0].pnl == pytest.approx(3.43)
    assert settled[0].market_result == 'no'


def test_storing_the_ledger_twice_converges(tmp_path):
    """Storing the same ledger three times leaves one fill and one settlement."""
    writer = _writer(tmp_path)
    for _ in range(3):
        asyncio.run(reconcile_with_venue(writer, _client()))
    assert len(writer.venue_fills()) == 1
    assert len(writer.venue_settlements()) == 1
    assert writer.venue_settlements()[0].pnl == pytest.approx(3.43)


def test_a_steady_state_cycle_writes_nothing_it_already_has(tmp_path):
    """A steady-state cycle writes nothing it already has.

    The upserts are idempotent either way; this pins the write cost.
    """
    writer = _writer(tmp_path)
    assert persist_venue_ledger(writer, fills=FILLS,
                                settlements=SETTLEMENTS) == (1, 1)
    assert persist_venue_ledger(writer, fills=FILLS,
                                settlements=SETTLEMENTS) == (0, 0)
    assert len(writer.venue_fills()) == 1
    assert len(writer.venue_settlements()) == 1


def test_an_incomplete_settlement_is_retried_so_it_can_heal(tmp_path):
    """A settlement stored with a null P&L is retried, so a later complete row replaces it."""
    writer = _writer(tmp_path)
    partial = [{'ticker': 'KXBTC15M-A', 'no_count_fp': '5.00'}]
    assert persist_venue_ledger(writer, fills=[], settlements=partial) == (0, 1)
    assert writer.venue_settlements()[0].pnl is None

    # Same ticker, now complete: written again.
    assert persist_venue_ledger(writer, fills=[], settlements=SETTLEMENTS) == (0, 1)
    assert writer.venue_settlements()[0].pnl == pytest.approx(3.43)
    # Once complete, it is no longer rewritten.
    assert persist_venue_ledger(writer, fills=[], settlements=SETTLEMENTS) == (0, 0)


def test_a_store_failure_does_not_stop_the_cycle(tmp_path):
    """A store write failure does not stop the cycle; the store is a record, not the account."""
    writer = _writer(tmp_path)

    def explode(_rows):
        raise RuntimeError('the store is on fire')

    writer.upsert_venue_fills = explode
    writer.upsert_venue_settlements = explode

    written = persist_venue_ledger(writer, fills=FILLS, settlements=SETTLEMENTS)
    assert written == (0, 0)


def test_a_settlement_we_never_booked_is_stored_with_a_null_of_our_own(tmp_path):
    """A settlement for a market we never booked is kept, with a null `our_pnl`.

    That is what an accepted order whose POST timed out leaves behind.
    """
    writer = _writer(tmp_path)
    persist_venue_ledger(writer, fills=[], settlements=SETTLEMENTS)
    row = writer.venue_settlements()[0]
    assert row.pnl == pytest.approx(3.43)
    assert row.our_pnl is None
    assert row.position_id is None


def test_the_balance_is_sampled_before_it_is_adopted(tmp_path):
    """The balance is sampled before it is adopted, or the recorded drift is always
    zero.

    A steady drift is a starting-balance mismatch; a growing one is an unrecorded fill.
    """
    writer = _writer(tmp_path)
    adopt_venue_balance(writer, 103.43, exchange_index=2)

    sample = writer.latest_venue_balance()
    assert sample.balance == pytest.approx(103.43)
    assert sample.our_bankroll == pytest.approx(100.0), 'ours, as it stood before'
    assert sample.drift == pytest.approx(3.43)
    assert sample.exchange_index == 2
    # The account took the venue's figure.
    assert writer.account().bankroll == pytest.approx(103.43)


def test_an_unreadable_balance_is_neither_sampled_nor_adopted(tmp_path):
    writer = _writer(tmp_path)
    adopt_venue_balance(writer, float('nan'))
    assert writer.latest_venue_balance() is None
    assert writer.account().bankroll == pytest.approx(100.0)


def test_the_curve_is_readable_from_what_a_cycle_stored(tmp_path):
    """End to end: reconcile, then the two series the account chart draws."""
    from core import venue_ledger

    writer = _writer(tmp_path)
    asyncio.run(reconcile_with_venue(writer, _client()))
    adopt_venue_balance(writer, 103.43, exchange_index=2)

    settlements = writer.venue_settlements()
    summary = venue_ledger.summarise(settlements)
    assert summary.realized_pnl == pytest.approx(3.43)
    assert summary.wins == 1 and summary.losses == 0
    assert summary.incomplete == 0

    points = venue_ledger.cumulative_curve(settlements)
    assert [p['cumulative_pnl'] for p in points] == pytest.approx([3.43])

    since = datetime.now(timezone.utc) - timedelta(hours=1)
    assert len(writer.venue_balances_since(since)) == 1

    check = venue_ledger.balance_check(
        venue_balance=103.43, settlements=settlements, fills=writer.venue_fills())
    assert check['implied_starting_balance'] == pytest.approx(100.0)
