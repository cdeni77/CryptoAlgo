"""Tests for one entry per (symbol, window) across restarts, and refusal of stale or bad
data.

Runs against a throwaway SQLite file, so no Postgres or network is needed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.config import Config
from core.decide import Reason, Side, WindowExposure, decide

W = pd.Timestamp('2026-08-23 00:30', tz='UTC')
WDT = datetime(2026, 8, 23, 0, 30, tzinfo=timezone.utc)


@pytest.fixture
def writer(tmp_path):
    """A real PgWriter over a throwaway SQLite file, so the real ORM and uniqueness logic
    run.
    """
    from core.pg_writer import PgWriter

    return PgWriter(f'sqlite:///{tmp_path / "live.db"}')


def row(**over):
    base = dict(symbol='BTC-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=9,
                baseline_probability=0.88, model_probability=0.93)
    base.update(over)
    return base


def position_fields(**over):
    base = dict(symbol='BTC-USD', window_open=WDT,
                settle_time=WDT + timedelta(minutes=15), offset_minutes=9,
                side=Side.UP.value, contracts=3, price=0.88, outlay=2.66,
                fee=0.02, model_probability=0.93, baseline_probability=0.88,
                edge=0.02)
    base.update(over)
    return base


# ------------------------------------------------ duplicate-order prevention

def test_a_second_position_for_the_same_symbol_and_window_is_refused(writer):
    """A duplicate position returns None instead of raising, so the loop does not die after
    the order is sent.
    """
    first = writer.open_position(**position_fields())
    assert first is not None
    second = writer.open_position(**position_fields(contracts=99, outlay=88.0))
    assert second is None, 'a second position was booked for the same window'
    assert len(writer.open_positions()) == 1
    assert writer.open_positions()[0].contracts == 3, 'the duplicate overwrote the first'


def test_a_second_ticket_for_the_same_symbol_and_window_returns_the_first(writer):
    """A duplicate ticket returns the first one."""
    fields = dict(symbol='BTC-USD', window_open=WDT,
                  settle_time=WDT + timedelta(minutes=15), offset_minutes=9,
                  side=Side.UP.value, contracts=3, limit_price=0.88,
                  max_price=0.89, expected_cost=2.66, model_probability=0.93,
                  edge=0.02, status='new')
    first = writer.write_ticket(**fields)
    second = writer.write_ticket(**{**fields, 'contracts': 99})
    assert first == second
    assert len(writer.open_tickets()) == 1


def test_another_symbol_in_the_same_window_is_not_blocked(writer):
    """The key is (symbol, window), so another symbol in the same window is allowed."""
    assert writer.open_position(**position_fields(symbol='BTC-USD')) is not None
    assert writer.open_position(**position_fields(symbol='ETH-USD')) is not None
    assert len(writer.open_positions()) == 2


# ------------------------------------------------------- restart recovery

def test_exposure_survives_a_process_restart(writer, tmp_path):
    """A fresh PgWriter over the same file sees the committed entry, and `decide` refuses.
    """
    from core.pg_writer import PgWriter

    writer.open_position(**position_fields())

    restarted = PgWriter(f'sqlite:///{tmp_path / "live.db"}')
    symbols, staked, count = restarted.entries_for_window(WDT)
    assert symbols == frozenset({'BTC-USD'})
    assert staked == pytest.approx(2.66)
    assert count == 1

    exposure = WindowExposure(stake=staked, positions=count, symbols_entered=symbols)
    decision = decide(row(), Config(), bankroll=100.0, exposure=exposure)
    assert decision.reason is Reason.ALREADY_ENTERED, (
        'after a restart the loop would have entered this window a second time'
    )


def test_a_ticket_alone_counts_as_an_entry_after_a_restart(writer, tmp_path):
    """A ticket with no position still blocks the window after a restart."""
    from core.pg_writer import PgWriter

    writer.write_ticket(symbol='BTC-USD', window_open=WDT,
                        settle_time=WDT + timedelta(minutes=15), offset_minutes=9,
                        side=Side.UP.value, contracts=3, limit_price=0.88,
                        max_price=0.89, expected_cost=2.66,
                        model_probability=0.93, edge=0.02, status='new')
    restarted = PgWriter(f'sqlite:///{tmp_path / "live.db"}')
    symbols, staked, count = restarted.entries_for_window(WDT)
    assert symbols == frozenset({'BTC-USD'})
    assert count == 1
    decision = decide(row(), Config(), bankroll=100.0,
                      exposure=WindowExposure(stake=staked, positions=count,
                                              symbols_entered=symbols))
    assert decision.reason is Reason.ALREADY_ENTERED


def test_a_skipped_ticket_does_not_block_the_window(writer):
    """A skipped ticket is not an entry."""
    writer.write_ticket(symbol='BTC-USD', window_open=WDT,
                        settle_time=WDT + timedelta(minutes=15), offset_minutes=9,
                        side=Side.UP.value, contracts=0, limit_price=0.88,
                        max_price=0.89, expected_cost=0.0,
                        model_probability=0.93, edge=0.02, status='skipped')
    symbols, staked, count = writer.entries_for_window(WDT)
    assert symbols == frozenset()
    assert count == 0


def test_the_previous_windows_entry_does_not_block_this_one(writer):
    """Exposure is per window; the previous window's entry does not block this one."""
    writer.open_position(**position_fields(
        window_open=WDT - timedelta(minutes=15),
        settle_time=WDT))
    symbols, staked, count = writer.entries_for_window(WDT)
    assert count == 0, 'the previous window is blocking this one'


# --------------------------------------------------------- stale data refused

def _run_main_with_argv(argv):
    """`scripts.live.main()` parses its own argv, so drive it through sys.argv."""
    import asyncio
    import sys

    from scripts.live import main

    saved = sys.argv
    sys.argv = list(argv)
    try:
        asyncio.run(main())
    finally:
        sys.argv = saved


def test_paper_mode_with_place_orders_is_refused_by_main():
    """`main()` refuses `--mode paper --place-orders`."""
    from scripts.live import build_parser

    args = build_parser().parse_args(['--mode', 'paper', '--place-orders'])
    assert args.place_orders and args.mode == 'paper'
    with pytest.raises(SystemExit, match='requires --mode live'):
        _run_main_with_argv(['live', '--mode', 'paper', '--place-orders'])


def test_forcing_past_the_gates_without_a_reason_is_refused_by_main():
    """`main()` refuses `--force` without `--reason`."""
    with pytest.raises(SystemExit, match='needs --reason'):
        _run_main_with_argv(['live', '--force'])


def test_a_quote_that_is_not_active_is_not_tradeable():
    """A market that is not active is not tradeable, even if it still returns a book."""
    from data_collection.kalshi_client import Quote

    live = Quote(ticker='T', yes_bid=0.19, yes_ask=0.20, no_bid=0.80, no_ask=0.81,
                 last_price=0.20, volume=10, open_interest=10,
                 close_time=None, status='active')
    assert live.tradeable()
    for status in ('settled', 'closed', 'paused', 'unknown', 'initialized'):
        stale = Quote(ticker='T', yes_bid=0.19, yes_ask=0.20, no_bid=0.80,
                      no_ask=0.81, last_price=0.20, volume=10, open_interest=10,
                      close_time=None, status=status)
        assert not stale.tradeable(), status


def test_a_one_sided_book_is_not_tradeable():
    """A one-sided book is not tradeable."""
    from data_collection.kalshi_client import Quote

    for yes_bid, yes_ask in ((None, 0.20), (0.19, None), (None, None)):
        quote = Quote(ticker='T', yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None,
                      no_ask=None, last_price=None, volume=0, open_interest=0,
                      close_time=None, status='active')
        assert not quote.tradeable(), (yes_bid, yes_ask)


def test_the_depth_of_a_side_is_read_from_that_side():
    """Each side's depth is read from that side, since `decide` caps the stake on it."""
    from data_collection.kalshi_client import Quote

    quote = Quote(ticker='T', yes_bid=0.19, yes_ask=0.20, no_bid=0.80, no_ask=0.81,
                  last_price=0.20, volume=10, open_interest=10, close_time=None,
                  status='active', yes_bid_size=1594.0, yes_ask_size=59.0)
    assert quote.ask_for('up') == pytest.approx(0.20)
    assert quote.ask_for('down') == pytest.approx(0.81)
    assert quote.size_for('up') == pytest.approx(59.0)
    assert quote.depth_dollars('up') == pytest.approx(0.20 * 59.0)
    assert quote.depth_dollars('up') < 25.0, (
        'the first live book was thinner than max_stake_dollars, which is the '
        'whole reason a measured depth is preferred'
    )
