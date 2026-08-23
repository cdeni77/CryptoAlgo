"""PROPOSED. One entry per (symbol, window) across a restart; stale data refused.

Drop into `backend/trader/tests/`.

`scripts/live.py` is 19.0% covered (measured; 82 of 431 statements) and
`tests/test_live.py` tests only `current_window`, `choose_offset`, three module
constants and the argparse defaults. The invariants that stop a restart loop
from placing the same bet twelve times are all in the uncovered part.

`core/pg_writer.py:603 entries_for_window` and `:527 open_position` carry the
whole restart story in their docstrings — including that the process *did* die
after putting a duplicate order on the wire and `restart: unless-stopped`
brought it back to do it again. None of that is tested. These tests run against
a throwaway SQLite file, so no Postgres and no network.

Also covered here: `test_live.py::test_place_orders_requires_live_mode` and
`::test_forcing_past_the_gates_requires_a_reason` assert only that the *parser
accepts* the dangerous combination, with a comment saying `main()` rejects it.
`scripts/live.py:563-566` is where the rejection lives and it never executes in
the suite.
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
    """A real PgWriter over a throwaway SQLite file.

    `_run_migrations` already tolerates a non-Postgres dialect, so this exercises
    the actual ORM and the actual uniqueness logic rather than a fake.
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
    """Not an IntegrityError that kills the loop — a None the caller can see.

    The failure this is designed against is documented in `open_position`: the
    bare insert raised, `scripts/live.py`'s loop caught only KeyboardInterrupt,
    the process died *after* the duplicate order was on the wire, and the
    container restarted it to do the same thing again.
    """
    first = writer.open_position(**position_fields())
    assert first is not None
    second = writer.open_position(**position_fields(contracts=99, outlay=88.0))
    assert second is None, 'a second position was booked for the same window'
    assert len(writer.open_positions()) == 1
    assert writer.open_positions()[0].contracts == 3, 'the duplicate overwrote the first'


def test_a_second_ticket_for_the_same_symbol_and_window_returns_the_first(writer):
    """A ticket exists from the moment an order is sent, so it has to dedupe too."""
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
    """The key is (symbol, window). Blocking on the window alone would abstain on
    two of three symbols every cycle."""
    assert writer.open_position(**position_fields(symbol='BTC-USD')) is not None
    assert writer.open_position(**position_fields(symbol='ETH-USD')) is not None
    assert len(writer.open_positions()) == 2


# ------------------------------------------------------- restart recovery

def test_exposure_survives_a_process_restart(writer, tmp_path):
    """The window does not restart when the process does.

    A fresh `PgWriter` over the same file must report the committed entry, and
    `decide` must then refuse — which is exactly the sequence
    `scripts/live.py:556-561` performs at the top of every cycle.
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
    """A crash between sending the order and booking the position leaves a ticket
    and no position. That must still block."""
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
    """An abstention is not an entry."""
    writer.write_ticket(symbol='BTC-USD', window_open=WDT,
                        settle_time=WDT + timedelta(minutes=15), offset_minutes=9,
                        side=Side.UP.value, contracts=0, limit_price=0.88,
                        max_price=0.89, expected_cost=0.0,
                        model_probability=0.93, edge=0.02, status='skipped')
    symbols, staked, count = writer.entries_for_window(WDT)
    assert symbols == frozenset()
    assert count == 0


def test_the_previous_windows_entry_does_not_block_this_one(writer):
    """Exposure is per window. Leaking it forward abstains forever."""
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
    """`test_live.py` asserts the *parser accepts* `--mode paper --place-orders`
    and comments that main() rejects it. That rejection is `scripts/live.py:563`
    and it never executes in the suite. Here it does."""
    from scripts.live import build_parser

    args = build_parser().parse_args(['--mode', 'paper', '--place-orders'])
    assert args.place_orders and args.mode == 'paper'
    with pytest.raises(SystemExit, match='requires --mode live'):
        _run_main_with_argv(['live', '--mode', 'paper', '--place-orders'])


def test_forcing_past_the_gates_without_a_reason_is_refused_by_main():
    """`scripts/live.py:565`. The existing test asserts only that the parser
    allows the pair."""
    with pytest.raises(SystemExit, match='needs --reason'):
        _run_main_with_argv(['live', '--force'])


def test_a_quote_that_is_not_active_is_not_tradeable():
    """A settled or paused market still returns a book. Trading it is trading a
    price that cannot be filled."""
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
    """The measured book was 0.19/0.20. A missing side is a parse failure or an
    empty market, and both used to read as a zero price."""
    from data_collection.kalshi_client import Quote

    for yes_bid, yes_ask in ((None, 0.20), (0.19, None), (None, None)):
        quote = Quote(ticker='T', yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None,
                      no_ask=None, last_price=None, volume=0, open_interest=0,
                      close_time=None, status='active')
        assert not quote.tradeable(), (yes_bid, yes_ask)


def test_the_depth_of_a_side_is_read_from_that_side():
    """`decide` prefers a measured depth over `max_stake_dollars`, so reading the
    wrong side's size caps the wrong order."""
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
