"""`act_on` is where money moves, and nothing tested it.

`scripts/live.py` sat at ~19% coverage and `act_on` at none of it, which is how
four defects lived here at once: an unresolved market booked a position without
sending an order, `--mode live --dry-run` did the same, a killed `fill_or_kill` was
recorded as a full fill, and a later refusal erased the record of a real one.

Every case below is asserted on the three things that matter — did an order go to
the wire, what position exists afterwards, and what happened to the bankroll.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from core.decide import Decision, Reason, Side
from data_collection.kalshi_client import KalshiError
from core.pg_writer import PgWriter
from scripts.live import act_on, filled_from_order, order_limit_price

WINDOW = datetime(2026, 8, 23, 0, 30, tzinfo=timezone.utc)
TICKER = 'KXBTC15M-26AUG230045'


class FakeKalshi:
    """Records what reached the wire; returns a scripted reply."""

    def __init__(self, reply=None, raises=None):
        self.reply, self.raises = reply, raises
        self.orders: list[dict] = []

    async def place_order(self, **body):
        self.orders.append(body)
        if self.raises is not None:
            raise self.raises
        return dict(self.reply or {})


def decision(*, ticker=TICKER, contracts=5, price=0.60) -> Decision:
    return Decision(
        symbol='BTC-USD', window_open=pd.Timestamp(WINDOW),
        settle_time=pd.Timestamp(WINDOW + timedelta(minutes=15)), offset=3,
        reason=Reason.TRADED, side=Side.UP, price=price,
        effective_cost=price + 0.0168, model_probability=0.72,
        baseline_probability=0.60, edge=0.08, contracts=contracts,
        stake=contracts * price + 0.14, fee=0.14, kelly_fraction=0.26,
        price_source='quote', market_ticker=ticker)


def args(**over) -> SimpleNamespace:
    base = dict(mode='live', place_orders=True, dry_run=False, reconcile=True)
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def writer(tmp_path) -> PgWriter:
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    w.ensure_account(100.0, mode='live')
    return w


def run(coro):
    return asyncio.run(coro)


def state(writer: PgWriter) -> tuple[int, float]:
    account = writer.account()
    return len(writer.open_positions()), float(account.bankroll)


class TestFillReadback:
    """`filled = int(order.get('count', contracts))` read the size *requested*."""

    @pytest.mark.parametrize('reply,expected', [
        ({'status': 'executed', 'order_id': 'a'}, 5),
        ({'status': 'filled', 'taker_fill_count': 5}, 5),
        ({'status': 'resting', 'remaining_count': 3}, 2),
        ({'status': 'canceled', 'count': 5}, 0),
        ({'status': 'cancelled', 'taker_fill_count': 0}, 0),
        ({'status': 'rejected'}, 0),
        ({}, 0),
        ({'count': 5}, 0),
    ])
    def test_the_fill_is_read_not_assumed(self, reply, expected):
        filled, _price = filled_from_order(reply, 5)
        assert filled == expected

    def test_a_killed_fill_or_kill_books_nothing(self, writer):
        kalshi = FakeKalshi({'status': 'canceled', 'count': 5, 'order_id': 'k'})
        before = state(writer)
        booked = run(act_on(args(), writer, kalshi, decision(), None))
        assert booked is False
        assert len(kalshi.orders) == 1, 'the order should still have been attempted'
        assert state(writer) == before, (
            'a killed order left a position and a debit; the documented claim was '
            'that it "leaves a ticket and no position"'
        )

    def test_a_partial_fill_books_only_what_filled(self, writer):
        kalshi = FakeKalshi({'status': 'executed', 'remaining_count': 3,
                             'order_id': 'p'})
        assert run(act_on(args(), writer, kalshi, decision(), None)) is True
        positions = writer.open_positions()
        assert len(positions) == 1
        assert positions[0].contracts == 2
        # And the debit is scaled to what was actually bought.
        assert float(positions[0].outlay) == pytest.approx(
            decision().stake * 2 / 5, abs=1e-9)

    def test_an_empty_body_is_not_a_fill(self, writer):
        """HTTP 200 with nothing in it used to record a complete fill."""
        before = state(writer)
        kalshi = FakeKalshi({})
        assert run(act_on(args(), writer, kalshi, decision(), None)) is False
        assert state(writer) == before


class TestNoOrderNoPosition:
    def test_an_unresolved_market_books_nothing(self, writer):
        """`market_ticker=None` used to fall through to `open_position`.

        Measured before the fix: 0 orders sent, a 5-contract position written,
        $3.10 debited — a holding the venue had never heard of, which `settle_due`
        then settled into invented PnL.
        """
        before = state(writer)
        kalshi = FakeKalshi({'status': 'executed'})
        assert run(act_on(args(), writer, kalshi,
                          decision(ticker=None), None)) is False
        assert kalshi.orders == []
        assert state(writer) == before

    def test_a_dry_run_books_nothing(self, writer):
        """`--dry-run` was declared and never read."""
        before = state(writer)
        kalshi = FakeKalshi({'status': 'executed'})
        assert run(act_on(args(place_orders=False, dry_run=True), writer, kalshi,
                          decision(), None)) is False
        assert kalshi.orders == []
        assert state(writer) == before

    def test_a_refused_order_books_nothing(self, writer):
        from data_collection.kalshi_client import KalshiError

        before = state(writer)
        kalshi = FakeKalshi(raises=KalshiError('insufficient balance'))
        assert run(act_on(args(), writer, kalshi, decision(), None)) is False
        assert state(writer) == before

    def test_an_in_flight_failure_books_nothing_and_does_not_retry(self, writer):
        """The request may have been accepted. Booking or retrying both risk a
        duplicate; the honest response is to record nothing and let
        reconciliation surface a venue position we do not hold."""
        before = state(writer)
        kalshi = FakeKalshi(raises=asyncio.TimeoutError())
        assert run(act_on(args(), writer, kalshi, decision(), None)) is False
        assert len(kalshi.orders) == 1, 'it must not have retried'
        assert state(writer) == before


class TestPaperMode:
    def test_paper_books_the_position_without_a_venue(self, writer, tmp_path):
        paper = PgWriter(database_url=f'sqlite:///{tmp_path}/paper.db')
        paper.ensure_account(100.0, mode='paper')
        booked = run(act_on(args(mode='paper', place_orders=False, dry_run=False),
                            paper, None, decision(), None))
        assert booked is True
        assert len(paper.open_positions()) == 1
        assert float(paper.account().bankroll) == pytest.approx(
            100.0 - decision().stake)


class TestIdempotency:
    def test_a_second_call_for_the_same_window_books_once(self, writer):
        """The DB constraint used to raise *after* the order was on the wire, and
        the loop caught only KeyboardInterrupt."""
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'x'})
        assert run(act_on(args(), writer, kalshi, decision(), None)) is True
        after_first = state(writer)
        # A second cycle at the same offset, as `choose_offset` produces.
        run(act_on(args(), writer, kalshi, decision(), None))
        assert len(writer.open_positions()) == 1, 'a second position was booked'
        assert state(writer)[0] == after_first[0]

    def test_a_later_refusal_does_not_erase_a_recorded_fill(self, writer):
        """`resolve_ticket` nulled filled_contracts/filled_price/filled_at,
        destroying the only local record of a real order."""
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'x'})
        run(act_on(args(), writer, kalshi, decision(), None))
        tickets = [t for t in writer.open_tickets(limit=50)] or None
        # Re-resolve as skipped, the way a duplicate submission would.
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import OrderTicket
            row = session.query(OrderTicket).one()
            ticket_id, filled = row.id, row.filled_contracts
        assert filled == 5
        writer.resolve_ticket(ticket_id, status='skipped', note='duplicate')
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import OrderTicket
            row = session.query(OrderTicket).one()
        assert row.status == 'skipped'
        assert row.filled_contracts == 5, 'the fill record was erased'


class TestLimitPrice:
    def test_the_limit_never_gives_away_the_whole_edge(self):
        """It was `price + edge` — the break-even price. Under fill_or_kill that
        lets a thin book walk the order to a zero-EV fill and call it a trade."""
        d = decision(price=0.60)
        limit = order_limit_price(d)
        assert limit > d.price
        assert limit < d.price + d.edge, (
            f'limit {limit:.4f} pays away the whole {d.edge:.4f} edge'
        )

    def test_the_limit_is_capped_in_cents(self):
        from core.config import DEFAULT_CONFIG

        generous = Decision(
            symbol='BTC-USD', window_open=pd.Timestamp(WINDOW),
            settle_time=pd.Timestamp(WINDOW + timedelta(minutes=15)), offset=3,
            reason=Reason.TRADED, side=Side.UP, price=0.30, effective_cost=0.32,
            model_probability=0.95, baseline_probability=0.30, edge=0.60,
            contracts=5, stake=1.6, fee=0.05, price_source='quote',
            market_ticker=TICKER)
        allowance = order_limit_price(generous) - generous.price
        assert allowance == pytest.approx(
            DEFAULT_CONFIG.max_slippage_cents / 100.0, abs=1e-9)


class TestTheIdempotencyKey:
    """The venue-side key must not enforce a stricter rule than the policy.

    Keyed on (symbol, window) alone it meant one order *attempt* per window,
    while the policy is one *position* per window. Those differ exactly when an
    order does not fill — and a fill_or_kill that kills still consumes the id, so
    the first thin-volume kill locked every later offset out of the window with
    `409 order_already_exists`.

    Measured over the first live night: 57 of 69 unfilled attempts were our own
    duplicate key rather than the venue refusing us, at a *higher* average
    claimed edge (8.37pp) than the attempts that filled (5.77pp). Only 9 were
    genuine `fill_or_kill_insufficient_resting_volume`, at 3.94pp.
    """

    def _key(self, writer, offset: int) -> str:
        kalshi = FakeKalshi(reply={'order': {'order_id': 'o', 'status': 'executed',
                                             'filled_count': '5.00'}})
        d = decision()
        d = replace(d, offset=offset)
        run(act_on(args(), writer, kalshi, d, None))
        return kalshi.orders[-1]['client_order_id']

    def test_two_offsets_in_one_window_are_two_different_orders(self, writer):
        """The regression. These were the same string, so the second was refused
        by the venue no matter how good the price had become."""
        assert self._key(writer, 3) != self._key(writer, 6)

    def test_the_same_offset_twice_is_the_same_order(self, writer):
        """Still idempotent where it matters: a cycle that runs twice, or a retry
        after a response we never saw, must not buy twice."""
        assert self._key(writer, 9) == self._key(writer, 9)

    def test_the_key_names_the_symbol_the_window_and_the_offset(self, writer):
        key = self._key(writer, 12)
        assert key.startswith('BTC-USD-'), key
        assert key.endswith('-12'), f'the offset must be in the key: {key}'
        assert f'{WINDOW:%Y%m%d%H%M}' in key, key

    def test_a_refused_order_leaves_the_window_open_to_the_next_offset(self, writer):
        """The other half of the mechanism, and it already worked: `skipped`
        means nothing was bought, so `entries_for_window` must not count it.
        Without this the new key would change nothing — the decision would never
        be made a second time."""
        kalshi = FakeKalshi(raises=KalshiError('409 fill_or_kill_insufficient_resting_volume'))
        assert run(act_on(args(), writer, kalshi, decision(), None)) is False
        symbols, staked, count = writer.entries_for_window(pd.Timestamp(WINDOW))
        assert count == 0 and symbols == frozenset(), (
            'a refused order blocked the window, so no later offset can try'
        )

    def test_a_crash_before_booking_still_blocks_the_window(self, writer):
        """The guard the old key was accidentally providing, which must survive
        the change. A ticket left in any status but `skipped` is an entry."""
        writer.write_ticket(
            symbol='BTC-USD', window_open=pd.Timestamp(WINDOW),
            settle_time=pd.Timestamp(WINDOW + timedelta(minutes=15)),
            offset_minutes=3, side=Side.UP.value, contracts=5, limit_price=0.60,
            max_price=0.61, expected_cost=3.14, model_probability=0.72,
            edge=0.08, status='new')
        symbols, staked, count = writer.entries_for_window(pd.Timestamp(WINDOW))
        assert count == 1 and symbols == frozenset({'BTC-USD'}), (
            'a ticket with no position must still count as an entry'
        )


class TestOutlayMatchesTheFill:
    """The bankroll must be debited what the venue charged, not what we intended.

    `outlay` and `fee` were pro-rated from `decision.stake` and `decision.fee`,
    both computed at the *decision* price. But `price` is booked from the fill,
    and a `fill_or_kill` fills at or better than its limit — so actual cash out is
    always <= the booked outlay and our bankroll read systematically low.

    Observed live 2026-08-25: `placing down 1 @ 0.08` then `filled 1 @ 0.0530`.
    The position recorded price 0.0530 and an outlay computed from 0.08, and the
    reconciler logged the difference as unexplained "balance drift".
    """

    def test_the_outlay_is_the_price_actually_paid(self, writer):
        from core.costs import trade_fee
        from core.config import Config

        # Decision at 0.60; the venue fills all 5 at 0.50.
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'f',
                             'average_fill_price_dollars': '0.5000'})
        assert run(act_on(args(), writer, kalshi, decision(contracts=5,
                                                          price=0.60), None)) is True
        position = writer.open_positions()[0]
        assert float(position.price) == pytest.approx(0.50), 'fill price not booked'
        # `outlay` is fee-inclusive, matching `decide()`'s `stake`.
        expected = 5 * 0.50 + float(trade_fee(5, 0.50, Config()))
        assert float(position.outlay) == pytest.approx(expected, abs=1e-6), (
            'outlay was pro-rated from the decision stake at 0.60 rather than '
            'computed from the 0.50 actually paid'
        )

    def test_the_fee_is_charged_on_the_fill_not_the_decision(self, writer):
        """A fee at the decision price is the wrong fee once the fill differs."""
        from core.costs import trade_fee
        from core.config import Config

        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'g',
                             'average_fill_price_dollars': '0.5000'})
        assert run(act_on(args(), writer, kalshi, decision(contracts=5,
                                                          price=0.60), None)) is True
        position = writer.open_positions()[0]
        assert float(position.fee) == pytest.approx(
            float(trade_fee(5, 0.50, Config())), abs=1e-6)

    def test_the_bankroll_falls_by_exactly_outlay_plus_fee(self, writer):
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'h',
                             'average_fill_price_dollars': '0.5000'})
        before = float(writer.account().bankroll)
        assert run(act_on(args(), writer, kalshi, decision(contracts=5,
                                                          price=0.60), None)) is True
        position = writer.open_positions()[0]
        after = float(writer.account().bankroll)
        assert before - after == pytest.approx(float(position.outlay), abs=1e-6)


class TestQuoteStaleness:
    """`max_quote_age_seconds` was declared and never checked anywhere.

    `run_cycle` reads the book once at the top of the cycle and stamps
    `quote_time`, then does a Coinbase fetch, four authenticated reconcile
    calls, inference and six 15-second quote calls before any order is sent —
    and nothing before this compared that elapsed time to anything. A quote
    that goes stale mid-cycle was traded as though it were still current.
    """

    def test_a_fresh_quote_places_the_order(self, writer):
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'fresh',
                             'average_fill_price_dollars': '0.6000'})
        quote_time = pd.Timestamp.now(tz='UTC')
        assert run(act_on(args(), writer, kalshi, decision(), None,
                          quote_time=quote_time)) is True
        assert len(kalshi.orders) == 1

    def test_a_stale_quote_refuses_without_sending_an_order(self, writer):
        from core.config import DEFAULT_CONFIG
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'stale',
                             'average_fill_price_dollars': '0.6000'})
        quote_time = pd.Timestamp.now(tz='UTC') - timedelta(
            seconds=DEFAULT_CONFIG.max_quote_age_seconds + 5)
        booked = run(act_on(args(), writer, kalshi, decision(), None,
                            quote_time=quote_time))
        assert booked is False
        assert len(kalshi.orders) == 0, 'a stale quote must never reach the wire'
        assert len(writer.open_positions()) == 0

    def test_omitting_quote_time_does_not_refuse(self, writer):
        """Backward compatible: no timestamp means no staleness claim to check."""
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'untimed',
                             'average_fill_price_dollars': '0.6000'})
        assert run(act_on(args(), writer, kalshi, decision(), None)) is True

    def test_a_stale_quote_still_writes_a_ticket_with_a_reason(self, writer):
        from core.config import DEFAULT_CONFIG
        kalshi = FakeKalshi({'status': 'executed', 'order_id': 'z'})
        quote_time = pd.Timestamp.now(tz='UTC') - timedelta(
            seconds=DEFAULT_CONFIG.max_quote_age_seconds + 5)
        run(act_on(args(), writer, kalshi, decision(), None, quote_time=quote_time))
        with writer._session() as session:  # noqa: SLF001
            from core.pg_writer import OrderTicket
            row = session.query(OrderTicket).one()
        assert row.status == 'skipped'
        assert 'stale' in (row.note or '').lower()

    def test_a_stale_quote_does_not_refuse_a_dry_run(self, writer):
        """Nothing is sent in dry-run anyway; staleness is not the reason to log."""
        from core.config import DEFAULT_CONFIG
        kalshi = FakeKalshi({})
        quote_time = pd.Timestamp.now(tz='UTC') - timedelta(
            seconds=DEFAULT_CONFIG.max_quote_age_seconds + 5)
        booked = run(act_on(args(mode='live', place_orders=False, dry_run=True),
                            writer, kalshi, decision(), None, quote_time=quote_time))
        assert booked is False
        assert len(kalshi.orders) == 0
