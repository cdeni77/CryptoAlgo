"""Positions must be re-settled when the venue graded them differently.

`settle_due` prefers the venue's settlement — but only if it already has one.
A window that matures before the venue's row arrives is graded from our
Coinbase bars, and nothing ever re-grades it: the same "only fills what is
empty" shape `set_window_outcome` had for predictions.

Measured 2026-10-03 over 216 settled trades: our books showed **130 winners
against the venue's 123**, a 3.2% disagreement matching the documented
Coinbase-vs-BRTI rate, all in near-ties. Worth $21.06 — our books read +$8.80
where the venue read -$12.26.

That is not just reporting. `account.realized_pnl` is what the daily-loss limit
and the drawdown breaker read, so both were keyed to a P&L optimistic by that
margin: the safety brakes under-react by exactly the amount our own label
flatters us.
"""

from __future__ import annotations

import datetime as dt

import pytest

OPEN = dt.datetime(2026, 10, 1, 12, 0, tzinfo=dt.timezone.utc)


@pytest.fixture
def writer(tmp_path):
    from core.pg_writer import PgWriter

    return PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')


def _position(writer, *, side='up', contracts=4, price=0.5, fee=0.05,
              settled_up=True):
    """One settled position, graded OUR way."""
    from core.pg_writer import Outcome, Position
    from core.book import won as _won

    outlay = contracts * price + fee
    win = _won(side=side, settled_up=settled_up)
    payout = float(contracts) if win else 0.0
    with writer._session() as session:                    # noqa: SLF001
        session.add(Position(
            symbol='BTC-USD', window_open=OPEN, settle_time=OPEN,
            offset_minutes=12, side=side, contracts=contracts, price=price,
            outlay=outlay, fee=fee, model_probability=0.6,
            baseline_probability=0.5, edge=0.02,
            outcome=Outcome.WON.value if win else Outcome.LOST.value,
            settled_up=settled_up, payout=payout, pnl=payout - outlay,
            settled_at=OPEN))
        session.commit()
    return outlay


def test_a_winner_the_venue_calls_a_loser_is_re_settled(writer):
    account = writer.ensure_account(500.0, mode='live')
    outlay = _position(writer, settled_up=True)       # we said UP, +$1.95
    before = writer.account().realized_pnl

    seen, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, False)])

    assert (seen, fixed) == (1, 1)
    assert delta == pytest.approx(-4.0), 'a 4-contract payout is withdrawn'
    # `realized_pnl` is RESTATED from the positions, not nudged by the delta --
    # see `test_realized_pnl_is_restated_not_nudged` for why. With one settled
    # position that is now a loss, the total is its own pnl.
    assert writer.account().realized_pnl == pytest.approx(-outlay)
    assert before == pytest.approx(0.0), 'sanity: the fixture starts flat'
    with writer._session() as session:                    # noqa: SLF001
        from core.pg_writer import Position
        row = session.query(Position).first()
        assert row.settled_up is False
        assert row.payout == 0.0
        assert row.pnl == pytest.approx(-outlay)


def test_agreement_re_settles_nothing(writer):
    """No position is re-graded when the venue agrees.

    `realized_pnl` is NOT asserted unchanged, because the restate runs every
    pass by design — it is a statement about what that field IS, so enforcing
    it only when something changed would leave existing drift in place. Here it
    corrects an inconsistency this fixture creates: `_position` inserts rows
    directly rather than through `settle_position`, which is what would
    normally accumulate the figure.
    """
    from core.pg_writer import Position

    writer.ensure_account(500.0, mode='live')
    _position(writer, settled_up=True)
    seen, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, True)])
    assert (fixed, delta) == (0, 0.0)
    with writer._session() as session:                    # noqa: SLF001
        expected = sum(float(r.pnl) for r in session.query(Position)
                       .filter(Position.settled_at.isnot(None)).all())
    assert writer.account().realized_pnl == pytest.approx(expected)


def test_it_is_idempotent(writer):
    """The reported count IS the disagreement, so a re-run must correct
    nothing."""
    writer.ensure_account(500.0, mode='live')
    _position(writer, settled_up=True)
    writer.correct_positions_from_venue([('BTC-USD', OPEN, False)])
    after_first = writer.account().realized_pnl
    _, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, False)])
    assert (fixed, delta) == (0, 0.0)
    assert writer.account().realized_pnl == pytest.approx(after_first)


def test_a_down_side_position_flips_the_other_way(writer):
    """`_won` is side-aware: a DOWN holding wins when the window settles
    down, so a venue 'up' must turn it into a loss."""
    writer.ensure_account(500.0, mode='live')
    _position(writer, side='down', settled_up=False)   # we said DOWN won
    _, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, True)])
    assert fixed == 1 and delta < 0


def test_bankroll_is_left_to_the_venue(writer):
    """`adopt_venue_balance` writes the venue's balance every cycle and the
    venue is the account of record. Correcting a figure it already owns would
    be a second source of truth."""
    writer.ensure_account(500.0, mode='live')
    _position(writer, settled_up=True)
    before = writer.account().bankroll
    writer.correct_positions_from_venue([('BTC-USD', OPEN, False)])
    assert writer.account().bankroll == pytest.approx(before)


def test_an_unsettled_position_is_not_touched(writer):
    """Only `settle_due` may grade a window the first time; this re-grades."""
    from core.pg_writer import Outcome, Position

    writer.ensure_account(500.0, mode='live')
    with writer._session() as session:                    # noqa: SLF001
        session.add(Position(
            symbol='BTC-USD', window_open=OPEN, settle_time=OPEN,
            offset_minutes=12, side='up', contracts=4, price=0.5,
            outlay=2.05, fee=0.05, model_probability=0.6,
            baseline_probability=0.5, edge=0.02,
            outcome=Outcome.PENDING.value, settled_at=None))
        session.commit()
    _, fixed, _ = writer.correct_positions_from_venue([('BTC-USD', OPEN, False)])
    assert fixed == 0


def test_realized_pnl_is_restated_not_nudged(writer):
    """`account.realized_pnl` covers the period since `account.reset_at`, while
    the venue labels reach further back — so applying the raw delta leaks
    pre-epoch corrections into a post-epoch total. On the first real run that
    put it at -$42.20 against post-epoch positions summing to -$12.20.

    The invariant it restores:
        realized_pnl == sum(positions.pnl where settled_at >= epoch)
    """
    from core.pg_writer import Account

    account = writer.ensure_account(500.0, mode='live')
    epoch = OPEN - dt.timedelta(hours=1)
    with writer._session() as session:                    # noqa: SLF001
        row = session.merge(account)
        row.reset_at = epoch
        row.realized_pnl = 0.0
        session.commit()

    # One position BEFORE the epoch and one after; both graded our way.
    _position(writer, settled_up=True)                     # at OPEN, post-epoch
    from core.pg_writer import Outcome, Position
    with writer._session() as session:                    # noqa: SLF001
        session.add(Position(
            symbol='ETH-USD', window_open=epoch - dt.timedelta(days=2),
            settle_time=epoch - dt.timedelta(days=2), offset_minutes=12,
            side='up', contracts=4, price=0.5, outlay=2.05, fee=0.05,
            model_probability=0.6, baseline_probability=0.5, edge=0.02,
            outcome=Outcome.WON.value, settled_up=True, payout=4.0, pnl=1.95,
            settled_at=epoch - dt.timedelta(days=2)))
        session.commit()

    # The venue disagrees with BOTH.
    writer.correct_positions_from_venue([
        ('BTC-USD', OPEN, False),
        ('ETH-USD', epoch - dt.timedelta(days=2), False),
    ])

    with writer._session() as session:                    # noqa: SLF001
        post = (session.query(Position)
                .filter(Position.settled_at >= epoch).all())
        expected = sum(float(r.pnl) for r in post)
    assert writer.account().realized_pnl == pytest.approx(expected), (
        'the pre-epoch correction must not leak into the post-epoch total'
    )
