"""Tests for `correct_positions_from_venue`, which re-settles positions the venue graded
differently.

A window that settles before the venue's result arrives is graded from our
Coinbase bars. `realized_pnl` feeds the daily-loss limit and the drawdown
breaker, so it must match the venue's grading.
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
    """One settled position, graded our way."""
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
    outlay = _position(writer, settled_up=True)  # we said UP, +$1.95
    before = writer.account().realized_pnl

    seen, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, False)])

    assert (seen, fixed) == (1, 1)
    assert delta == pytest.approx(-4.0), 'a 4-contract payout is withdrawn'
    # `realized_pnl` is restated from the positions, not adjusted by the delta (see
    # `test_realized_pnl_is_restated_not_nudged`). With one settled losing position,
    # the total is its own pnl.
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

    `realized_pnl` is restated on every pass by design. Here it corrects a
    mismatch the fixture creates by inserting rows directly.
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
    """A second run corrects nothing."""
    writer.ensure_account(500.0, mode='live')
    _position(writer, settled_up=True)
    writer.correct_positions_from_venue([('BTC-USD', OPEN, False)])
    after_first = writer.account().realized_pnl
    _, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, False)])
    assert (fixed, delta) == (0, 0.0)
    assert writer.account().realized_pnl == pytest.approx(after_first)


def test_a_down_side_position_flips_the_other_way(writer):
    """A DOWN position flips the other way: a venue 'up' makes it a loss."""
    writer.ensure_account(500.0, mode='live')
    _position(writer, side='down', settled_up=False)   # we said DOWN won
    _, fixed, delta = writer.correct_positions_from_venue(
        [('BTC-USD', OPEN, True)])
    assert fixed == 1 and delta < 0


def test_bankroll_is_left_to_the_venue(writer):
    """The bankroll is left alone; `adopt_venue_balance` already takes it from the venue.
    """
    writer.ensure_account(500.0, mode='live')
    _position(writer, settled_up=True)
    before = writer.account().bankroll
    writer.correct_positions_from_venue([('BTC-USD', OPEN, False)])
    assert writer.account().bankroll == pytest.approx(before)


def test_an_unsettled_position_is_not_touched(writer):
    """An unsettled position is not touched; only `settle_due` grades a window first."""
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
    """`realized_pnl` is restated from post-epoch positions, so pre-epoch corrections do
    not leak in.

    Invariant: realized_pnl == sum(positions.pnl where settled_at >= epoch).
    """
    from core.pg_writer import Account

    account = writer.ensure_account(500.0, mode='live')
    epoch = OPEN - dt.timedelta(hours=1)
    with writer._session() as session:                    # noqa: SLF001
        row = session.merge(account)
        row.reset_at = epoch
        row.realized_pnl = 0.0
        session.commit()

    # One position before the epoch and one after; both graded our way.
    _position(writer, settled_up=True)  # at OPEN, post-epoch
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

    # The venue disagrees with both.
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
