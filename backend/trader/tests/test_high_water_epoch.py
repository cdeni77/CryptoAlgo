"""Peak and current realised P&L must span the same period.

`realised_high_water()` is half of the live drawdown breaker; the other half is
`account.realized_pnl`. `scripts/reset_dashboard.py --rebase` zeroes the second
at a new epoch — and this walked every settlement ever recorded, so the breaker
compared a high-water from BEFORE the reset against a P&L counted AFTER it.

Measured on the live account 2026-09-24: peak $322.38 against current $173.02,
a phantom **46.3% drawdown** that halted an account whose realised P&L since its
epoch was **-$26.98**. The halt is sticky and cleared by hand, so a false one
costs real trading days.
"""

from __future__ import annotations

import datetime as dt

import pytest

EPOCH = dt.datetime(2026, 9, 24, 12, 0, tzinfo=dt.timezone.utc)


def _settle(writer, pnl, when):
    """One settled position with a known pnl at a known instant."""
    from core.pg_writer import Outcome, Position

    with writer._session() as session:                    # noqa: SLF001
        session.add(Position(
            symbol='BTC-USD', window_open=when, settle_time=when,
            offset_minutes=12, side='up', contracts=1, price=0.5,
            outlay=0.5, fee=0.0, model_probability=0.6,
            baseline_probability=0.5, edge=0.02,
            outcome=Outcome.WON.value if pnl > 0 else Outcome.LOST.value,
            pnl=pnl, settled_at=when))
        session.commit()


@pytest.fixture
def writer(tmp_path):
    from core.pg_writer import PgWriter

    return PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')


def test_without_an_epoch_it_spans_everything(writer):
    writer.ensure_account(100.0, mode='paper')
    _settle(writer, +50.0, EPOCH - dt.timedelta(days=5))
    _settle(writer, -10.0, EPOCH + dt.timedelta(hours=1))
    assert writer.realised_high_water() == pytest.approx(50.0)


def test_an_epoch_excludes_the_earlier_peak(writer):
    """The bug: a +$50 run before the reset must not become the peak that a
    post-reset -$10 is measured against."""
    account = writer.ensure_account(100.0, mode='paper')
    _settle(writer, +50.0, EPOCH - dt.timedelta(days=5))
    _settle(writer, -10.0, EPOCH + dt.timedelta(hours=1))
    with writer._session() as session:                    # noqa: SLF001
        row = session.merge(account)
        row.reset_at = EPOCH
        session.commit()
    assert writer.realised_high_water() == pytest.approx(0.0), (
        'nothing after the epoch was ever in profit, so the high-water is 0'
    )


def test_a_post_epoch_peak_still_counts(writer):
    account = writer.ensure_account(100.0, mode='paper')
    _settle(writer, +80.0, EPOCH - dt.timedelta(days=5))
    _settle(writer, +12.0, EPOCH + dt.timedelta(hours=1))
    _settle(writer, -4.0, EPOCH + dt.timedelta(hours=2))
    with writer._session() as session:                    # noqa: SLF001
        row = session.merge(account)
        row.reset_at = EPOCH
        session.commit()
    assert writer.realised_high_water() == pytest.approx(12.0), (
        'the breaker must still see a genuine post-reset peak'
    )
