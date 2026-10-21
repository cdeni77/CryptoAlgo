"""The fee is recorded from the venue and subtracted from P&L.

Two defects, measured 2026-10-20 and fixed together:

  * `settle_position` computed `pnl = payout - outlay`, and `outlay` is
    `contracts * price` exactly, so the fee was never in the arithmetic. Our
    books reported GROSS against the venue's net: of 419 settled markets, 409
    differed from the venue by precisely the fee and nothing else.
  * `Position.fee` was 0.0000 on every row, because `maker_entry` implies
    `assume_maker` and that estimate is made at DECISION time -- before it is
    known whether the order rests or crosses. ~19% of fills cross and pay about
    1.18c per contract.
"""
from __future__ import annotations

import pytest

from core.pg_writer import Account, Outcome, PgWriter, Position


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')
    w.ensure_account(mode='paper', starting_bankroll=100.0)
    return w


WINDOW = __import__('pandas').Timestamp('2026-10-20 12:00', tz='UTC')


def _open(w, *, contracts=2, price=0.70, fee=0.0):
    import pandas as pd
    return w.open_position(
        symbol='BTC-USD', window_open=pd.Timestamp('2026-10-20 12:00', tz='UTC'),
        settle_time=pd.Timestamp('2026-10-20 12:15', tz='UTC'),
        offset_minutes=3, side='up', contracts=contracts, price=price,
        outlay=contracts * price, fee=fee, model_probability=0.75,
        baseline_probability=0.70, edge=0.05)


def test_the_venue_fee_is_subtracted_from_pnl(writer):
    pid = _open(writer, contracts=2, price=0.70)
    pnl = writer.settle_position(pid, settled_up=True, fee=0.0236)
    # payout 2.00 - outlay 1.40 - fee 0.0236
    assert pnl == pytest.approx(0.5764, abs=1e-9)


def test_the_venue_fee_is_stored_on_the_position(writer):
    pid = _open(writer)
    writer.settle_position(pid, settled_up=True, fee=0.0236)
    with writer._session() as s:
        assert float(s.get(Position, pid).fee) == pytest.approx(0.0236)


def test_a_loss_also_pays_the_fee(writer):
    """The fee is charged on entry, so a losing market pays stake AND fee."""
    pid = _open(writer, contracts=2, price=0.70)
    pnl = writer.settle_position(pid, settled_up=False, fee=0.0236)
    assert pnl == pytest.approx(-1.4236, abs=1e-9)


def test_a_zero_fee_is_honoured_not_ignored(writer):
    """A maker fill really is charged $0.00. Treating 0 as 'absent' would make
    every maker settlement fall back to a stale estimate."""
    pid = _open(writer, contracts=2, price=0.70, fee=0.05)
    pnl = writer.settle_position(pid, settled_up=True, fee=0.0)
    assert pnl == pytest.approx(0.60, abs=1e-9)
    with writer._session() as s:
        assert float(s.get(Position, pid).fee) == 0.0


def test_no_venue_fee_falls_back_to_the_booked_estimate(writer):
    """Paper mode, or a settlement the venue has not posted yet. The
    arithmetic must still be NET, just against our own estimate."""
    pid = _open(writer, contracts=2, price=0.70, fee=0.03)
    pnl = writer.settle_position(pid, settled_up=True, fee=None)
    assert pnl == pytest.approx(0.57, abs=1e-9)


def test_fees_paid_moves_by_the_difference_not_the_total(writer):
    """The entry already booked an estimate. Adding the venue's figure on top
    of it would count the fee twice."""
    pid = _open(writer, contracts=2, price=0.70, fee=0.01)
    with writer._session() as s:
        before = float(s.query(Account).one().fees_paid)
    writer.settle_position(pid, settled_up=True, fee=0.0236)
    with writer._session() as s:
        after = float(s.query(Account).one().fees_paid)
    assert after - before == pytest.approx(0.0136, abs=1e-9)


def test_settling_twice_is_still_idempotent(writer):
    pid = _open(writer)
    first = writer.settle_position(pid, settled_up=True, fee=0.0236)
    assert first is not None
    assert writer.settle_position(pid, settled_up=True, fee=0.0236) is None


def test_realized_pnl_on_the_account_is_net(writer):
    """The dashboard's "ours" figure reads `account.realized_pnl` directly, so
    it is this number that was reading gross against the venue's net."""
    pid = _open(writer, contracts=2, price=0.70)
    writer.settle_position(pid, settled_up=True, fee=0.0236)
    with writer._session() as s:
        assert float(s.query(Account).one().realized_pnl) == pytest.approx(
            0.5764, abs=1e-9)


def test_the_bankroll_is_credited_the_payout_not_the_net(writer):
    """The fee leaves the account at ENTRY, not at settlement. Deducting it
    from the payout here as well would debit it twice — and live, the venue
    balance overwrites ours each cycle, so the error would surface as a growing
    drift rather than as an obvious wrong number."""
    pid = _open(writer, contracts=2, price=0.70)
    with writer._session() as s:
        before = float(s.query(Account).one().bankroll)
    writer.settle_position(pid, settled_up=True, fee=0.0236)
    with writer._session() as s:
        after = float(s.query(Account).one().bankroll)
    assert after - before == pytest.approx(2.0, abs=1e-9)


# --- the venue's fee is PER MARKET, not per position -------------------------
#
# Multi-entry puts several positions on one market. Handing the venue's whole
# `fee_cost` to each of them charged it once per position: measured 2026-10-20
# on KXBTC15M-26OCT201130-30, our fee came to 0.0714 against the venue's
# 0.0357 — exactly double across two positions — and that 3.5c was the entire
# residual `pnl_gap` after everything else reconciled.

def test_one_position_in_a_market_takes_the_whole_fee(writer):
    from livetrade.ledger import _fee_share
    pid = _open(writer, contracts=2)
    with writer._session() as s:
        from core.pg_writer import Position
        assert _fee_share(writer, s.get(Position, pid)) == 1.0


def test_two_positions_split_the_fee_by_contracts(writer):
    """Three contracts split 2/1 must take 2/3 and 1/3 of the market's fee —
    summing to exactly one fee, never two."""
    import pandas as pd

    from core.pg_writer import Position
    from livetrade.ledger import _fee_share

    a = writer.open_position(
        symbol='BTC-USD', window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=3,
        side='up', contracts=2, price=0.70, outlay=1.40, fee=0.0,
        model_probability=0.75, baseline_probability=0.70, edge=0.05)
    b = writer.open_position(
        symbol='BTC-USD', window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=6,
        side='up', contracts=1, price=0.72, outlay=0.72, fee=0.0,
        model_probability=0.75, baseline_probability=0.70, edge=0.05)
    with writer._session() as s:
        share_a = _fee_share(writer, s.get(Position, a))
        share_b = _fee_share(writer, s.get(Position, b))
    assert share_a == pytest.approx(2 / 3)
    assert share_b == pytest.approx(1 / 3)
    # The property that actually matters.
    assert share_a + share_b == pytest.approx(1.0)


def test_a_zero_contract_market_splits_evenly_rather_than_dividing_by_zero(writer):
    import pandas as pd

    from core.pg_writer import Position
    from livetrade.ledger import _fee_share

    ids = [
        writer.open_position(
            symbol='BTC-USD', window_open=WINDOW,
            settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=off,
            side='up', contracts=0, price=0.70, outlay=0.0, fee=0.0,
            model_probability=0.75, baseline_probability=0.70, edge=0.05)
        for off in (3, 6)
    ]
    with writer._session() as s:
        shares = [_fee_share(writer, s.get(Position, i)) for i in ids]
    assert sum(shares) == pytest.approx(1.0)


# --- the venue re-grade path must also be net of the fee --------------------
#
# `correct_outcomes` re-grades a position when the venue settled it differently
# from our bars. It recomputed `pnl = payout - outlay` while `settle_position`
# uses `payout - outlay - fee`, so any re-graded position silently became
# fee-free — and `account.realized_pnl` is restated as `sum(positions.pnl)` in
# the same function, so the error went straight to the dashboard headline.

def test_a_regraded_position_keeps_its_fee(writer):
    """Flip the outcome via the venue and the fee must still be subtracted."""
    import pandas as pd

    from core.pg_writer import Position

    pid = _open(writer, contracts=2, price=0.70, fee=0.03)
    writer.settle_position(pid, settled_up=True, fee=0.03)
    with writer._session() as s:
        assert float(s.get(Position, pid).pnl) == pytest.approx(0.57, abs=1e-9)

    # The venue says it actually settled DOWN.
    writer.correct_positions_from_venue([('BTC-USD', pd.Timestamp(WINDOW), False)])

    with writer._session() as s:
        row = s.get(Position, pid)
        assert row.outcome == 'lost'
        # payout 0 - outlay 1.40 - fee 0.03, NOT -1.40.
        assert float(row.pnl) == pytest.approx(-1.43, abs=1e-9)
