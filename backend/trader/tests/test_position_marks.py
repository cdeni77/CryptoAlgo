"""Open positions are marked to the VENUE's bid, never to our own forecast.

`at risk` on the dashboard is cost, deliberately: marking an open binary at the
probability the model believes books conviction as profit, which is how a
losing system draws a rising equity curve. The venue's bid is a different
claim — what someone will actually pay for the contract right now — so it is
safe to show moving.
"""
from __future__ import annotations

import pandas as pd
import pytest

from core.pg_writer import Outcome, PgWriter, Position

WINDOW = pd.Timestamp('2026-10-20 14:30', tz='UTC')
AT = pd.Timestamp('2026-10-20 14:36', tz='UTC').to_pydatetime()


@pytest.fixture
def writer(tmp_path):
    w = PgWriter(database_url=f'sqlite:///{tmp_path}/t.db')
    w.ensure_account(mode='paper', starting_bankroll=100.0)
    return w


def _open(w, *, offset=3, side='up', contracts=1, price=0.70):
    return w.open_position(
        symbol='BTC-USD', window_open=WINDOW,
        settle_time=WINDOW + pd.Timedelta(minutes=15), offset_minutes=offset,
        side=side, contracts=contracts, price=price, outlay=contracts * price,
        fee=0.0, model_probability=0.75, baseline_probability=0.70, edge=0.05)


def test_a_mark_is_written_with_its_timestamp(writer):
    pid = _open(writer)
    assert writer.mark_open_positions({('BTC-USD', WINDOW): 0.78}, at=AT) == 1
    with writer._session() as s:
        row = s.get(Position, pid)
        assert float(row.mark_price) == 0.78
        assert row.mark_at is not None


def test_a_settled_position_is_never_marked(writer):
    """It already has a payout. A mark beside that is a second, contradictory
    answer to the same question."""
    pid = _open(writer)
    writer.settle_position(pid, settled_up=True, fee=0.0)
    assert writer.mark_open_positions({('BTC-USD', WINDOW): 0.78}, at=AT) == 0
    with writer._session() as s:
        assert s.get(Position, pid).mark_price is None


def test_a_zero_bid_is_a_real_mark(writer):
    """Nobody bidding is a real state, and a position about to expire worthless
    is exactly when the number matters most."""
    pid = _open(writer)
    writer.mark_open_positions({('BTC-USD', WINDOW): 0.0}, at=AT)
    with writer._session() as s:
        assert float(s.get(Position, pid).mark_price) == 0.0


def test_a_missing_quote_leaves_the_previous_mark_alone(writer):
    """So the surface can show a mark going stale rather than jumping to zero
    or vanishing when one cycle fails to read the book."""
    pid = _open(writer)
    writer.mark_open_positions({('BTC-USD', WINDOW): 0.78}, at=AT)
    writer.mark_open_positions({}, at=AT)
    with writer._session() as s:
        assert float(s.get(Position, pid).mark_price) == 0.78


def test_each_position_in_a_window_is_marked(writer):
    """Multi-entry puts several positions on one (symbol, window); all of them
    are the same contract and all take the same bid."""
    a = _open(writer, offset=3)
    b = _open(writer, offset=6, price=0.74)
    assert writer.mark_open_positions({('BTC-USD', WINDOW): 0.80}, at=AT) == 2
    with writer._session() as s:
        assert float(s.get(Position, a).mark_price) == 0.80
        assert float(s.get(Position, b).mark_price) == 0.80


def test_marking_nothing_is_a_no_op(writer):
    _open(writer)
    assert writer.mark_open_positions({}, at=AT) == 0
