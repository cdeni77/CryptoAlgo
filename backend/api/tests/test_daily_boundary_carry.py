"""A day is its settlements plus what crossed its midnights.

Kalshi marks open positions to market, so a window bought before midnight has
already shown part of its gain by the time it settles the next day. Realised
P&L books the whole thing on the settlement day, so the two disagree by exactly
the unrealised carried across the boundary.

Measured on a flat book: three positions were open at one Eastern close, 51
contracts costing $40.05, marked by the venue at $50.94. Our settled figure for
the next day was $35.32 and the venue's app showed $24.43 --
35.32 + 0.00 - 10.89 = 24.43.

The carry is `the venue's mark - what the open positions cost`. Both sides come
from stored rows; nothing here is estimated.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

EASTERN = ZoneInfo('America/New_York')


def _session():
    from database import SessionLocal

    return SessionLocal()


def _clear():
    from models.serving import Account, Position, VenueBalance, VenueSettlement

    with _session() as db:
        for model in (VenueSettlement, VenueBalance, Position, Account):
            db.query(model).delete()
        db.commit()


@pytest.fixture(autouse=True)
def _clean(client):
    """The schema is created by the app lifespan, which the `client` fixture
    enters, so this must run after it rather than beside it.
    """
    _clear()
    yield
    _clear()


def _close_of(day):
    """The instant an Eastern day ends, in UTC."""
    return datetime.combine(day + timedelta(days=1), time(0, 0),
                            tzinfo=EASTERN).astimezone(timezone.utc)


def _settlement(when, pnl, ticker):
    from models.serving import VenueSettlement

    return VenueSettlement(
        ticker=ticker, market_result='no', yes_contracts=0.0, no_contracts=5.0,
        yes_cost=None, no_cost=1.50, revenue=5.0, fee_cost=0.07, pnl=pnl,
        settled_time=when)


def _balance(when, cash, mark):
    from models.serving import VenueBalance

    return VenueBalance(timestamp=when, balance=cash, total_balance=cash,
                        portfolio_value=mark, exchange_index=2)


def _position(window_open, settled_at, contracts, outlay, fee, pnl):
    from models.serving import Position

    return Position(
        symbol='BTC-USD', window_open=window_open,
        settle_time=window_open + timedelta(minutes=15), offset_minutes=3,
        side='up', contracts=contracts, price=0.78, outlay=outlay, fee=fee,
        model_probability=0.8, baseline_probability=0.78, edge=0.02,
        outcome='won', settled_at=settled_at, pnl=pnl)


def _seed():
    """A book carrying $10.89 of unrealised across one Eastern midnight."""
    today = (datetime.now(timezone.utc).astimezone(EASTERN).date()
             - timedelta(days=1))
    yesterday = today - timedelta(days=1)
    boundary = _close_of(yesterday)          # the midnight that is crossed

    with _session() as db:
        # A flat book at the START of the earlier day, so that day has a carry
        # at both ends and is not reported as unmeasurable.
        db.add(_balance(_close_of(yesterday - timedelta(days=1))
                        - timedelta(seconds=1), 953.58, 0.0))
        # Settled on the earlier day, so it is not part of today's figure.
        db.add(_settlement(boundary - timedelta(hours=6), 4.00, 'KXBTC15M-Y'))
        # The three that crossed the boundary and settled after it.
        db.add(_settlement(boundary + timedelta(minutes=10), 35.32, 'KXBTC15M-T'))
        # Open at the boundary: 51 contracts, cost 40.05, the venue marks 50.94.
        db.add(_position(boundary - timedelta(minutes=5),
                         boundary + timedelta(minutes=10),
                         51, 40.00, 0.05, 10.95))
        db.add(_balance(boundary - timedelta(seconds=1), 957.58, 50.94))
        # The book is flat by the end of the later day.
        db.add(_balance(_close_of(today) - timedelta(seconds=1), 1032.95, 0.0))
        db.commit()
    return yesterday.isoformat(), today.isoformat()


def test_the_day_is_adjusted_for_what_crossed_its_midnights(client):
    """35.32 settled, 10.89 of it already shown the day before -> 24.43."""
    _, today = _seed()
    body = client.get('/account/venue/daily?days=5&all_time=true').json()
    cell = next(d for d in body['days'] if d['date'] == today)

    assert cell['pnl'] == 35.32, 'the settled primitive must not move'
    assert cell['carried_in'] == 10.89
    assert cell['carried_out'] == 0.0
    assert cell['pnl_marked'] == 24.43


def test_the_earlier_day_is_credited_with_what_it_carried_out(client):
    """The same $10.89 belongs to the day the mark moved, not to neither day."""
    yesterday, _ = _seed()
    body = client.get('/account/venue/daily?days=5&all_time=true').json()
    cell = next(d for d in body['days'] if d['date'] == yesterday)

    assert cell['pnl'] == 4.00
    assert cell['carried_in'] == 0.0
    assert cell['carried_out'] == 10.89
    assert cell['pnl_marked'] == 14.89


def test_a_day_with_no_boundary_mark_falls_back_to_settled(client):
    """Before the sampler existed there is no mark, so there is no carry."""
    _clear()
    long_ago = datetime.now(timezone.utc) - timedelta(days=20)
    with _session() as db:
        db.add(_settlement(long_ago, 7.50, 'KXBTC15M-OLD'))
        db.commit()

    body = client.get('/account/venue/daily?days=30&all_time=true').json()
    cell = next(d for d in body['days']
                if d['date'] == long_ago.astimezone(EASTERN).date().isoformat())
    assert cell['pnl'] == 7.50
    assert cell['pnl_marked'] is None
    assert cell['carried_in'] is None


def test_a_mark_too_big_for_the_crypto_book_is_flagged(client):
    """A binary pays at most $1, so a mark above the open contract count is foreign.

    This account is shared; a position in another series sits in the venue's
    `portfolio_value` and would otherwise be carried in as strategy P&L.
    """
    _clear()
    today = (datetime.now(timezone.utc).astimezone(EASTERN).date()
             - timedelta(days=1))
    yesterday = today - timedelta(days=1)
    boundary = _close_of(yesterday)
    with _session() as db:
        db.add(_settlement(boundary + timedelta(minutes=10), 5.00, 'KXBTC15M-F'))
        db.add(_position(boundary - timedelta(minutes=5),
                         boundary + timedelta(minutes=10), 10, 7.00, 0.05, 2.0))
        # 10 contracts can be worth at most $10; $400 is somebody else's bet.
        db.add(_balance(boundary - timedelta(seconds=1), 900.0, 400.0))
        db.add(_balance(_close_of(today) - timedelta(seconds=1), 905.0, 0.0))
        db.commit()

    body = client.get('/account/venue/daily?days=5&all_time=true').json()
    cell = next(d for d in body['days'] if d['date'] == today.isoformat())
    assert cell['foreign_mark'] is True
    assert cell['pnl_marked'] is None, 'a contaminated carry must not be shown'
    assert cell['pnl'] == 5.00
