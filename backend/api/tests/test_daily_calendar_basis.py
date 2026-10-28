"""The day-by-day calendar measures crypto settlements and nothing else.

This Kalshi account is shared with markets this system does not trade, and it
takes deposits. A day measured on the account balance therefore reports a
football bet and a bank transfer as the strategy's result: one day's balance
moved -$25.64 while the crypto markets settled -$134.10, the difference being a
+$90.09 position in another series and the unrealised carried over midnight.
Across the ledger the other series come to -$2,128 and the deposits to $3,146.

`venue_settlements` holds the crypto series alone, and its per-day totals match
the venue's own `/portfolio/positions` ledger to the cent. So that is the
measure, in both scopes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

EASTERN = ZoneInfo('America/New_York')


def _session():
    from database import SessionLocal

    return SessionLocal()


def _clear():
    from models.serving import Account, VenueBalance, VenueSettlement

    with _session() as db:
        for model in (VenueSettlement, VenueBalance, Account):
            db.query(model).delete()
        db.commit()


@pytest.fixture(autouse=True)
def _clean(client):
    """The schema is created by the app lifespan, which the `client` fixture
    enters, so this must run after it rather than beside it.

    The SQLite file is shared by the whole session. An `account` row left
    behind here carries a `reset_at` that would hide every other module's
    fixtures from the epoch filter, so the table is emptied on the way out
    as well as on the way in.
    """
    _clear()
    yield
    _clear()


def _settlement(when, pnl, ticker='KXBTC15M-A'):
    from models.serving import VenueSettlement

    return VenueSettlement(
        ticker=ticker, market_result='no', yes_contracts=0.0,
        no_contracts=5.0, yes_cost=None, no_cost=1.50, revenue=5.0,
        fee_cost=0.07, pnl=pnl, settled_time=when)


def _balance(when, cash, mark=0.0):
    from models.serving import VenueBalance

    return VenueBalance(timestamp=when, balance=cash, total_balance=cash,
                        portfolio_value=mark, exchange_index=2)


def _seed(epoch=None):
    """One settled day, and a balance that moved for an unrelated reason."""
    from models.serving import Account

    # Noon Eastern on each day, so the Eastern bucket is unambiguous whatever
    # the host clock is set to.
    yesterday = datetime.now(timezone.utc) - timedelta(days=2, hours=12)
    today = yesterday + timedelta(days=1)
    with _session() as db:
        db.add(_settlement(yesterday, -4.0))
        db.add(_settlement(today, -10.0, ticker='KXBTC15M-B'))
        # +$500 of someone else's money across the same two days: a deposit, a
        # bet in another series, or both. None of it is this strategy.
        db.add(_balance(yesterday, 100.0))
        db.add(_balance(today, 600.0))
        if epoch is not None:
            db.add(Account(mode='live', starting_bankroll=100.0,
                           bankroll=100.0, reset_at=epoch))
        db.commit()
    # The bucket key is the EASTERN date, which is not the UTC one for half
    # the day. Deriving it the same way the server does keeps the test from
    # passing or failing on the hour it is run at.
    return today.astimezone(EASTERN).date().isoformat()


def test_all_time_counts_only_crypto_settlements(client):
    day = _seed()
    body = client.get('/account/venue/daily?days=30&all_time=true').json()
    cell = next(d for d in body['days'] if d['date'] == day)
    assert cell['pnl'] == -10.0
    assert body['total'] == -14.0


def test_since_reset_counts_only_crypto_settlements(client):
    """The epoch-scoped view must mean the same thing as the all-time one.

    It was measured on the change in account value, which is why the two
    scopes disagreed: the same day read -$10 under one heading and +$490
    under the other.
    """
    epoch = datetime.now(timezone.utc) - timedelta(days=3)
    day = _seed(epoch=epoch)
    body = client.get('/account/venue/daily?days=30').json()
    cell = next(d for d in body['days'] if d['date'] == day)
    assert cell['pnl'] == -10.0
    assert body['total'] == -14.0


def test_both_scopes_report_the_same_day_identically(client):
    """One quantity under one label, whichever scope is selected."""
    epoch = datetime.now(timezone.utc) - timedelta(days=3)
    day = _seed(epoch=epoch)
    scoped = client.get('/account/venue/daily?days=30').json()
    every = client.get('/account/venue/daily?days=30&all_time=true').json()
    a = next(d for d in scoped['days'] if d['date'] == day)
    b = next(d for d in every['days'] if d['date'] == day)
    assert a == b
    assert scoped['total'] == every['total']
