"""The venue-ledger routes: what they serve, and what they refuse to invent.

The failure being designed against is not a 500. It is `$0.00 realised` on a
paper account that has never traded, rendering identically to a measured zero —
the same class of error as the `pr_auc = holdout_auc - 0.06` this whole surface
was rewritten to remove.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _session():
    from database import SessionLocal

    return SessionLocal()


def _clear():
    from models.serving import VenueBalance, VenueFill, VenueSettlement

    with _session() as db:
        for model in (VenueSettlement, VenueFill, VenueBalance):
            db.query(model).delete()
        db.commit()


def _settlement(**overrides):
    from models.serving import VenueSettlement

    row = {
        'ticker': 'KXBTC15M-A', 'market_result': 'no', 'yes_contracts': 0.0,
        'no_contracts': 5.0, 'yes_cost': None, 'no_cost': 1.50,
        'revenue': 5.0, 'fee_cost': 0.07, 'pnl': 3.43,
        'settled_time': datetime(2026, 8, 25, 10, 15, tzinfo=timezone.utc),
    }
    row.update(overrides)
    return VenueSettlement(**row)


def test_an_unsynced_ledger_says_so_rather_than_reporting_zero(client):
    """A paper account has no venue ledger, and the response must not imply one."""
    _clear()
    body = client.get('/account/venue').json()

    assert body['available'] is False
    assert body['reason']
    assert 'sync_venue' in body['reason']
    # Nulls with reasons, never zeros.
    for field in ('realized_pnl', 'fees', 'balance', 'win_rate'):
        assert body[field]['value'] is None
        assert body[field]['reason']


def test_the_ledger_totals_what_the_venue_paid(client):
    _clear()
    with _session() as db:
        db.add(_settlement())
        db.add(_settlement(ticker='KXBTC15M-B', market_result='yes',
                           revenue=0.0, pnl=-1.57,
                           settled_time=datetime(2026, 8, 25, 10, 30,
                                                 tzinfo=timezone.utc)))
        db.commit()

    body = client.get('/account/venue').json()
    assert body['available'] is True
    assert body['settlements'] == 2
    assert body['realized_pnl']['value'] == 3.43 - 1.57
    assert body['fees']['value'] == 0.14
    assert body['contracts'] == 10.0
    # We held NO in both; the venue resolved one 'no' and one 'yes'.
    assert (body['wins'], body['losses']) == (1, 1)
    assert body['win_rate']['value'] == 0.5


def test_a_winning_favourite_that_nets_negative_still_counts_as_a_win(client):
    """The win rate reads `market_result`, not the sign of the P&L.

    100 contracts at 97c returns $100 on $97 of cost; a fee above $3 makes the net
    negative. This system buys favourites, so this is where most of its trades are,
    and classifying them by P&L sign would put the win rate at odds with the
    venue's own settlement record.
    """
    _clear()
    with _session() as db:
        db.add(_settlement(ticker='KXBTC15M-FAV', market_result='no',
                           no_contracts=100.0, no_cost=97.0, revenue=100.0,
                           fee_cost=3.5, pnl=-0.5))
        db.commit()

    body = client.get('/account/venue').json()
    assert body['wins'] == 1 and body['losses'] == 0
    assert body['realized_pnl']['value'] == -0.5

    row = client.get('/venue/settlements').json()['settlements'][0]
    assert row['won'] is True
    assert row['pnl'] == -0.5


def test_an_incomplete_settlement_is_counted_not_added_as_zero(client):
    """A row the venue left a gap in makes the total short by an unknown amount."""
    _clear()
    with _session() as db:
        db.add(_settlement())
        db.add(_settlement(ticker='KXBTC15M-GAP', market_result=None,
                           revenue=None, fee_cost=None, no_cost=None, pnl=None))
        db.commit()

    body = client.get('/account/venue').json()
    assert body['settlements'] == 2
    assert body['incomplete'] == 1
    assert body['realized_pnl']['value'] == 3.43, 'the gap is excluded, not zeroed'
    assert body['undecided'] == 1

    row = next(r for r in client.get('/venue/settlements').json()['settlements']
               if r['ticker'] == 'KXBTC15M-GAP')
    assert row['pnl'] is None
    assert row['won'] is None


def test_every_field_null_totals_to_null_rather_than_to_zero(client):
    _clear()
    with _session() as db:
        db.add(_settlement(revenue=None, fee_cost=None, no_cost=None, pnl=None,
                           market_result=None))
        db.commit()

    body = client.get('/account/venue').json()
    assert body['available'] is True
    assert body['realized_pnl']['value'] is None
    assert body['realized_pnl']['reason']
    assert body['win_rate']['value'] is None


def test_the_gap_against_our_own_books_is_served_as_a_measurement(client):
    """Ours and theirs, side by side. The disagreement is the point."""
    from models.serving import Account, VenueBalance

    _clear()
    with _session() as db:
        db.query(Account).delete()
        db.add(Account(mode='live', starting_bankroll=100.0, bankroll=103.0,
                       realized_pnl=3.0, fees_paid=0.05))
        db.add(_settlement(our_pnl=3.0))
        db.add(VenueBalance(timestamp=datetime.now(timezone.utc), balance=103.43,
                            exchange_index=2, our_bankroll=103.0, drift=0.43))
        db.commit()

    body = client.get('/account/venue').json()
    assert body['mode'] == 'live', 'a live figure must never render as a paper one'
    assert body['our_realized_pnl']['value'] == 3.0
    assert round(body['pnl_gap']['value'], 2) == 0.43
    assert body['balance']['value'] == 103.43
    assert body['balance_drift']['value'] == 0.43
    assert body['exchange_index'] == 2

    row = client.get('/venue/settlements').json()['settlements'][0]
    assert round(row['pnl_gap'], 2) == 0.43

    with _session() as db:
        db.query(Account).delete()
        db.commit()


def test_a_market_we_never_booked_has_a_null_gap_and_a_reason(client):
    """Not a zero gap: we have no figure to disagree with."""
    _clear()
    with _session() as db:
        db.add(_settlement(our_pnl=None))
        db.commit()

    row = client.get('/venue/settlements').json()['settlements'][0]
    assert row['our_pnl'] is None
    assert row['pnl_gap'] is None


def test_the_curve_is_cumulative_within_its_window_and_says_what_it_excluded(client):
    """A partial total must never be shown as the account's lifetime total."""
    _clear()
    now = datetime.now(timezone.utc)
    with _session() as db:
        # Well outside a 7-day window.
        db.add(_settlement(ticker='OLD', pnl=10.0, settled_time=now - timedelta(days=40)))
        db.add(_settlement(ticker='NEW-1', pnl=1.0, settled_time=now - timedelta(hours=2)))
        db.add(_settlement(ticker='NEW-2', pnl=-0.5, settled_time=now - timedelta(hours=1)))
        db.commit()

    body = client.get('/account/venue/equity?days=7').json()
    assert [p['ticker'] for p in body['points']] == ['NEW-1', 'NEW-2']
    assert [p['cumulative_pnl'] for p in body['points']] == [1.0, 0.5]
    assert body['pnl_before_window'] == 10.0
    assert body['realized_pnl_in_window'] == 0.5


def test_the_curve_skips_a_settlement_with_no_pnl(client):
    """An unknown step is absent, not flat. A flat step looks like a measurement."""
    _clear()
    now = datetime.now(timezone.utc)
    with _session() as db:
        db.add(_settlement(ticker='A', pnl=2.0, settled_time=now - timedelta(hours=3)))
        db.add(_settlement(ticker='B', pnl=None, settled_time=now - timedelta(hours=2)))
        db.add(_settlement(ticker='C', pnl=1.0, settled_time=now - timedelta(hours=1)))
        db.commit()

    points = client.get('/account/venue/equity?days=7').json()['points']
    assert [p['ticker'] for p in points] == ['A', 'C']
    assert [p['cumulative_pnl'] for p in points] == [2.0, 3.0]


def test_the_balance_series_rides_along_for_the_live_end_of_the_chart(client):
    """Settlements step every fifteen minutes at most; the balance is per cycle."""
    from models.serving import VenueBalance

    _clear()
    now = datetime.now(timezone.utc)
    with _session() as db:
        for i in range(3):
            db.add(VenueBalance(timestamp=now - timedelta(minutes=i),
                                balance=100.0 + i, our_bankroll=100.0,
                                drift=float(i), exchange_index=2))
        db.commit()

    body = client.get('/account/venue/equity?days=1').json()
    assert len(body['balances']) == 3
    assert [b['balance'] for b in body['balances']] == [102.0, 101.0, 100.0]
    assert body['points'] == []


def test_a_no_fill_is_served_at_what_it_cost_and_not_at_the_yes_price(client):
    """0.30 for a 30c NO, not the 0.70 the venue's single YES book quotes."""
    from models.serving import VenueFill

    _clear()
    with _session() as db:
        db.add(VenueFill(trade_id='f1', order_id='o1', ticker='KXBTC15M-A',
                         side='down', action='buy', contracts=5.0, price=0.30,
                         is_taker=True,
                         created_time=datetime.now(timezone.utc)))
        db.commit()

    row = client.get('/venue/fills').json()['fills'][0]
    assert row['side'] == 'down'
    assert row['price'] == 0.30
    assert row['cost'] == 1.5


def test_the_venue_routes_validate_their_arguments(client):
    assert client.get('/account/venue/equity?days=0').status_code == 422
    assert client.get('/account/venue/equity?days=400').status_code == 422
    assert client.get('/venue/settlements?limit=0').status_code == 422
    assert client.get('/venue/fills?limit=5000').status_code == 422
    # `days` is optional on the table routes: the whole ledger is the default.
    assert client.get('/venue/settlements').status_code == 200
    assert client.get('/venue/fills').status_code == 200
