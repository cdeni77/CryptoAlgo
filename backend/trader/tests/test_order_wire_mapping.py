"""PROPOSED. The UP/DOWN -> yes/no mapping on the wire. HIGHEST SEVERITY GAP.

Drop into `backend/trader/tests/`.

Measured: flipping `data_collection/kalshi_client.py`

    'side': 'yes' if side == 'up' else 'no',
    ('yes_price' if side == 'up' else 'no_price'): cents,

to its opposite leaves all 230 tests passing. That mutation buys the exact
opposite side of every trade with real money. `core/decide.py`'s side *label*
is covered (flipping it kills three tests); the boundary where the label becomes
a venue order body is not covered at all, and `kalshi_client.py:437-449` never
executes in the suite.

The test intercepts `_request`, so it never touches the network.
"""

from __future__ import annotations

import asyncio

import pytest

from data_collection.kalshi_client import KalshiClient

KEY = None  # filled by the fixture below


@pytest.fixture
def key_pem():
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def captured_body(key_pem, **kwargs) -> dict:
    """Place an order against a stubbed transport and return the JSON body."""
    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=True)
    seen = {}

    async def fake_request(method, path, *, body=None, **_):
        seen['method'] = method
        seen['path'] = path
        seen['body'] = body
        return {'order': {'order_id': 'o1', 'status': 'executed'}}

    client._request = fake_request
    asyncio.run(client.place_order(**kwargs))
    return seen


def test_an_up_decision_buys_yes(key_pem):
    """UP means "settles at or above the strike", which is the YES contract.

    If this ever inverts, every trade is placed on the losing side and the PnL
    is the mirror image of the forecast — which looks like a broken model, not a
    broken mapping.
    """
    seen = captured_body(key_pem, ticker='KXBTC15M-26AUG230030', side='up',
                         contracts=4, limit_price=0.87)
    body = seen['body']
    assert seen['method'] == 'POST'
    assert seen['path'] == '/portfolio/orders'
    assert body['side'] == 'yes'
    assert body['action'] == 'buy'
    assert body['yes_price'] == 87
    assert 'no_price' not in body
    assert body['count'] == 4
    assert body['ticker'] == 'KXBTC15M-26AUG230030'


def test_a_down_decision_buys_no(key_pem):
    seen = captured_body(key_pem, ticker='KXBTC15M-26AUG230030', side='down',
                         contracts=2, limit_price=0.13)
    body = seen['body']
    assert body['side'] == 'no'
    assert body['no_price'] == 13
    assert 'yes_price' not in body
    assert body['count'] == 2


def test_the_price_field_and_the_side_field_never_disagree(key_pem):
    """A body carrying `side: no` and a `yes_price` is a half-applied flip.

    Which is the shape the bug actually takes when someone edits one of the two
    expressions and not the other.
    """
    for side, expected in (('up', 'yes'), ('down', 'no')):
        body = captured_body(key_pem, ticker='T', side=side, contracts=1,
                             limit_price=0.5)['body']
        assert body['side'] == expected
        assert f'{expected}_price' in body
        other = 'no' if expected == 'yes' else 'yes'
        assert f'{other}_price' not in body


def test_the_limit_price_is_whole_cents_of_the_dollar_price(key_pem):
    """Dollars in, integer cents out. A stray factor of 100 here is a 100x order."""
    for dollars, cents in ((0.01, 1), (0.13, 13), (0.5, 50), (0.87, 87), (0.99, 99)):
        body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                             limit_price=dollars)['body']
        assert body['yes_price'] == cents, dollars


def test_the_default_time_in_force_is_fill_or_kill(key_pem):
    """A 15-minute market is a wasting asset; a resting order fills against a
    barrier probability that no longer holds."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.5)['body']
    assert body['time_in_force'] == 'fill_or_kill'
    assert body['type'] == 'limit'


def test_every_order_carries_an_idempotency_key(key_pem):
    """`client_order_id` is the only thing standing between a retried request and
    a doubled position."""
    first = captured_body(key_pem, ticker='T', side='up', contracts=1,
                          limit_price=0.5)['body']
    second = captured_body(key_pem, ticker='T', side='up', contracts=1,
                           limit_price=0.5)['body']
    assert first['client_order_id']
    assert first['client_order_id'] != second['client_order_id']
    # And an explicit one is honoured rather than overwritten.
    explicit = captured_body(key_pem, ticker='T', side='up', contracts=1,
                             limit_price=0.5,
                             client_order_id='BTC-USD|2026-08-23T00:30|9')['body']
    assert explicit['client_order_id'] == 'BTC-USD|2026-08-23T00:30|9'
