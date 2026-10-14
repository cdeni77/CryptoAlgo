"""Tests for the UP/DOWN mapping onto the Kalshi V2 order body.

V2 quotes a single YES book: `bid` buys YES; `ask` sells YES, which is buying NO
at (1 - price). So DOWN converts the price, not only the label: paying 0.31 for
NO is an ask at 0.69. Rounding also inverts: a bid rounds up and an ask rounds
down, so the order can still fill. The tests intercept `_request`, so nothing
reaches the network.
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


def test_an_up_decision_bids_on_the_yes_book(key_pem):
    """UP is `side: bid` at the YES price we pay, sent to the V2 endpoint."""
    seen = captured_body(key_pem, ticker='KXBTC15M-X', side='up',
                         contracts=5, limit_price=0.87)
    assert seen['method'] == 'POST'
    assert seen['path'] == '/portfolio/events/orders', (
        'V1 /portfolio/orders returns 410 deprecated_v1_order_endpoint'
    )
    body = seen['body']
    assert body['side'] == 'bid'
    assert float(body['price']) == pytest.approx(0.87)
    # V2 has no action or type field, and no separate yes/no price.
    for gone in ('action', 'type', 'yes_price', 'no_price'):
        assert gone not in body, f'{gone} is a V1 field'


def test_a_down_decision_asks_on_the_yes_book_at_one_minus_the_price(key_pem):
    """DOWN is an ask on the YES book at `1 - price`: paying 0.13 for NO is selling YES at
    0.87.
    """
    seen = captured_body(key_pem, ticker='KXBTC15M-X', side='down',
                         contracts=5, limit_price=0.13)
    body = seen['body']
    assert body['side'] == 'ask'
    assert float(body['price']) == pytest.approx(0.87), (
        f"paying 0.13 for NO must become selling YES at 0.87, got {body['price']}"
    )


def test_the_two_sides_are_complements_of_each_other(key_pem):
    """A half-applied flip shows here: both sides of the same economic trade give the same
    YES limit.
    """
    up = captured_body(key_pem, ticker='T', side='up', contracts=1,
                       limit_price=0.40)['body']
    down = captured_body(key_pem, ticker='T', side='down', contracts=1,
                         limit_price=0.60)['body']
    assert up['side'] == 'bid' and down['side'] == 'ask'
    assert float(up['price']) == pytest.approx(float(down['price'])), (
        'the same economic trade produced two different YES limits'
    )


@pytest.mark.parametrize('side,dollars,expected_cents', [
    # A bid rounds up: it fills against asks at or below it. An ask rounds down, for
    # the mirror reason.
    ('up', 0.8650, 87),
    ('up', 0.8600, 86),
    ('down', 0.1350, 86),  # 1 - 0.135 = 0.865 -> floor -> 86c
    ('down', 0.1400, 86),  # 1 - 0.14 = 0.860 -> 86c
])
def test_the_limit_rounds_so_the_order_can_still_fill(key_pem, side, dollars,
                                                     expected_cents):
    body = captured_body(key_pem, ticker='T', side=side, contracts=1,
                         limit_price=dollars)['body']
    assert round(float(body['price']) * 100) == expected_cents, (
        f"{side} at {dollars} sent {body['price']}"
    )


def test_count_and_price_are_fixed_point_strings(key_pem):
    """V2 needs count and price as decimal strings."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=7,
                         limit_price=0.50)['body']
    assert isinstance(body['count'], str) and float(body['count']) == 7.0
    assert isinstance(body['price'], str)


def test_self_trade_prevention_is_set_to_cross_the_spread(key_pem):
    """`self_trade_prevention_type` is `taker_at_cross`, which V2 requires."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50)['body']
    assert body['self_trade_prevention_type'] == 'taker_at_cross'


def test_the_default_time_in_force_takes_partial_fills(key_pem):
    """The default time in force is `immediate_or_cancel`.

    A resting order can fill after the forecast expires, and `fill_or_kill` returns
    nothing when the touch is thin. `immediate_or_cancel` takes a partial fill.
    """
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50)['body']
    assert body['time_in_force'] == 'immediate_or_cancel'


def test_every_order_carries_the_idempotency_key_it_was_given(key_pem):
    """The order carries the `client_order_id` it was given; it prevents duplicate real
    orders.
    """
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50, client_order_id='BTC-USD-202608230030')['body']
    assert body['client_order_id'] == 'BTC-USD-202608230030'


def test_a_limit_that_cannot_be_expressed_is_refused(key_pem):
    """A limit that implies a YES price outside 1c..99c raises."""
    with pytest.raises(ValueError, match='outside 1c'):
        captured_body(key_pem, ticker='T', side='down', contracts=1,
                      limit_price=0.0)
