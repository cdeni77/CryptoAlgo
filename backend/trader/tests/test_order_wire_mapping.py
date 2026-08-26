"""The UP/DOWN mapping on the wire. HIGHEST SEVERITY GAP, now on the V2 API.

Measured on the V1 body: flipping the side and price fields to their opposite left
all 230 tests passing, while buying the exact opposite side of every trade with
real money. `core/decide.py`'s side *label* was covered; the boundary where that
label becomes a venue order body was not covered at all.

**V2 makes this sharper, not safer.** The first real live cycle returned
`410 deprecated_v1_order_endpoint`, and V2 is not a renamed path — it quotes a
SINGLE book from the YES side:

    bid  = buy YES
    ask  = sell YES, which is economically buying NO at (1 - price)

So the price must be converted, not just relabelled. `decide()` produces what we
would PAY for the chosen side, and paying 0.31 for NO is selling YES at 0.69.
Sending 0.31 as an `ask` offers to sell YES for thirty-one cents — a strictly
worse error than inverting the side, because it also looks plausible.

The rounding direction inverts too: a bid fills against asks at or below it, an
ask against bids at or above it. Round the wrong way under `fill_or_kill` and the
order cannot fill at all.

The tests intercept `_request`, so they never touch the network.
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
    """UP means buy YES, which is `side: bid` at the YES price we would pay."""
    seen = captured_body(key_pem, ticker='KXBTC15M-X', side='up',
                         contracts=5, limit_price=0.87)
    assert seen['method'] == 'POST'
    assert seen['path'] == '/portfolio/events/orders', (
        'V1 /portfolio/orders returns 410 deprecated_v1_order_endpoint'
    )
    body = seen['body']
    assert body['side'] == 'bid'
    assert float(body['price']) == pytest.approx(0.87)
    # V2 has no action/type, and no separate yes/no price.
    for gone in ('action', 'type', 'yes_price', 'no_price'):
        assert gone not in body, f'{gone} is a V1 field'


def test_a_down_decision_asks_on_the_yes_book_at_one_minus_the_price(key_pem):
    """DOWN means buy NO, expressed as selling YES at `1 - what we pay`.

    This is the assertion that matters most in the file. Paying 0.13 for NO is
    selling YES at 0.87; sending 0.13 would offer to sell YES for thirteen cents.
    """
    seen = captured_body(key_pem, ticker='KXBTC15M-X', side='down',
                         contracts=5, limit_price=0.13)
    body = seen['body']
    assert body['side'] == 'ask'
    assert float(body['price']) == pytest.approx(0.87), (
        f"paying 0.13 for NO must become selling YES at 0.87, got {body['price']}"
    )


def test_the_two_sides_are_complements_of_each_other(key_pem):
    """A half-applied flip shows up here even if each side looks right alone."""
    up = captured_body(key_pem, ticker='T', side='up', contracts=1,
                       limit_price=0.40)['body']
    down = captured_body(key_pem, ticker='T', side='down', contracts=1,
                         limit_price=0.60)['body']
    assert up['side'] == 'bid' and down['side'] == 'ask'
    # Paying 0.40 for YES and 0.60 for NO are the same economic price, so both
    # must land on the same YES limit.
    assert float(up['price']) == pytest.approx(float(down['price'])), (
        'the same economic trade produced two different YES limits'
    )


@pytest.mark.parametrize('side,dollars,expected_cents', [
    # A bid rounds UP: it fills against asks at or below it, so rounding down
    # could make it unfillable. An ask rounds DOWN, for the mirror reason.
    ('up', 0.8650, 87),
    ('up', 0.8600, 86),
    ('down', 0.1350, 86),   # 1 - 0.135 = 0.865 -> floor -> 86c
    ('down', 0.1400, 86),   # 1 - 0.14  = 0.860 -> 86c
])
def test_the_limit_rounds_so_the_order_can_still_fill(key_pem, side, dollars,
                                                     expected_cents):
    body = captured_body(key_pem, ticker='T', side=side, contracts=1,
                         limit_price=dollars)['body']
    assert round(float(body['price']) * 100) == expected_cents, (
        f"{side} at {dollars} sent {body['price']}"
    )


def test_count_and_price_are_fixed_point_strings(key_pem):
    """V2 rejects a bare integer count; both are decimal strings."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=7,
                         limit_price=0.50)['body']
    assert isinstance(body['count'], str) and float(body['count']) == 7.0
    assert isinstance(body['price'], str)


def test_self_trade_prevention_is_set_to_cross_the_spread(key_pem):
    """Required in V2. `maker` would rest, which is the opposite of a
    fill_or_kill on a wasting fifteen-minute market."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50)['body']
    assert body['self_trade_prevention_type'] == 'taker_at_cross'


def test_the_default_time_in_force_takes_partial_fills(key_pem):
    """Neither resting nor all-or-nothing.

    A resting order fills against a barrier probability that has expired, so
    `good_till_canceled` is wrong on a fifteen-minute market. But `fill_or_kill`
    is all-or-nothing: nine contracts wanted against five resting returns
    nothing, which is how `fill_or_kill_insufficient_resting_volume` kept
    appearing in the live log while the touch held single digits.
    `immediate_or_cancel` takes the five, and with a positive edge a partial fill
    strictly beats a kill.
    """
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50)['body']
    assert body['time_in_force'] == 'immediate_or_cancel'


def test_every_order_carries_the_idempotency_key_it_was_given(key_pem):
    """Deterministic per (symbol, window), and the only thing standing between a
    duplicated cycle and a duplicated real order."""
    body = captured_body(key_pem, ticker='T', side='up', contracts=1,
                         limit_price=0.50, client_order_id='BTC-USD-202608230030')['body']
    assert body['client_order_id'] == 'BTC-USD-202608230030'


def test_a_limit_that_cannot_be_expressed_is_refused(key_pem):
    """1c..99c. A DOWN order at a price implying 0c or 100c YES must raise
    rather than send something the venue will reject or misread."""
    with pytest.raises(ValueError, match='outside 1c'):
        captured_body(key_pem, ticker='T', side='down', contracts=1,
                      limit_price=0.0)
