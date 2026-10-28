"""A cancel must name the shard, or the venue cannot find a live order.

`order_id` alone does not identify an order: it does not carry the exchange
shard, and this account's crypto markets are on `exchange_index` 2. The V2
resource is `DELETE /portfolio/events/orders/{id}`, and it auto-routes only
when `market_ticker` comes with it. Without the ticker the lookup runs on the
wrong shard and returns 404 for an order that is alive.

Measured: order 01a17f47 was created 21:33:01 and the venue's own record has
it alive until 21:40:56. The sweep "cancelled" it at 21:36:01, got the 404 and
closed the ticket, leaving a live order nobody tracked for five more minutes.
A fill after that lands as a position we have no record of.

The bare `/portfolio/orders/{id}` is the legacy family. The venue answers 410
there now, so a cancel sent to it cannot succeed for any order.
"""

from __future__ import annotations

import asyncio

import pytest

from data_collection.kalshi_client import KalshiClient


@pytest.fixture
def key_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


ORDER = '01a17f47-e6c8-7bd0-8142-688729eb76d9'


def _cancel(key_pem, **kwargs) -> dict:
    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=True)
    seen: dict = {}

    async def _request(method, path, *, params=None, body=None, **kw):
        seen.update(method=method, path=path, params=params or {})
        return {}

    client._request = _request
    asyncio.run(client.cancel_order(ORDER, **kwargs))
    return seen


def test_cancel_uses_the_v2_order_resource(key_pem):
    seen = _cancel(key_pem, market_ticker='KXBTCD-26OCT28-B1')
    assert seen['method'] == 'DELETE'
    assert seen['path'] == f'/portfolio/events/orders/{ORDER}'


def test_cancel_sends_the_market_ticker_so_the_venue_can_route(key_pem):
    """Without it the venue cannot pick the shard and 404s a live order."""
    seen = _cancel(key_pem, market_ticker='KXBTCD-26OCT28-B1')
    assert seen['params'].get('market_ticker') == 'KXBTCD-26OCT28-B1'


def test_cancel_does_not_use_the_legacy_family(key_pem):
    """`/portfolio/orders/{id}` is retired; the venue answers 410 there."""
    seen = _cancel(key_pem, market_ticker='KXBTCD-26OCT28-B1')
    assert not seen['path'].startswith(f'/portfolio/orders/'), (
        'the legacy order family is deprecated and returns 410, so a cancel '
        'sent there can never succeed')


def test_an_absent_ticker_is_omitted_rather_than_sent_as_null(key_pem):
    """A null query value is not the same as no query value."""
    seen = _cancel(key_pem)
    assert 'market_ticker' not in seen['params']
