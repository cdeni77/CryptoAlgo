"""`post_only` is a separate decision from resting, and it has a price.

Resting at our own touch is how the entry price is won. `post_only` is the
extra guarantee that an order the book has moved onto is REFUSED rather than
filled as a taker. They were one flag, so turning the guarantee off meant
giving up the resting price too.

Measured 2026-10-18..25 with both kinds running side by side on the same
account: resting fills realised -0.48pp per contract and crossing fills
-0.13pp net of the taker fee. The maker entry buys an 8pp better price and
pays 9.7pp of win rate for it, because an order only comes to you when the
market has moved against it. Over the same span post_only refused 23.7% of
orders, and the resting orders that never filled would have won 95.3% of the
time.

So "make when we can, take when we can" is a configuration the account must
be able to express. These tests pin the wire payload, because the venue's
behaviour turns on that field and nothing else reveals it.
"""

from __future__ import annotations

import asyncio

import pytest

from core.config import Config
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


def _payload(key_pem, **kwargs) -> dict:
    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=True)
    seen: dict = {}

    async def _request(method, path, *, body=None, params=None, **kw):
        seen.update(body or {})
        return {'order': {'order_id': 'x'}}

    client._request = _request
    asyncio.run(client.place_order(
        ticker='KXBTC15M-26OCT2820-T1', side='up', contracts=3,
        limit_price=0.61, **kwargs))
    return seen


def test_resting_still_sends_post_only_by_default(key_pem):
    """The default must not change: omitting the argument keeps the guarantee."""
    assert _payload(key_pem, rest=True).get('post_only') is True


def test_resting_without_post_only_lets_the_order_take(key_pem):
    """**The whole point.** Still resting, but a crossing book fills instead
    of refusing."""
    body = _payload(key_pem, rest=True, post_only=False)
    assert 'post_only' not in body, (
        'post_only was still sent, so the venue would refuse a crossing order '
        'and the configuration would do nothing')
    assert body.get('self_trade_prevention_type') == 'maker', (
        'the order must still REST -- dropping post_only must not quietly '
        'turn this into a taker order priced at the ask')


def test_a_taker_order_never_sends_post_only(key_pem):
    assert 'post_only' not in _payload(key_pem, rest=False)


def test_the_config_default_keeps_the_guarantee():
    assert Config().maker_post_only is True
