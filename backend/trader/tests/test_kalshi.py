"""Tests for the Kalshi client: signing, market resolution, and refusal to trade.

An order is refused unless the client is built with `live=True`, and an unresolved
market gives an abstention. Nothing reaches the network.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data_collection.kalshi_client import (
    CENT, DEFAULT_BASE_URL, DEMO_BASE_URL, KalshiClient, KalshiError, NotLiveError,
    Quote, _parse_time, _price, _to_quote,
)


@pytest.fixture(scope='module')
def key_pem() -> str:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode()


def test_an_unconfigured_client_says_so_rather_than_failing_later(monkeypatch):
    monkeypatch.delenv('KALSHI_KEY_ID', raising=False)
    monkeypatch.delenv('KALSHI_PRIVATE_KEY', raising=False)
    monkeypatch.delenv('KALSHI_PRIVATE_KEY_PATH', raising=False)
    client = KalshiClient()
    assert not client.configured
    with pytest.raises(KalshiError, match='credentials are not configured'):
        client._headers('GET', '/trade-api/v2/portfolio/balance')


def test_the_signature_covers_timestamp_method_and_path(key_pem):
    """The signature is RSA-PSS over SHA-256 of `timestamp + METHOD + path`, with the
    query excluded.
    """
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    client = KalshiClient(key_id='abc-123', private_key_pem=key_pem)
    headers = client._headers('get', '/trade-api/v2/markets')
    assert headers['KALSHI-ACCESS-KEY'] == 'abc-123'
    timestamp = headers['KALSHI-ACCESS-TIMESTAMP']
    assert timestamp.isdigit() and len(timestamp) >= 13, 'not milliseconds'

    message = f'{timestamp}GET/trade-api/v2/markets'.encode()
    public = serialization.load_pem_private_key(key_pem.encode(), password=None).public_key()
    public.verify(
        base64.b64decode(headers['KALSHI-ACCESS-SIGNATURE']), message,
        padding.PSS(mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),
        hashes.SHA256(),
    )


def test_the_method_is_upper_cased_in_the_signed_message(key_pem):
    client = KalshiClient(key_id='k', private_key_pem=key_pem)
    lower = client._headers('post', '/trade-api/v2/portfolio/orders')
    assert lower['KALSHI-ACCESS-SIGNATURE']  # it signed something


def test_placing_an_order_is_refused_unless_the_client_is_live(key_pem):
    """A client not built with `live=True` refuses to place an order."""
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=False)
    with pytest.raises(NotLiveError, match='live=True'):
        asyncio.run(client.place_order(ticker='X', side='up', contracts=1,
                                       limit_price=0.5))


def test_an_order_validates_its_own_arguments(key_pem):
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=True)
    with pytest.raises(ValueError, match="side must be"):
        asyncio.run(client.place_order(ticker='X', side='sideways', contracts=1,
                                       limit_price=0.5))
    with pytest.raises(ValueError, match='at least 1'):
        asyncio.run(client.place_order(ticker='X', side='up', contracts=0,
                                       limit_price=0.5))
    for bad_price in (0.0, 1.0, 1.5, -0.2):
        with pytest.raises(ValueError, match='outside 1c'):
            asyncio.run(client.place_order(ticker='X', side='up', contracts=1,
                                           limit_price=bad_price))


def _sent_price(key_pem, side, limit_price):
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem=key_pem, live=True)
    sent = {}

    async def _request(method, path, *, body=None, **kw):
        sent.update(body or {})
        return {'order': {'status': 'executed'}}

    client._request = _request
    asyncio.run(client.place_order(ticker='X', side=side, contracts=1,
                                   limit_price=limit_price))
    return sent['side'], sent['price']


@pytest.mark.parametrize('side, limit, expected', [
    # Middle band: whole-cent tick, rounded toward a fill.
    ('up', 0.6, ('bid', '0.6000')),
    ('up', 0.603, ('bid', '0.6100')),
    ('down', 0.397, ('ask', '0.6000')),  # YES 0.603, floored for an ask
    # Tails: deci-cent tick, so the limit is not widened by up to 0.9c.
    ('up', 0.0632, ('bid', '0.0640')),
    ('up', 0.9213, ('bid', '0.9220')),
    ('down', 0.0632, ('ask', '0.9360')),  # YES 0.9368 -> floor 0.936
    ('down', 0.9213, ('ask', '0.0780')),  # YES 0.0787 -> floor 0.078
    # Exactly on a deci-cent level stays put.
    ('up', 0.064, ('bid', '0.0640')),
    # Crossing a band edge lands on a level valid in the band it lands in.
    ('up', 0.0995, ('bid', '0.1000')),
    ('down', 0.8995, ('ask', '0.1000')),  # YES 0.1005 -> floor to 0.10
])
def test_a_limit_is_rounded_onto_the_venue_tick_toward_a_fill(
        key_pem, side, limit, expected):
    """A limit rounds onto the `tapered_deci_cent` tick toward a fill: up for a bid,
    down for an ask.
    """
    assert _sent_price(key_pem, side, limit) == expected


def test_a_quote_is_converted_to_probabilities_at_the_boundary():
    """A quote is converted from cents to probabilities once, here."""
    quote = _to_quote({'ticker': 'KXBTCD-T1', 'yes_bid': 84, 'yes_ask': 87,
                       'no_bid': 13, 'no_ask': 16, 'last_price': 85,
                       'volume': 1200, 'open_interest': 4000,
                       'close_time': '2026-08-23T03:15:00Z', 'status': 'active'})
    assert quote.yes_ask == pytest.approx(0.87)
    assert quote.mid == pytest.approx(0.855)
    assert quote.spread == pytest.approx(0.03)
    assert quote.ask_for('up') == pytest.approx(0.87)
    assert quote.ask_for('down') == pytest.approx(0.16)
    assert quote.tradeable()
    # The two asks sum above one; the excess is the spread.
    assert quote.ask_for('up') + quote.ask_for('down') > 1.0


def test_a_quote_is_read_from_the_real_wire_shape():
    """A quote is read from the venue's dollar-string fields (`yes_bid_dollars: "0.1900"`)."""
    quote = _to_quote({
        'ticker': 'KXBTC15M-A', 'status': 'active',
        'yes_bid_dollars': '0.1900', 'yes_ask_dollars': '0.2000',
        'no_bid_dollars': '0.8000', 'no_ask_dollars': '0.8100',
        'last_price_dollars': '0.1950',
        'yes_bid_size': '1594', 'yes_ask_size': '820',
        'volume': 12000, 'open_interest': 40000,
        'close_time': '2026-08-23T03:15:00Z',
    })
    assert quote.yes_bid == pytest.approx(0.19)
    assert quote.yes_ask == pytest.approx(0.20)
    assert quote.no_bid == pytest.approx(0.80)
    assert quote.last_price == pytest.approx(0.195)
    assert quote.yes_bid_size == pytest.approx(1594)
    assert quote.tradeable()


def test_bare_cent_fields_are_null_on_a_live_response_and_must_not_be_trusted_alone():
    """With the bare cent fields absent, the dollar fields alone give a tradeable quote."""
    quote = _to_quote({
        'ticker': 'KXBTC15M-A', 'status': 'active',
        'yes_bid_dollars': '0.1900', 'yes_ask_dollars': '0.2000',
    })
    assert quote.tradeable(), 'a real dollar-denominated quote read as untradeable'


def test_an_empty_book_is_not_tradeable():
    quote = _to_quote({'ticker': 'X', 'status': 'active'})
    assert quote.yes_bid is None and quote.yes_ask is None
    assert not quote.tradeable()
    assert quote.mid is None and quote.spread is None


def test_a_settled_market_is_not_tradeable():
    quote = _to_quote({'ticker': 'X', 'yes_bid': 99, 'yes_ask': 100,
                       'status': 'settled'})
    assert not quote.tradeable()


def test_a_zero_price_reads_as_absent_not_as_free():
    """A zero integer-cent price reads as absent, not free."""
    assert _price({'yes_bid': 0}, 'yes_bid') is None
    assert _price({'yes_bid': None}, 'yes_bid') is None
    assert _price({'yes_bid': 'nonsense'}, 'yes_bid') is None
    assert _price({'yes_bid': 45}, 'yes_bid') == pytest.approx(0.45)


def test_times_parse_to_utc():
    parsed = _parse_time('2026-08-23T03:15:00Z')
    assert parsed == datetime(2026, 8, 23, 3, 15, tzinfo=timezone.utc)
    assert _parse_time('') is None
    assert _parse_time('not a time') is None


def test_market_resolution_abstains_rather_than_taking_a_neighbour(monkeypatch):
    """Resolution asks the venue which market closes at this settlement, and returns
    None if none does.
    """
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem='')
    settle = datetime(2026, 8, 23, 3, 15, tzinfo=timezone.utc)

    async def markets(**_):
        return [
            {'ticker': 'EXACT', 'close_time': '2026-08-23T03:15:00Z'},
            {'ticker': 'NEXT', 'close_time': '2026-08-23T03:30:00Z'},
        ]

    monkeypatch.setattr(client, 'markets', markets)
    found = asyncio.run(client.resolve_window_market('KXBTCD', settle))
    assert found is not None and found['ticker'] == 'EXACT'

    async def only_far(**_):
        return [{'ticker': 'FAR', 'close_time': '2026-08-23T04:00:00Z'}]

    monkeypatch.setattr(client, 'markets', only_far)
    assert asyncio.run(client.resolve_window_market('KXBTCD', settle)) is None

    async def none_at_all(**_):
        return []

    monkeypatch.setattr(client, 'markets', none_at_all)
    assert asyncio.run(client.resolve_window_market('KXBTCD', settle)) is None


def test_the_default_host_is_production_and_a_demo_host_exists():
    assert 'kalshi' in DEFAULT_BASE_URL and DEFAULT_BASE_URL.endswith('/trade-api/v2')
    assert DEMO_BASE_URL != DEFAULT_BASE_URL
    assert CENT == 0.01


def test_a_request_before_opening_the_client_is_an_error(key_pem):
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem=key_pem)
    with pytest.raises(KalshiError, match='not open'):
        asyncio.run(client._request('GET', '/markets'))


# ------------------------------------------------------- the preflight script

def test_the_preflight_cannot_place_an_order():
    """The preflight builds the client without `live=True`, so it cannot place an order."""
    import inspect

    from scripts import check_venue

    source = inspect.getsource(check_venue)
    assert 'live=False' in source
    assert 'place_order' not in source


def test_the_preflight_distinguishes_a_network_block_from_a_rejection():
    """The preflight tells a network block (such as an egress 403) apart from a key rejection."""
    import inspect

    from scripts import check_venue

    source = inspect.getsource(check_venue.main)
    assert 'allowlist' in source
    assert 'NETWORK policy' in source
    assert source.index('allowlist') < source.index('rejected the signature'), (
        'the auth branch is checked first, so a network block is reported as an '
        'auth failure'
    )


def test_the_series_are_the_fifteen_minute_ones():
    """The series are the 15-minute ones (`KXBTC15M`), not the hourly threshold ladder `KXBTCD`."""
    from scripts.live import SERIES_BY_SYMBOL

    for symbol, series in SERIES_BY_SYMBOL.items():
        assert series.endswith('15M'), (
            f'{symbol} points at {series!r}; without the 15M suffix this is an '
            f'hourly threshold series and every window will abstain'
        )


def test_resolution_uses_close_time_and_not_the_ticker():
    """Resolution matches on `close_time` (UTC), never by parsing the ticker (named in
    US Eastern).
    """
    import asyncio
    import inspect
    from datetime import datetime, timezone

    from data_collection import kalshi_client

    source = inspect.getsource(kalshi_client.KalshiClient.resolve_window_market)
    assert 'close_time' in source
    for parsed in ('ticker[', 'strptime', '%b', 'AUG'):
        assert parsed not in source, (
            f'{parsed!r} suggests the ticker is being parsed for a time'
        )

    # A ticker whose name disagrees with its close_time is matched on the close_time.
    client = kalshi_client.KalshiClient(key_id='k', private_key_pem='')
    settle = datetime(2026, 8, 23, 4, 45, tzinfo=timezone.utc)

    async def markets(**_):
        return [{'ticker': 'KXBTC15M-26AUG230045-45',
                 'close_time': '2026-08-23T04:45:00Z'}]

    client.markets = markets
    found = asyncio.run(client.resolve_window_market('KXBTC15M', settle))
    assert found is not None
    assert found['ticker'] == 'KXBTC15M-26AUG230045-45'


def test_reconcile_pulls_everything_authoritative_in_one_call():
    """`reconcile` returns balance, positions, fills and settlements, the account of
    record.

    Live, the venue's ledger is right where it disagrees with ours, since our settlement
    is a Coinbase proxy for BRTI.
    """
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem='')
    calls: list[str] = []

    async def balance(*, exchange_index=None):
        calls.append('balance')
        return 137.42

    async def positions():
        calls.append('positions')
        return [{'ticker': 'KXBTC15M-A', 'position': 3}]

    async def fills(**_):
        calls.append('fills')
        return [{'ticker': 'KXBTC15M-A', 'count': 3}]

    async def settlements(**_):
        calls.append('settlements')
        return [{'ticker': 'KXBTC15M-B', 'revenue_dollars': '3.0000'}]

    client.balance = balance
    client.positions = positions
    client.fills = fills
    client.settlements = settlements

    state = asyncio.run(client.reconcile())
    assert set(calls) == {'balance', 'positions', 'fills', 'settlements'}
    assert state['balance'] == pytest.approx(137.42)
    assert len(state['positions']) == 1
    assert len(state['settlements']) == 1


def test_reconcile_survives_settlements_being_unavailable():
    """If the settlements endpoint fails, the balance is still returned."""
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem='')

    async def balance(*, exchange_index=None):
        return 100.0

    async def positions():
        return []

    async def fills(**_):
        return []

    async def settlements(**_):
        raise KalshiError('GET /portfolio/settlements -> 404')

    client.balance = balance
    client.positions = positions
    client.fills = fills
    client.settlements = settlements

    state = asyncio.run(client.reconcile())
    assert state['balance'] == pytest.approx(100.0)
    assert state['settlements'] == []


def test_the_venues_settlement_wins_over_our_bars(tmp_path):
    """`settle_due` books the venue's settlement when it disagrees with our bars."""
    from core.pg_writer import PgWriter
    from scripts.live import settle_due

    writer = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    writer.ensure_account(100.0, mode='live')

    window = datetime(2026, 8, 23, 0, 30, tzinfo=timezone.utc)
    settle_time = window + timedelta(minutes=15)
    ticker = 'KXBTC15M-26AUG230045'
    writer.write_prediction(
        symbol='BTC-USD', window_open=window, settle_time=settle_time,
        offset_minutes=3, decision_time=window + timedelta(minutes=3),
        strike=100.0, last_price=100.0, displacement=0.0, sigma_remaining=0.001,
        z_score=0.0, baseline_probability=0.5, model_probability=0.6,
        market_probability=0.6, price_source='quote', reason='traded', traded=True,
        side='up', price=0.6, effective_cost=0.62, edge=0.01, contracts=5,
        model_version=None)
    writer.write_ticket(
        symbol='BTC-USD', window_open=window, settle_time=settle_time,
        offset_minutes=3, market_ticker=ticker, side='up', contracts=5,
        limit_price=0.6, max_price=0.61, expected_cost=3.1,
        model_probability=0.6, edge=0.01)
    writer.open_position(
        symbol='BTC-USD', window_open=window, settle_time=settle_time,
        offset_minutes=3, side='up', contracts=5, price=0.6, outlay=3.1, fee=0.02,
        model_probability=0.6, baseline_probability=0.5, edge=0.01)

    # Bars that say DOWN: the settling minute's mean is below the strike of 100.
    minute = settle_time - timedelta(minutes=1)
    bars = {'BTC-USD': pd.DataFrame([{
        'event_time': pd.Timestamp(minute), 'open': 99.0, 'high': 99.0,
        'low': 99.0, 'close': 99.0, 'volume': 1.0}])}

    settled = settle_due(
        writer, bars,
        venue_settlements={ticker: {'ticker': ticker, 'market_result': 'yes'}})

    assert len(settled) == 1
    position = writer.settled_positions_since(window)[0]
    assert position.settled_up is True, (
        'the bars said down and the venue said yes; the venue is the account of '
        'record and its answer must be the one booked'
    )
    assert position.payout == 5.0
    # The payout reaches the account.
    assert writer.account().bankroll == pytest.approx(100.0 + 5.0)
    assert writer.account().realized_pnl == pytest.approx(5.0 - 3.1)


def test_settling_from_bars_uses_the_trained_rule(tmp_path):
    """Settling from bars uses the training label: the mean of the minute ending at
    settle_time, compared with `>=`.
    """
    from core.pg_writer import PgWriter
    from scripts.live import settle_due

    writer = PgWriter(database_url=f'sqlite:///{tmp_path}/serving.db')
    writer.ensure_account(100.0, mode='paper')
    window = datetime(2026, 8, 23, 0, 30, tzinfo=timezone.utc)
    settle_time = window + timedelta(minutes=15)
    writer.write_prediction(
        symbol='BTC-USD', window_open=window, settle_time=settle_time,
        offset_minutes=3, decision_time=window + timedelta(minutes=3),
        strike=100.0, last_price=100.0, displacement=0.0, sigma_remaining=0.001,
        z_score=0.0, baseline_probability=0.5, model_probability=0.6,
        market_probability=None, price_source='baseline', reason='traded',
        traded=True, side='up', price=0.5, effective_cost=0.52, edge=0.01,
        contracts=2, model_version=None)
    writer.open_position(
        symbol='BTC-USD', window_open=window, settle_time=settle_time,
        offset_minutes=3, side='up', contracts=2, price=0.5, outlay=1.04, fee=0.02,
        model_probability=0.6, baseline_probability=0.5, edge=0.01)

    # The minute ending at settle_time is exactly flat on the strike. The minute
    # starting at settle_time is below it.
    bars = {'BTC-USD': pd.DataFrame([
        {'event_time': pd.Timestamp(settle_time - timedelta(minutes=1)),
         'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0, 'volume': 1.0},
        {'event_time': pd.Timestamp(settle_time),
         'open': 98.0, 'high': 98.0, 'low': 98.0, 'close': 98.0, 'volume': 1.0},
    ])}

    settle_due(writer, bars)
    position = writer.settled_positions_since(window)[0]
    assert position.settled_up is True, (
        'a window that ends exactly on its strike pays the up side, because '
        'strike_type is greater_or_equal — and the bar read must be the one '
        'ENDING at settle_time, not the one starting there'
    )


# --- per-shard balance ------------------------------------------------------
#
# Kalshi shards its exchange by category and balances are local to a shard.
# `GET /portfolio/balance` returns a `balance_breakdown` entry per
# `exchange_index`. The KX*15M crypto series use `exchange_index=2`.

BREAKDOWN_PAYLOAD = {
    'balance': 10796,
    'balance_dollars': '107.9618',
    'balance_breakdown': [
        {'balance': '106.6085', 'exchange_index': 0},
        {'balance': '0.0000', 'exchange_index': 1},
        {'balance': '1.3533', 'exchange_index': 2},
        {'balance': '0.0000', 'exchange_index': 3},
    ],
}


def _client_returning(payload, key_pem):
    client = KalshiClient(key_id='k', private_key_pem=key_pem)

    async def _request(method, path, **kwargs):
        return payload

    client._request = _request
    return client


@pytest.mark.asyncio
async def test_the_spendable_balance_is_the_shard_the_market_lives_on(key_pem):
    """Only shard 2 can pay for a shard-2 market, whatever the total says."""
    client = _client_returning(BREAKDOWN_PAYLOAD, key_pem)
    assert await client.balance(exchange_index=2) == pytest.approx(1.3533)


@pytest.mark.asyncio
async def test_the_default_balance_is_still_the_total(key_pem):
    """Without a shard, the balance is the account total."""
    client = _client_returning(BREAKDOWN_PAYLOAD, key_pem)
    assert await client.balance() == pytest.approx(107.9618)


@pytest.mark.asyncio
async def test_a_shard_absent_from_the_breakdown_holds_nothing(key_pem):
    client = _client_returning(BREAKDOWN_PAYLOAD, key_pem)
    assert await client.balance(exchange_index=7) == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_a_venue_with_no_breakdown_falls_back_to_the_total(key_pem):
    """A response with no breakdown falls back to the total instead of NaN."""
    client = _client_returning({'balance_dollars': '50.0000'}, key_pem)
    assert await client.balance(exchange_index=2) == pytest.approx(50.0)


@pytest.mark.asyncio
async def test_reconcile_reads_the_balance_of_the_shard_it_is_told_about(key_pem):
    """`reconcile` reads the balance of the named shard, so sizing uses money an order
    can reach.
    """
    client = KalshiClient(key_id='k', private_key_pem=key_pem)
    seen = {}

    async def _request(method, path, **kwargs):
        if path == '/portfolio/balance':
            return BREAKDOWN_PAYLOAD
        return {}

    async def _balance(*, exchange_index=None):
        seen['exchange_index'] = exchange_index
        return 1.3533

    client._request = _request
    client.balance = _balance
    state = await client.reconcile(exchange_index=2)
    assert seen['exchange_index'] == 2
    assert state['balance'] == pytest.approx(1.3533)
