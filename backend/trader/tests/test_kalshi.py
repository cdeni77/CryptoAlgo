"""The venue client: signing, market resolution, and the refusal to trade.

Nothing here reaches the network. What is tested is the shape of what would be
sent and — more importantly — the two behaviours that make an unattended script
safe: an order is refused unless the client was constructed for it, and a market
that cannot be resolved produces an abstention rather than a neighbouring
contract.
"""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from data_collection.kalshi_client import (
    CENT, DEFAULT_BASE_URL, DEMO_BASE_URL, KalshiClient, KalshiError, NotLiveError,
    Quote, _cents, _parse_time, _to_quote,
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
    """RSA-PSS over SHA-256, not an HMAC secret.

    The signed string is `timestamp + METHOD + path`, and the path excludes the
    query — so signing the final URL instead would authenticate a different
    message than the one sent.
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
    """The failure being designed against is a script meant to observe that trades.

    Which cannot be undone, unlike a missed trade.
    """
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


def test_a_quote_is_converted_to_probabilities_at_the_boundary():
    """Everything above this module reasons on the probability scale.

    A stray factor of 100 between cents and dollars is the classic bug in a
    binary system, so the conversion happens once, here.
    """
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
    # The two asks sum above one: that difference is the spread being crossed.
    assert quote.ask_for('up') + quote.ask_for('down') > 1.0


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
    assert _cents(0) is None
    assert _cents(None) is None
    assert _cents('nonsense') is None
    assert _cents(45) == pytest.approx(0.45)


def test_times_parse_to_utc():
    parsed = _parse_time('2026-08-23T03:15:00Z')
    assert parsed == datetime(2026, 8, 23, 3, 15, tzinfo=timezone.utc)
    assert _parse_time('') is None
    assert _parse_time('not a time') is None


def test_market_resolution_abstains_rather_than_taking_a_neighbour(monkeypatch):
    """A ticker built from a pattern fails by finding the *wrong* contract.

    So resolution asks the venue which market closes when this window settles,
    and returns nothing when none does.
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
    """It constructs the client without live=True, so ordering is impossible.

    A read-only check that could trade would be worse than no check at all.
    """
    import inspect

    from scripts import check_venue

    source = inspect.getsource(check_venue)
    assert 'live=False' in source
    assert 'place_order' not in source


def test_the_preflight_distinguishes_a_network_block_from_a_rejection():
    """Both surface as an HTTP error and the fix is completely different.

    Measured: this container's own egress policy returns 403 "Host not in
    allowlist", which under the previous message read as "the key id and the
    private key are not a pair" — sending the reader to rotate a credential that
    was never even seen.
    """
    import inspect

    from scripts import check_venue

    source = inspect.getsource(check_venue.main)
    assert 'allowlist' in source
    assert 'NETWORK policy' in source
    assert source.index('allowlist') < source.index('rejected the signature'), (
        'the auth branch is checked first, so a network block is reported as an '
        'auth failure'
    )


def test_the_preflight_is_launchable_from_the_dashboard():
    """It is read-only, so the job allow-list should include it."""
    import sys
    from pathlib import Path

    api_root = Path(__file__).resolve().parents[2] / 'api'
    if not api_root.exists():
        pytest.skip('API package not present')
    sys.path.insert(0, str(api_root))
    try:
        for name in list(sys.modules):
            if name.startswith(('endpoints', 'security')):
                del sys.modules[name]
        from endpoints.jobs import JOBS

        assert 'scripts.check_venue' in JOBS
        # And the live loop is deliberately NOT launchable from a web request:
        # two copies racing over one account.
        assert 'scripts.live' not in JOBS
    finally:
        sys.path.remove(str(api_root))
        for name in list(sys.modules):
            if name.startswith(('endpoints', 'security')):
                del sys.modules[name]


def test_the_series_are_the_fifteen_minute_ones():
    """`KXBTCD` is the hourly series, and pointing at it abstains on every window.

    Measured against the live venue: `KXBTCD` returns 200 open markets, all
    closing on the hour, with an explicit strike in the ticker
    (`KXBTCD-26AUG2317-T86749.99`) — a threshold ladder, not an up/down market.
    The 15-minute series is `KXBTC15M`, whose tickers are series + date + HHMM
    with no strike, because the strike is the price at the window's open.

    Worth a test rather than a comment: the failure was silent in the sense that
    everything reported healthy — credentials fine, series present, hundreds of
    markets — and only the resolution step said no.
    """
    from scripts.live import SERIES_BY_SYMBOL

    for symbol, series in SERIES_BY_SYMBOL.items():
        assert series.endswith('15M'), (
            f'{symbol} points at {series!r}; without the 15M suffix this is an '
            f'hourly threshold series and every window will abstain'
        )


def test_resolution_uses_close_time_and_not_the_ticker():
    """The ticker is named in Eastern; `close_time` is UTC.

    `KXBTC15M-26AUG230045` settles at 04:45Z, because 00:45 EDT is 04:45 UTC.
    Parsing the ticker for its settlement would mean hardcoding the venue's
    timezone and its daylight-saving rule, and being wrong twice a year — so this
    asserts the resolver never looks at the ticker string.
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

    # And behaviourally: a ticker whose name disagrees with its close_time must
    # be matched on the close_time.
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
    """Balance, positions, fills and settlements — the account of record.

    In paper mode the bankroll is arithmetic and settlement comes from our own
    bars. Live, both are estimates of someone else's ledger: we approximate sixty
    seconds of CF Benchmarks BRTI with a one-minute OHLC mean of Coinbase bars, so
    a position settled from bars can disagree with what was actually paid. Where
    they disagree the venue is right.
    """
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem='')
    calls: list[str] = []

    async def balance():
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
    """An endpoint that moves must not take the balance check down with it."""
    import asyncio

    client = KalshiClient(key_id='k', private_key_pem='')

    async def balance():
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
    """Behaviour, not source text.

    This pair of tests used to read `inspect.getsource` and assert that certain
    strings appeared in it — `'reconcile_with_venue' in source`,
    `'update_account(bankroll=venue_balance)' in source`. Both passed for the
    whole time the settlement reconciliation they describe **did not exist**:
    `reconcile_with_venue` built a dict of the venue's settlements, logged its
    length, and dropped it, and `settle_position` was called from exactly one
    place — `settle_due`, off our own bars, with the wrong rule. A test that
    checks a function contains a substring is a test of the comment.

    So: construct a position, hand `settle_due` a venue settlement that disagrees
    with the bars, and assert the venue's answer is the one booked.
    """
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
    # And the payout reached the account, which it never used to.
    assert writer.account().bankroll == pytest.approx(100.0 + 5.0)
    assert writer.account().realized_pnl == pytest.approx(5.0 - 3.1)


def test_settling_from_bars_uses_the_trained_rule(tmp_path):
    """The mean over the minute ENDING at settle_time, compared with `>=`.

    `settle_due` read the `open` of the bar *starting* at `settle_time` and used a
    strict `>`. Three deviations from the label the model is trained against, and
    measured on real bars they disagreed on 3.4-8.2% of windows. A dead-flat
    window is the cheapest case that separates them: `>=` pays the up side and `>`
    does not.
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
    # *starting* at settle_time is below it — which is what the old code read.
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
