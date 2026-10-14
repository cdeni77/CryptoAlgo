"""The settlement-index proxy: parsers, the median rule, the recorder's sockets."""
from __future__ import annotations

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import pytest

from core.index_proxy import (
    EXCHANGES, HANDLERS, MID_COLUMNS, ExchangeState, IndexProxyCache, SideBook,
    proxy_row, subscribe_messages,
)

FIXTURES = json.loads((Path(__file__).parent / 'fixtures' / 'ws'
                       / 'index_exchanges.json').read_text())
T0 = 1_791_981_000.0
SYMBOLS = ('BTC-USD', 'ETH-USD', 'SOL-USD')


def _fed(exchange: str, at: float = T0) -> ExchangeState:
    state = ExchangeState(exchange, SYMBOLS)
    for i, message in enumerate(FIXTURES[exchange]):
        assert HANDLERS[exchange](message, state, at + 0.01 * i) is None
    return state


def _fresh(exchange, bid, ask, at=T0):
    state = ExchangeState(exchange, SYMBOLS)
    state.alive(at)
    state.set_bbo('BTC-USD', bid, ask, at)
    return state


# ---------------------------------------------------------------------------
# parsers, on the venue's own message shapes
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('exchange,bid,ask,last', [
    # The snapshot's best bid 62000 is deleted by the update; a new best ask arrives.
    ('coinbase', 61999.5, 62000.8, 62000.8),
    ('kraken', 62000.3, 62000.5, 62000.5),
    ('bitstamp', 61999.0, 62001.0, 62001.0),
    # The first l2_updates is the whole book; the second removes the best ask.
    ('gemini', 61999.0, 62004.0, 62003.0),
])
def test_each_parser_tracks_the_top_of_book_and_the_last_trade(exchange, bid, ask, last):
    state = _fed(exchange)
    quote = state.quotes['BTC-USD']
    assert (quote.bid, quote.ask) == (bid, ask)
    assert quote.mid == pytest.approx((bid + ask) / 2)
    assert quote.last_trade == last
    assert state.alive_at is not None
    # Nothing leaks onto a symbol the messages never named.
    assert state.quotes['ETH-USD'].mid is None


def test_every_exchange_has_a_parser_and_a_subscription():
    for exchange in EXCHANGES:
        assert exchange in HANDLERS
        assert subscribe_messages(exchange, SYMBOLS)


def test_kraken_asks_for_a_bbo_triggered_ticker():
    ticker = subscribe_messages('kraken', SYMBOLS)[0]['params']
    assert ticker['channel'] == 'ticker' and ticker['event_trigger'] == 'bbo'
    assert ticker['symbol'] == ['BTC/USD', 'ETH/USD', 'SOL/USD']


def test_coinbase_subscribes_to_ticker_and_heartbeats_only():
    """Coinbase uses `ticker` (the touch on every message), not `level2`, to avoid folding
    a full book.
    """
    channels = {m['channel'] for m in subscribe_messages('coinbase', SYMBOLS)}
    assert channels == {'ticker', 'heartbeats'}


def test_a_coinbase_ticker_for_another_product_is_ignored():
    state = ExchangeState('coinbase', ('BTC-USD',))
    message = dict(FIXTURES['coinbase'][1])
    message['events'] = [{'type': 'update', 'tickers': [
        dict(message['events'][0]['tickers'][0], product_id='DOGE-USD')]}]
    HANDLERS['coinbase'](message, state, T0)
    assert state.quotes['BTC-USD'].mid is None


def test_bitstamp_subscribes_to_the_snapshot_book_per_pair():
    names = {m['data']['channel'] for m in subscribe_messages('bitstamp', SYMBOLS)}
    assert {'order_book_btcusd', 'order_book_ethusd', 'order_book_solusd'} <= names


@pytest.mark.parametrize('exchange,message', [
    ('coinbase', {'type': 'error', 'message': 'failure to subscribe'}),
    ('kraken', {'method': 'subscribe', 'success': False, 'error': 'Currency pair not supported'}),
    ('bitstamp', {'event': 'bts:request_reconnect', 'channel': '', 'data': ''}),
    ('bitstamp', {'event': 'bts:error', 'channel': '', 'data': {'code': 4009}}),
    ('gemini', {'result': 'error', 'reason': 'InvalidJson'}),
])
def test_a_venue_error_asks_for_a_reconnect(exchange, message):
    assert HANDLERS[exchange](message, ExchangeState(exchange, SYMBOLS), T0)


def test_a_crossed_book_is_not_a_price():
    state = _fresh('kraken', 62001.0, 62000.0)
    assert state.mid('BTC-USD', T0, max_staleness=5) is None


def test_side_book_recomputes_only_when_the_best_leaves():
    bids = SideBook(bids=True)
    for price, size in ((10.0, 1.0), (12.0, 1.0), (11.0, 1.0)):
        bids.set(price, size)
    assert bids.best() == 12.0
    bids.set(11.0, 0.0)
    assert bids.best() == 12.0
    bids.set(12.0, 0.0)
    assert bids.best() == 10.0
    bids.set(12.0, 0.0)          # removing an absent level is harmless
    asks = SideBook(bids=False)
    asks.set(5.0, 1.0)
    asks.set(4.0, 1.0)
    asks.set(4.0, 0.0)
    assert asks.best() == 5.0
    asks.set(5.0, 0.0)
    assert asks.best() is None


def test_a_disconnect_forgets_every_quote_and_book():
    state = _fed('gemini')
    state.disconnect()
    assert state.quotes['BTC-USD'].mid is None
    assert state.mid('BTC-USD', T0, max_staleness=5) is None
    assert state.book('BTC-USD')[0].best() is None


# ---------------------------------------------------------------------------
# the median, staleness and the two-exchange rule
# ---------------------------------------------------------------------------

def test_the_proxy_is_the_median_of_fresh_mids():
    states = [_fed(e) for e in EXCHANGES]
    row = proxy_row(states, 'BTC-USD', T0 + 1.0, max_staleness=5)
    assert row['n_exchanges'] == 4
    mids = sorted([62000.15, 62000.4, 62000.0, 62001.5])
    assert row['proxy'] == pytest.approx((mids[1] + mids[2]) / 2)
    assert set(MID_COLUMNS) <= set(row)
    assert row['mid_kraken'] == pytest.approx(62000.4)


def test_an_odd_count_takes_the_middle_mid():
    states = [_fresh('kraken', 99, 101), _fresh('bitstamp', 199, 201),
              _fresh('gemini', 109, 111)]
    assert proxy_row(states, 'BTC-USD', T0, max_staleness=5)['proxy'] == 110.0


def test_one_fresh_exchange_is_not_a_proxy():
    states = [_fresh('kraken', 99, 101), _fresh('bitstamp', 99, 101, at=T0 - 30)]
    row = proxy_row(states, 'BTC-USD', T0, max_staleness=5, min_exchanges=2)
    assert row['n_exchanges'] == 1 and math.isnan(row['proxy'])
    assert math.isnan(row['mid_bitstamp']) and row['mid_kraken'] == 100.0
    assert math.isnan(row['mid_coinbase']), 'an exchange not connected is NaN'


def test_staleness_is_judged_on_the_connection_not_the_quote():
    """Staleness is judged on the connection: a quiet book is unchanged while its socket
    keeps talking.
    """
    state = _fresh('kraken', 99, 101, at=T0)
    state.alive(T0 + 20)                       # heartbeats continue
    assert state.mid('BTC-USD', T0 + 21, max_staleness=5) == 100.0
    assert state.mid('BTC-USD', T0 + 30, max_staleness=5) is None   # socket quiet


def test_a_book_stuck_behind_a_live_heartbeat_ages_out():
    state = _fresh('kraken', 99, 101, at=T0)
    state.alive(T0 + 120)
    assert state.mid('BTC-USD', T0 + 121, max_staleness=5) is None


# ---------------------------------------------------------------------------
# the cache
# ---------------------------------------------------------------------------

def test_the_cache_keeps_a_bounded_history_in_the_stored_shape():
    cache = IndexProxyCache(history_seconds=3)
    for i in range(5):
        cache.record({'t': T0 + i, 'symbol': 'BTC-USD', 'proxy': 100.0 + i,
                      'n_exchanges': 3})
    snap = cache.snapshot('BTC-USD')
    assert list(snap.columns) == ['symbol', 'event_time', 'proxy', 'n_exchanges']
    assert list(snap['proxy']) == [102.0, 103.0, 104.0]
    assert cache.latest('BTC-USD')['proxy'] == 104.0
    assert cache.latest('ETH-USD') is None
    assert cache.snapshot('ETH-USD').empty


# ---------------------------------------------------------------------------
# the recorder: sockets, timeouts, reconnects, rows
# ---------------------------------------------------------------------------

from scripts import record_index  # noqa: E402


class FakeWS:
    """Replays messages, then either closes or goes silent forever."""

    def __init__(self, messages=(), *, then='close'):
        self.queue = [json.dumps(m) for m in messages]
        self.then = then
        self.sent = []
        self.closed = False

    async def receive(self):
        if self.queue:
            return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=self.queue.pop(0))
        if self.then == 'close':
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None)
        await asyncio.Event().wait()            # silent forever

    def exception(self):
        return None

    async def send_json(self, message):
        self.sent.append(message)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False


def test_a_silent_socket_is_noticed_on_a_timeout():
    async def drive():
        state = ExchangeState('kraken', SYMBOLS)
        ws = FakeWS(FIXTURES['kraken'], then='silent')
        reason = await asyncio.wait_for(record_index.consume(
            ws, 'kraken', state, idle_timeout=0.05), timeout=5.0)
        return reason, state
    reason, state = asyncio.run(drive())
    assert reason.startswith('silent')
    assert state.alive_at is None, 'the books of a dead socket are forgotten'


def test_a_venue_error_ends_the_connection_with_its_reason():
    async def drive():
        ws = FakeWS([{'event': 'bts:request_reconnect'}], then='silent')
        return await record_index.consume(ws, 'bitstamp',
                                          ExchangeState('bitstamp', SYMBOLS),
                                          idle_timeout=5.0)
    assert 'reconnect' in asyncio.run(drive())


def test_a_coinbase_sequence_gap_keeps_the_connection():
    """A Coinbase ticker gap keeps the connection, since each message carries the whole
    touch.
    """
    messages = FIXTURES['coinbase'][:2] + [dict(FIXTURES['coinbase'][2], sequence_num=9)]

    async def drive():
        state = ExchangeState('coinbase', SYMBOLS)
        return await record_index.consume(
            FakeWS(messages, then='silent'), 'coinbase', state, idle_timeout=0.05)
    assert asyncio.run(drive()).startswith('silent')


def test_a_large_message_is_decoded_off_the_loop_and_still_applied():
    async def drive():
        state = ExchangeState('gemini', SYMBOLS)
        ws = FakeWS(FIXTURES['gemini'][:1], then='silent')
        seen = {}
        original = ExchangeState.disconnect

        def spy(self):
            seen['bid'] = self.quotes['BTC-USD'].bid
            original(self)
        ExchangeState.disconnect = spy
        try:
            await record_index.consume(ws, 'gemini', state, idle_timeout=0.05,
                                       large_bytes=10)
        finally:
            ExchangeState.disconnect = original
        return seen
    assert asyncio.run(drive())['bid'] == 61998.0


class FakeSession:
    def __init__(self, sockets):
        self.sockets = list(sockets)
        self.connects = 0

    def ws_connect(self, url, **kwargs):
        self.connects += 1
        sockets = self.sockets

        async def connect():
            if not sockets:
                await asyncio.Event().wait()
            ws = sockets.pop(0)
            if isinstance(ws, Exception):
                raise ws
            return ws
        return connect()


def _args(**over):
    args = record_index.build_parser().parse_args([])
    args.backoff_start = 0.001
    args.backoff_max = 0.002
    args.idle_timeout = 0.05
    args.connect_timeout = 0.2
    for key, value in over.items():
        setattr(args, key, value)
    return args


def test_the_exchange_task_reconnects_after_a_close_an_error_and_a_hang():
    async def drive():
        first = FakeWS(FIXTURES['kraken'], then='close')
        second = FakeWS(FIXTURES['kraken'], then='silent')
        session = FakeSession([first, OSError('connection refused'), second])
        task = asyncio.create_task(record_index.run_exchange(
            session, 'kraken', ExchangeState('kraken', SYMBOLS), _args()))
        for _ in range(200):
            await asyncio.sleep(0.01)
            if session.connects >= 4:
                break
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return session, first
    session, first = asyncio.run(drive())
    # close -> refused -> silent -> a connect that hangs and times out
    assert session.connects >= 4
    assert first.sent == subscribe_messages('kraken', SYMBOLS)


def test_sample_stamps_the_second_and_records_every_symbol():
    states = [_fresh('kraken', 99, 101), _fresh('bitstamp', 99.5, 100.5)]
    rows = record_index.sample(states, SYMBOLS, T0 + 0.73, max_staleness=5,
                               min_exchanges=2)
    assert [r['symbol'] for r in rows] == list(SYMBOLS)
    btc = rows[0]
    assert btc['t'] == T0 and btc['computed'] == T0 + 0.73
    assert btc['proxy'] == 100.0 and btc['n_exchanges'] == 2
    assert math.isnan(rows[1]['proxy'])


def test_spooled_rows_compact_into_the_index_table(tmp_path):
    from core.datastore import ResearchStore
    from core.spool import compact_table, table_spool

    states = [_fresh('kraken', 99, 101), _fresh('bitstamp', 99.5, 100.5)]
    spool = table_spool(tmp_path / 'spool', 'index_proxy_1s', 'index_proxy')
    for i in range(3):
        for row in record_index.sample(states, SYMBOLS, T0 + i + 0.5,
                                       max_staleness=5, min_exchanges=2):
            spool.append(record_index.spool_record(row))
    spool.close()
    compact_table(tmp_path / 'spool', tmp_path / 'store', 'index_proxy_1s',
                  keep_days=0, now=T0 + 3 * 3600)
    got = ResearchStore(tmp_path / 'store').read('index_proxy_1s', min_quality=None)
    assert len(got) == 9
    btc = got[got['symbol'] == 'BTC-USD'].sort_values('event_time')
    assert list(btc['proxy']) == [100.0, 100.0, 100.0]
    assert (btc['available_time'] > btc['event_time']).all()
    assert btc['mid_coinbase'].isna().all() and (btc['mid_kraken'] == 100.0).all()
    eth = got[got['symbol'] == 'ETH-USD']
    assert eth['proxy'].isna().all() and (eth['n_exchanges'] == 0).all()


def test_the_sampler_feeds_the_cache_spool_and_heartbeat(tmp_path, monkeypatch):
    from core import heartbeat
    monkeypatch.setattr(heartbeat, 'ROOT', tmp_path)
    clock = [T0 + 0.2]

    async def fake_sleep(seconds):
        clock[0] += seconds
        if clock[0] > T0 + 4.5:
            raise asyncio.CancelledError
        await asyncio.sleep(0)

    class Spool:
        rows, flushes = [], 0

        def extend(self, rows):
            self.rows.extend(rows)

        def flush(self):
            Spool.flushes += 1

    states = {'kraken': _fresh('kraken', 99, 101), 'gemini': _fresh('gemini', 99, 101)}
    for state in states.values():
        state.alive(T0 + 10)
    cache, spool = IndexProxyCache(), Spool()

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await record_index.sample_loop(states, cache, spool, _args(),
                                           clock=lambda: clock[0], sleep=fake_sleep)
    asyncio.run(drive())
    stamps = sorted({r['t'] for r in spool.rows})
    assert stamps == [T0 + 1, T0 + 2, T0 + 3, T0 + 4]
    assert all(r['proxy'] is None or isinstance(r['proxy'], float) for r in spool.rows)
    assert cache.latest('BTC-USD')['proxy'] == 100.0
    assert heartbeat.age('index') is not None
    assert Spool.flushes >= 1


def test_the_index_component_is_wired_supervised_and_has_a_limit():
    from core import heartbeat
    from scripts.run_live import NAMES, Component
    assert 'index' in NAMES and 'index' in heartbeat.MAX_AGE
    assert 'index' not in Component.selected(NAMES, disable=('index',))


def test_the_exchange_list_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv('INDEX_PROXY_EXCHANGES', 'kraken, gemini')
    assert record_index.build_parser().parse_args([]).exchanges == ['kraken', 'gemini']
