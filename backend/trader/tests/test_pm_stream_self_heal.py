"""A stuck Polymarket stream cache is torn down and resubscribed on request.

The trading loop counts consecutive peer refusals per symbol. After
`peer_resubscribe_after_cycles` of them it asks `record_pm_stream` to drop that
symbol's state and subscribe again. The heartbeat is touched only for a
two-sided book of the current window.
"""
from __future__ import annotations

import asyncio
import time

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from livetrade import features
from livetrade.features import cross_venue_row
from scripts import record_pm_stream as rec

CONFIG = DEFAULT_CONFIG.with_overrides(peer_resubscribe_after_cycles=3)


class _Quote:
    yes_bid, yes_ask = 0.50, 0.52


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    rec.CACHE.clear()
    rec._WINDOWS.clear()
    rec._RESUBSCRIBE.clear()
    features.reset_peer_tracking()
    beats: list = []
    monkeypatch.setattr(rec.heartbeat, 'touch', lambda name: beats.append(name))
    monkeypatch.setattr(rec, '_LAST_BEAT', {'at': -1e18})
    yield beats
    rec.CACHE.clear()
    rec._WINDOWS.clear()
    rec._RESUBSCRIBE.clear()
    features.reset_peer_tracking()


# ---- the trading loop's side ------------------------------------------------

def _decide(pm, now, window):
    return cross_venue_row(pm, _Quote(), now=now, window_open=window,
                           symbol='BTC-USD', config=CONFIG)


def test_n_stuck_cycles_request_a_resubscribe():
    now = pd.Timestamp('2026-10-14 12:03:02', tz='UTC')
    window = pd.Timestamp('2026-10-14 12:00', tz='UTC')
    stale = {'best_bid': 40.0, 'best_ask': 42.0, 'window': window,
             'at': now - pd.Timedelta(minutes=5)}
    _decide(stale, now, window)
    _decide(None, now, window)
    assert not rec._RESUBSCRIBE, 'two cycles are under the threshold'
    _decide(stale, now, window)
    assert rec._RESUBSCRIBE == {'BTC-USD': 'too_old'}


def test_a_used_book_breaks_the_streak():
    now = pd.Timestamp('2026-10-14 12:03:02', tz='UTC')
    window = pd.Timestamp('2026-10-14 12:00', tz='UTC')
    stale = {'best_bid': 40.0, 'best_ask': 42.0, 'window': window,
             'at': now - pd.Timedelta(minutes=5)}
    fresh = dict(stale, at=now - pd.Timedelta(seconds=1))
    for pm in (stale, stale, fresh, stale, stale):
        _decide(pm, now, window)
    assert not rec._RESUBSCRIBE


def test_a_one_sided_book_is_not_a_stuck_cache():
    now = pd.Timestamp('2026-10-14 12:03:02', tz='UTC')
    window = pd.Timestamp('2026-10-14 12:00', tz='UTC')
    lone = {'best_bid': 40.0, 'best_ask': float('nan'), 'window': window,
            'at': now - pd.Timedelta(seconds=1)}
    for _ in range(5):
        _decide(lone, now, window)
    assert not rec._RESUBSCRIBE


def test_forget_drops_the_book_and_the_slug():
    rec.CACHE['BTC-USD'] = {'best_bid': 40.0}
    rec.CACHE['ETH-USD'] = {'best_bid': 40.0}
    rec._WINDOWS.update({'BTC-USD': 'btc-slug', 'ETH-USD': 'eth-slug'})
    rec.forget(['BTC-USD'])
    assert 'BTC-USD' not in rec.CACHE and 'BTC-USD' not in rec._WINDOWS
    assert 'ETH-USD' in rec.CACHE and 'ETH-USD' in rec._WINDOWS


# ---- the recorder's side ----------------------------------------------------

def test_consume_returns_on_a_request_while_frames_flow():
    async def _run():
        async def flood():
            await asyncio.sleep(0.005)
            return {'event_type': 'price_change', 'price_changes': []}

        async def ask_later():
            await asyncio.sleep(0.05)
            rec.request_resubscribe('BTC-USD', 'too_old')

        asyncio.ensure_future(ask_later())
        return await rec.consume(_Stream(flood), until=30.0, silence_seconds=30.0)

    reason = asyncio.new_event_loop().run_until_complete(_run())
    assert reason.startswith('resubscribe requested'), reason


def test_consume_returns_on_a_request_while_the_socket_is_silent():
    async def _run():
        async def never():
            await asyncio.sleep(3600)

        rec.request_resubscribe('BTC-USD', 'no_book')
        return await rec.consume(_Stream(never), until=30.0, silence_seconds=30.0)

    reason = asyncio.new_event_loop().run_until_complete(_run())
    assert 'BTC-USD (no_book)' in reason


def _ms(ts: pd.Timestamp) -> str:
    return str(int(ts.timestamp() * 1000))


def _run_recorder(monkeypatch, *, fresh: bool) -> int:
    """Run `record_pm_stream.run` against fakes and return how many times it resolved
    tokens.

    The live side decides three times in between.
    """
    from scripts.record_pm_ladder import slug_for

    resolved = {'n': 0}

    async def fake_resolve(session, now):
        resolved['n'] += 1
        rec._WINDOWS['BTC-USD'] = slug_for('btc', now)
        return {'T1': 'BTC-USD'}

    class _Session:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

    opened = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=10)

    class _FakeSocket:
        def __init__(self, session, tokens):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def events(self):
            async def _gen():
                yield {'event_type': 'book', 'asset_id': 'T1',
                       'timestamp': _ms(opened),
                       'bids': [{'price': '0.44', 'size': '250'}],
                       'asks': [{'price': '0.46', 'size': '150'}]}
                while True:
                    await asyncio.sleep(0.005)
                    now = pd.Timestamp.now(tz='UTC')
                    changes = ([{'asset_id': 'T1', 'price': '0.44',
                                 'size': '260', 'side': 'BUY'}] if fresh else [])
                    yield {'event_type': 'price_change', 'timestamp': _ms(now),
                           'price_changes': changes}
            return _gen()

    import aiohttp
    monkeypatch.setattr(aiohttp, 'ClientSession', _Session)
    monkeypatch.setattr(rec, 'resolve_tokens', fake_resolve)
    monkeypatch.setattr(rec, '_Socket', _FakeSocket)

    async def _scenario():
        task = asyncio.ensure_future(rec.run())
        try:
            deadline = time.monotonic() + 5.0
            while 'BTC-USD' not in rec.CACHE and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            await asyncio.sleep(0.05)
            for _ in range(3):
                now = pd.Timestamp.now(tz='UTC')
                _decide(rec.CACHE.get('BTC-USD'), now, rec.current_window(now))
                await asyncio.sleep(0.02)
            deadline = time.monotonic() + 2.0
            while resolved['n'] < 2 and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            if not fresh:
                return resolved['n']
            await asyncio.sleep(0.2)
            return resolved['n']
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    return asyncio.new_event_loop().run_until_complete(_scenario())


def test_a_stuck_cache_is_resubscribed(monkeypatch):
    assert _run_recorder(monkeypatch, fresh=False) >= 2, (
        'three too_old cycles must make the recorder resolve and subscribe again')


def test_a_healthy_cache_is_left_alone(monkeypatch):
    assert _run_recorder(monkeypatch, fresh=True) == 1
    assert not rec._RESUBSCRIBE


# ---- the heartbeat ----------------------------------------------------------

def _book():
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}],
                'asks': [{'price': '0.46', 'size': '150'}]})
    return book


def test_the_heartbeat_needs_the_current_window(_clean):
    now = pd.Timestamp('2026-10-14 12:03:02', tz='UTC')
    rec.publish(_book(), {'T': 'BTC-USD'},
                window=pd.Timestamp('2026-10-14 11:45', tz='UTC'), now=now)
    assert _clean == [], 'a book for the previous window is not liveness'
    rec.publish(_book(), {'T': 'BTC-USD'},
                window=pd.Timestamp('2026-10-14 12:00', tz='UTC'), now=now)
    assert _clean == ['pm_stream']


def test_the_heartbeat_needs_a_two_sided_book(_clean):
    now = pd.Timestamp('2026-10-14 12:03:02', tz='UTC')
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}], 'asks': []})
    rec.publish(book, {'T': 'BTC-USD'},
                window=pd.Timestamp('2026-10-14 12:00', tz='UTC'), now=now)
    assert _clean == []


def test_the_heartbeat_needs_a_known_window(_clean):
    rec.publish(_book(), {'T': 'BTC-USD'})
    assert _clean == []


class _Stream:
    def __init__(self, factory):
        self._factory = factory

    def events(self):
        factory = self._factory

        async def _gen():
            while True:
                out = await factory()
                if out is not None:
                    yield out
        return _gen()
