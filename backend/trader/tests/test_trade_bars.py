"""Bars folded from the Coinbase trade stream, and when live may use them.

The stream replaces the ~3s REST candle fetch on the decision path, so it must
produce exactly what REST would, and must never be used for a minute it did
not see in full: a reconnect, a sequence gap or a late print all send that
minute back to REST.
"""
from __future__ import annotations

import time

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.trade_bars import MINUTE, TradeTape, minute_of, parse_time
from livetrade import bars as live_bars
from scripts.record_spot_stream import SequenceTracker, handle

A = 1_700_000_040          # a minute boundary
B, C = A + MINUTE, A + 2 * MINUTE


def _tape(now=None, **kw):
    clock = {'now': now if now is not None else C + MINUTE + 5.0}
    tape = TradeTape(now=lambda: clock['now'], **kw)
    return tape, clock


class TestCandleSemantics:
    def test_trades_make_the_hand_computed_candle(self):
        tape, _ = _tape()
        tape.connected(at=A - 30)
        # Out of order, with a duplicate.
        tape.add('BTC-USD', '1', '100.0', '1.0', A + 5, received=A + 5.1)
        tape.add('BTC-USD', '3', '103.0', '0.5', A + 50, received=A + 50.1)
        tape.add('BTC-USD', '2', '99.0', '2.0', A + 20, received=A + 50.2)
        assert tape.add('BTC-USD', '2', '99.0', '2.0', A + 20,
                        received=A + 50.3) is False
        tape.add('BTC-USD', '7', '101.5', '0.25', C + 1, received=C + 1.1)

        assert tape.bar('BTC-USD', A) == {
            'minute': A, 'open': 100.0, 'high': 103.0, 'low': 99.0,
            'close': 103.0, 'volume': 3.5, 'trades': 3}
        # A minute with no trades has no bar, exactly as REST omits it.
        assert tape.bar('BTC-USD', B) is None
        assert tape.span('BTC-USD', A, C) == [
            tape.bar('BTC-USD', A), tape.bar('BTC-USD', C)]
        assert tape.bar('BTC-USD', C)['open'] == tape.bar('BTC-USD', C)['close'] == 101.5

    def test_a_tied_timestamp_orders_on_the_trade_id(self):
        tape, _ = _tape()
        tape.connected(at=A - 30)
        tape.add('ETH-USD', '10', '2.0', '1', A + 9, received=A + 9.1)
        tape.add('ETH-USD', '9', '1.0', '1', A + 9, received=A + 9.2)
        bar = tape.bar('ETH-USD', A)
        assert (bar['open'], bar['close']) == (1.0, 2.0)

    def test_iso_times_with_nanoseconds_parse(self):
        assert parse_time('2023-11-14T22:14:00.123456789Z') == pytest.approx(
            1700000040.123456789, abs=1e-6)
        assert parse_time('2023-11-14T22:14:00Z') == 1700000040.0


class TestFinalityAndCoverage:
    def test_a_minute_is_final_only_after_its_close_plus_grace(self):
        tape, _ = _tape(grace_seconds=0.75)
        tape.connected(at=A - 30)
        assert not tape.reliable('BTC-USD', A, now=A + MINUTE + 0.5)
        assert tape.reliable('BTC-USD', A, now=A + MINUTE + 0.75)

    def test_a_minute_that_began_before_coverage_is_never_served(self):
        tape, _ = _tape()
        tape.connected(at=A + 10)
        assert not tape.reliable('BTC-USD', A)
        assert tape.reliable('BTC-USD', B)

    def test_a_disconnect_and_reconnect_lose_the_minutes_between(self):
        tape, _ = _tape()
        tape.connected(at=A - 30)
        tape.disconnected(at=B + 10)
        tape.connected(at=B + 40)
        assert tape.reliable('BTC-USD', A)
        assert not tape.reliable('BTC-USD', B)
        assert tape.reliable('BTC-USD', C)
        assert tape.span('BTC-USD', A, C) is None

    def test_a_sequence_gap_breaks_coverage(self):
        tape, _ = _tape()
        tape.connected(at=A - 30)
        tape.gap(since=B + 5, at=B + 6)
        assert not tape.reliable('BTC-USD', B)
        assert tape.reliable('BTC-USD', C)

    def test_a_late_print_taints_its_minute(self):
        tape, _ = _tape()
        tape.connected(at=A - 30)
        tape.add('BTC-USD', '1', '100', '1', A + 5, received=A + 5.1)
        assert tape.reliable('BTC-USD', A)
        tape.add('BTC-USD', '2', '100', '1', A + 59, received=A + MINUTE + 3)
        assert not tape.reliable('BTC-USD', A)

    def test_an_old_snapshot_trade_does_not_taint_an_uncovered_minute(self):
        tape, _ = _tape()
        tape.add('BTC-USD', '1', '100', '1', A + 5, received=C, snapshot=True)
        tape.connected(at=C)
        assert tape.late_trades == 0

    def test_retention_bounds_what_is_served(self):
        tape, _ = _tape(retain_minutes=5)
        tape.connected(at=A - 30)
        assert not tape.reliable('BTC-USD', A, now=A + 30 * MINUTE)


class TestTheSocketMessages:
    def test_trades_are_folded_and_coverage_starts_with_them(self):
        tape, _ = _tape()
        seqs = SequenceTracker()
        handle({'channel': 'market_trades', 'sequence_num': 0, 'events': [{
            'type': 'update', 'trades': [{
                'trade_id': '5', 'product_id': 'BTC-USD', 'price': '100.5',
                'size': '0.1', 'side': 'BUY', 'time': '2023-11-14T22:14:05Z'}]}]},
            tape, seqs, received=A + 5.2, last_received=None)
        assert tape.is_connected
        assert tape.bar('BTC-USD', A)['close'] == 100.5

    @pytest.mark.parametrize('order,gap', [
        # One counter across the connection: a jump in it is a lost message.
        ([('heartbeats', 1), ('market_trades', 2), ('heartbeats', 3),
          ('market_trades', 5)], True),
        ([('heartbeats', 1), ('market_trades', 2), ('heartbeats', 3),
          ('market_trades', 4)], False),
        # One counter per channel: interleaving is not a gap.
        ([('heartbeats', 10), ('market_trades', 1), ('heartbeats', 11),
          ('market_trades', 2)], False),
        ([('heartbeats', 10), ('market_trades', 1), ('heartbeats', 11),
          ('market_trades', 4)], True),
    ])
    def test_a_gap_is_judged_under_either_reading(self, order, gap):
        seqs = SequenceTracker()
        seen = [seqs.observe(channel, seq) for channel, seq in order]
        assert any(seen) is gap


# ---------------------------------------------------------------------------
# Live's use of it
# ---------------------------------------------------------------------------

def _trades_for(minute: int, k: int):
    """Three trades a minute, deterministic."""
    base = 100.0 + k
    return [(f'{k}a', base, 1.0, minute + 3), (f'{k}b', base + 2, 0.5, minute + 30),
            (f'{k}c', base - 1, 0.25, minute + 50)]


def _rest_row(minute: int, trades) -> dict:
    prices = [p for _, p, _, _ in trades]
    return {'event_time': pd.Timestamp(minute, unit='s', tz='UTC'),
            'open': trades[0][1], 'high': max(prices), 'low': min(prices),
            'close': trades[-1][1], 'volume': sum(s for _, _, s, _ in trades),
            'quote_volume': None, 'trade_count': None}


def _scenario(now: float, *, n: int = 20, fetched_ago: float = 30.0):
    """A REST cache fetched 30s ago and a tape that covered all of it."""
    last = minute_of(now) - MINUTE
    minutes = [last - (n - 1 - i) * MINUTE for i in range(n)]
    tape = TradeTape(now=lambda: now)
    tape.connected(at=minutes[0] - 5)
    rows = []
    fetched_at = now - fetched_ago
    for k, minute in enumerate(minutes):
        trades = _trades_for(minute, k)
        for tid, price, size, at in trades:
            tape.add('BTC-USD', tid, price, size, at, received=at + 0.05)
        if minute + MINUTE <= fetched_at:
            rows.append(_rest_row(minute, trades))
    return tape, pd.DataFrame(rows), fetched_at, minutes


class TestStreamBars:
    def test_rest_then_stream_equals_the_rest_candles_it_replaces(self):
        now = minute_of(time.time()) + 1.0
        tape, rest, fetched_at, minutes = _scenario(now)
        out = live_bars.stream_bars(rest, tape, 'BTC-USD', fetched_at=fetched_at,
                               now=now, floor=pd.Timestamp(0, unit='s', tz='UTC'))
        assert out is not None
        expected = pd.DataFrame([_rest_row(m, _trades_for(m, k))
                                 for k, m in enumerate(minutes)])
        assert list(out['event_time']) == list(expected['event_time'])
        for column in ('open', 'high', 'low', 'close', 'volume'):
            assert out[column].tolist() == pytest.approx(expected[column].tolist())
        # Still the minute before the decision, as REST would give.
        assert out['event_time'].iloc[-1] == pd.Timestamp(minutes[-1], unit='s', tz='UTC')

    def test_an_unfinished_rest_candle_after_the_anchor_is_replaced(self):
        now = minute_of(time.time()) + 1.0
        # Fetched 6s after the newest REST candle closed: unsettled.
        tape, rest, fetched_at, minutes = _scenario(now, fetched_ago=55.0)
        # The newest REST candle was read seconds after it closed and is short.
        rest.loc[rest.index[-1], 'volume'] -= 0.25
        out = live_bars.stream_bars(rest, tape, 'BTC-USD', fetched_at=fetched_at,
                               now=now, floor=pd.Timestamp(0, unit='s', tz='UTC'))
        assert out is not None
        row = out[out['event_time'] == rest['event_time'].iloc[-1]].iloc[0]
        assert row['volume'] == pytest.approx(1.75)

    def test_a_disagreement_with_a_settled_rest_candle_refuses(self, caplog):
        now = minute_of(time.time()) + 1.0
        tape, rest, fetched_at, _ = _scenario(now)
        rest.loc[rest.index[2], 'high'] += 1.0
        with caplog.at_level('WARNING'):
            assert live_bars.stream_bars(
                rest, tape, 'BTC-USD', fetched_at=fetched_at, now=now,
                floor=pd.Timestamp(0, unit='s', tz='UTC')) is None
        assert 'disagrees with REST' in caplog.text

    def test_a_gap_after_the_anchor_refuses(self):
        now = minute_of(time.time()) + 1.0
        tape, rest, fetched_at, minutes = _scenario(now)
        tape.gap(since=minutes[-2] + 10, at=minutes[-2] + 11)
        assert live_bars.stream_bars(rest, tape, 'BTC-USD', fetched_at=fetched_at,
                                now=now,
                                floor=pd.Timestamp(0, unit='s', tz='UTC')) is None

    def test_the_newest_minute_not_yet_final_refuses(self):
        now = minute_of(time.time()) + 0.2
        tape, rest, fetched_at, _ = _scenario(now)
        assert live_bars.stream_bars(rest, tape, 'BTC-USD', fetched_at=fetched_at,
                                now=now,
                                floor=pd.Timestamp(0, unit='s', tz='UTC')) is None


class _Bar:
    def __init__(self, row):
        self.event_time = row['event_time'].tz_convert(None).to_pydatetime()
        for key in ('open', 'high', 'low', 'close', 'volume'):
            setattr(self, key, row[key])


class _Client:
    def __init__(self, frames):
        self.frames, self.calls = frames, []

    async def get_candles_range(self, symbol, timeframe, start, end):
        self.calls.append(symbol)
        return [_Bar(r) for _, r in self.frames[symbol].iterrows()]

    async def close(self):
        pass


@pytest.fixture
def _clean_caches():
    live_bars._BAR_CACHE.clear()
    live_bars._BAR_FETCHED_AT.clear()
    yield
    live_bars._BAR_CACHE.clear()
    live_bars._BAR_FETCHED_AT.clear()


@pytest.mark.asyncio
async def test_a_covered_universe_needs_no_rest_call(monkeypatch, _clean_caches):
    now = time.time()
    tape = TradeTape()
    frames = {}
    for i, symbol in enumerate(DEFAULT_CONFIG.symbols):
        tape_i, rest, fetched_at, _ = _scenario(now)
        # One tape, every symbol on it.
        for minute, acc in tape_i._minutes['BTC-USD'].items():
            tape._minutes.setdefault(symbol, {})[minute] = acc
        tape._coverage = tape_i._coverage
        live_bars._BAR_CACHE[symbol] = rest
        live_bars._BAR_FETCHED_AT[symbol] = fetched_at
        frames[symbol] = rest
    client = _Client(frames)
    monkeypatch.setattr(live_bars, 'CoinbaseRESTClient', lambda **kw: client)
    monkeypatch.setattr(live_bars, '_spot_tape', lambda: tape)
    out = await live_bars.fetch_bars(DEFAULT_CONFIG, minutes=60, prefer_stream=True)
    assert set(out) == set(DEFAULT_CONFIG.symbols)
    assert client.calls == [], 'the decision path still went to REST'


@pytest.mark.asyncio
async def test_a_gap_falls_back_to_rest_for_that_symbol(monkeypatch, _clean_caches):
    now = time.time()
    tape = TradeTape()
    frames = {}
    for symbol in DEFAULT_CONFIG.symbols:
        tape_i, rest, fetched_at, minutes = _scenario(now)
        for minute, acc in tape_i._minutes['BTC-USD'].items():
            tape._minutes.setdefault(symbol, {})[minute] = acc
        tape._coverage = tape_i._coverage
        live_bars._BAR_CACHE[symbol] = rest
        live_bars._BAR_FETCHED_AT[symbol] = fetched_at
        frames[symbol] = rest
    gapped = DEFAULT_CONFIG.symbols[1]
    tape._tainted[gapped] = {minutes[-1]}
    client = _Client(frames)
    monkeypatch.setattr(live_bars, 'CoinbaseRESTClient', lambda **kw: client)
    monkeypatch.setattr(live_bars, '_spot_tape', lambda: tape)
    out = await live_bars.fetch_bars(DEFAULT_CONFIG, minutes=60, prefer_stream=True)
    assert set(out) == set(DEFAULT_CONFIG.symbols)
    assert client.calls == [gapped]


@pytest.mark.asyncio
async def test_no_stream_means_rest_exactly_as_before(monkeypatch, _clean_caches):
    now = time.time()
    frames = {s: _scenario(now)[1] for s in DEFAULT_CONFIG.symbols}
    client = _Client(frames)
    monkeypatch.setattr(live_bars, 'CoinbaseRESTClient', lambda **kw: client)
    monkeypatch.setattr(live_bars, '_spot_tape', lambda: None)
    out = await live_bars.fetch_bars(DEFAULT_CONFIG, minutes=60, prefer_stream=True)
    assert sorted(client.calls) == sorted(DEFAULT_CONFIG.symbols)
    assert set(out) == set(DEFAULT_CONFIG.symbols)
