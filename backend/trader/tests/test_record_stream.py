"""The stream component's contract with the rest of the live process.

The gate assertion is the load-bearing one. Every other recorder waits on
`TradingGate.idle()`; this one must not, because a stream that pauses goes stale
and staleness is the failure the book cache exists to prevent.
"""
from __future__ import annotations

import asyncio
import inspect
import time
from datetime import datetime, timezone

import pytest

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from scripts import record_stream, run_live


def snap(ticker='K', **over):
    base = dict(venue='kalshi', market_ticker=ticker, kind='snapshot',
                received=0.0, seq=1, yes=[(0.3, 10.0)], no=[(0.65, 4.0)])
    base.update(over)
    return BookEvent(**base)


def test_stream_is_a_known_component():
    assert 'stream' in run_live.NAMES


def test_the_stream_can_be_disabled_and_everything_else_still_runs():
    selected = run_live.Component.selected(run_live.NAMES, disable=['stream'])
    assert 'stream' not in selected and 'trade' in selected


def test_an_unknown_component_is_still_refused():
    with pytest.raises(ValueError):
        run_live.Component.selected(run_live.NAMES, disable=['strem'])


def test_the_stream_never_waits_on_the_trading_gate():
    """Exactly one gate wait, and it sits inside the flush branch.

    Every other recorder gates its whole cycle. Gating the read loop here would
    let frames queue in the socket while a decision runs, and the book would be
    stale by exactly the amount the gate held — which is the failure the cache
    exists to prevent.
    """
    lines = inspect.getsource(record_stream.consume).splitlines()
    waits = [i for i, line in enumerate(lines) if 'gate.idle()' in line]
    assert len(waits) == 1, f'expected one gate wait, found {len(waits)}'
    branch = [i for i, line in enumerate(lines) if 'next_flush' in line and 'if' in line]
    assert branch and branch[0] < waits[0], 'the gate wait must be inside the flush branch'
    following = '\n'.join(lines[waits[0]:waits[0] + 3])
    assert 'flush' in following, 'the gate must guard the flush and nothing else'


def test_settled_markets_are_forgotten_rather_than_sampled_as_live():
    """A settled Kalshi market serves an EMPTY ladder, not an error."""
    cache = BookCache(now=lambda: 0.0)
    cache.apply(snap('OLD'))
    cache.apply(snap('NEW'))
    record_stream.retire(cache, keep={'NEW'})
    assert cache.ladder('OLD') is None
    assert cache.ladder('NEW') is not None


def test_retiring_nothing_when_everything_is_still_open():
    cache = BookCache(now=lambda: 0.0)
    cache.apply(snap('A'))
    record_stream.retire(cache, keep={'A', 'B'})
    assert cache.tickers() == ['A']


def test_the_process_wide_cache_is_the_one_consumers_read():
    assert isinstance(record_stream.CACHE, BookCache)


def test_the_component_is_wired_into_the_factories():
    gate = run_live.TradingGate()
    args = run_live.build_parser().parse_args([])
    assert 'stream' in run_live.build_factories(args, gate)


# -- the silent-socket deadlock ---------------------------------------------
#
# Observed live on 2026-08-27 at 23:30:00, a window boundary. The subscribed
# markets settled, the venue stopped sending, and `consume` sat forever in
# `async for` — because the refresh deadline was only checked inside the loop
# body, and the loop body needed a frame that was never coming. Nothing raised,
# `supervise` saw a coroutine legitimately awaiting, and the recorder was dead
# behind a container reporting healthy.

class _FakeStream:
    """A stream that yields `events` then goes quiet forever."""

    def __init__(self, events=(), quiet_forever=True):
        self._events = list(events)
        self._quiet = quiet_forever

    async def events(self):
        for event in self._events:
            yield event
        if self._quiet:
            await asyncio.Event().wait()      # never set


class _NullSpool:
    def extend(self, rows):
        return sum(1 for _ in rows)

    def flush(self):
        pass


@pytest.mark.asyncio
async def test_a_socket_that_never_speaks_does_not_hang_forever():
    reason = await asyncio.wait_for(
        record_stream.consume(
            _FakeStream(), BookCache(now=lambda: 0.0), _NullSpool(), {},
            until=time.monotonic() + 300.0, idle_timeout=0.05),
        timeout=5.0)
    assert reason.startswith('silent'), reason


@pytest.mark.asyncio
async def test_a_socket_that_goes_quiet_after_a_burst_is_still_caught():
    """The live failure exactly: frames, then settlement, then nothing."""
    reason = await asyncio.wait_for(
        record_stream.consume(
            _FakeStream([snap('A'), snap('B')]), BookCache(now=lambda: 0.0),
            _NullSpool(), {}, until=time.monotonic() + 300.0,
            idle_timeout=0.05),
        timeout=5.0)
    assert reason.startswith('silent'), reason


@pytest.mark.asyncio
async def test_the_refresh_deadline_is_honoured_without_any_frame():
    """A deadline that only fires on frame arrival is not a deadline."""
    reason = await asyncio.wait_for(
        record_stream.consume(
            _FakeStream(), BookCache(now=lambda: 0.0), _NullSpool(), {},
            until=time.monotonic() + 0.05, idle_timeout=300.0),
        timeout=5.0)
    assert reason == 'subscription refresh', reason


@pytest.mark.asyncio
async def test_a_closed_socket_ends_the_loop_rather_than_waiting():
    reason = await asyncio.wait_for(
        record_stream.consume(
            _FakeStream([snap('A')], quiet_forever=False),
            BookCache(now=lambda: 0.0), _NullSpool(), {},
            until=time.monotonic() + 300.0, idle_timeout=300.0),
        timeout=5.0)
    assert reason == 'socket closed', reason


@pytest.mark.asyncio
async def test_a_sequence_gap_stops_the_loop_so_the_caller_resubscribes():
    cache = BookCache(now=lambda: 0.0)
    events = [snap('A', seq=1),
              snap('A', kind='delta', seq=99, absolute=False,
                   yes=[(0.3, 1.0)], no=[])]
    reason = await asyncio.wait_for(
        record_stream.consume(_FakeStream(events, quiet_forever=False), cache,
                              _NullSpool(), {}, until=time.monotonic() + 300.0,
                              idle_timeout=300.0),
        timeout=5.0)
    assert reason == 'sequence gap', reason


# -- a market the venue still calls open but has already closed --------------

def _market(ticker, closes):
    return {'ticker': ticker, 'close_time': closes}


def test_a_market_past_its_close_is_not_subscribed():
    """Observed live at a 10:30 boundary: `status=open` still listed the market
    that had just settled, so the recorder subscribed to a dead market, heard
    nothing, declared the socket silent and rebuilt — twice — leaving the book
    empty for ~30s at every window boundary."""
    now = datetime(2026, 8, 27, 14, 30, 5, tzinfo=timezone.utc)
    assert record_stream._closed(_market('OLD', '2026-08-27T14:30:00Z'), now)
    assert not record_stream._closed(_market('NEW', '2026-08-27T14:45:00Z'), now)


def test_a_market_closing_exactly_now_is_treated_as_closed():
    now = datetime(2026, 8, 27, 14, 30, 0, tzinfo=timezone.utc)
    assert record_stream._closed(_market('EDGE', '2026-08-27T14:30:00Z'), now)


def test_an_unparseable_or_absent_close_time_keeps_the_market():
    """Dropping a market because its timestamp was odd would be worse than
    subscribing to one that is about to settle."""
    now = datetime(2026, 8, 27, 14, 30, 0, tzinfo=timezone.utc)
    assert not record_stream._closed({'ticker': 'X'}, now)
    assert not record_stream._closed(_market('X', 'not-a-time'), now)
    assert not record_stream._closed(_market('X', None), now)


@pytest.mark.asyncio
async def test_open_tickers_filters_the_settled_market_out():
    now = datetime(2026, 8, 27, 14, 30, 5, tzinfo=timezone.utc)

    class FakeClient:
        async def _request(self, method, path, params=None):
            if params['series_ticker'] != 'KXBTC15M':
                return {'markets': []}
            return {'markets': [_market('KXBTC15M-OLD', '2026-08-27T14:30:00Z'),
                                _market('KXBTC15M-NEW', '2026-08-27T14:45:00Z')]}

    got = await record_stream.open_tickers(FakeClient(), now=now)
    assert list(got) == ['KXBTC15M-NEW']
