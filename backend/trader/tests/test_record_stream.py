"""Tests for the stream component's contract with the rest of the live process.

Other recorders wait on `TradingGate.idle()`. The stream must not, because a
paused stream goes stale.
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


@pytest.mark.asyncio
async def test_the_stream_keeps_folding_while_a_decision_holds_the_gate():
    """A held gate defers the flush but never pauses the read loop."""
    gate = run_live.TradingGate()
    cache = BookCache(now=lambda: 0.0)
    flushes = []

    class Spool(_NullSpool):
        def flush(self):
            flushes.append(cache.tickers())

    async with gate.deciding():
        reason = await asyncio.wait_for(
            record_stream.consume(
                _FakeStream([snap('A'), snap('B'), snap('C')],
                            quiet_forever=False),
                cache, Spool(), {}, gate=gate, flush_every=0.0,
                until=time.monotonic() + 300.0, idle_timeout=300.0),
            timeout=5.0)
    assert reason == 'socket closed', reason
    assert cache.tickers() == ['A', 'B', 'C'], 'every frame folded under the gate'
    assert flushes == [], 'no flush while a decision is in flight'


@pytest.mark.asyncio
async def test_a_deferred_flush_runs_once_the_gate_reopens():
    gate = run_live.TradingGate()
    flushed = []

    class Spool(_NullSpool):
        def flush(self):
            flushed.append(True)

    async def frames():
        async with gate.deciding():
            yield snap('A')
        yield snap('B')

    class Stream:
        def events(self):
            return frames()

    await asyncio.wait_for(
        record_stream.consume(
            Stream(), BookCache(now=lambda: 0.0), Spool(), {}, gate=gate,
            flush_every=0.0, until=time.monotonic() + 300.0,
            idle_timeout=300.0),
        timeout=5.0)
    assert flushed == [True]


def test_settled_markets_are_forgotten_rather_than_sampled_as_live():
    """A settled Kalshi market serves an empty ladder, not an error, so it is forgotten.
    """
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
# When the subscribed markets settle the venue stops sending. Every exit
# condition is enforced on a timeout, so a quiet socket cannot block `consume`
# forever while the container still reports healthy.

class _FakeStream:
    """A stream that yields `events` then goes quiet forever."""

    def __init__(self, events=(), quiet_forever=True):
        self._events = list(events)
        self._quiet = quiet_forever

    async def events(self):
        for event in self._events:
            yield event
        if self._quiet:
            await asyncio.Event().wait()  # never set


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
    """Frames, then settlement, then silence: `consume` still returns."""
    reason = await asyncio.wait_for(
        record_stream.consume(
            _FakeStream([snap('A'), snap('B')]), BookCache(now=lambda: 0.0),
            _NullSpool(), {}, until=time.monotonic() + 300.0,
            idle_timeout=0.05),
        timeout=5.0)
    assert reason.startswith('silent'), reason


@pytest.mark.asyncio
async def test_the_refresh_deadline_is_honoured_without_any_frame():
    """The refresh deadline fires with no frame at all."""
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


# -- the venue's `status` field lags its own clock ---------------------------

def _market(ticker, opens, closes, status='active'):
    return {'ticker': ticker, 'open_time': opens, 'close_time': closes,
            'status': status}


NOW = datetime(2026, 8, 27, 15, 15, 5, tzinfo=timezone.utc)


def test_a_market_past_its_close_is_not_subscribed():
    """A market past its close is not subscribed, even while the venue still lists it as
    open.
    """
    assert not record_stream._trading(
        _market('OLD', '2026-08-27T15:00:00Z', '2026-08-27T15:15:00Z'), NOW)


def test_the_current_market_is_subscribed_even_while_status_says_initialized():
    """The current market is subscribed while its status is still `initialized`.

    The replacement market opens at the boundary but stays `initialized` for about
    40s, so filtering on status would leave the book empty at the start of a window.
    """
    assert record_stream._trading(
        _market('NEW', '2026-08-27T15:15:00Z', '2026-08-27T15:30:00Z',
                status='initialized'), NOW)


def test_a_market_whose_window_has_not_started_is_not_subscribed():
    assert not record_stream._trading(
        _market('NEXT', '2026-08-27T15:30:00Z', '2026-08-27T15:45:00Z'), NOW)


def test_the_boundary_instant_belongs_to_the_new_window():
    boundary = datetime(2026, 8, 27, 15, 15, 0, tzinfo=timezone.utc)
    assert not record_stream._trading(
        _market('OLD', '2026-08-27T15:00:00Z', '2026-08-27T15:15:00Z'), boundary)
    assert record_stream._trading(
        _market('NEW', '2026-08-27T15:15:00Z', '2026-08-27T15:30:00Z'), boundary)


def test_a_market_missing_a_timestamp_is_kept():
    """A market with a missing or bad timestamp is kept."""
    assert record_stream._trading({'ticker': 'X'}, NOW)
    assert record_stream._trading(
        _market('X', 'not-a-time', '2026-08-27T15:30:00Z'), NOW)


@pytest.mark.asyncio
async def test_open_tickers_queries_by_close_time_not_status():
    seen = {}

    class FakeClient:
        async def _request(self, method, path, params=None):
            seen.update(params)
            if params['series_ticker'] != 'KXBTC15M':
                return {'markets': []}
            return {'markets': [
                _market('KXBTC15M-OLD', '2026-08-27T15:00:00Z', '2026-08-27T15:15:00Z'),
                _market('KXBTC15M-NOW', '2026-08-27T15:15:00Z', '2026-08-27T15:30:00Z',
                        status='initialized'),
                _market('KXBTC15M-NEXT', '2026-08-27T15:30:00Z', '2026-08-27T15:45:00Z'),
            ]}

    got = await record_stream.open_tickers(FakeClient(), now=NOW)
    assert list(got) == ['KXBTC15M-NOW'], (
        'exactly the market whose window contains now, whatever its status')
    assert 'status' not in seen, "status lags the venue's own clock"
    assert seen['min_close_ts'] == int(NOW.timestamp())
    assert seen['max_close_ts'] == int(NOW.timestamp()) + 1800


def test_the_empty_market_retry_is_fast_not_the_refresh_cadence():
    """The retry after an empty market set is fast, not the full refresh interval.

    At a window boundary the old market has closed and the new one may not be listed
    yet.
    """
    args = record_stream.build_parser().parse_args([])
    assert args.empty_retry_seconds <= 5.0
    assert args.empty_retry_seconds < args.refresh_seconds


def test_the_empty_branch_waits_on_the_fast_retry():
    source = inspect.getsource(record_stream.run)
    branch = source[source.index('if not symbols:'):]
    branch = branch[:branch.index('continue')]
    assert 'empty_retry_seconds' in branch
    assert 'refresh_seconds' not in branch, (
        'the boundary gap must not be paid at the refresh cadence')


def test_an_incomplete_market_set_is_rechecked_soon():
    """An incomplete market set is rechecked soon; the venue lists each series' next market
    at a different moment.
    """
    source = inspect.getsource(record_stream.run)
    body = source[source.index('covered = set(symbols.values())'):]
    body = body[:body.index('reason = await consume')]
    assert 'empty_retry_seconds' in body and 'refresh_seconds' in body, (
        'the wait must choose between the fast retry and the normal cadence')
    assert 'series_to_symbol()' in body, (
        'completeness is measured against the configured series, not a constant')
