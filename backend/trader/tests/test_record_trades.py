"""The tape recorder's contract: complete minutes, written once, nothing lost.

`venue_trades` is the only evidence that answers whether a RESTING order would
have filled, and until 2026-10-20 its only writer was a manual backfill. The
failures worth testing are all of the silent kind:

* a minute written from a partial fetch understates how deep an aggressor went,
  which is the direction that makes a fill simulation look better than it was;
* a watermark advanced past a fetch that FAILED turns a transient venue error
  into a permanent hole;
* a market dropped the instant it closes loses the end of every window, which is
  the most valuable part of it;
* and a row written with a `quality` the store does not rank is a row
  `ResearchStore.read` silently returns none of — which is true of every
  backfilled row on disk today.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core import heartbeat
from data_collection.kalshi_client import Trade
from scripts import record_trades, run_live

WINDOW = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)


def print_at(minute, *, price, side='bid', contracts=10.0, second=5):
    return Trade(
        trade_id=f'{minute}-{side}-{price}', ticker='KXBTC15M-26OCT201200-15',
        contracts=contracts, yes_price=price, no_price=None,
        created_time=WINDOW + timedelta(minutes=minute, seconds=second),
        is_block_trade=False, taker_side='up', taker_book_side=side)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_trades_is_a_known_component():
    assert 'trades' in run_live.NAMES


def test_the_tape_recorder_can_be_disabled_and_everything_else_still_runs():
    selected = run_live.Component.selected(run_live.NAMES, disable=['trades'])
    assert 'trades' not in selected and 'trade' in selected


def test_it_is_not_confused_with_the_trading_loop():
    """`trade` and `trades` are one character apart and do unrelated jobs."""
    selected = run_live.Component.selected(run_live.NAMES, disable=['trades'])
    assert 'trade' in selected
    selected = run_live.Component.selected(run_live.NAMES, disable=['trade'])
    assert 'trades' in selected


def test_it_has_a_factory_and_a_staleness_limit():
    """The known failure: a name in COMPONENTS and not in `build_factories` is
    a KeyError at startup, after the banner has printed the component as `on`.
    `test_healthcheck` asserts this for every component; asserting it here too
    is cheap and names the one being added."""
    import argparse

    args = argparse.Namespace(trade_args=[], disable=[], verbose=False)
    assert 'trades' in run_live.build_factories(args, run_live.TradingGate())
    assert 'trades' in heartbeat.MAX_AGE


def test_the_phase_is_read_from_the_declaration():
    """`pm_ladder` drifted this way once: the declaration was changed and the
    factory's literal was not, so the banner printed a phase the poll did not
    use."""
    source = inspect.getsource(run_live.build_factories)
    body = source.split('async def trades(')[1].split('async def ')[0]
    assert 'COMPONENTS' in body and 'phase' in body


def test_it_does_not_hold_a_second_websocket_subscription():
    """The decision is REST, and it is load-bearing rather than incidental.

    Kalshi's `seq` is global per SUBSCRIPTION, so trade frames on the book
    socket advance the counter the book cache gap-checks — and `any_gapped()`
    forces a full resubscribe of the one component the trading path reads.
    """
    code = '\n'.join(inspect.getsource(obj) for obj in
                     (record_trades.run, record_trades.poll_markets,
                      record_trades._tape))
    assert 'KalshiStream' not in code
    assert 'orderbook_delta' not in code
    assert 'market_trades' in code
    # And nothing the module imported is a stream.
    assert not [name for name in vars(record_trades) if 'Stream' in name]


def test_the_loop_sleeps_through_the_aligned_helper():
    """A fixed `sleep(interval)` makes the real period `interval + work`, so
    the phase walks through the minute — measured on `venue_ladder` and fixed
    there."""
    source = inspect.getsource(record_trades.run)
    assert 'seconds_to_next_mark(' in source
    assert 'asyncio.sleep(args.interval)' not in source


def test_it_does_not_ask_for_status_open():
    source = inspect.getsource(record_trades.poll_markets)
    assert "'status': 'open'" not in source
    assert 'min_close_ts' in source and 'max_close_ts' in source


# ---------------------------------------------------------------------------
# which minutes may be written
# ---------------------------------------------------------------------------

def test_the_minute_holding_the_lookback_edge_is_not_written():
    """It was only partly fetched. Emitting it understates `n_prints` and can
    raise `min_price` above the real low — the one direction of error a fill
    simulation must not make."""
    lo, hi = record_trades.complete_minutes(
        window_open=WINDOW,
        since=WINDOW + timedelta(minutes=3, seconds=20),
        now=WINDOW + timedelta(minutes=8),
        watermark=None, window_minutes=15)
    assert lo == 4 and hi == 7


def test_an_exactly_aligned_lookback_edge_keeps_its_minute():
    """`since` on the minute mark means that minute WAS fully fetched."""
    lo, _ = record_trades.complete_minutes(
        window_open=WINDOW, since=WINDOW + timedelta(minutes=3),
        now=WINDOW + timedelta(minutes=8), watermark=None, window_minutes=15)
    assert lo == 3


def test_the_minute_in_progress_is_not_written():
    _, hi = record_trades.complete_minutes(
        window_open=WINDOW, since=WINDOW,
        now=WINDOW + timedelta(minutes=6, seconds=30),
        watermark=None, window_minutes=15)
    assert hi == 5


def test_a_minute_already_written_is_not_written_again():
    lo, hi = record_trades.complete_minutes(
        window_open=WINDOW, since=WINDOW,
        now=WINDOW + timedelta(minutes=9), watermark=6, window_minutes=15)
    assert (lo, hi) == (7, 8)


def test_nothing_is_due_before_the_first_minute_has_elapsed():
    lo, hi = record_trades.complete_minutes(
        window_open=WINDOW, since=WINDOW - timedelta(minutes=4),
        now=WINDOW + timedelta(seconds=30), watermark=None, window_minutes=15)
    assert lo > hi


def test_it_never_runs_past_the_end_of_the_window():
    """A market is polled past its close, so `now` keeps advancing while the
    window does not. Without the cap the range would run to minute 20 and fold
    the next market's prints into this one."""
    _, hi = record_trades.complete_minutes(
        window_open=WINDOW, since=WINDOW,
        now=WINDOW + timedelta(minutes=21), watermark=None, window_minutes=15)
    assert hi == 15


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------

def test_the_extremes_are_kept_per_minute_and_per_aggressor_side():
    """A resting bid at P is hit when a SELLING aggressor trades at or below P,
    so `min_price` on that side is the whole fill test. A mean answers nothing."""
    rows = record_trades.fold_prints(
        [print_at(2, price=0.40, side='bid'),
         print_at(2, price=0.37, side='bid'),
         print_at(2, price=0.46, side='ask'),
         print_at(3, price=0.50, side='bid')],
        ticker='T', symbol='BTC', window_open=WINDOW,
        fetched_at=WINDOW + timedelta(minutes=5),
        first_minute=0, last_minute=4)
    by_key = {(r['offset_minutes'], r['taker_book_side']): r for r in rows}
    assert by_key[(2, 'bid')]['min_price'] == pytest.approx(0.37)
    assert by_key[(2, 'bid')]['max_price'] == pytest.approx(0.40)
    assert by_key[(2, 'bid')]['n_prints'] == 2
    assert by_key[(2, 'ask')]['min_price'] == pytest.approx(0.46)
    assert set(by_key) == {(2, 'bid'), (2, 'ask'), (3, 'bid')}


def test_prints_outside_the_requested_minutes_are_dropped():
    rows = record_trades.fold_prints(
        [print_at(1, price=0.3), print_at(5, price=0.3), print_at(9, price=0.3)],
        ticker='T', symbol='BTC', window_open=WINDOW,
        fetched_at=WINDOW, first_minute=4, last_minute=8)
    assert [r['offset_minutes'] for r in rows] == [5]


def test_a_print_belonging_to_the_next_market_is_dropped():
    """Minute 16 is the next window's market. Folding it in here would put a
    later market's tape inside this one."""
    rows = record_trades.fold_prints(
        [print_at(16, price=0.3)], ticker='T', symbol='BTC',
        window_open=WINDOW, fetched_at=WINDOW,
        first_minute=0, last_minute=15)
    assert rows == []


def test_the_vwap_is_weighted_by_contracts():
    rows = record_trades.fold_prints(
        [print_at(1, price=0.20, contracts=1.0),
         print_at(1, price=0.60, contracts=3.0)],
        ticker='T', symbol='BTC', window_open=WINDOW, fetched_at=WINDOW,
        first_minute=0, last_minute=5)
    assert rows[0]['vwap'] == pytest.approx(0.50)
    assert rows[0]['contracts'] == pytest.approx(4.0)


def test_a_print_the_venue_did_not_price_or_stamp_is_skipped():
    unpriced = Trade(trade_id='x', ticker='T', contracts=1.0, yes_price=None,
                     no_price=None, created_time=WINDOW + timedelta(minutes=1),
                     is_block_trade=False, taker_book_side='bid')
    unstamped = Trade(trade_id='y', ticker='T', contracts=1.0, yes_price=0.3,
                      no_price=None, created_time=None, is_block_trade=False,
                      taker_book_side='bid')
    rows = record_trades.fold_prints(
        [unpriced, unstamped], ticker='T', symbol='BTC', window_open=WINDOW,
        fetched_at=WINDOW, first_minute=0, last_minute=5)
    assert rows == []


def test_an_unnamed_aggressor_side_is_recorded_rather_than_discarded():
    """`taker_book_side` is part of the event key, so it cannot be null — but a
    print the venue did not attribute is still a print, and dropping it would
    understate the tape."""
    anonymous = Trade(trade_id='z', ticker='T', contracts=2.0, yes_price=0.33,
                      no_price=None,
                      created_time=WINDOW + timedelta(minutes=1),
                      is_block_trade=False, taker_book_side=None)
    rows = record_trades.fold_prints(
        [anonymous], ticker='T', symbol='BTC', window_open=WINDOW,
        fetched_at=WINDOW, first_minute=0, last_minute=5)
    assert [r['taker_book_side'] for r in rows] == ['unknown']


def test_available_time_is_never_before_the_event():
    rows = record_trades.fold_prints(
        [print_at(2, price=0.4)], ticker='T', symbol='BTC',
        window_open=WINDOW, fetched_at=WINDOW, first_minute=0, last_minute=5)
    assert rows[0]['available_time'] >= rows[0]['event_time']


def test_a_live_row_survives_a_default_read_where_a_backfilled_one_does_not(tmp_path):
    """**The one that matters for the dataset being usable.**

    `QUALITY_LEVELS` is ('invalid', 'unvalidated', 'suspicious', 'valid') and
    `ResearchStore.read` defaults to `min_quality='valid'`. `backfill_trades`
    writes `quality='backfill'`, which ranks nowhere — so a default read of
    `venue_trades` returns ZERO rows today, which also means its own
    `_already()` resume check always comes back empty. Live rows must not have
    that property.
    """
    from core.datastore import ResearchStore

    store = ResearchStore(str(tmp_path))
    rows = record_trades.fold_prints(
        [print_at(2, price=0.4)], ticker='KXBTC15M-26OCT201200-15',
        symbol='BTC', window_open=WINDOW,
        fetched_at=WINDOW + timedelta(minutes=4),
        first_minute=0, last_minute=3)
    store.write('venue_trades', pd.DataFrame(rows))

    back = store.read('venue_trades')
    assert back is not None and len(back) == 1
    assert back.iloc[0]['taker_book_side'] == 'bid'
    assert float(back.iloc[0]['min_price']) == pytest.approx(0.4)


# ---------------------------------------------------------------------------
# the poll set
# ---------------------------------------------------------------------------

def test_a_settled_market_is_polled_through_its_grace_period():
    """The listing is bounded by close time, so a settled market vanishes from
    it immediately — while the recorder is still a minute or two behind its
    last prints. Dropping it on the close loses minutes 13 and 14 of every
    window, silently."""
    tracked = {'T': {'symbol': 'BTC', 'window_open': WINDOW,
                     'close_time': WINDOW + timedelta(minutes=15)}}
    dropped = record_trades.retire(
        tracked, now=WINDOW + timedelta(minutes=17), grace=300.0)
    assert dropped == [] and 'T' in tracked

    dropped = record_trades.retire(
        tracked, now=WINDOW + timedelta(minutes=21), grace=300.0)
    assert dropped == ['T'] and 'T' not in tracked


@pytest.mark.asyncio
async def test_a_market_without_an_open_time_is_skipped_not_guessed():
    class Client:
        async def _request(self, method, path, params=None):
            return {'markets': [{'ticker': 'A', 'open_time': None},
                                {'ticker': 'B',
                                 'open_time': '2026-10-20T12:00:00Z',
                                 'close_time': '2026-10-20T12:15:00Z'}]}

    out = await record_trades.poll_markets(Client(), now=WINDOW)
    assert set(out) == {'B'}
    assert out['B']['close_time'] == WINDOW + timedelta(minutes=15)


@pytest.mark.asyncio
async def test_a_missing_close_time_falls_back_to_the_window_length():
    class Client:
        async def _request(self, method, path, params=None):
            return {'markets': [{'ticker': 'B',
                                 'open_time': '2026-10-20T12:00:00Z'}]}

    out = await record_trades.poll_markets(Client(), now=WINDOW)
    assert out['B']['close_time'] == WINDOW + timedelta(minutes=15)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

class FakeStore:
    def __init__(self):
        self.frames = []
        self.wrote = asyncio.Event()

    def write(self, dataset, frame):
        assert dataset == 'venue_trades'
        self.frames.append(frame)
        self.wrote.set()
        return len(frame)

    @property
    def offsets(self):
        if not self.frames:
            return []
        return sorted(int(o) for f in self.frames for o in f.offset_minutes)


class FakeClient:
    """Three markets' worth of listing, one market's worth of tape."""

    def __init__(self, open_time, *, fail_first=0, prints=None):
        self.open_time = open_time
        self.fail_first = fail_first
        self.prints = prints
        self.calls = 0
        self.listings = 0

    async def _request(self, method, path, params=None):
        self.listings += 1
        return {'markets': [{
            'ticker': 'KXBTC15M-26OCT201200-15',
            'open_time': self.open_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
            'close_time': (self.open_time + timedelta(minutes=15)).strftime(
                '%Y-%m-%dT%H:%M:%SZ')}]}

    async def market_trades(self, *, ticker, since=None, limit=100):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError('too_many_requests')
        return list(self.prints)


def tape(opened, minutes, *, price=0.30, side='bid'):
    """One print in each named minute of a window that opened at `opened`."""
    return [dataclasses.replace(
        print_at(0, price=price, side=side),
        trade_id=f'{m}',
        created_time=opened + timedelta(minutes=m, seconds=5))
        for m in minutes]


def _args(**over):
    """The recorder's OWN defaults, overridden — the same rule `run_live`
    follows, so a default changed in `build_parser` is not silently bypassed
    here."""
    ns = record_trades.build_parser().parse_args([])
    for key, value in over.items():
        assert hasattr(ns, key), key
        setattr(ns, key, value)
    return ns


async def _run_until_write(store, client, *, gate=None, timeout=5.0, **over):
    task = asyncio.create_task(
        record_trades.run(_args(**over), gate=gate, store=store, client=client))
    try:
        await asyncio.wait_for(store.wrote.wait(), timeout)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    return task


@pytest.fixture
def instant_cycles(monkeypatch):
    """Collapse the inter-cycle sleep so a test runs several cycles quickly."""
    monkeypatch.setattr(record_trades, 'seconds_to_next_mark',
                        lambda *a, **k: 0.0)
    monkeypatch.setattr(record_trades, 'SERIES', {'KXBTC15M': 'BTC'})


@pytest.mark.asyncio
async def test_it_writes_only_the_minutes_that_are_complete(instant_cycles):
    """Ten minutes in, with the whole window inside the lookback: minutes 0-9
    are elapsed and fetched, minute 10 is still running. The tape carries
    prints for 0-12, and the three that are not complete must not appear."""
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    store = FakeStore()
    client = FakeClient(opened, prints=tape(opened, range(13)))
    await _run_until_write(store, client, lookback_seconds=900.0, batch_rows=1)

    assert store.offsets == list(range(10)), (
        'minute 10 was still in progress and must not be written')


@pytest.mark.asyncio
async def test_a_minute_is_written_once(instant_cycles):
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    prints = tape(opened, range(9))
    store, client = FakeStore(), FakeClient(opened, prints=prints)

    task = asyncio.create_task(record_trades.run(
        _args(lookback_seconds=900.0, batch_rows=1),
        store=store, client=client))
    try:
        await asyncio.wait_for(store.wrote.wait(), 5.0)
        # Let many more cycles run.
        await asyncio.sleep(0.2)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert client.listings > 1, 'the loop should have cycled more than once'
    # The watermark stops the FETCH, not only the write: once every complete
    # minute of a market has been recorded there is nothing left to ask the
    # venue for until the next one elapses, so a quiet cycle costs one listing.
    assert client.calls == 1
    assert store.offsets == list(range(9))


@pytest.mark.asyncio
async def test_a_failed_fetch_does_not_advance_past_the_minutes_it_missed(
        instant_cycles):
    """A transient venue error must not become a permanent hole. The backfill
    already recorded that an ERROR is not an empty tape; here the cost is
    higher, because nothing re-asks."""
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    prints = tape(opened, range(9))
    store = FakeStore()
    client = FakeClient(opened, fail_first=2, prints=prints)

    await _run_until_write(store, client, lookback_seconds=900.0, batch_rows=1)

    assert client.calls >= 3
    assert store.offsets == list(range(9)), (
        'the minutes covered by the failed fetches must still be written')


@pytest.mark.asyncio
async def test_a_listing_failure_does_not_lose_the_markets_already_tracked(
        instant_cycles):
    """The end of a window is fetched after the market has left the listing, so
    a listing error must not drop it."""
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    prints = tape(opened, range(9))
    store = FakeStore()
    client = FakeClient(opened, prints=prints)
    calls = {'n': 0}
    original = client._request

    async def flaky(method, path, params=None):
        calls['n'] += 1
        if calls['n'] > 1:
            raise RuntimeError('listing is down')
        return await original(method, path, params=params)

    client._request = flaky
    client.fail_first = 1          # nothing written on the first cycle

    await _run_until_write(store, client, lookback_seconds=900.0, batch_rows=1)
    assert store.offsets == list(range(9))


@pytest.mark.asyncio
async def test_it_waits_for_the_trading_gate(instant_cycles):
    """Every recorder defers to a decision in flight. This one is a Parquet
    write on a 60s cadence working a minute behind the tape, so deferral costs
    it nothing."""
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    prints = tape(opened, [5])
    store, client = FakeStore(), FakeClient(opened, prints=prints)
    gate = run_live.TradingGate()

    async with gate.deciding():
        task = asyncio.create_task(record_trades.run(
            _args(lookback_seconds=900.0, batch_rows=1),
            gate=gate, store=store, client=client))
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(asyncio.shield(store.wrote.wait()), 0.3)
        assert client.calls == 0

    try:
        await asyncio.wait_for(store.wrote.wait(), 5.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert store.offsets == [5]


@pytest.mark.asyncio
async def test_the_buffer_is_flushed_when_the_task_is_cancelled(instant_cycles):
    """`CancelledError` is a BaseException, so an `except Exception` handler
    never sees a shutdown — the hole already found in `record_ladder`, on a
    dataset with the same property: an unrecorded minute is gone."""
    now = datetime.now(timezone.utc)
    opened = now - timedelta(minutes=10)
    prints = tape(opened, [5])
    store, client = FakeStore(), FakeClient(opened, prints=prints)

    # A batch far larger than one cycle produces, so nothing is written until
    # the cancellation flushes it.
    task = asyncio.create_task(record_trades.run(
        _args(lookback_seconds=900.0, batch_rows=10_000),
        store=store, client=client))
    while client.calls == 0:
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert not store.frames
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.offsets == [5]


@pytest.mark.asyncio
async def test_the_heartbeat_is_stamped_even_when_nothing_printed(
        instant_cycles, monkeypatch, tmp_path):
    """A fifteen-minute market that never traded in a minute is normal, so
    silence must not read as a wedge."""
    monkeypatch.setattr(heartbeat, 'ROOT', tmp_path / 'alive')
    now = datetime.now(timezone.utc)
    client = FakeClient(now - timedelta(minutes=10), prints=[])
    store = FakeStore()

    task = asyncio.create_task(record_trades.run(
        _args(lookback_seconds=900.0), store=store, client=client))
    try:
        for _ in range(200):
            if heartbeat.age('trades') is not None:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert heartbeat.age('trades') is not None
    assert not store.frames
