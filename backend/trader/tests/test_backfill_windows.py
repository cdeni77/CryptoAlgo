"""Tests for what the backfill fetches, and under which venue.

Covers venue-scoped watermarks, interior gap detection, the incremental window in
hours, deeper history on a populated store, and write-failure reporting.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data_collection.ingest import Ingestor
from data_collection.models import OHLCVBar
from data_collection.storage import SQLiteDatabase

T0 = datetime(2026, 1, 1)


@pytest.fixture
def database(tmp_path) -> SQLiteDatabase:
    db = SQLiteDatabase(str(tmp_path / 'trading.db'))
    db.initialize()
    return db


def _bars(hours, symbol='BTC-PERP'):
    return [
        OHLCVBar(symbol=symbol, timeframe='1h', event_time=T0 + timedelta(hours=h),
                 available_time=T0 + timedelta(hours=h + 1),
                 open=100.0, high=101.0, low=99.0, close=100.0, volume=10.0)
        for h in hours
    ]


# ---------------------------------------------------------------------------
# Venue-scoped watermarks
# ---------------------------------------------------------------------------


def test_one_venues_coverage_does_not_hide_anothers_absence(database):
    """Coverage on one venue does not mark another venue as covered."""
    Ingestor(database).ingest_bars(_bars(range(50)), venue='coinbase')

    assert database.get_latest_ohlcv_time('BTC-PERP', '1h', venue='coinbase') is not None
    assert database.get_latest_ohlcv_time('BTC-PERP', '1h', venue='binance') is None, (
        'Binance reads as covered because Coinbase is, so its bars are never fetched'
    )


def test_an_unscoped_watermark_still_answers_for_any_venue(database):
    """Without a venue, the watermark answers for any venue."""
    Ingestor(database).ingest_bars(_bars(range(50)), venue='coinbase')

    assert database.get_latest_ohlcv_time('BTC-PERP', '1h') is not None


# ---------------------------------------------------------------------------
# Interior gaps
# ---------------------------------------------------------------------------


def test_an_interior_hole_is_found(database):
    """A hole between the earliest and latest bar is found."""
    Ingestor(database).ingest_bars(
        _bars(list(range(10)) + list(range(20, 30))), venue='coinbase')

    gaps = database.find_gaps(
        'ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert len(gaps) == 1
    start, end = gaps[0]
    assert start == T0 + timedelta(hours=10)
    assert end == T0 + timedelta(hours=20)


def test_a_complete_range_reports_no_gaps(database):
    """A complete range has no gaps, so a cycle does not refetch everything."""
    Ingestor(database).ingest_bars(_bars(range(30)), venue='coinbase')

    assert database.find_gaps(
        'ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30), step_seconds=3600) == []


def test_an_empty_table_is_one_whole_gap(database):
    gaps = database.find_gaps(
        'ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert gaps == [(T0, T0 + timedelta(hours=30))]


def test_leading_and_trailing_gaps_are_both_found(database):
    Ingestor(database).ingest_bars(_bars(range(10, 20)), venue='coinbase')

    gaps = database.find_gaps(
        'ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert len(gaps) == 2
    assert gaps[0][0] == T0
    assert gaps[-1][1] == T0 + timedelta(hours=30)


def test_gaps_are_venue_scoped(database):
    """A hole in one venue is not hidden by coverage on another."""
    ingestor = Ingestor(database)
    ingestor.ingest_bars(_bars(range(30)), venue='coinbase')
    ingestor.ingest_bars(_bars(range(10)), venue='binance_proxy')

    assert database.find_gaps('ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30),
                              step_seconds=3600, venue='coinbase') == []
    assert database.find_gaps('ohlcv', 'BTC-PERP', T0, T0 + timedelta(hours=30),
                              step_seconds=3600, venue='binance_proxy')


def test_an_unknown_table_is_refused(database):
    """The table name goes into SQL, so only known tables are accepted."""
    with pytest.raises(ValueError):
        database.find_gaps('drop_me', 'BIP', T0, T0 + timedelta(hours=1),
                           step_seconds=3600)


# ---------------------------------------------------------------------------
# The incremental window
# ---------------------------------------------------------------------------


def test_the_scrape_window_can_be_expressed_in_hours():
    """`--backfill-hours` parses as a float, so a 6-hour window is not rounded up to a day.
    """
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-hours', '6'])
    assert args.backfill_hours == 6.0
    assert isinstance(args.backfill_hours, float), (
        'an integral number of hours must not quantise the window'
    )


def test_the_scrape_window_accepts_a_fraction_of_a_day():
    """The window accepts a fraction of a day."""
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-days', '0.25'])
    assert args.backfill_days == pytest.approx(0.25)


def test_hours_override_days_rather_than_adding_to_them():
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-days', '400', '--backfill-hours', '6'])
    assert args.backfill_days == 400 and args.backfill_hours == 6.0
    # `main` prefers hours when both are given; this pins that order in the source.
    import inspect

    import scripts.scrape as scrape

    source = inspect.getsource(scrape.main)
    assert 'backfill_hours' in source
    assert source.index('backfill_hours') < source.index('backfill_days'), (
        'days is consulted before hours, so hours cannot be the override'
    )


def test_the_default_window_is_five_years():
    """The default first-cycle window is five years, since the first cycle defines the
    dataset.
    """
    from scripts.scrape import build_parser

    assert build_parser().parse_args([]).backfill_days == 1825


# ---------------------------------------------------------------------------
# Deeper history on a populated store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deeper_request_fetches_history_older_than_the_store(database):
    """A deeper `--backfill-days` on a populated store fetches the older missing history.

    The requested start must not be replaced by the newest stored bar. The forward
    gap is still fetched as well.
    """
    from datetime import timezone

    from data_collection.pipeline import DataPipeline

    stored = _bars(range(24))  # one day, at T0
    Ingestor(database).ingest_bars(stored, venue='coinbase')

    requested = []

    class _Pipe(DataPipeline):
        def __init__(self):  # no connector, no config
            self._database = database
            self._quality_tracker = _NullQuality()

        def _venue_name(self):
            return 'coinbase'

        def _granularity_to_seconds(self, timeframe):
            return 3_600

        async def _fetch_bars(self, symbol, timeframe, start, end):
            requested.append((start, end))
            hours = int((end - start).total_seconds() // 3_600)
            offset = int((start - T0).total_seconds() // 3_600)
            return _bars(range(offset, offset + max(hours, 0)), symbol=symbol)

        def _process_and_insert_bars(self, bars, symbol, timeframe, venue):
            if not bars:
                return 0
            return Ingestor(database).ingest_bars(bars, venue=venue)

    deep_start = T0 - timedelta(days=30)
    # `end` is past the stored day, so a forward gap exists as well as the prepend.
    await _Pipe().backfill(start=deep_start, end=T0 + timedelta(days=3),
                           symbols=['BTC-PERP'], timeframes=['1h'])

    assert requested, 'nothing was fetched at all'
    prepends = [(a, b) for a, b in requested if a < T0]
    assert prepends, (
        f'no request older than the stored history; the deeper start was '
        f'discarded. Requests made: {requested}'
    )
    first_start, first_end = prepends[0]
    assert first_start == deep_start, (
        f'prepend began at {first_start}, not the requested {deep_start}'
    )
    assert first_end < T0, 'the prepend must stop before the stored span begins'

    # The forward gap is still fetched alongside the prepend.
    assert any(a >= T0 for a, _ in requested), f'no forward fetch: {requested}'

    earliest = database.get_earliest_ohlcv_time('BTC-PERP', '1h', venue='coinbase')
    assert earliest is not None and earliest <= deep_start + timedelta(hours=1)


@pytest.mark.asyncio
async def test_a_request_inside_the_stored_span_still_fetches_nothing(database):
    """A range already covered by the store fetches nothing, so the hourly loop stays
    incremental.
    """
    from data_collection.pipeline import DataPipeline

    Ingestor(database).ingest_bars(_bars(range(48)), venue='coinbase')
    requested = []

    class _Pipe(DataPipeline):
        def __init__(self):
            self._database = database
            self._quality_tracker = _NullQuality()

        def _venue_name(self):
            return 'coinbase'

        def _granularity_to_seconds(self, timeframe):
            return 3_600

        async def _fetch_bars(self, symbol, timeframe, start, end):
            requested.append((start, end))
            return []

        def _process_and_insert_bars(self, bars, symbol, timeframe, venue):
            return 0

    # Entirely inside the stored span.
    await _Pipe().backfill(start=T0 + timedelta(hours=6),
                           end=T0 + timedelta(hours=40),
                           symbols=['BTC-PERP'], timeframes=['1h'])

    assert not requested, f'a covered range triggered fetches: {requested}'


class _NullQuality:
    def get_summary(self):
        return {}


# ---------------------------------------------------------------------------
# A write failure is not a fact about the market
# ---------------------------------------------------------------------------


def test_a_readonly_database_raises_instead_of_reporting_zero(tmp_path, monkeypatch):
    """A read-only database raises `StorageWriteError` instead of reporting zero bars
    inserted.

    A failure on every row is one connection-level problem, so it raises on the
    first one. `PRAGMA query_only` simulates read-only because root ignores chmod.
    """
    import sqlite3
    from contextlib import contextmanager

    from data_collection import storage as storage_module
    from data_collection.storage import SQLiteDatabase, StorageWriteError

    path = tmp_path / 'trading.db'
    db = SQLiteDatabase(str(path))
    db.initialize()

    original = SQLiteDatabase._get_connection

    @contextmanager
    def read_only(self):
        with original(self) as conn:
            conn.execute('PRAGMA query_only = ON')
            yield conn

    monkeypatch.setattr(SQLiteDatabase, '_get_connection',
                        contextmanager(read_only.__wrapped__))

    with pytest.raises(StorageWriteError) as caught:
        Ingestor(db).ingest_bars(_bars(range(10)), venue='coinbase')

    message = str(caught.value)
    assert 'cannot write' in message
    assert str(path) in message, 'the message must name the file that failed'
    assert 'writable' in message, 'the message must say what to check'
    # The message states how far the write got, so a partial write is not read as none.
    assert '0 of 10' in message, message

    # The same insert succeeds once writes are allowed, so the failure is the write
    # path, not the schema.
    monkeypatch.setattr(SQLiteDatabase, '_get_connection', original)
    assert Ingestor(db).ingest_bars(_bars(range(10)), venue='coinbase').inserted == 10


@pytest.mark.asyncio
async def test_fetched_but_unwritten_history_is_reported_as_a_write_failure(database, caplog):
    """A span that is fetched but not stored is logged as a write failure, not as missing
    history.
    """
    import logging

    from data_collection.pipeline import DataPipeline

    Ingestor(database).ingest_bars(_bars(range(24)), venue='coinbase')

    class _Pipe(DataPipeline):
        def __init__(self):
            self._database = database
            self._quality_tracker = _NullQuality()

        def _venue_name(self):
            return 'coinbase'

        def _granularity_to_seconds(self, timeframe):
            return 3_600

        async def _fetch_bars(self, symbol, timeframe, start, end):
            # Deep history exists and is returned.
            return _bars(range(-48, -24), symbol=symbol)

        def _process_and_insert_bars(self, bars, symbol, timeframe, venue):
            return 0  # ...and nothing is stored.

    with caplog.at_level(logging.ERROR):
        await _Pipe().backfill(start=T0 - timedelta(days=2), end=T0 + timedelta(days=2),
                               symbols=['BTC-PERP'], timeframes=['1h'])

    errors = ' '.join(r.message for r in caplog.records if r.levelno >= logging.ERROR)
    assert 'write failure' in errors, (
        f'a fetched-but-unwritten span was not reported as a write failure: {errors}'
    )
    assert 'not missing history' in errors


# ---------------------------------------------------------------------------
# Empty batches, and recovering from them
# ---------------------------------------------------------------------------

def test_an_empty_batch_is_retried_with_backoff_not_immediately():
    """Empty batches retry with growing backoff, since an instant retry cannot clear a rate
    limit.
    """
    import inspect

    from data_collection.coinbase_connector import (
        EMPTY_BATCH_BACKOFF_SECONDS, CoinbaseRESTClient,
    )

    assert len(EMPTY_BATCH_BACKOFF_SECONDS) >= 3, 'one retry is not a strategy'
    assert list(EMPTY_BATCH_BACKOFF_SECONDS) == sorted(EMPTY_BATCH_BACKOFF_SECONDS), (
        'the waits must grow, or the later attempts add nothing'
    )

@pytest.mark.asyncio
async def test_an_empty_batch_is_retried_with_growing_waits():
    """Counts the attempts and waits: an empty batch retries through the whole backoff
    ladder.
    """
    import asyncio

    from data_collection.coinbase_connector import (
        EMPTY_BATCH_BACKOFF_SECONDS, CoinbaseRESTClient,
    )

    calls: list[tuple[int, int]] = []
    waits: list[float] = []

    async def always_empty(method, path, params=None, **kwargs):
        calls.append((int(params['start']), int(params['end'])))
        return 200, {'candles': []}

    async def record_sleep(seconds):
        waits.append(seconds)

    client = CoinbaseRESTClient(api_key=None, api_secret=None)
    client._request = always_empty                      # noqa: SLF001
    original, asyncio.sleep = asyncio.sleep, record_sleep
    try:
        start = datetime(2025, 1, 1)
        await client.get_candles_range(
            'BTC-USD', '1m', start, start + timedelta(minutes=600))
    finally:
        asyncio.sleep = original

    assert len(calls) > len(EMPTY_BATCH_BACKOFF_SECONDS), (
        f'{len(calls)} requests for {len(EMPTY_BATCH_BACKOFF_SECONDS)} configured '
        f'backoffs: the empty batch was not retried through the whole ladder'
    )
    assert waits, 'it retried without waiting at all'
    assert waits[:len(EMPTY_BATCH_BACKOFF_SECONDS)] == list(EMPTY_BATCH_BACKOFF_SECONDS), (
        f'waited {waits[:4]}, expected {list(EMPTY_BATCH_BACKOFF_SECONDS)}'
    )


@pytest.mark.asyncio
async def test_a_window_that_never_answers_is_recorded_on_the_client():
    """A window that never returns data is recorded in `last_skipped_windows` for the
    caller.
    """
    from data_collection.coinbase_connector import CoinbaseRESTClient

    async def always_empty(method, path, params=None, **kwargs):
        return 200, {'candles': []}

    async def no_wait(seconds):
        return None

    import asyncio

    client = CoinbaseRESTClient(api_key=None, api_secret=None)
    client._request = always_empty                      # noqa: SLF001
    original, asyncio.sleep = asyncio.sleep, no_wait
    try:
        start = datetime(2025, 1, 1)
        bars = await client.get_candles_range(
            'BTC-USD', '1m', start, start + timedelta(minutes=600))
    finally:
        asyncio.sleep = original

    assert bars == []
    assert client.last_skipped_windows, (
        'nothing came back and nothing was recorded, so a caller cannot tell a '
        'quiet market from a venue that stopped answering'
    )


def test_gap_detection_finds_a_multi_hour_hole(tmp_path):
    """Finds a 300-minute block, the shape a failed batch leaves."""
    import sqlite3
    from datetime import datetime, timedelta

    from scripts.scrape import find_gaps

    database = tmp_path / 'gaps.db'
    connection = sqlite3.connect(database)
    connection.execute(
        'CREATE TABLE ohlcv (symbol TEXT, timeframe TEXT, venue TEXT, '
        'event_time TEXT)')
    start = datetime(2025, 10, 25, 12, 0)
    rows = [
        ('BTC-USD', '1m', 'coinbase_spot', (start + timedelta(minutes=i)).isoformat(' '))
        for i in range(600)
        if not (194 <= i < 494) and i != 550  # a 300-min hole, and one lone minute
    ]
    connection.executemany('INSERT INTO ohlcv VALUES (?,?,?,?)', rows)
    connection.commit()
    connection.close()

    gaps = find_gaps(str(database), 'BTC-USD', '1m', 'coinbase_spot', min_minutes=2)
    assert len(gaps) == 1
    span = int((gaps[0][1] - gaps[0][0]).total_seconds() // 60)
    assert span == 300
    assert gaps[0][0] == datetime(2025, 10, 25, 15, 14)


def test_min_minutes_controls_whether_a_lone_missing_minute_is_a_gap():
    """`min_minutes` is the threshold, and 1 includes isolated single missing minutes.
    """
    import sqlite3
    import tempfile
    from datetime import datetime, timedelta
    from pathlib import Path

    from scripts.scrape import find_gaps

    with tempfile.TemporaryDirectory() as directory:
        database = Path(directory) / 'one.db'
        connection = sqlite3.connect(database)
        connection.execute(
            'CREATE TABLE ohlcv (symbol TEXT, timeframe TEXT, venue TEXT, '
            'event_time TEXT)')
        start = datetime(2025, 1, 1)
        rows = [('BTC-USD', '1m', 'coinbase_spot',
                 (start + timedelta(minutes=i)).isoformat(' '))
                for i in range(100) if i != 50]
        connection.executemany('INSERT INTO ohlcv VALUES (?,?,?,?)', rows)
        connection.commit()
        connection.close()

        assert find_gaps(str(database), 'BTC-USD', '1m', 'coinbase_spot',
                         min_minutes=2) == []
        assert len(find_gaps(str(database), 'BTC-USD', '1m', 'coinbase_spot',
                             min_minutes=1)) == 1


@pytest.mark.asyncio
async def test_a_forced_backfill_fetches_an_interior_range():
    """`force=True` fetches an interior range that the resuming path skips.

    Without `force`, `backfill` resumes from the store's boundaries, so a hole in the
    middle is never fetched. `--fill-gaps` relies on `force`.
    """
    from datetime import datetime as dt

    from data_collection.pipeline import DataPipeline

    fetched: list[tuple] = []

    async def fake_fetch(self, symbol, timeframe, start, end):
        fetched.append((start, end))
        return []

    class FakeDB:
        def get_latest_ohlcv_time(self, *a, **k):
            return dt(2026, 8, 23, 4, 17)

        def get_earliest_ohlcv_time(self, *a, **k):
            return dt(2021, 8, 24, 4, 18)

    class FakeConfig:
        symbols = ['BTC-USD']
        timeframes = ['1m']
        backfill_days = 1825

    pipeline = DataPipeline.__new__(DataPipeline)
    pipeline._database = FakeDB()
    pipeline.config = FakeConfig()
    pipeline._venue_name = lambda: 'coinbase_spot'
    pipeline._granularity_to_seconds = lambda tf: 60
    pipeline._quality_tracker = type('Q', (), {'get_summary': lambda s: {}})()
    pipeline._process_and_insert_bars = lambda *a, **k: True

    hole_start, hole_end = dt(2023, 5, 3, 5, 59), dt(2023, 5, 3, 6, 1)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DataPipeline, '_fetch_bars', fake_fetch)

        await pipeline.backfill(start=hole_start, end=hole_end,
                                symbols=['BTC-USD'], timeframes=['1m'])
        assert fetched == [], (
            'the resuming path fetched an interior range; if that is now intended, '
            'this test no longer describes the defect'
        )

        await pipeline.backfill(start=hole_start, end=hole_end,
                                symbols=['BTC-USD'], timeframes=['1m'], force=True)
        assert fetched == [(hole_start, hole_end)], (
            f'a forced backfill must fetch exactly the range asked for; got {fetched}'
        )


def test_gaps_are_coalesced_into_requests_a_batch_can_cover():
    """`coalesce` merges gaps that fit in one request, and keeps apart gaps that do not.
    """
    from datetime import datetime as dt

    from scripts.scrape import coalesce

    base = dt(2025, 1, 1)
    spaced = [(base + timedelta(minutes=301 * i), base + timedelta(minutes=301 * i + 1))
              for i in range(5)]
    assert len(coalesce(spaced, span_minutes=299)) == 5, (
        'holes 301 apart cannot share a 300-minute request'
    )

    clustered = [(base + timedelta(minutes=10 * i), base + timedelta(minutes=10 * i + 2))
                 for i in range(20)]
    assert len(coalesce(clustered, span_minutes=299)) == 1

    assert coalesce([], span_minutes=299) == []
