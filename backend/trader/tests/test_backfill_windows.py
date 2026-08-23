"""What the backfill decides to fetch, and under which venue.

Three defects in one code path, all of which made a scrape quietly incomplete:

- Watermarks were keyed on `(symbol, timeframe)` with no venue. So once Coinbase
  covered a span, the CCXT prepend was skipped and the reference-venue bars the
  cross-venue basis and lead-lag features need were never collected at all.
- Missing windows were built from MIN and MAX only, giving `(start, MIN)` and
  `(MAX, end)`. Any interior hole — a failed request mid-history — was
  permanently bracketed and invisible, and the features were computed straight
  across it.
- `INCREMENTAL_BACKFILL_HOURS=6` was quantised with `ceil(hours / 24)`, so every
  value from 1 to 24 fetched a whole day: four times the API calls for the same
  data.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from data_collection.ingest import Ingestor
from data_collection.models import FundingRate, OHLCVBar
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


def _rates(hours, symbol='BIP'):
    return [
        FundingRate(symbol=symbol, event_time=T0 + timedelta(hours=h),
                    available_time=T0 + timedelta(hours=h),
                    rate=1e-5, mark_price=1.0, index_price=1.0)
        for h in hours
    ]


# ---------------------------------------------------------------------------
# Venue-scoped watermarks
# ---------------------------------------------------------------------------


def test_one_venues_coverage_does_not_hide_anothers_absence(database):
    """The defect that emptied every cross-venue feature."""
    Ingestor(database).ingest_bars(_bars(range(50)), venue='coinbase')

    assert database.get_latest_ohlcv_time('BTC-PERP', '1h', venue='coinbase') is not None
    assert database.get_latest_ohlcv_time('BTC-PERP', '1h', venue='binance') is None, (
        'Binance reads as covered because Coinbase is, so its bars are never fetched'
    )


def test_an_unscoped_watermark_still_answers_for_any_venue(database):
    """Omitting the venue keeps the old meaning, for callers that want it."""
    Ingestor(database).ingest_bars(_bars(range(50)), venue='coinbase')

    assert database.get_latest_ohlcv_time('BTC-PERP', '1h') is not None


@pytest.mark.parametrize('getter,venue_present', [
    ('get_latest_funding_time', 'coinbase'),
    ('get_earliest_funding_time', 'coinbase'),
])
def test_funding_watermarks_are_venue_scoped(database, getter, venue_present):
    Ingestor(database).ingest_funding(_rates(range(30)), venue=venue_present)

    assert getattr(database, getter)('BIP', venue=venue_present) is not None
    assert getattr(database, getter)('BIP', venue='binance_proxy') is None


# ---------------------------------------------------------------------------
# Interior gaps
# ---------------------------------------------------------------------------


def test_an_interior_hole_is_found(database):
    """MIN/MAX bracketing could not see this, so it stayed a hole forever."""
    Ingestor(database).ingest_funding(
        _rates(list(range(10)) + list(range(20, 30))), venue='coinbase')

    gaps = database.find_gaps(
        'funding_rates', 'BIP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert len(gaps) == 1
    start, end = gaps[0]
    assert start == T0 + timedelta(hours=10)
    assert end == T0 + timedelta(hours=20)


def test_a_complete_range_reports_no_gaps(database):
    """Otherwise every cycle refetches everything."""
    Ingestor(database).ingest_funding(_rates(range(30)), venue='coinbase')

    assert database.find_gaps(
        'funding_rates', 'BIP', T0, T0 + timedelta(hours=30), step_seconds=3600) == []


def test_an_empty_table_is_one_whole_gap(database):
    gaps = database.find_gaps(
        'funding_rates', 'BIP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert gaps == [(T0, T0 + timedelta(hours=30))]


def test_leading_and_trailing_gaps_are_both_found(database):
    Ingestor(database).ingest_funding(_rates(range(10, 20)), venue='coinbase')

    gaps = database.find_gaps(
        'funding_rates', 'BIP', T0, T0 + timedelta(hours=30), step_seconds=3600)

    assert len(gaps) == 2
    assert gaps[0][0] == T0
    assert gaps[-1][1] == T0 + timedelta(hours=30)


def test_gaps_are_venue_scoped(database):
    """A hole in the reference venue must not be hidden by the trade venue."""
    ingestor = Ingestor(database)
    ingestor.ingest_funding(_rates(range(30)), venue='coinbase')
    ingestor.ingest_funding(_rates(range(10)), venue='binance_proxy')

    assert database.find_gaps('funding_rates', 'BIP', T0, T0 + timedelta(hours=30),
                              step_seconds=3600, venue='coinbase') == []
    assert database.find_gaps('funding_rates', 'BIP', T0, T0 + timedelta(hours=30),
                              step_seconds=3600, venue='binance_proxy')


def test_an_unknown_table_is_refused(database):
    """The table name is interpolated into SQL, so it must be a known one."""
    with pytest.raises(ValueError):
        database.find_gaps('drop_me', 'BIP', T0, T0 + timedelta(hours=1),
                           step_seconds=3600)


# ---------------------------------------------------------------------------
# The incremental window
# ---------------------------------------------------------------------------


def test_the_scrape_window_can_be_expressed_in_hours():
    """`ceil(hours / 24)` once made a 6-hour incremental fetch pull 24.

    Asserted by parsing the actual parser rather than by grepping the source for a
    quoted flag name: the earlier version searched for `"--backfill-hours"` with
    double quotes and broke on a module that uses single ones, which is a test
    failing on its own formatting assumption rather than on the behaviour.
    """
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-hours', '6'])
    assert args.backfill_hours == 6.0
    assert isinstance(args.backfill_hours, float), (
        'an integral number of hours must not quantise the window'
    )


def test_the_scrape_window_accepts_a_fraction_of_a_day():
    """The hourly cycle wants 0.25 days, not 1."""
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-days', '0.25'])
    assert args.backfill_days == pytest.approx(0.25)


def test_hours_override_days_rather_than_adding_to_them():
    from scripts.scrape import build_parser

    args = build_parser().parse_args(['--backfill-days', '400', '--backfill-hours', '6'])
    assert args.backfill_days == 400 and args.backfill_hours == 6.0
    # `main` prefers hours when both are given; this pins the parser half of it.
    import inspect

    import scripts.scrape as scrape

    source = inspect.getsource(scrape.main)
    assert 'backfill_hours' in source
    assert source.index('backfill_hours') < source.index('backfill_days'), (
        'days is consulted before hours, so hours cannot be the override'
    )


def test_the_default_window_is_five_years():
    """An empty store's first cycle *defines* the dataset.

    `--backfill-days` in compose is the FIRST cycle only; every cycle after it
    uses the incremental window. It was 30 once, which at this window size is far
    too little to fit anything, and nothing in the log said why.
    """
    from scripts.scrape import build_parser

    assert build_parser().parse_args([]).backfill_days == 1825


# ---------------------------------------------------------------------------
# Deeper history on a populated store
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_deeper_request_fetches_history_older_than_the_store(database):
    """`--backfill-days 1825` against a 400-day store must fetch the 1,425 missing.

    It did not. `append_start` was set to the newest stored bar whenever anything
    was stored, so the requested `start` was discarded and the scrape fetched only
    the forward gap — then logged "already up to date". Every deeper request on a
    populated store was a no-op, which is exactly the situation once a loop has
    been running: the only way to get more history was to delete the database.

    The prepend branch had been removed along with CCXT, and correctly so for the
    reason it existed: it filled the span before a contract was listed with
    another exchange's bars for the same underlying. But the defect there was the
    *source*, not the direction. This asserts the direction is back.
    """
    from datetime import timezone

    from data_collection.pipeline import DataPipeline

    stored = _bars(range(24))                      # one day, at T0
    Ingestor(database).ingest_bars(stored, venue='coinbase')

    requested = []

    class _Pipe(DataPipeline):
        def __init__(self):                        # no connector, no config
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
    # `end` past the stored day, so there is a genuine forward gap too and the
    # test can show the prepend did not replace the append.
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

    # The forward gap is still fetched, so restoring the prepend did not replace
    # the append.
    assert any(a >= T0 for a, _ in requested), f'no forward fetch: {requested}'

    earliest = database.get_earliest_ohlcv_time('BTC-PERP', '1h', venue='coinbase')
    assert earliest is not None and earliest <= deep_start + timedelta(hours=1)


@pytest.mark.asyncio
async def test_a_request_inside_the_stored_span_still_fetches_nothing(database):
    """The fix must not turn every run into a full re-fetch.

    An incremental cycle asks for a few hours against a store that already covers
    them. That has to stay a no-op, or the hourly loop re-downloads its whole
    history every hour.
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
    """The defect that nearly concluded Coinbase has no history before 2025.

    `insert_ohlcv_batch` caught every per-bar exception and returned a count. On a
    read-only file that produced 34,060 identical ERROR lines and a return of 0,
    and a caller cannot distinguish 0-because-nothing-was-served from
    0-because-nothing-could-be-written. The scrape logged the second as the first
    — "the request pre-dates the contract, so nothing is missing" — while 34,060
    fetched bars sat in memory unwritten.

    A failure affecting every row is one connection-level problem, not N row
    problems, so it raises on the first occurrence with the cause attached.

    Read-only is induced with `PRAGMA query_only` rather than `chmod`, because
    these tests run as root and root ignores file permissions — the chmod version
    of this test passed against the unfixed code.
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
    # And it names how far it got, so a partial write is not mistaken for none.
    assert '0 of 10' in message, message

    # The same insert succeeds once writes are allowed again, so this is
    # exercising the write path rather than a schema problem.
    monkeypatch.setattr(SQLiteDatabase, '_get_connection', original)
    assert Ingestor(db).ingest_bars(_bars(range(10)), venue='coinbase').inserted == 10


@pytest.mark.asyncio
async def test_fetched_but_unwritten_history_is_reported_as_a_write_failure(database, caplog):
    """The log has to distinguish "no data" from "data we threw away".

    When a prepend fetches bars and stores none of them, that is an error about
    this process, not a discovery about the venue. Asserted on the message,
    because the message is the only thing a human reads at 3am during a scrape.
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
            return 0                      # ...and nothing is stored.

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
    """A rate limit answers an instant retry exactly as it answered the first ask.

    The previous version retried once with no delay, which cannot recover the
    most likely cause of an empty batch. On a five-year backfill of ~8,700
    batches per symbol, two gave up that way and left two five-hour holes.
    """
    import inspect

    from data_collection.coinbase_connector import (
        EMPTY_BATCH_BACKOFF_SECONDS, CoinbaseRESTClient,
    )

    assert len(EMPTY_BATCH_BACKOFF_SECONDS) >= 3, 'one retry is not a strategy'
    assert list(EMPTY_BATCH_BACKOFF_SECONDS) == sorted(EMPTY_BATCH_BACKOFF_SECONDS), (
        'the waits must grow, or the later attempts add nothing'
    )
    source = inspect.getsource(CoinbaseRESTClient.get_candles_range)
    assert 'EMPTY_BATCH_BACKOFF_SECONDS' in source
    assert 'asyncio.sleep(delay)' in source, 'it retries without waiting'


def test_skipped_windows_are_recorded_on_the_client():
    """So a caller can act on them rather than parse a log line."""
    import inspect

    from data_collection.coinbase_connector import CoinbaseRESTClient

    source = inspect.getsource(CoinbaseRESTClient.get_candles_range)
    assert 'self.last_skipped_windows' in source


def test_gap_detection_finds_a_multi_hour_hole(tmp_path):
    """The shape of the real failure: a 300-minute block from one failed batch."""
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
        if not (194 <= i < 494) and i != 550        # a 300-min hole, and one lone minute
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
    """`min_minutes` is the threshold, and 1 includes isolated single minutes.

    This test used to be called
    `test_a_single_missing_minute_is_not_a_gap_worth_refetching`, asserting that
    a lone hole is "a minute in which nothing traded". That premise was false —
    86% of the isolated holes in the real store were a client-side pagination
    off-by-one, spaced exactly 301 minutes apart and co-occurring across symbols
    177x more often than chance allows. The mechanism the body checks is correct
    and unchanged; only the claim in the name was wrong, and the default moved
    from 2 to 1 as a result.
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
