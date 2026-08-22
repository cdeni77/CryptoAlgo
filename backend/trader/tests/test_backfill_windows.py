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


def test_the_incremental_window_is_expressed_in_hours():
    """`ceil(hours / 24)` made 6 hours fetch 24."""
    import sys

    from scripts.live_orchestrator import parse_args

    argv = sys.argv
    sys.argv = ['live_orchestrator', '--incremental-backfill-hours', '6']
    try:
        args = parse_args()
    finally:
        sys.argv = argv

    assert args.incremental_backfill_hours == 6
    # And the scrape step takes hours, so nothing rounds it back up to a day.
    import inspect

    from scripts.live_orchestrator import _scrape

    assert 'backfill-hours' in inspect.getsource(_scrape)


def test_run_pipeline_accepts_a_fractional_window():
    """The hourly cycle needs sub-day windows."""
    import sys

    from scripts.run_pipeline import main  # noqa: F401  (import guard)

    # Parse only; running it would hit the network.
    import argparse
    import scripts.run_pipeline as rp

    source = inspect_source = __import__('inspect').getsource(rp)
    assert '"--backfill-hours"' in source
    assert 'type=float' in source


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
