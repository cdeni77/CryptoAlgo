"""Storage ingest invariants.

`data_collection.ingest` is the only door into the database, and the only place
that sets `quality`. Before it existed there were two ingest paths: OHLCV went
through the validated pipeline, while funding and open interest were inserted
directly. Open interest was constructed with `quality=DataQuality.VALID` written
in by hand, so the column claimed the data was checked when nothing had checked
it, and `DataValidator.validate_open_interest` had zero callers.

These tests pin the properties that stop that from recurring.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data_collection.ingest import Ingestor, unvalidated_row_counts
from data_collection.models import DataQuality, FundingRate, OHLCVBar, OpenInterest
from data_collection.storage import SQLiteDatabase

T0 = datetime(2026, 1, 1)


@pytest.fixture
def database(tmp_path) -> SQLiteDatabase:
    db = SQLiteDatabase(str(tmp_path / 'trading.db'))
    db.initialize()
    return db


@pytest.fixture
def ingestor(database) -> Ingestor:
    return Ingestor(database)


def _bar(hour: int, close: float = 100.0) -> OHLCVBar:
    return OHLCVBar(
        event_time=T0 + timedelta(hours=hour),
        available_time=T0 + timedelta(hours=hour + 1),
        symbol='BIP', timeframe='1h',
        open=close, high=close * 1.01, low=close * 0.99, close=close, volume=1000.0,
    )


def _funding(hour: int, rate: float) -> FundingRate:
    return FundingRate(
        event_time=T0 + timedelta(hours=hour),
        available_time=T0 + timedelta(hours=hour),
        symbol='BIP', rate=rate, mark_price=100.0, index_price=100.0,
    )


def _open_interest(hour: int, contracts: float) -> OpenInterest:
    return OpenInterest(
        event_time=T0 + timedelta(hours=hour),
        available_time=T0 + timedelta(hours=hour),
        symbol='BIP', open_interest_contracts=contracts, open_interest_usd=contracts * 100,
    )


def test_quality_defaults_to_unvalidated():
    """The default must not claim the data was checked.

    This is what let hand-constructed records masquerade as validated ones.
    """
    assert _bar(0).quality is DataQuality.UNVALIDATED
    assert _funding(0, 0.0001).quality is DataQuality.UNVALIDATED
    assert _open_interest(0, 100.0).quality is DataQuality.UNVALIDATED


def test_ingest_stamps_quality_and_venue(ingestor):
    result = ingestor.ingest_bars([_bar(i, 100 + i * 0.1) for i in range(10)], venue='coinbase')

    assert result.inserted == 10
    assert result.rejected == 0
    assert all(bar.quality is DataQuality.VALID for bar in result.stored)
    assert all(bar.venue == 'coinbase' for bar in result.stored)


def test_inconsistent_bar_is_rejected(ingestor):
    broken = _bar(20)
    broken.high, broken.low = 90.0, 110.0        # high below low

    result = ingestor.ingest_bars([broken], venue='coinbase')

    assert result.inserted == 0
    assert result.rejected == 1


def test_rejected_bar_does_not_poison_the_next_comparison(ingestor):
    """A dropped bar must not become the reference for the following bar.

    The previous implementation advanced `prev_bar` unconditionally, so one
    corrupt record became the baseline for the next gap and price-jump check and
    cascaded into rejecting healthy bars after it.
    """
    corrupt = _bar(31)
    corrupt.high, corrupt.low = 1.0, 2.0

    result = ingestor.ingest_bars([_bar(30), corrupt, _bar(32, 100.5)], venue='coinbase')

    assert result.rejected == 1
    assert result.inserted == 2, 'a healthy bar was rejected because of its neighbour'


def test_open_interest_is_validated(ingestor):
    """validate_open_interest had no callers; a negative figure went straight in."""
    result = ingestor.ingest_open_interest(
        [_open_interest(1, 5000.0), _open_interest(2, -100.0)], venue='bybit'
    )

    assert result.inserted == 1
    assert result.rejected == 1


def test_implausible_funding_is_flagged_not_silently_accepted(ingestor):
    """A 50%-per-hour rate is stored, but marked, so a reader can exclude it.

    The validator flags rather than drops, which is the right call for storage —
    but it means the quality column has to survive into the research store, or
    the carry features average it in regardless.
    """
    result = ingestor.ingest_funding([_funding(1, 0.00001), _funding(2, 0.5)], venue='coinbase')

    assert result.inserted == 2
    assert result.suspicious == 1
    qualities = {rate.quality for rate in result.stored}
    assert DataQuality.SUSPICIOUS in qualities


def test_nothing_reaches_storage_unvalidated(ingestor, database):
    ingestor.ingest_bars([_bar(i) for i in range(5)], venue='coinbase')
    ingestor.ingest_funding([_funding(1, 0.00001)], venue='coinbase')
    ingestor.ingest_open_interest([_open_interest(1, 5000.0)], venue='bybit')

    assert all(count == 0 for count in unvalidated_row_counts(database).values())


def test_venue_is_persisted_per_row(ingestor, database):
    """The backfill blends venues on purpose; the boundary must be recoverable."""
    ingestor.ingest_bars([_bar(i) for i in range(10)], venue='coinbase')
    ingestor.ingest_bars([_bar(i, 200.0) for i in range(50, 55)], venue='binance')

    connection = sqlite3.connect(database.db_path)
    try:
        counts = dict(
            connection.execute('SELECT venue, COUNT(*) FROM ohlcv GROUP BY venue').fetchall()
        )
    finally:
        connection.close()

    assert counts == {'coinbase': 10, 'binance': 5}


def test_research_store_excludes_flagged_data_by_default(ingestor, database, tmp_path):
    """Quality survives the migration, and `read` filters on it.

    This is the loop closing: validated at ingest, flagged in storage, excluded
    from the feature build unless explicitly requested.
    """
    from core.datastore import ResearchStore, from_sqlite

    ingestor.ingest_funding([_funding(1, 0.00001), _funding(2, 0.5)], venue='coinbase')

    store = ResearchStore(tmp_path / 'research')
    from_sqlite(store, database.db_path, venue='coinbase')

    clean = store.read('funding')
    everything = store.read('funding', min_quality=None)

    assert len(clean) == 1, 'suspicious funding leaked into a default read'
    assert len(everything) == 2
    assert set(everything['quality']) == {'valid', 'suspicious'}
