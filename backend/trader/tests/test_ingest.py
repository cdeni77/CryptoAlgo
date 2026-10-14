"""Tests for storage ingest invariants.

`data_collection.ingest` is the only path into the database and the only place that sets
`quality`. Records default to `DataQuality.UNVALIDATED`.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import pytest

from data_collection.ingest import Ingestor, unvalidated_row_counts
from data_collection.models import DataQuality, OHLCVBar
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


def test_quality_defaults_to_unvalidated():
    """The default quality does not claim the data was checked."""
    assert _bar(0).quality is DataQuality.UNVALIDATED


def test_ingest_stamps_quality_and_venue(ingestor):
    result = ingestor.ingest_bars([_bar(i, 100 + i * 0.1) for i in range(10)], venue='coinbase')

    assert result.inserted == 10
    assert result.rejected == 0
    assert all(bar.quality is DataQuality.VALID for bar in result.stored)
    assert all(bar.venue == 'coinbase' for bar in result.stored)


def test_inconsistent_bar_is_rejected(ingestor):
    broken = _bar(20)
    broken.high, broken.low = 90.0, 110.0  # high below low

    result = ingestor.ingest_bars([broken], venue='coinbase')

    assert result.inserted == 0
    assert result.rejected == 1


def test_rejected_bar_does_not_poison_the_next_comparison(ingestor):
    """A rejected bar is not used as the reference for the next bar's checks."""
    corrupt = _bar(31)
    corrupt.high, corrupt.low = 1.0, 2.0

    result = ingestor.ingest_bars([_bar(30), corrupt, _bar(32, 100.5)], venue='coinbase')

    assert result.rejected == 1
    assert result.inserted == 2, 'a healthy bar was rejected because of its neighbour'


def test_nothing_reaches_storage_unvalidated(ingestor, database):
    ingestor.ingest_bars([_bar(i) for i in range(5)], venue='coinbase')

    assert all(count == 0 for count in unvalidated_row_counts(database).values())


def test_venue_is_persisted_per_row(ingestor, database):
    """Venue is stored per row, so blended backfill sources stay recoverable."""
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


def test_two_venues_coexist_for_the_same_bar(tmp_path):
    """Bars from two venues for the same instant both survive; the unique key includes venue."""
    from datetime import datetime, timedelta, timezone

    from data_collection.models import DataQuality, OHLCVBar
    from data_collection.storage import SQLiteDatabase

    db = SQLiteDatabase(str(tmp_path / 'venues.db'))
    db.initialize()

    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def bar(venue: str, close: float) -> OHLCVBar:
        return OHLCVBar(
            symbol='BIP', timeframe='1h', venue=venue,
            event_time=stamp, available_time=stamp + timedelta(hours=1),
            open=close, high=close * 1.01, low=close * 0.99, close=close,
            volume=1_000.0, quality=DataQuality.VALID,
        )

    assert db.insert_ohlcv(bar('coinbase', 60_000.0))
    assert db.insert_ohlcv(bar('binance', 60_030.0))

    rows = _bars_at(db, 'BIP', stamp)

    assert len(rows) == 2, f'one venue overwrote the other: {rows}'
    assert {r['venue'] for r in rows} == {'coinbase', 'binance'}
    assert {round(r['close']) for r in rows} == {60_000, 60_030}


def test_the_same_venue_still_deduplicates(tmp_path):
    """Within one venue, a rewritten bar updates its row instead of duplicating it."""
    from datetime import datetime, timedelta, timezone

    from data_collection.models import DataQuality, OHLCVBar
    from data_collection.storage import SQLiteDatabase

    db = SQLiteDatabase(str(tmp_path / 'dedupe.db'))
    db.initialize()
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for close in (60_000.0, 60_500.0):
        db.insert_ohlcv(OHLCVBar(
            symbol='BIP', timeframe='1h', venue='coinbase',
            event_time=stamp, available_time=stamp + timedelta(hours=1),
            open=close, high=close, low=close, close=close,
            volume=1_000.0, quality=DataQuality.VALID,
        ))

    rows = _bars_at(db, 'BIP', stamp)

    assert len(rows) == 1
    assert round(rows[0]['close']) == 60_500, 'the revision did not replace the original'


def test_a_legacy_database_is_rebuilt_onto_the_venue_key(tmp_path):
    """A database keyed without venue is rebuilt onto the venue key.

    SQLite cannot alter a UNIQUE constraint, so `initialize()` rebuilds the table.
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    from data_collection.models import DataQuality, OHLCVBar
    from data_collection.storage import SQLiteDatabase

    path = tmp_path / 'legacy.db'
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # The old schema, as it was in the part that matters.
    con = sqlite3.connect(str(path))
    con.execute("""
        CREATE TABLE ohlcv (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL, timeframe TEXT NOT NULL,
            venue TEXT DEFAULT 'unknown',
            event_time TIMESTAMP NOT NULL, available_time TIMESTAMP NOT NULL,
            open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
            close REAL NOT NULL, volume REAL NOT NULL,
            quote_volume REAL, trade_count INTEGER,
            quality TEXT DEFAULT 'valid', quality_notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, event_time)
        )
    """)
    con.execute(
        "INSERT INTO ohlcv (symbol, timeframe, venue, event_time, available_time, "
        "open, high, low, close, volume) VALUES (?,?,?,?,?,?,?,?,?,?)",
        ('BIP', '1h', 'coinbase', stamp, stamp, 1.0, 1.0, 1.0, 60_000.0, 1.0),
    )
    con.commit()
    con.close()

    db = SQLiteDatabase(str(path))
    db.initialize()

    # The pre-existing row survived the rebuild.
    rows = _bars_at(db, 'BIP', stamp)
    assert len(rows) == 1 and rows[0]['venue'] == 'coinbase'

    # The second venue can now be written beside it.
    db.insert_ohlcv(OHLCVBar(
        symbol='BIP', timeframe='1h', venue='binance',
        event_time=stamp, available_time=stamp + timedelta(hours=1),
        open=60_030.0, high=60_030.0, low=60_030.0, close=60_030.0,
        volume=1.0, quality=DataQuality.VALID,
    ))

    rows = _bars_at(db, 'BIP', stamp)
    assert {r['venue'] for r in rows} == {'coinbase', 'binance'}


def _bars_at(db, symbol: str, stamp) -> list[dict]:
    with db._get_connection() as conn:
        return [
            dict(row)
            for row in conn.execute(
                'SELECT venue, close FROM ohlcv WHERE symbol = ? AND event_time = ?',
                (symbol, stamp),
            ).fetchall()
        ]


def test_no_module_imports_ccxt():
    """No module imports CCXT; spot bars come from the native Coinbase client."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    offenders = []
    for path in root.rglob('*.py'):
        if '__pycache__' in path.parts or path.name == Path(__file__).name:
            continue
        text = path.read_text()
        if 'import ccxt' in text or 'CCXTConnector' in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f'CCXT survives in: {offenders}'

    # Both requirement files must also be free of ccxt.
    for requirements in (root / 'requirements.txt',
                         root.parent / 'api' / 'requirements.txt'):
        if not requirements.exists():
            continue
        declared = [
            line.strip() for line in requirements.read_text().splitlines()
            if line.strip().lower().startswith('ccxt')
        ]
        assert not declared, f'{requirements.name} still declares {declared}'


def test_the_minute_bar_dataset_rejects_a_flagged_row_by_default():
    """The minute-bar dataset excludes flagged rows by default."""
    import pandas as pd
    from core.datastore import DEFAULT_MIN_QUALITY, ResearchStore
    import tempfile

    with tempfile.TemporaryDirectory() as root:
        store = ResearchStore(root)
        when = pd.Timestamp('2026-01-01', tz='UTC')
        store.write('minute_bars', pd.DataFrame([
            {'venue': 'coinbase_spot', 'symbol': 'BTC-USD', 'event_time': when,
             'available_time': when + pd.Timedelta(minutes=1), 'quality': 'valid',
             'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0,
             'quote_volume': None, 'trade_count': 1},
            {'venue': 'coinbase_spot', 'symbol': 'BTC-USD',
             'event_time': when + pd.Timedelta(minutes=1),
             'available_time': when + pd.Timedelta(minutes=2), 'quality': 'suspicious',
             'open': 99.0, 'high': 99.0, 'low': 99.0, 'close': 99.0, 'volume': 1.0,
             'quote_volume': None, 'trade_count': 1},
        ]))
        assert DEFAULT_MIN_QUALITY == 'valid'
        clean = store.read('minute_bars', venue='coinbase_spot', symbols=['BTC-USD'])
        assert len(clean) == 1 and clean['close'].iloc[0] == 1.0
        everything = store.read('minute_bars', venue='coinbase_spot',
                                symbols=['BTC-USD'], min_quality='unvalidated')
        assert len(everything) == 2
