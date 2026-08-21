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


# ---------------------------------------------------------------------------
# Venue is part of the key
# ---------------------------------------------------------------------------


def test_two_venues_coexist_for_the_same_bar(tmp_path):
    """Both venues' bars for the same hour must survive. Neither may replace the other.

    The unique key was `(symbol, timeframe, event_time)` with venue outside it,
    against `INSERT OR REPLACE`. So writing a Coinbase bar and then a Binance bar
    for the same instrument and hour silently discarded the first — and the
    cross-venue features (basis, lead-lag) need exactly that pair, so they would
    have produced no rows at all while the venue column sat there looking correct.
    """
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
    """Venue widens the key; it must not disable de-duplication within a venue.

    Re-running a backfill re-writes the same bars, and each should update its row
    rather than accumulate a duplicate.
    """
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


def test_funding_and_open_interest_are_venue_keyed(tmp_path):
    """Funding differs materially between venues, so a proxy must not overwrite it.

    Funding is the mechanism this system is mostly betting on — 2bp/hour is
    48bp/day against a 5-54bp round trip — so a Binance rate silently stored as
    Coinbase's own is not a labelling nit.
    """
    from datetime import datetime, timedelta, timezone

    from data_collection.models import DataQuality, FundingRate, OpenInterest
    from data_collection.storage import SQLiteDatabase

    db = SQLiteDatabase(str(tmp_path / 'venue_keys.db'))
    db.initialize()
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    for venue, rate in (('coinbase', 2e-5), ('binance', 5e-5)):
        db.insert_funding_rate(FundingRate(
            symbol='BIP', event_time=stamp, available_time=stamp,
            rate=rate, mark_price=60_000.0, index_price=60_000.0,
            venue=venue, quality=DataQuality.VALID,
        ))
        db.insert_open_interest(OpenInterest(
            symbol='BIP', event_time=stamp, available_time=stamp,
            open_interest_contracts=1_000.0, venue=venue,
            quality=DataQuality.VALID,
        ))

    with db._get_connection() as conn:
        funding = conn.execute(
            'SELECT venue, rate FROM funding_rates WHERE symbol = ?', ('BIP',)
        ).fetchall()
        oi = conn.execute(
            'SELECT venue FROM open_interest WHERE symbol = ?', ('BIP',)
        ).fetchall()

    assert {r['venue'] for r in funding} == {'coinbase', 'binance'}, (
        f'funding rates collapsed across venues: {[dict(r) for r in funding]}'
    )
    assert {r['venue'] for r in oi} == {'coinbase', 'binance'}


def test_a_legacy_database_is_rebuilt_onto_the_venue_key(tmp_path):
    """An existing database keyed without venue must be migrated, not left broken.

    SQLite cannot alter a UNIQUE constraint, so `initialize()` rebuilds the table.
    Rows already collapsed by the old key are unrecoverable — re-running the
    backfill is the only way to get the second venue back — but from the rebuild
    onward both venues coexist.
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    from data_collection.models import DataQuality, OHLCVBar
    from data_collection.storage import SQLiteDatabase

    path = tmp_path / 'legacy.db'
    stamp = datetime(2026, 1, 1, tzinfo=timezone.utc)

    # The old schema, verbatim in the part that matters.
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

    # And the second venue can now be written alongside it.
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


def test_the_ingestor_stamps_the_venue_on_funding(database):
    """The code path, not just the schema key.

    `test_funding_and_open_interest_are_venue_keyed` constructs
    `FundingRate(..., venue=venue)` by hand and calls `db.insert_funding_rate`
    directly, so it proves the unique key works and says nothing about whether
    anything sets the field. `ingest_funding` did not: it set `funding_source` and
    left `venue` at its `'unknown'` default, so every rate collided on
    `(symbol, 'unknown', event_time)` and `INSERT OR REPLACE` kept the last
    writer — a Binance proxy rate overwriting Coinbase's own, then read back as
    Coinbase's. Removing the stamp again left all 13 tests in this file passing.
    """
    ingestor = Ingestor(database)
    for rate, venue in ((0.0001, 'coinbase'), (0.0009, 'binance_proxy')):
        ingestor.ingest_funding([FundingRate(
            symbol='BIP-20DEC30-CDE', event_time=T0, available_time=T0,
            rate=rate, mark_price=1.0, index_price=1.0,
        )], venue=venue)

    with database._get_connection() as connection:
        rows = connection.execute(
            'SELECT venue, rate FROM funding_rates ORDER BY venue'
        ).fetchall()

    stamped = {row[0]: row[1] for row in rows}
    assert 'unknown' not in stamped, 'ingest_funding did not stamp the venue'
    assert stamped == {'binance_proxy': 0.0009, 'coinbase': 0.0001}, (
        f'a proxy rate overwrote the venue\'s own: {stamped}'
    )


def test_the_ingestor_stamps_the_venue_on_open_interest(database):
    """Same property for the third dataset, so all three are covered."""
    ingestor = Ingestor(database)
    for value, venue in ((1000.0, 'bybit'), (2000.0, 'binance')):
        ingestor.ingest_open_interest([OpenInterest(
            symbol='BIP-20DEC30-CDE', event_time=T0, available_time=T0,
            open_interest_contracts=value,
        )], venue=venue)

    with database._get_connection() as connection:
        venues = {
            row[0] for row in
            connection.execute('SELECT venue FROM open_interest').fetchall()
        }

    assert 'unknown' not in venues and len(venues) == 2, venues
