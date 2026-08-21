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


# ---------------------------------------------------------------------------
# The funding snapshot is not a gap fill
# ---------------------------------------------------------------------------


def test_the_funding_snapshot_is_taken_outside_the_gap_loop():
    """CDE publishes no funding history, so the snapshot is the whole series.

    It was fetched inside `for win_start, win_end in windows` and stored only if
    `win_start <= event_time <= win_end`. Both halves broke collection:

    * `funding_time` is the settlement the rate applies to, and the backfill
      window ends at "now" — so a next-hour settlement fell outside every window
      and was dropped.
    * With no gaps in the range, the loop body never runs. That is the steady
      state once an hour has a row, so the series would stop growing.

    Either way the failure is silent, on data that cannot be re-fetched later.
    Checked against the parse tree rather than the text: an indentation heuristic
    measured the `try`/`if` nesting around the call instead of the loop it needed
    to be outside of, and passed or failed for the wrong reason.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(
        (Path(__file__).resolve().parents[1] / 'scripts' / 'run_pipeline.py').read_text()
    )
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'backfill_funding_rates'
    )

    # `get_contract_snapshot` is the accessor now: funding and open interest ride
    # one product payload, so they cannot straddle a settlement. Both spellings
    # count — the property under test is where the call sits, not its name.
    SNAPSHOT_CALLS = {'get_contract_snapshot', 'get_funding_rate'}

    def calls_current_funding(node) -> bool:
        return any(
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr in SNAPSHOT_CALLS
            for n in ast.walk(node)
        )

    window_loops = [
        node for node in ast.walk(func)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Tuple)
        and {getattr(e, 'id', '') for e in node.target.elts} == {'win_start', 'win_end'}
    ]
    assert window_loops, 'the per-window loop is gone; this test needs rewriting'

    for loop in window_loops:
        assert not calls_current_funding(loop), (
            'the current-funding snapshot is inside the per-window loop again. '
            'It must run once per symbol: funding_time can fall outside every '
            'window, and a range with no gaps never enters the loop at all.'
        )

    assert calls_current_funding(func), (
        'nothing fetches the current funding rate, so the series cannot grow'
    )

    # And no window test may guard it anywhere in the function.
    compares = [
        n for n in ast.walk(func)
        if isinstance(n, ast.Compare)
        and 'win_start' in ast.dump(n) and 'event_time' in ast.dump(n)
    ]
    assert not compares, (
        'a window comparison against event_time is back; funding_time can sit '
        'after the end of the backfill range'
    )


def test_spot_symbols_get_no_funding_product():
    """Funding is a perpetual cash flow. Spot has none, so it must not be mapped.

    `_extract_coin_code('BTC-USD')` resolves to 'BIP', so the funding product map
    happily pointed every spot symbol at the corresponding CDE perp. A spot
    scrape then fetched the *perp's* funding rate and filed it under the spot
    symbol — the right number under a key that has no such thing, once per
    settlement, indistinguishable in the store from a real observation.

    Confirmed in a live store: `BTC-USD`, `ETH-USD` and sixteen others each
    carried a funding row alongside their CDE counterparts.
    """
    from scripts.run_pipeline import SPOT_QUOTES, _extract_coin_code

    for spot in ('BTC-USD', 'ETH-USD', 'PEPE-USD', 'SOL-USDC', 'XRP-USDT'):
        assert SPOT_QUOTES.search(spot), f'{spot} not recognised as spot'
        # The trap: the code resolves, which is why the guard has to be explicit.
        assert _extract_coin_code(spot), (
            f'{spot} resolves to a perp code, so skipping it cannot rely on '
            f'resolution failing'
        )

    for perp in ('BIP-20DEC30-CDE', 'HYP-20DEC30-CDE', 'BTC-PERP'):
        assert not SPOT_QUOTES.search(perp), f'{perp} wrongly treated as spot'


def test_the_funding_snapshot_is_filed_under_the_run_venue():
    """A hardcoded venue misfiles funding on any run with a venue label.

    The snapshot insert passed `venue='coinbase'` literally, so a spot run's rows
    landed on the perp venue. Combined with the mapping bug above, that put
    perp funding under a spot symbol on the perp venue — two wrongs in one row.
    """
    import inspect

    from scripts import run_pipeline

    source = inspect.getsource(run_pipeline.backfill_funding_rates)
    assert 'venue_label' in inspect.signature(
        run_pipeline.backfill_funding_rates
    ).parameters, 'the funding backfill no longer takes a venue label'
    assert "ingest_funding(\n                        [current], venue=venue_label" in source \
        or 'venue=venue_label' in source, (
        'the snapshot is not filed under the run venue'
    )


def test_open_interest_rides_the_funding_snapshot():
    """One request, one instant, two records — and neither from another exchange.

    Open interest used to come from CCXT, because this client had no method for
    it and a comment recorded that absence as "Coinbase exposes no open-interest
    endpoint". It is on the product payload, under
    `future_product_details.open_interest`, on the contract actually traded:
    268,164 for BIP-20DEC30-CDE against 21,579,279 for gate's BTC/USDT:USDT.

    Fetching the two separately would double the request count and let the funding
    row keyed to 22:00 be paired with open interest read after the 22:00 print.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / 'scripts' / 'run_pipeline.py').read_text()
    tree = ast.parse(source)
    func = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == 'backfill_funding_rates'
    )

    attributes = {
        n.func.attr for n in ast.walk(func)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert 'get_contract_snapshot' in attributes, (
        'the snapshot no longer comes from one product request'
    )
    assert 'ingest_open_interest' in attributes, (
        'the open interest half of the snapshot is fetched and then dropped'
    )

    # Nothing reaches for another exchange. Prose mentioning CCXT is fine and
    # wanted — the comments explaining why it was removed are the record — so the
    # check is on imports and calls, which is what test_no_module_imports_ccxt does.
    assert 'CCXTConnector' not in {
        n.func.id for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }


def test_no_module_imports_ccxt():
    """The dependency is gone, not merely unused.

    CCXT served three purposes and every one is now native: perp bars and spot
    bars come from Coinbase (`coinbase` and `coinbase_spot`), open interest comes
    from the product endpoint, and cross-venue funding was never permissible —
    `proxy_funding_symbols` is a promotion gate with a threshold of zero, so the
    fallback could only ever write rows the gates then refused.
    """
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

    # Both requirement files. The API declared ccxt and never imported it, which
    # is how a dependency outlives every use of it.
    for requirements in (root / 'requirements.txt',
                         root.parent / 'api' / 'requirements.txt'):
        if not requirements.exists():
            continue
        declared = [
            line.strip() for line in requirements.read_text().splitlines()
            if line.strip().lower().startswith('ccxt')
        ]
        assert not declared, f'{requirements.name} still declares {declared}'


def test_the_snapshot_stores_funding_and_open_interest_together():
    """The glue, exercised rather than inspected.

    The AST tests above pin *where* the snapshot call sits; this runs it. Both
    halves of one product payload have to land, under the run's venue label, keyed
    on the symbol the caller asked for rather than the Coinbase product id —
    every one of those was a bug at some point today.
    """
    import asyncio
    from unittest import mock

    from data_collection.storage import SQLiteDatabase
    from scripts import run_pipeline

    payload = {
        'product_id': 'BIP-20DEC30-CDE', 'price': '77105',
        'future_product_details': {
            'funding_rate': '0.000009', 'funding_interval': '3600s',
            'funding_time': '2026-08-21T22:00:00Z',
            'open_interest': '268164', 'index_price': '77118.16',
        },
    }

    class FakeClient:
        def __init__(self):
            self.requests = 0

        async def get_contract_snapshot(self, product_id):
            self.requests += 1
            from data_collection.coinbase_connector import CoinbaseRESTClient
            parse = CoinbaseRESTClient.__dict__
            unbound = object.__new__(CoinbaseRESTClient)
            return (parse['_parse_funding'](unbound, product_id, payload),
                    parse['_parse_open_interest'](unbound, product_id, payload))

        async def get_funding_rate_history(self, *a, **k):
            return []

        async def close(self):
            pass

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        database = SQLiteDatabase(f'{tmp}/trading.db')
        database.initialize()
        client = FakeClient()

        with mock.patch.object(run_pipeline, 'CoinbaseRESTClient', lambda *a: client), \
             mock.patch.object(run_pipeline, 'resolve_coinbase_funding_product_map',
                               mock.AsyncMock(return_value={'BIP': 'BIP-20DEC30-CDE'})):
            asyncio.run(run_pipeline.backfill_funding_rates(
                ['BIP'], T0, T0 + timedelta(hours=1), database,
                api_key='k', api_secret='s', venue_label='coinbase',
            ))

        with database._get_connection() as conn:
            funding = [dict(r) for r in conn.cursor().execute(
                'SELECT symbol, venue, rate FROM funding_rates')]
            interest = [dict(r) for r in conn.cursor().execute(
                'SELECT symbol, venue, open_interest_contracts FROM open_interest')]

    assert client.requests == 1, f'{client.requests} product requests for one symbol'

    assert len(funding) == 1, f'funding rows: {funding}'
    assert funding[0]['symbol'] == 'BIP', 'stored under the product id, not the symbol'
    assert funding[0]['venue'] == 'coinbase'
    assert funding[0]['rate'] == pytest.approx(9e-6)

    assert len(interest) == 1, (
        f'open interest rows: {interest} — the snapshot half was dropped'
    )
    assert interest[0]['symbol'] == 'BIP'
    assert interest[0]['venue'] == 'coinbase'
    assert interest[0]['open_interest_contracts'] == pytest.approx(268164.0)


def test_the_open_interest_snapshot_is_keyed_on_the_hour():
    """A microsecond stamp broke dedup, alignment and the pairing with funding.

    The store's key is (symbol, venue, event_time). `utc_now()` mints a new one
    every call, so a second run inside one hour appended duplicate rows instead of
    upserting — funding, keyed on the settlement hour, did not have that problem,
    and the two halves of a single request disagreeing about their timestamp is
    what fetching them together exists to prevent.

    `_align` also reindexes onto the hourly bar grid with `method='ffill'`, so a
    reading at 21:21:06 landed on the 22:00 bar rather than 21:00.
    """
    import asyncio
    from unittest import mock

    from data_collection.coinbase_connector import CoinbaseRESTClient

    client = CoinbaseRESTClient('k', 's')
    body = {'price': '77105', 'future_product_details': {
        'funding_rate': '0.000009', 'funding_interval': '3600s',
        'funding_time': '2026-08-21T22:00:00Z', 'open_interest': '268164'}}

    with mock.patch.object(client, '_request', mock.AsyncMock(return_value=(200, body))):
        first = asyncio.run(client.get_contract_snapshot('BIP-20DEC30-CDE'))
        second = asyncio.run(client.get_contract_snapshot('BIP-20DEC30-CDE'))

    for funding, interest in (first, second):
        assert interest.event_time.minute == 0
        assert interest.event_time.second == 0
        assert interest.event_time.microsecond == 0
        # available_time keeps the real instant: known at 21:21, describes hour 21.
        assert interest.available_time >= interest.event_time
        assert funding.event_time.minute == 0

    # Two reads in the same hour must produce the same key, or they accumulate.
    assert first[1].event_time == second[1].event_time, (
        'repeat reads inside one hour produce different keys, so they duplicate '
        'instead of upserting'
    )


def test_repeat_snapshots_in_one_hour_upsert(database):
    """The consequence of the above, at the storage layer."""
    from data_collection.models import OpenInterest

    ingestor = Ingestor(database)
    stamp = datetime(2026, 8, 21, 21, 0, 0)
    for contracts in (268164.0, 268900.0, 269500.0):
        ingestor.ingest_open_interest([OpenInterest(
            symbol='BIP-20DEC30-CDE', event_time=stamp,
            available_time=stamp + timedelta(minutes=21),
            open_interest_contracts=contracts,
        )], venue='coinbase')

    with database._get_connection() as conn:
        rows = [dict(r) for r in conn.cursor().execute(
            'SELECT event_time, open_interest_contracts FROM open_interest')]

    assert len(rows) == 1, f'three reads in one hour produced {len(rows)} rows'
    assert rows[0]['open_interest_contracts'] == 269500.0, 'the latest read must win'
