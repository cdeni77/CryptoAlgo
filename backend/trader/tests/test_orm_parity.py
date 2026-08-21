"""The trader and API ORM models describe the same tables. Prove it.

`backend/trader/core/pg_writer.py` and `backend/api/models/` declare the same
Postgres tables twice, on purpose: the two run in separate containers and neither
imports the other's package. That duplication is a deliberate isolation choice,
but "keep both in sync" written in a doc is not a mechanism — it is a hope, and
it had already failed. `wallet.balance` defaulted to 10,000 on the trader side and
100,000 on the API side, so whichever container created the row decided the
paper account's starting balance, and the equity curve began from a different
number than the dashboard reported.

This test is the mechanism. It compares every shared table column by column,
including nullability and defaults, and fails on the next divergence rather than
leaving it to be discovered from a wrong number on a screen.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any

import pytest

TRADER_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = TRADER_ROOT.parent / 'api'


def _load_api_models() -> Any:
    """Import the API's declarative Base with its own module search path.

    The API package uses bare imports (`from models.base import Base`), so its
    root has to be on `sys.path` for the duration. It is inserted ahead of the
    trader root and removed afterwards, so the trader's own `core`/`scripts`
    imports are unaffected in the rest of the session.
    """
    if not API_ROOT.exists():
        pytest.skip(f'API package not present at {API_ROOT}')

    inserted = str(API_ROOT)
    sys.path.insert(0, inserted)
    try:
        for module in ('models.base', 'models.signals', 'models.trade', 'models.wallet'):
            importlib.import_module(module)
        return importlib.import_module('models.base').Base
    finally:
        if inserted in sys.path:
            sys.path.remove(inserted)


def _load_api_module(dotted: str) -> Any:
    """Import an API module with the API root on `sys.path`, then restore it.

    Same mechanism as `_load_api_models`: the API package uses bare imports, so
    its root has to lead the path for the duration and be removed afterwards.
    """
    if not API_ROOT.exists():
        pytest.skip(f'API package not present at {API_ROOT}')

    inserted = str(API_ROOT)
    sys.path.insert(0, inserted)
    try:
        return importlib.import_module(dotted)
    finally:
        if inserted in sys.path:
            sys.path.remove(inserted)


def _tables(base: Any) -> dict[str, Any]:
    return dict(base.metadata.tables)


def _column_shape(column: Any) -> dict[str, Any]:
    """The parts of a column definition that a divergence would actually break.

    Three earlier blind spots, each demonstrated by mutation:

    - `server_default` was compared as a boolean, so `text('100000')` against
      `text('10000')` produced identical shapes — the exact `wallet.balance`
      drift this file exists to catch, expressed server-side instead.
    - callable defaults collapsed to the string `'<callable>'`, so
      `default=list` and `default=dict` were interchangeable.
    - `index` was not compared at all, so an index on one side only passed.
    """
    default = getattr(column.default, 'arg', None) if column.default is not None else None
    if callable(default):
        # The function's identity, not the fact that it is one. `list` and `dict`
        # are both callables and produce very different columns.
        default = f'<callable:{getattr(default, "__name__", repr(default))}>'

    server_default = column.server_default
    if server_default is not None:
        # The value, not its presence.
        arg = getattr(server_default, 'arg', server_default)
        server_default = str(getattr(arg, 'text', arg))

    return {
        'type': str(column.type).upper(),
        'nullable': bool(column.nullable),
        'primary_key': bool(column.primary_key),
        'default': default,
        'server_default': server_default,
        'index': bool(column.index),
        'unique': bool(column.unique),
    }


@pytest.fixture(scope='module')
def schemas() -> tuple[dict[str, Any], dict[str, Any]]:
    from core.pg_writer import Base as trader_base

    api_base = _load_api_models()
    return _tables(trader_base), _tables(api_base)


def test_the_shared_tables_are_the_same_set(schemas):
    """Neither side should own a table the other does not know about.

    A table only one side declares is not automatically a bug — but it is always
    worth knowing about, because `create_all` on the other container will not
    create it and a read against it fails at runtime rather than at startup.
    """
    trader, api = schemas
    shared = set(trader) & set(api)

    assert shared, 'the two ORMs share no tables at all, which means one failed to import'

    # Every table either side declares should be shared. If a genuinely
    # single-sided table is ever added, list it here with a reason.
    single_sided = (set(trader) ^ set(api))
    assert not single_sided, (
        f'tables declared on only one side: {sorted(single_sided)}. '
        f'Whichever container does not declare it will not create it.'
    )


def test_every_shared_table_has_the_same_columns(schemas):
    trader, api = schemas

    problems: list[str] = []
    for name in sorted(set(trader) & set(api)):
        trader_columns = set(trader[name].columns.keys())
        api_columns = set(api[name].columns.keys())
        only_trader = sorted(trader_columns - api_columns)
        only_api = sorted(api_columns - trader_columns)
        if only_trader:
            problems.append(f'{name}: only in trader: {only_trader}')
        if only_api:
            problems.append(f'{name}: only in api: {only_api}')

    assert not problems, (
        'the duplicated ORM models have diverged:\n  ' + '\n  '.join(problems)
        + '\n\nA column one side writes and the other cannot read is silent: the '
          'read returns nothing rather than failing.'
    )


def test_shared_columns_have_the_same_definition(schemas):
    """Same name is not enough — the default is what bit us.

    `wallet.balance` existed on both sides with the same type and nullability,
    and still produced two different starting balances, because the defaults
    differed by a factor of ten.
    """
    trader, api = schemas

    problems: list[str] = []
    for name in sorted(set(trader) & set(api)):
        shared_columns = set(trader[name].columns.keys()) & set(api[name].columns.keys())
        for column in sorted(shared_columns):
            left = _column_shape(trader[name].columns[column])
            right = _column_shape(api[name].columns[column])
            for key in left:
                if left[key] != right[key]:
                    problems.append(
                        f'{name}.{column}.{key}: trader={left[key]!r} api={right[key]!r}'
                    )

    assert not problems, (
        'shared columns differ in definition:\n  ' + '\n  '.join(problems)
    )


def test_the_migration_lists_match(schemas):
    """Both containers run the same idempotent ALTERs, so the lists must agree.

    Whichever service starts second has to add any column the first one's
    `create_all` did not, or a writer will insert into a column the reader's
    model expects and cannot find.
    """
    import re

    trader_source = (TRADER_ROOT / 'core' / 'pg_writer.py').read_text()
    api_source = (API_ROOT / 'app.py').read_text()

    # Table, column AND type. Capturing only the column name meant
    # `cost_bps VARCHAR` on one side against `DOUBLE PRECISION` on the other
    # passed, as did the right column added to the wrong table — which is the
    # drift class this check exists for.
    pattern = re.compile(
        r'ALTER TABLE\s+(\w+)\s+ADD COLUMN IF NOT EXISTS\s+(\w+)\s+([A-Za-z ]+)',
        re.IGNORECASE,
    )
    trader_columns = set(pattern.findall(trader_source))
    api_columns = set(pattern.findall(api_source))

    assert trader_columns, 'no migrations found in pg_writer.py'
    assert trader_columns == api_columns, (
        f'migration lists differ — only in trader: '
        f'{sorted(trader_columns - api_columns)}; '
        f'only in api: {sorted(api_columns - trader_columns)}'
    )


def test_the_index_migrations_match(schemas):
    """Indexes drift the same way columns do, and are invisible to create_all.

    `create_all` only creates missing *tables*, so `index=True` added to an
    existing model never reaches a database that already has the table. Both
    containers therefore carry explicit `CREATE INDEX IF NOT EXISTS`
    statements, and the column check above cannot see them — its pattern
    matches `ALTER TABLE` only.
    """
    import re

    trader_source = (TRADER_ROOT / 'core' / 'pg_writer.py').read_text()
    api_source = (API_ROOT / 'app.py').read_text()

    # Statements are wrapped across string literals, so join them first.
    def indexes(source: str) -> set[tuple[str, str, str]]:
        flat = re.sub(r'"\s*\n\s*"', '', source)
        return set(
            re.findall(
                r'CREATE INDEX IF NOT EXISTS\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]*)\)',
                flat,
                re.IGNORECASE,
            )
        )

    trader_indexes = indexes(trader_source)
    api_indexes = indexes(api_source)

    assert trader_indexes, 'no index migrations found in pg_writer.py'
    assert trader_indexes == api_indexes, (
        f'index migrations differ — only in trader: '
        f'{sorted(trader_indexes - api_indexes)}; '
        f'only in api: {sorted(api_indexes - trader_indexes)}'
    )


def test_every_index_migration_names_a_column_the_model_indexes(schemas):
    """A migration index and a model index have to be the same index.

    Two ways this goes wrong: the migration names a column the model does not
    mark `index=True`, so a fresh database lacks the index an upgraded one has;
    or the name does not follow SQLAlchemy's `ix_<table>_<column>` convention,
    so `create_all` and the migration each create their own copy of it.
    """
    import re

    trader, _ = schemas
    flat = re.sub(r'"\s*\n\s*"', '', (TRADER_ROOT / 'core' / 'pg_writer.py').read_text())

    problems: list[str] = []
    for name, table, columns in re.findall(
        r'CREATE INDEX IF NOT EXISTS\s+(\w+)\s+ON\s+(\w+)\s*\(([^)]*)\)', flat, re.IGNORECASE
    ):
        column = columns.strip()
        if table not in trader:
            problems.append(f'{name}: table {table} is not declared')
            continue
        if column not in trader[table].columns:
            problems.append(f'{name}: {table}.{column} is not a column')
            continue
        if not trader[table].columns[column].index:
            problems.append(
                f'{name}: {table}.{column} is not index=True, so a fresh '
                f'create_all would not build this index'
            )
        expected = f'ix_{table}_{column}'
        if name != expected:
            problems.append(f'{name}: SQLAlchemy would name this {expected}')

    assert not problems, 'index migrations disagree with the models:\n  ' + '\n  '.join(problems)


# ---------------------------------------------------------------------------
# Contract sizes
# ---------------------------------------------------------------------------


def test_the_api_contract_sizes_match_the_cost_model():
    """Contract size is money, and it had drifted 2x-5x in three places.

    `core/costs.py` is the single source of truth for money (CLAUDE.md), but the
    API serves its own `CDE_PRODUCTS` table to the frontend, which prefers the
    API value over its own fallback. AVAX read 5 against 10, LINK 10 against 50,
    LTC 1 against 5 — and contract size multiplies straight into notional, fee as
    a fraction of notional, margin, liquidation price, participation rate and
    PnL. The five instruments actually traded happened to agree, which is why
    nothing surfaced it.

    Duplication here is deliberate (the containers do not import each other), so
    the guard is the same one `test_orm_parity` applies to the ORMs: compare them
    and fail on divergence.
    """
    from core.costs import CONTRACT_UNITS

    api_products = _load_api_module('endpoints.coins').CDE_PRODUCTS

    mismatches = []
    for asset, product in api_products.items():
        expected = CONTRACT_UNITS.get(asset)
        if expected is None:
            continue
        actual = float(product['units_per_contract'])
        if abs(actual - float(expected)) > 1e-9:
            mismatches.append(
                f'{asset} ({product.get("code")}): API {actual} vs core/costs.py {expected}'
            )

    assert not mismatches, (
        'contract sizes disagree, so notional and PnL differ between the '
        'services:\n  ' + '\n  '.join(mismatches)
    )


def test_every_api_product_is_known_to_the_cost_model():
    """A product the cost model has never heard of gets a default contract size.

    Asserted through `get_contract_spec`, which is what production calls, rather
    than through set membership in `CONTRACT_UNITS`. The two differ: the meme
    contracts are keyed `1000PEPE` / `1000SHIB` there while the API and
    `core/profiles.py` call them `PEPE` / `SHIB`, and `resolve_base` bridges that
    with an alias. The literal version flagged both as unknown when they price
    correctly — a false alarm on a naming convention, which is worse than no
    check because it trains you to ignore it.
    """
    from core.costs import DEFAULT_UNITS, get_contract_spec

    api_products = _load_api_module('endpoints.coins').CDE_PRODUCTS

    # `base == 'UNKNOWN'` is the fallback, not `units == DEFAULT_UNITS`: BCH is
    # genuinely one unit per contract (confirmed against the venue's own
    # `contract_size`), so a units-based check condemns a correct entry forever.
    unpriced = [
        name for name, detail in api_products.items()
        # Resolve from the product id the API serves, since that is what a
        # caller passes back in.
        if get_contract_spec(detail.get('symbol') or name).base == 'UNKNOWN'
    ]
    assert not unpriced, (
        f'served by the API but unresolvable by get_contract_spec, so priced at '
        f'the {DEFAULT_UNITS} fallback: {sorted(unpriced)}'
    )

    # And the size the API advertises must be the size the cost model uses.
    mismatches = [
        f"{name}: api={detail['units_per_contract']:g} "
        f"cost_model={get_contract_spec(detail['symbol']).units:g}"
        for name, detail in api_products.items()
        if float(detail['units_per_contract'])
        != get_contract_spec(detail['symbol']).units
    ]
    assert not mismatches, 'contract size differs from the cost model:\n  ' + '\n  '.join(mismatches)


def test_no_table_is_declared_without_a_writer(schemas):
    """A table nothing writes is a surface that serves fabricated aggregates.

    `trades` was declared on both sides, exposed at `/trades` with a
    `/trades/stats` that computed win rate over an empty table and reported
    `0.0` — indistinguishable from a measured 0% — while the only writers,
    `PgWriter.open_trade` and `.close_trade`, had no callers. The paper engine
    keeps its ledger in `paper_positions`. When live execution lands it should
    extend those tables (which carry funding, exit reason and TP/SL) rather
    than resurrect a parallel schema.
    """
    trader, api = schemas
    assert 'trades' not in trader and 'trades' not in api, (
        'the `trades` table is back. Nothing wrote it before; if something '
        'writes it now, delete this test and say what.'
    )


def test_the_venue_schedule_agrees_with_the_contract_units_table():
    """Three sources declare contract size. All three have to say the same thing.

    Contract size multiplies straight into notional, fee-as-a-fraction-of-
    notional, margin, liquidation price, participation rate and PnL — so a
    disagreement is a silent multiplier on every number the system produces.

    `test_the_api_contract_sizes_match_the_cost_model` above compares two of the
    three: the API's `CDE_PRODUCTS` against `core/costs.py:CONTRACT_UNITS`. It
    passed while the file both are meant to derive from — the venue fee schedule
    in `configs/exchange/` — disagreed on three instruments:

        AVAX   CONTRACT_UNITS 10   schedule 5    (2x)
        LINK   CONTRACT_UNITS 50   schedule 10   (5x)
        LTC    CONTRACT_UNITS  5   schedule 1    (5x)

    `ExchangeCostAssumptions.contract_sizes` is parsed out of that file and read
    by nothing — `get_contract_spec` always uses `CONTRACT_UNITS` — so the
    schedule's sizes were dead data and the disagreement had no way to surface.

    Resolved by asking the venue: `future_product_details.contract_size` on the
    product endpoint reports 10 / 50 / 5 for all three, agreeing with
    `CONTRACT_UNITS` across all sixteen contracts — so the schedule was the wrong
    one and is corrected. `probe_funding --sizes-only` reproduces that table.
    The edit was inert (those sizes are read by nothing), which is exactly why
    nothing caught it: this test and the loader's warning are the mechanism now.
    """
    import json

    from core.costs import CONTRACT_UNITS, resolve_base

    schedule_path = (
        TRADER_ROOT / 'configs' / 'exchange' / 'coinbase_us_perps_cde_v202602.json'
    )
    if not schedule_path.exists():
        pytest.skip(f'no venue schedule at {schedule_path}')

    declared = json.loads(schedule_path.read_text()).get('contract_sizes') or {}
    assert declared, 'the venue schedule declares no contract sizes'

    mismatches = []
    for key, size in sorted(declared.items()):
        # The schedule keys on both the asset (AVAX) and the contract code (AVP);
        # both resolve to the same base, which is what CONTRACT_UNITS keys on.
        base = resolve_base(key)
        if base is None or base not in CONTRACT_UNITS:
            continue
        if float(CONTRACT_UNITS[base]) != float(size):
            mismatches.append(
                f'{key} (base {base}): schedule={size} CONTRACT_UNITS={CONTRACT_UNITS[base]} '
                f'({float(CONTRACT_UNITS[base]) / float(size):.3g}x)'
            )

    assert not mismatches, (
        'contract sizes disagree between the venue schedule and the money '
        'module:\n  ' + '\n  '.join(mismatches)
        + '\n\nCheck Coinbase\'s published contract specs and fix whichever is '
          'wrong. Contract size multiplies into notional, fees, margin, '
          'liquidation price and PnL, so this is not a cosmetic difference.'
    )


def test_the_api_serves_every_instrument_the_trader_models():
    """One universe, or an explicit reason why not.

    The trader modelled sixteen contracts (`core/profiles.py`) while the API and
    frontend served nine. Nothing was wrong with either list on its own, and that
    is the problem: the nine got mistaken for the traded universe when writing
    the spot scrape, which would have left seven instruments with no cross-venue
    features and no dashboard row.

    The API cannot import the trader (separate containers), so the tables are
    duplicated on purpose and this is the mechanism — the same one
    `test_the_api_contract_sizes_match_the_cost_model` applies to sizes.
    """
    from core.costs import SPOT_PRODUCTS, resolve_base
    from core.profiles import COIN_PROFILES

    coins = _load_api_module('endpoints.coins')
    modelled = {resolve_base(name) for name in COIN_PROFILES}

    served_cde = {resolve_base(name) for name in coins.CDE_PRODUCTS}
    missing = sorted(b for b in modelled - served_cde if b)
    assert not missing, (
        f'the trader models these but the API serves no CDE product for them: '
        f'{missing}. They will have no dashboard row and no /coins/cde-specs '
        f'entry, so the frontend falls back for their contract size.'
    )

    served_spot = {resolve_base(name) for name in coins.COINBASE_PRODUCTS}
    missing_spot = sorted(b for b in modelled - served_spot if b)
    assert not missing_spot, (
        f'no spot product served for {missing_spot}; spot is the reference venue '
        f'the cross-venue features use'
    )

    # And the trader's own spot table has to cover the same set, since that is
    # what `--spot-universe` scrapes from.
    uncovered = sorted(b for b in modelled if b and b not in SPOT_PRODUCTS)
    assert not uncovered, f'SPOT_PRODUCTS has no entry for {uncovered}'


def test_the_spot_universe_covers_every_contract_once():
    """`--spot-universe` is what stops the list being hand-typed. It has to be
    complete and free of duplicates, or a silently short scrape is back."""
    from core.costs import spot_universe
    from core.profiles import COIN_PROFILES

    products = spot_universe(sorted(COIN_PROFILES))

    assert len(products) == len(COIN_PROFILES), (
        f'{len(products)} spot products for {len(COIN_PROFILES)} contracts'
    )
    assert len(set(products)) == len(products), 'a spot product appears twice'
    assert all(p.endswith('-USD') for p in products), products
    # The meme contracts are keyed 1000PEPE/1000SHIB but spot is plain.
    assert 'PEPE-USD' in products and 'SHIB-USD' in products
    assert not any('1000' in p for p in products), products


def test_every_modelled_contract_has_an_explicit_fee_floor():
    """A missing floor silently charges the BTC/ETH rate to everything.

    `fee_floor` falls back to `config.min_fee_per_contract`, which the CDE
    schedule sets to 0.75 — the group-A rate for BIP and ETP. Every other
    contract is 0.10. Seven of the eighteen (BCP, NER, SUP, XLP, POP, SHP, PEP)
    were never listed, so they were priced 7.5x too expensive.

    That is not a rounding error. SHIB's round trip read 287bp instead of 42bp,
    DOT 172 instead of 26, and the cost enters the *target* through
    `core/targets.py` as well as the hurdle `decide()` compares against — so
    those instruments looked untradeable, and the cross-sectional cost features
    ranked them against the others on a fiction.

    `load_dataset` warns about this, and a warning was not enough: it fires on a
    run nobody is reading closely, and the number it protects is money.
    """
    from core.config import Config, find_cost_config
    from core.costs import symbols_missing_fee_schedule
    from core.profiles import COIN_PROFILES

    path = find_cost_config()
    if path is None:
        pytest.skip('no cost config on the search path')
    config = Config().with_cost_assumptions(path)

    # Every spelling a profile answers to, *and* the decorated product ids the
    # store actually holds — `BIP-20DEC30-CDE`, not `BIP`. Testing bare codes
    # alone missed a false positive on the decorated form: the check compared
    # the resolved fee against the default, so BIP and ETP at an explicit $0.75
    # were reported as uncovered while pricing correctly.
    codes = sorted({p for profile in COIN_PROFILES.values() for p in profile.prefixes})
    decorated = [f'{c}-20DEC30-CDE' for c in codes]
    missing = symbols_missing_fee_schedule(codes + decorated, config)

    assert not missing, (
        f'no explicit per-contract fee for {missing}. They fall back to '
        f'${config.min_fee_per_contract:.2f}/contract, the group-A rate, which '
        f'overstates their round trip several-fold.'
    )
