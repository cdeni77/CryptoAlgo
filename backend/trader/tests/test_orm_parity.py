"""The trader and API ORM models describe the same tables. Prove it.

`backend/trader/core/pg_writer.py` and `backend/api/models/` declare the same
Postgres tables twice, on purpose: the two run in separate containers and neither
imports the other's package. That duplication is a deliberate isolation choice,
but "keep both in sync" written in a doc is not a mechanism — it is a hope, and
it had already failed. `wallet.balance` defaulted to 10,000 on the trader side and
100,000 on the API side, so whichever container created the row decided the
account's starting balance, and the equity curve began from a different number
than the dashboard reported.

The API copy is now *generated* from the trader's module rather than transcribed,
which removes the transcription failure mode — but not this test's job. A
generated copy can still fall out of date, and the migration lists are written by
hand on both sides.

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
        for module in ('models.base', 'models.serving'):
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
    """The two containers must apply exactly the same migrations.

    Compared by *value*, not by parsing the two source files for string literals.
    That earlier version was fragile in the way that matters: it broke when the
    quoting style changed and it could not see through the API now referencing
    the mirror's list instead of restating it — so it would have failed on a
    change that made divergence structurally impossible.

    What is asserted is the invariant itself: whatever the API applies is
    character-for-character what the trader applies, in the same order. Order
    counts because an index created before the table it names is an error, not a
    no-op.
    """
    trader, api = schemas
    from core.pg_writer import MIGRATIONS as trader_migrations

    api_serving = _load_api_module('models.serving')
    api_migrations = getattr(api_serving, 'MIGRATIONS', None)
    assert api_migrations is not None, (
        'the API mirror declares no MIGRATIONS; the trader has '
        f'{len(trader_migrations)}'
    )
    assert list(api_migrations) == list(trader_migrations), (
        'migration lists differ.\n'
        f'  trader only: {sorted(set(trader_migrations) - set(api_migrations))}\n'
        f'  api only:    {sorted(set(api_migrations) - set(trader_migrations))}'
    )

    app_source = (API_ROOT / 'app.py').read_text()
    assert 'POSTGRES_MIGRATIONS' in app_source, (
        'app.py no longer defines POSTGRES_MIGRATIONS, so nothing applies them'
    )


def test_the_api_applies_the_migrations_it_declares(monkeypatch):
    """`app.py` must run the same list the mirror holds, not a copy of it.

    The failure this rules out is a second literal list in `app.py` that drifts
    from the mirror's — which is exactly the shape of the bug the whole module
    exists for, one level up.
    """
    from core.pg_writer import MIGRATIONS as trader_migrations

    # `app.py` imports `database`, which refuses to import without a URL.
    # A URL nothing is listening on is fine: `create_engine` is lazy, and this
    # test never opens a connection.
    monkeypatch.setenv('DATABASE_URL', 'postgresql://nobody:nothing@127.0.0.1:1/absent')
    app_module = _load_api_module('app')
    applied = list(getattr(app_module, 'POSTGRES_MIGRATIONS', ()))
    assert applied == list(trader_migrations), (
        'app.py applies a different list from the one the models declare'
    )


def test_the_migrations_are_dialect_portable():
    """Every migration must run on SQLite as well as Postgres.

    Not a portability nicety: the API test suite runs on SQLite, so a
    Postgres-only statement is a migration the tests never execute. The previous
    list was all `ADD COLUMN IF NOT EXISTS`, which meant a dialect guard skipped
    the whole thing under test and the migrations production ran had never been
    run anywhere else.
    """
    from core.pg_writer import MIGRATIONS

    for statement in MIGRATIONS:
        assert statement.upper().startswith('CREATE INDEX IF NOT EXISTS'), (
            f'{statement!r} is not a portable, idempotent index creation. '
            f'Anything else needs a dialect guard, and a guard means the test '
            f'suite stops exercising it.'
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
