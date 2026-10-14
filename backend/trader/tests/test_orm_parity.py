"""Tests that the trader and API ORM models describe the same tables.

`core/pg_writer.py` and `backend/api/models/` declare the same Postgres tables,
because the two run in separate containers. The API copy is generated from the
trader module. This test compares every shared table column by column, including
defaults, plus the migration lists, so any drift fails here.
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
    """Imports the API's declarative Base with the API root on `sys.path`, then removes it.

    The API uses bare imports (`from models.base import Base`), so its root must lead
    the path for the import.
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
    """Imports an API module with the API root on `sys.path`, then restores the path."""
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
    """The parts of a column definition whose divergence would matter.

    Compares `server_default` by value, callable defaults by name, and `index`.
    """
    default = getattr(column.default, 'arg', None) if column.default is not None else None
    if callable(default):
        # Compare the function's name: `list` and `dict` are both callables but give
        # different columns.
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
    """Every table is declared on both sides, since `create_all` on the other container
    would not create it.
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
    """Shared columns have the same definition, including defaults."""
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
    """Both containers apply the same migrations, in the same order.

    Compared by value. Order matters because an index created before its table is an
    error.
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
    """`app.py` runs the mirror's migration list, not a separate copy."""
    from core.pg_writer import MIGRATIONS as trader_migrations

    # `app.py` imports `database`, which needs a URL. Nothing listens on this one, but
    # `create_engine` is lazy and the test never connects.
    monkeypatch.setenv('DATABASE_URL', 'postgresql://nobody:nothing@127.0.0.1:1/absent')
    app_module = _load_api_module('app')
    applied = list(getattr(app_module, 'POSTGRES_MIGRATIONS', ()))
    assert applied == list(trader_migrations), (
        'app.py applies a different list from the one the models declare'
    )


def test_the_migrations_are_dialect_portable():
    """Every migration is `CREATE INDEX IF NOT EXISTS`, which runs on SQLite and Postgres.

    The API tests run on SQLite, so a Postgres-only statement would never be tested.
    """
    from core.pg_writer import MIGRATIONS

    for statement in MIGRATIONS:
        assert statement.upper().startswith('CREATE INDEX IF NOT EXISTS'), (
            f'{statement!r} is not a portable, idempotent index creation. '
            f'Anything else needs a dialect guard, and a guard means the test '
            f'suite stops exercising it.'
        )


def test_every_index_migration_names_a_column_the_model_indexes(schemas):
    """Each migration index matches a model column with `index=True` and SQLAlchemy's
    `ix_<table>_<column>` name.
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
# Tables without a writer
# ---------------------------------------------------------------------------


def test_no_table_is_declared_without_a_writer(schemas):
    """The `trades` table is not declared: nothing writes it, so it would serve empty
    aggregates.
    """
    trader, api = schemas
    assert 'trades' not in trader and 'trades' not in api, (
        'the `trades` table is back. Nothing wrote it before; if something '
        'writes it now, delete this test and say what.'
    )
