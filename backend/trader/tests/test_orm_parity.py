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


def _tables(base: Any) -> dict[str, Any]:
    return dict(base.metadata.tables)


def _column_shape(column: Any) -> dict[str, Any]:
    """The parts of a column definition that a divergence would actually break."""
    default = getattr(column.default, 'arg', None) if column.default is not None else None
    if callable(default):
        default = '<callable>'
    return {
        'type': str(column.type).upper(),
        'nullable': bool(column.nullable),
        'primary_key': bool(column.primary_key),
        'default': default,
        'server_default': column.server_default is not None,
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

    pattern = re.compile(r'ADD COLUMN IF NOT EXISTS (\w+)')
    trader_columns = set(pattern.findall(trader_source))
    api_columns = set(pattern.findall(api_source))

    assert trader_columns, 'no migrations found in pg_writer.py'
    assert trader_columns == api_columns, (
        f'migration lists differ — only in trader: '
        f'{sorted(trader_columns - api_columns)}; '
        f'only in api: {sorted(api_columns - trader_columns)}'
    )
