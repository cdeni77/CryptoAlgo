"""Schema creation belongs to startup, not to module import.

The API runs under `uvicorn --workers 4` (see `backend/api/Dockerfile`). Every
worker imports `app.py`, and `create_all` used to run there: four processes
each checking whether a table exists and then creating it, which is a race
whose loser dies with DuplicateTable before FastAPI exists to report it.

These tests pin the two properties that fix it — import does not touch the
database, and the tables exist once the app has started.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from sqlalchemy import inspect


@pytest.fixture
def fresh_app(monkeypatch):
    """Import `app` against a given URL, then put the session back as it was.

    Two things have to be undone or the rest of the suite inherits them: the
    `DATABASE_URL` this points at a deliberately dead host, and the evicted
    `app`/`database` modules that every other test's `client` fixture imports.
    Setting the variable with bare `os.environ` and leaving the modules popped
    made the whole API suite order-dependent — it passed only because this file
    sorts last, and adding `-n auto` (xdist does not preserve file grouping) or
    renaming the file broke five tests in other files with
    `OperationalError: port 1 failed`.
    """
    # Evicting `app` and `database` alone is not enough. Every controller and
    # endpoint module did `from database import get_db`, capturing a function
    # closed over that module's `SessionLocal` — so a cached controller keeps
    # using the engine it was first imported with, and restoring two entries
    # leaves the dead one reachable through all of them.
    roots = ('app', 'database', 'security')
    packages = ('models', 'controllers', 'endpoints')

    def _owned(name: str) -> bool:
        return name in roots or name.split('.')[0] in packages

    original = {name: module for name, module in sys.modules.items() if _owned(name)}

    def _load(url: str):
        for name in list(sys.modules):
            if _owned(name):
                del sys.modules[name]
        monkeypatch.setenv('DATABASE_URL', url)
        return importlib.import_module('app')

    yield _load

    for name in list(sys.modules):
        if _owned(name):
            del sys.modules[name]
    sys.modules.update(original)


def test_importing_the_app_does_not_connect(tmp_path, monkeypatch, fresh_app):
    """A `DATABASE_URL` nothing is listening on must still import cleanly.

    `create_engine` is lazy, so this passes as long as no top-level statement
    opens a connection. It fails the moment `create_all()` or a migration moves
    back to module scope.
    """
    monkeypatch.setenv('API_TOKEN', 'x')
    module = fresh_app('postgresql://nobody:nothing@127.0.0.1:1/absent')
    assert module.app is not None

    with pytest.raises(Exception):
        # Proof the URL really is unreachable, so the clean import above was
        # laziness rather than a live connection to something else. `inspect()`
        # is what opens the connection in SQLAlchemy 2.x, so it goes inside.
        inspect(module.engine).get_table_names()


def test_startup_creates_the_tables(tmp_path, monkeypatch, fresh_app):
    monkeypatch.setenv('API_TOKEN', 'x')
    database = tmp_path / 'startup.db'
    module = fresh_app(f'sqlite:///{database}')

    assert inspect(module.engine).get_table_names() == [], (
        'tables exist before startup, so something created them at import'
    )

    from fastapi.testclient import TestClient

    with TestClient(module.app) as client:
        assert client.get('/').status_code == 200
        tables = set(inspect(module.engine).get_table_names())

    for expected in ('predictions', 'positions', 'account', 'equity_curve',
                     'model_runs', 'calibration'):
        assert expected in tables, f'{expected} missing after startup: {sorted(tables)}'


def test_bootstrap_is_idempotent(tmp_path, monkeypatch, fresh_app):
    """Every worker runs it. Running it twice must be a no-op, not an error."""
    monkeypatch.setenv('API_TOKEN', 'x')
    database = tmp_path / 'twice.db'
    module = fresh_app(f'sqlite:///{database}')

    module.bootstrap_schema()
    first = set(inspect(module.engine).get_table_names())
    module.bootstrap_schema()
    assert set(inspect(module.engine).get_table_names()) == first
