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


def _fresh_app(url: str):
    """Import `app` against `url`, evicting anything that cached an engine."""
    for module in ('app', 'database'):
        sys.modules.pop(module, None)
    import os
    os.environ['DATABASE_URL'] = url
    return importlib.import_module('app')


def test_importing_the_app_does_not_connect(tmp_path, monkeypatch):
    """A `DATABASE_URL` nothing is listening on must still import cleanly.

    `create_engine` is lazy, so this passes as long as no top-level statement
    opens a connection. It fails the moment `create_all()` or a migration moves
    back to module scope.
    """
    monkeypatch.setenv('API_TOKEN', 'x')
    module = _fresh_app('postgresql://nobody:nothing@127.0.0.1:1/absent')
    assert module.app is not None

    with pytest.raises(Exception):
        # Proof the URL really is unreachable, so the clean import above was
        # laziness rather than a live connection to something else. `inspect()`
        # is what opens the connection in SQLAlchemy 2.x, so it goes inside.
        inspect(module.engine).get_table_names()


def test_startup_creates_the_tables(tmp_path, monkeypatch):
    monkeypatch.setenv('API_TOKEN', 'x')
    database = tmp_path / 'startup.db'
    module = _fresh_app(f'sqlite:///{database}')

    assert inspect(module.engine).get_table_names() == [], (
        'tables exist before startup, so something created them at import'
    )

    from fastapi.testclient import TestClient

    with TestClient(module.app) as client:
        assert client.get('/').status_code == 200
        tables = set(inspect(module.engine).get_table_names())

    for expected in ('signals', 'paper_positions', 'wallet'):
        assert expected in tables, f'{expected} missing after startup: {sorted(tables)}'


def test_bootstrap_is_idempotent(tmp_path, monkeypatch):
    """Every worker runs it. Running it twice must be a no-op, not an error."""
    monkeypatch.setenv('API_TOKEN', 'x')
    database = tmp_path / 'twice.db'
    module = _fresh_app(f'sqlite:///{database}')

    module.bootstrap_schema()
    first = set(inspect(module.engine).get_table_names())
    module.bootstrap_schema()
    assert set(inspect(module.engine).get_table_names()) == first
