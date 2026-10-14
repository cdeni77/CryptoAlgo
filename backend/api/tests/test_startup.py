"""Schema creation runs at startup, not at module import.

Several uvicorn workers import `app.py`. Import must not touch the database,
and the tables must exist once the app has started.
"""

from __future__ import annotations

import importlib
import sys

import pytest
from sqlalchemy import inspect


@pytest.fixture
def fresh_app(monkeypatch):
    """Import `app` against a given URL, then restore the session.

    Restores `DATABASE_URL` and the evicted modules, so other tests do not
    inherit a dead database.
    """
    # Evict the packages too: each `from database import get_db` captures the
    # engine it was first imported with.
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

    `create_engine` is lazy, so this fails only if a top-level statement opens
    a connection.
    """
    module = fresh_app('postgresql://nobody:nothing@127.0.0.1:1/absent')
    assert module.app is not None

    with pytest.raises(Exception):
        # Proves the URL is unreachable. `inspect()` opens the connection in
        # SQLAlchemy 2.x.
        inspect(module.engine).get_table_names()


def test_startup_creates_the_tables(tmp_path, monkeypatch, fresh_app):
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
    database = tmp_path / 'twice.db'
    module = fresh_app(f'sqlite:///{database}')

    module.bootstrap_schema()
    first = set(inspect(module.engine).get_table_names())
    module.bootstrap_schema()
    assert set(inspect(module.engine).get_table_names()) == first
