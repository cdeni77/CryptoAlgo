"""Test harness for the API.

The app runs against SQLite here rather than Postgres. That is a real limitation
— the `ADD COLUMN IF NOT EXISTS` migrations in `app.py` are Postgres syntax and
are skipped — but the alternative is no API tests at all, which is where this
started. What these tests cover is the layer above SQL: authentication, argument
validation, response shapes, and whether a missing measurement comes back as null
instead of an invented number. Those are the behaviours that were broken.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))


@pytest.fixture(scope='session', autouse=True)
def sqlite_database(tmp_path_factory):
    """Point the app at a throwaway SQLite file before anything imports it."""
    path = tmp_path_factory.mktemp('api') / 'test.db'
    os.environ['DATABASE_URL'] = f'sqlite:///{path}'
    os.environ.setdefault('TRADER_DIR', str(API_ROOT.parent / 'trader'))
    yield path


@pytest.fixture
def clean_token(monkeypatch):
    """No API token configured — the fail-closed state."""
    monkeypatch.delenv('API_TOKEN', raising=False)


@pytest.fixture
def with_token(monkeypatch):
    monkeypatch.setenv('API_TOKEN', 'test-token-value')
    return 'test-token-value'


@pytest.fixture
def client(sqlite_database):
    """A TestClient over the real app.

    Nothing is stubbed. `app.run_migrations` checks the dialect and skips its
    Postgres-only ALTERs on SQLite, which is why this works without patching the
    engine — the guard belongs in the app, not in the harness.

    Entered as a context manager because schema creation moved from module
    import into the lifespan hook, and `TestClient` only runs lifespan inside
    `with`. Constructed bare, the tables would never exist.
    """
    from fastapi.testclient import TestClient

    import app as app_module

    with TestClient(app_module.app) as test_client:
        yield test_client


@pytest.fixture
def empty_models_dir(tmp_path, monkeypatch):
    """A models directory with no artifact and no ledger."""
    directory = tmp_path / 'models'
    directory.mkdir()
    monkeypatch.setenv('MODELS_DIR', str(directory))
    return directory
