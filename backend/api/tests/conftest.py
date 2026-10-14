"""Test harness for the API.

The app runs against SQLite, not Postgres. The tests cover the layer above SQL:
response shapes, and a missing measurement returned as null with a reason.
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
    yield path


@pytest.fixture
def client(sqlite_database):
    """A TestClient over the real app, with nothing stubbed.

    Entered with `with`, because `TestClient` runs the lifespan hook, which
    creates the schema, only inside it.
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
