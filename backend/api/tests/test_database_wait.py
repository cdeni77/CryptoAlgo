"""Startup waits for the database instead of exiting.

A host reboot restarts every container at once through the restart policy,
which does not honour `depends_on`. Postgres answers "the database system is
starting up" while it recovers. Exiting on that left uvicorn's reloader
running as the container's main process with no application behind it, so the
container read `Up` and the proxy returned 502 for ten hours.
"""

from __future__ import annotations

import time

import pytest
from sqlalchemy.exc import OperationalError


@pytest.fixture
def app_module(sqlite_database):
    """Imported late: `conftest` sets DATABASE_URL in a session fixture, and
    `database.py` reads it at import time."""
    import app

    return app


class FakeEngine:
    """Fails `fail_times` times, then connects."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.attempts = 0

    def connect(self):
        self.attempts += 1
        if self.attempts <= self.fail_times:
            raise OperationalError(
                "SELECT 1", {}, Exception("the database system is starting up"))
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, *a, **k):
        return None


@pytest.fixture
def no_sleep(monkeypatch):
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    return slept


def test_it_retries_until_the_database_answers(app_module, monkeypatch, no_sleep):
    engine = FakeEngine(fail_times=3)
    monkeypatch.setattr(app_module, "engine", engine)

    app_module.wait_for_database(timeout_seconds=60, interval_seconds=0.01)

    assert engine.attempts == 4, "it must keep trying, then succeed"
    assert len(no_sleep) == 3, "it must pause between attempts"


def test_it_returns_immediately_when_the_database_is_up(app_module, monkeypatch, no_sleep):
    engine = FakeEngine(fail_times=0)
    monkeypatch.setattr(app_module, "engine", engine)

    app_module.wait_for_database(timeout_seconds=60)

    assert engine.attempts == 1 and no_sleep == []


def test_it_still_fails_loudly_when_the_database_never_comes(app_module,
                                                             monkeypatch,
                                                             no_sleep):
    """A wait that never ends would hide a real outage."""
    engine = FakeEngine(fail_times=10_000)
    monkeypatch.setattr(app_module, "engine", engine)

    with pytest.raises(OperationalError):
        app_module.wait_for_database(timeout_seconds=0, interval_seconds=0.01)


def test_the_lifespan_waits_before_it_touches_the_schema(app_module, monkeypatch):
    """Order matters: bootstrap_schema is the call that was dying."""
    calls = []
    monkeypatch.setattr(app_module, "wait_for_database",
                        lambda *a, **k: calls.append("wait"))
    monkeypatch.setattr(app_module, "bootstrap_schema",
                        lambda: calls.append("bootstrap"))

    import asyncio

    async def drive():
        async with app_module.lifespan(None):
            pass

    asyncio.run(drive())
    assert calls == ["wait", "bootstrap"]
