"""The container healthcheck must assert the loop is working.

The one it replaces opened a Postgres connection and closed it. It never
touched the trading loop, the venue, the stream or the model — and nothing
`depends_on` it, with no autoheal, so under `restart: unless-stopped` even a
FAILING check changed nothing. The process could be wedged, abstaining on every
window, or scoring all-NaN features while `docker compose ps` read
`Up (healthy)`. Verified: five hours of exactly that.

`supervise` restarts a component that raises or returns and gets both right.
What it cannot distinguish is a coroutine legitimately awaiting from one that
will await forever — the documented websocket deadlock, and the store-sync
subprocess that had no timeout. Both presented as a healthy container with one
log line missing, and an absent log line is not something anyone is required to
notice.
"""

from __future__ import annotations

import time

import pytest

from core import heartbeat
from scripts.healthcheck import main


@pytest.fixture(autouse=True)
def _root(tmp_path, monkeypatch):
    monkeypatch.setattr(heartbeat, 'ROOT', tmp_path)
    return tmp_path


def test_a_fresh_component_passes():
    heartbeat.touch('ladder')
    assert main(['--components', 'ladder', '--quiet']) == 0


def test_a_component_that_never_reported_fails():
    """"Not started yet" and "died before its first cycle" look identical from
    outside, and the second is the one worth catching."""
    assert main(['--components', 'ladder', '--quiet']) == 1


def test_a_stale_component_fails(_root):
    (_root / 'ladder').write_text(str(time.time() - 10_000))
    assert main(['--components', 'ladder', '--quiet']) == 1


def test_one_dead_component_fails_the_whole_check():
    heartbeat.touch('ladder')
    heartbeat.touch('trade')
    assert main(['--components', 'ladder,trade', '--quiet']) == 0
    assert main(['--components', 'ladder,trade,stream', '--quiet']) == 1


def test_only_the_components_asked_for_are_checked():
    """`--disable stream` must not fail the container for a component nobody
    asked to run."""
    heartbeat.touch('trade')
    assert main(['--components', 'trade', '--quiet']) == 0


def test_the_environment_can_name_the_components(monkeypatch):
    heartbeat.touch('trade')
    monkeypatch.setenv('LIVE_COMPONENTS', 'trade')
    assert main(['--quiet']) == 0
    monkeypatch.setenv('LIVE_COMPONENTS', 'trade,ladder')
    assert main(['--quiet']) == 1


def test_a_write_failure_does_not_raise(monkeypatch):
    """A heartbeat that can break the thing it measures is worse than none."""
    monkeypatch.setattr(heartbeat, 'ROOT', heartbeat.Path('/proc/nonexistent/x'))
    heartbeat.touch('trade')       # must not raise
    assert heartbeat.age('trade') is None


def test_every_component_has_a_limit():
    from scripts.run_live import NAMES

    missing = [n for n in NAMES if n not in heartbeat.MAX_AGE]
    assert not missing, f'no staleness limit for {missing}'
