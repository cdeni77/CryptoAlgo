"""Tests for the container healthcheck, which reads each component's heartbeat.

`supervise` restarts a component that raises or returns, but cannot tell a
coroutine that is waiting from one that will wait forever. A stale heartbeat
catches the second case.
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
    """A component that never reported fails the check."""
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
    """Only the requested components are checked."""
    heartbeat.touch('trade')
    assert main(['--components', 'trade', '--quiet']) == 0


def test_the_environment_can_name_the_components(monkeypatch):
    heartbeat.touch('trade')
    monkeypatch.setenv('LIVE_COMPONENTS', 'trade')
    assert main(['--quiet']) == 0
    monkeypatch.setenv('LIVE_COMPONENTS', 'trade,ladder')
    assert main(['--quiet']) == 1


def test_a_write_failure_does_not_raise(monkeypatch):
    """A heartbeat write failure does not raise into the component."""
    monkeypatch.setattr(heartbeat, 'ROOT', heartbeat.Path('/proc/nonexistent/x'))
    heartbeat.touch('trade')  # must not raise
    assert heartbeat.age('trade') is None


def test_every_component_has_a_limit():
    from scripts.run_live import NAMES

    missing = [n for n in NAMES if n not in heartbeat.MAX_AGE]
    assert not missing, f'no staleness limit for {missing}'
