"""Tests that the Polymarket ladder recorder retries a transient 403.

`clob.polymarket.com` sometimes returns 403 briefly (an edge throttle) and then
recovers. A bounded retry keeps that minute's book.
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.record_pm_ladder import _get


class FakeResponse:
    def __init__(self, status: int, body: str = '{}'):
        self.status = status
        self._body = body

    async def text(self) -> str:
        return self._body

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise RuntimeError(f'HTTP {self.status}')

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    """Replays a scripted sequence of responses, one per call to .get()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def get(self, url, headers=None):
        self.calls += 1
        if not self._responses:
            raise AssertionError('FakeSession ran out of scripted responses')
        return self._responses.pop(0)


async def _sleepless(_seconds):
    return None


def test_a_single_transient_403_is_retried_and_recovered(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([
        FakeResponse(403, '{"error":"forbidden"}'),
        FakeResponse(200, '{"bids": [], "asks": []}'),
    ])
    result = asyncio.run(_get(session, 'https://clob.polymarket.com/book?token_id=x'))
    assert result == {'bids': [], 'asks': []}
    assert session.calls == 2, 'must have retried once after the 403'


def test_a_persistent_403_eventually_raises_rather_than_looping_forever(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(403)] * 5)
    with pytest.raises(RuntimeError):
        asyncio.run(_get(session, 'https://clob.polymarket.com/book?token_id=x'))
    assert session.calls <= 4, 'retries must be bounded, not unlimited'


def test_a_clean_200_is_not_retried(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(200, '{"ok": true}')])
    result = asyncio.run(_get(session, 'https://gamma-api.polymarket.com/markets'))
    assert result == {'ok': True}
    assert session.calls == 1


def test_a_genuine_404_is_not_retried_as_if_it_were_transient(monkeypatch):
    """A 404 is not retried; repeating a bad request gives the same result."""
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(404, '{"error":"not found"}')])
    with pytest.raises(RuntimeError):
        asyncio.run(_get(session, 'https://gamma-api.polymarket.com/markets?slug=bogus'))
    assert session.calls == 1, 'a 404 must fail immediately, not retry'
