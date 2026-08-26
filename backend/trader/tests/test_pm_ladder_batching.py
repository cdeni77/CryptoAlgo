"""Batch the per-cycle Polymarket requests instead of one per asset.

Measured live: the recorder was making 6 requests a cycle — 3 separate
`GET {GAMMA}/markets?slug=` lookups and 3 separate `GET {CLOB}/book?token_id=`
lookups, one pair per asset. `clob.polymarket.com` then 403'd on the book
lookup for one particular window for over ten minutes straight (18:30-18:41),
which the existing per-request retry cannot help with — it is built for a
few-second blip, not a ten-minute one.

Two things verified directly against both live hosts before writing this:
  * `GET {GAMMA}/markets?slug=A&slug=B` returns all matching markets in one
    call, so the three per-asset lookups become one.
  * `POST {CLOB}/books` with `[{"token_id": ...}, ...]` returns every
    requested book in one call, each tagged with its own `asset_id` so the
    response can be matched back to the request without relying on order.
  * That the batch response is unordered was also verified directly, not
    assumed: `_match_books` below is tested against a reply given in the
    reverse order it was asked in.

Cutting 6 requests to 2 does not fix a genuine ten-minute outage, but it cuts
our own footprint by 3x against the specific host that has been blocking us,
which is the one lever a client actually has over "please stop throttling me".

The ten-minute failure had a second, independent cause worth fixing on its
own: the token this recorder tried for that window was 17 digits
(`10042263157051608`), not the market's real ~78-digit CTF token id
(`100422631570516080548716502776163316714562570823066703621126328689672049562711`,
confirmed live from both the host and the running container, using the exact
same parsing code, minutes after the failures started). A market minutes old
can have gamma serving a placeholder id before the real on-chain token
attaches, and CLOB legitimately has nothing to serve for it. `_valid_token`
rejects an implausibly short id before spending a request on it, so a
still-initializing market gets skipped and re-checked next cycle instead of
retried against a token that cannot ever work.
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.record_pm_ladder import _match_books, _post, _valid_token

REAL_TOKEN = ('100422631570516080548716502776163316714562570823066703621126'
              '328689672049562711')
PLACEHOLDER_TOKEN = '10042263157051608'  # observed live: 17 digits, 403s every time


def test_a_freshly_opened_markets_placeholder_token_is_rejected():
    assert _valid_token(PLACEHOLDER_TOKEN) is False


def test_a_real_ctf_token_is_accepted():
    assert _valid_token(REAL_TOKEN) is True


def test_a_missing_token_is_rejected():
    assert _valid_token(None) is False
    assert _valid_token('') is False


def test_match_books_pairs_by_asset_id_not_by_response_order():
    wanted = [('BTC-USD', 'tok-btc'), ('ETH-USD', 'tok-eth')]
    # Deliberately the reverse of `wanted`'s order — the batch endpoint does
    # not promise to preserve request order, and this is that verified live.
    books = [{'asset_id': 'tok-eth', 'bids': []}, {'asset_id': 'tok-btc', 'asks': []}]
    matched = _match_books(wanted, books)
    assert matched['BTC-USD']['asset_id'] == 'tok-btc'
    assert matched['ETH-USD']['asset_id'] == 'tok-eth'


def test_match_books_drops_a_token_the_batch_response_omitted():
    wanted = [('BTC-USD', 'tok-btc'), ('SOL-USD', 'tok-sol')]
    books = [{'asset_id': 'tok-btc', 'bids': []}]  # sol's book never came back
    matched = _match_books(wanted, books)
    assert 'BTC-USD' in matched
    assert 'SOL-USD' not in matched


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
    """Replays a scripted sequence of responses, one per call to .post()."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0
        self.last_json = None

    def post(self, url, headers=None, json=None):
        self.calls += 1
        self.last_json = json
        if not self._responses:
            raise AssertionError('FakeSession ran out of scripted responses')
        return self._responses.pop(0)


async def _sleepless(_seconds):
    return None


def test_post_retries_a_transient_403(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([
        FakeResponse(403, '{"error":"forbidden"}'),
        FakeResponse(200, '[{"asset_id": "x"}]'),
    ])
    result = asyncio.run(
        _post(session, 'https://clob.polymarket.com/books', [{'token_id': 'x'}]))
    assert result == [{'asset_id': 'x'}]
    assert session.calls == 2, 'must have retried once after the 403'


def test_post_a_persistent_403_eventually_raises_rather_than_looping_forever(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(403)] * 5)
    with pytest.raises(RuntimeError):
        asyncio.run(_post(session, 'https://clob.polymarket.com/books', []))
    assert session.calls <= 4, 'retries must be bounded, not unlimited'


def test_post_a_clean_200_is_not_retried(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(200, '[]')])
    result = asyncio.run(_post(session, 'https://clob.polymarket.com/books', []))
    assert result == []
    assert session.calls == 1


def test_post_sends_the_batched_payload_unchanged(monkeypatch):
    monkeypatch.setattr('scripts.record_pm_ladder.asyncio.sleep', _sleepless)
    session = FakeSession([FakeResponse(200, '[]')])
    payload = [{'token_id': 'a'}, {'token_id': 'b'}]
    asyncio.run(_post(session, 'https://clob.polymarket.com/books', payload))
    assert session.last_json == payload
