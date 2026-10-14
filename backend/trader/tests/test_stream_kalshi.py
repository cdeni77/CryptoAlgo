"""Tests for the Kalshi stream adapter against real captured frames.

`tests/fixtures/ws/kalshi_capture.jsonl.gz` pairs 65s of frames with REST snapshots, so
the folded book can be checked against the venue's own.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from core.stream_book import BookCache
from data_collection.stream.kalshi import parse_frame, rest_levels

FIXTURE = Path(__file__).parent / 'fixtures' / 'ws' / 'kalshi_capture.jsonl.gz'
pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason='kalshi_capture.jsonl.gz fixture is missing')


def records():
    with gzip.open(FIXTURE, 'rt') as fh:
        for line in fh:
            yield json.loads(line)


@pytest.fixture(scope='module')
def capture():
    recs = list(records())
    return ([r for r in recs if r['kind'] == 'ws'],
            [r for r in recs if r['kind'] == 'rest'])


def test_no_captured_frame_makes_the_parser_raise(capture):
    ws, _ = capture
    for rec in ws:
        parse_frame(rec['payload'], rec['t'])


def test_the_capture_holds_both_snapshots_and_deltas(capture):
    ws, _ = capture
    kinds = {e.kind for rec in ws
             if (e := parse_frame(rec['payload'], rec['t'])) is not None}
    assert kinds == {'snapshot', 'delta'}


def test_deltas_are_signed_changes_and_snapshots_are_absolute(capture):
    ws, _ = capture
    events = [e for rec in ws
              if (e := parse_frame(rec['payload'], rec['t'])) is not None]
    assert all(e.absolute for e in events if e.is_snapshot)
    assert not any(e.absolute for e in events if e.is_delta)
    assert any(s < 0 for e in events if e.is_delta for _, s in e.yes + e.no), (
        'delta_fp is a signed change; a capture with no negative would not '
        'exercise removal at all')


def test_a_non_book_frame_is_ignored_rather_than_parsed():
    assert parse_frame({'type': 'subscribed', 'id': 1,
                        'msg': {'channel': 'orderbook_delta', 'sid': 1}}, 1.0) is None
    assert parse_frame({}, 1.0) is None
    assert parse_frame(None, 1.0) is None
    assert parse_frame({'type': 'orderbook_delta', 'msg': {}}, 1.0) is None


def test_the_stream_uses_different_field_names_from_rest():
    """The stream sends `msg.yes_dollars_fp` where REST sends
    `orderbook_fp.yes_dollars`.

    Reading one shape against the other gives an empty book and no exception.
    """
    event = parse_frame({'type': 'orderbook_snapshot', 'seq': 1, 'msg': {
        'market_ticker': 'K', 'yes_dollars_fp': [['0.30', '10.00']],
        'no_dollars_fp': [['0.65', '4.00']]}}, 1.0)
    assert event.yes == [(0.30, 10.0)] and event.no == [(0.65, 4.0)]


def test_the_folded_book_matches_the_venues_own_orderbook(capture):
    """The folded book matches the venue's REST orderbook at each snapshot.

    Top of book must match exactly, since every book feature is built from it. The
    ladder may differ by at most one price per side: a REST response can catch a level
    that a market maker toggles rapidly in its other state.
    """
    ws, rest = capture
    cache = BookCache(now=lambda: 0.0)
    index = 0
    compared = 0
    for sample in rest:
        while index < len(ws) and ws[index]['t'] <= sample['t']:
            event = parse_frame(ws[index]['payload'], ws[index]['t'])
            if event is not None:
                cache.apply(event)
            index += 1
        ticker = sample['payload']['ticker']
        ladder = cache.ladder(ticker)
        if ladder is None or cache.gapped(ticker):
            continue
        want_yes, want_no = rest_levels(sample['payload'])
        for side, mine, theirs in (('YES', ladder.yes, want_yes),
                                   ('NO', ladder.no, want_no)):
            assert max(p for p, _ in mine) == max(p for p, _ in theirs), (
                f'{ticker}: best {side} bid diverged')
            drift = {p for p, _ in mine} ^ {p for p, _ in theirs}
            assert len(drift) <= 1, (
                f'{ticker}: {len(drift)} {side} prices diverged ({sorted(drift)}); '
                'more than one is a fold error, not a flickering order')
        compared += 1
    assert compared >= 9, f'only {compared} comparisons; capture a longer window'


def test_a_level_emptied_by_deltas_leaves_no_float_residue(capture):
    """Levels emptied by deltas leave no float residue that would read as a best bid."""
    ws, _ = capture
    cache = BookCache(now=lambda: 0.0)
    for rec in ws:
        event = parse_frame(rec['payload'], rec['t'])
        if event is not None:
            cache.apply(event)
    for ticker in cache.tickers():
        ladder = cache.ladder(ticker)
        tiny = [(p, s) for p, s in ladder.yes + ladder.no if s < 0.01]
        assert not tiny, f'{ticker}: residue left resting at {tiny[:3]}'
