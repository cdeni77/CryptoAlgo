"""The adapter is tested against real captured frames, not invented ones.

`tests/fixtures/ws/kalshi_capture.jsonl.gz` is 65 seconds of the live venue:
40,759 `orderbook_delta`/`orderbook_snapshot` frames for three markets, plus 12
`GET /markets/{t}/orderbook` snapshots taken during the same window. That
pairing is the point — it lets a test assert that folding the stream reproduces
the venue's own answer, which is the only evidence delta application is correct
rather than merely plausible.

It has already earned itself twice. It caught that the documented field names
are wrong (`price_dollars`/`delta_fp`, not `price`/`delta`), and it caught a
float-residue bug that put a phantom best bid three cents above the real touch.
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
    not FIXTURE.exists(), reason='capture a fixture with scripts.probe_ws first')


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
    """A live trap: REST says `orderbook_fp.yes_dollars`, the stream says
    `msg.yes_dollars_fp`. Reading one shape against the other gives an empty
    book and no exception."""
    event = parse_frame({'type': 'orderbook_snapshot', 'seq': 1, 'msg': {
        'market_ticker': 'K', 'yes_dollars_fp': [['0.30', '10.00']],
        'no_dollars_fp': [['0.65', '4.00']]}}, 1.0)
    assert event.yes == [(0.30, 10.0)] and event.no == [(0.65, 4.0)]


def test_the_folded_book_matches_the_venues_own_orderbook(capture):
    """THE test. Fold the stream up to each REST sample and compare.

    Measured over this fixture: **11 of 11 comparisons agree exactly on the best
    bid on BOTH sides**, and the worst whole-ladder disagreement is a single
    price out of ~100.

    That one is understood rather than tolerated. It is a NO level at 0.036 that
    a market maker toggles on and off **fourteen times in sixty-five seconds**
    (-2000, +2000, -2000, ...). A REST response reflects the server tens of
    milliseconds before it lands, so it legitimately catches a flickering level
    in the other state. It sits far from the touch and never moves the best bid.

    So the bounds are: the top of book must match EXACTLY, because it is the
    number every book feature is built from; the ladder may differ by at most
    one price per side, which is a flickering order, not a fold error. Widening
    either bound would let a real defect through.
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
    """Regression: emptied levels held 2.4e-12 and read as a phantom best bid."""
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
