"""The stream component's contract with the rest of the live process.

The gate assertion is the load-bearing one. Every other recorder waits on
`TradingGate.idle()`; this one must not, because a stream that pauses goes stale
and staleness is the failure the book cache exists to prevent.
"""
from __future__ import annotations

import inspect

import pytest

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent
from scripts import record_stream, run_live


def snap(ticker='K', **over):
    base = dict(venue='kalshi', market_ticker=ticker, kind='snapshot',
                received=0.0, seq=1, yes=[(0.3, 10.0)], no=[(0.65, 4.0)])
    base.update(over)
    return BookEvent(**base)


def test_stream_is_a_known_component():
    assert 'stream' in run_live.NAMES


def test_the_stream_can_be_disabled_and_everything_else_still_runs():
    selected = run_live.Component.selected(run_live.NAMES, disable=['stream'])
    assert 'stream' not in selected and 'trade' in selected


def test_an_unknown_component_is_still_refused():
    with pytest.raises(ValueError):
        run_live.Component.selected(run_live.NAMES, disable=['strem'])


def test_the_stream_never_waits_on_the_trading_gate():
    """Exactly one gate wait, and it sits inside the flush branch.

    Every other recorder gates its whole cycle. Gating the read loop here would
    let frames queue in the socket while a decision runs, and the book would be
    stale by exactly the amount the gate held — which is the failure the cache
    exists to prevent.
    """
    lines = inspect.getsource(record_stream.consume).splitlines()
    waits = [i for i, line in enumerate(lines) if 'gate.idle()' in line]
    assert len(waits) == 1, f'expected one gate wait, found {len(waits)}'
    branch = [i for i, line in enumerate(lines) if 'next_flush' in line and 'if' in line]
    assert branch and branch[0] < waits[0], 'the gate wait must be inside the flush branch'
    following = '\n'.join(lines[waits[0]:waits[0] + 3])
    assert 'flush' in following, 'the gate must guard the flush and nothing else'


def test_settled_markets_are_forgotten_rather_than_sampled_as_live():
    """A settled Kalshi market serves an EMPTY ladder, not an error."""
    cache = BookCache(now=lambda: 0.0)
    cache.apply(snap('OLD'))
    cache.apply(snap('NEW'))
    record_stream.retire(cache, keep={'NEW'})
    assert cache.ladder('OLD') is None
    assert cache.ladder('NEW') is not None


def test_retiring_nothing_when_everything_is_still_open():
    cache = BookCache(now=lambda: 0.0)
    cache.apply(snap('A'))
    record_stream.retire(cache, keep={'A', 'B'})
    assert cache.tickers() == ['A']


def test_the_process_wide_cache_is_the_one_consumers_read():
    assert isinstance(record_stream.CACHE, BookCache)


def test_the_component_is_wired_into_the_factories():
    gate = run_live.TradingGate()
    args = run_live.build_parser().parse_args([])
    assert 'stream' in run_live.build_factories(args, gate)
