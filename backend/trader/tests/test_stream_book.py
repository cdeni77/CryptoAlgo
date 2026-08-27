"""The cache holds the state a wrong answer would come from, so it is tested hard.

The sequence tests encode a MEASURED fact rather than a documented one: Kalshi's
`seq` is contiguous per subscription and not per market. A per-market check
would flag every delta as a gap.
"""
from __future__ import annotations

import pytest

from core.stream_book import BookCache
from data_collection.stream.base import BookEvent


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def snap(**over):
    base = dict(venue='kalshi', market_ticker='K', kind='snapshot',
                received=1000.0, seq=1, yes=[(0.30, 100.0), (0.31, 50.0)],
                no=[(0.68, 20.0)])
    base.update(over)
    return BookEvent(**base)


def test_a_snapshot_replaces_the_book_rather_than_merging():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(seq=2, yes=[(0.40, 7.0)], no=[]))
    assert cache.ladder('K').yes == [(0.40, 7.0)]
    assert cache.ladder('K').no == []


def test_an_absolute_delta_sets_the_size_at_a_price():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, yes=[(0.30, 5.0)], no=[]))
    assert dict(cache.ladder('K').yes)[0.30] == 5.0
    assert dict(cache.ladder('K').yes)[0.31] == 50.0, 'untouched levels survive'


def test_a_signed_delta_adds_to_the_resting_size():
    cache = BookCache(now=Clock())
    cache.apply(snap())                                    # 0.30 -> 100
    cache.apply(snap(kind='delta', seq=2, absolute=False,
                     yes=[(0.30, -40.0)], no=[]))
    assert dict(cache.ladder('K').yes)[0.30] == 60.0


def test_a_signed_delta_that_empties_a_level_removes_it():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, absolute=False,
                     yes=[(0.30, -100.0)], no=[]))
    assert 0.30 not in dict(cache.ladder('K').yes)


def test_an_absolute_zero_removes_the_level():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.apply(snap(kind='delta', seq=2, yes=[(0.30, 0.0)], no=[]))
    assert 0.30 not in dict(cache.ladder('K').yes)


def test_an_unknown_ticker_is_none_not_an_empty_book():
    assert BookCache(now=Clock()).ladder('NOPE') is None, (
        'an empty ladder and no ladder are different claims')


def test_age_is_measured_from_the_last_event_and_marks_stale():
    clock = Clock()
    cache = BookCache(max_age_seconds=10.0, now=clock)
    cache.apply(snap(received=clock.t))
    clock.t += 4.0
    assert cache.ladder('K').age_seconds == pytest.approx(4.0)
    assert not cache.ladder('K').stale
    clock.t += 7.0
    assert cache.ladder('K').stale


def test_a_delta_for_a_book_we_never_snapshotted_is_refused():
    cache = BookCache(now=Clock())
    cache.apply(snap(kind='delta', seq=5, yes=[(0.30, 1.0)], no=[]))
    assert cache.ladder('K') is None, (
        'folding a delta into nothing invents a book from one level')


# -- sequence: the measured semantics ---------------------------------------

def test_interleaved_markets_on_one_connection_are_not_a_gap():
    """MEASURED: seq is contiguous across the subscription, not within a market.

    BTC reads 1, 9, 10 while the connection reads 1..N. A per-market check would
    call every one of these a gap.
    """
    cache = BookCache(now=Clock())
    cache.apply(snap(market_ticker='BTC', seq=1))
    cache.apply(snap(market_ticker='ETH', seq=2))
    cache.apply(snap(market_ticker='SOL', seq=3))
    cache.apply(snap(market_ticker='BTC', kind='delta', seq=4,
                     absolute=False, yes=[(0.30, 1.0)], no=[]))
    cache.apply(snap(market_ticker='ETH', kind='delta', seq=5,
                     absolute=False, yes=[(0.30, 1.0)], no=[]))
    assert not cache.any_gapped()


def test_a_gap_marks_every_book_on_the_connection():
    cache = BookCache(now=Clock())
    cache.apply(snap(market_ticker='BTC', seq=1))
    cache.apply(snap(market_ticker='ETH', seq=2))
    cache.apply(snap(market_ticker='BTC', kind='delta', seq=9,
                     absolute=False, yes=[(0.30, 1.0)], no=[]))
    assert cache.gapped('BTC') and cache.gapped('ETH'), (
        'the missed frame could have belonged to either book')


def test_a_snapshot_clears_the_gap_for_its_own_book():
    cache = BookCache(now=Clock())
    cache.apply(snap(seq=1))
    cache.apply(snap(kind='delta', seq=9, absolute=False,
                     yes=[(0.30, 1.0)], no=[]))
    assert cache.gapped('K')
    cache.apply(snap(seq=10))
    assert not cache.gapped('K')


def test_a_sequence_restart_after_reconnect_is_not_a_gap():
    cache = BookCache(now=Clock())
    cache.apply(snap(seq=500))
    cache.apply(snap(market_ticker='B', seq=1))
    assert not cache.any_gapped()


def test_no_seq_means_no_gap_detection_rather_than_a_false_gap():
    cache = BookCache(now=Clock())
    cache.apply(snap(seq=None))
    cache.apply(snap(kind='delta', seq=None, absolute=False,
                     yes=[(0.30, 1.0)], no=[]))
    assert not cache.any_gapped()


def test_forgetting_a_settled_market_makes_it_unreadable():
    cache = BookCache(now=Clock())
    cache.apply(snap())
    cache.forget('K')
    assert cache.ladder('K') is None


def test_the_cache_never_branches_on_a_venue_name():
    import inspect

    import core.stream_book as mod
    source = inspect.getsource(mod)
    body = source[source.index('class BookCache'):]
    assert 'kalshi' not in body.lower() and 'polymarket' not in body.lower(), (
        'the cache must stay venue-blind; the convention rides on the event')
