"""The shard is remembered so the book can be read after reconciliation.

Kalshi shards its exchange by category and balances are LOCAL to a shard, so the
balance query has to name the right one. The only place it was ever read from
was the quotes — which is what forced the book to be fetched first, and cost
about a second of staleness on every order.
"""
from __future__ import annotations

import pytest

from scripts import live


class _Q:
    def __init__(self, exchange_index):
        self.exchange_index = exchange_index


@pytest.fixture(autouse=True)
def _clear():
    live._LAST_EXCHANGE_INDEX = None
    yield
    live._LAST_EXCHANGE_INDEX = None


def _quotes(index):
    return {'BTC-USD': (_Q(index), 'KXBTC15M-X')}


def test_nothing_is_known_before_a_book_has_been_read():
    assert live._LAST_EXCHANGE_INDEX is None, (
        'until a book is seen the balance query falls back to its own default, '
        'which is what already happens on a cycle where no symbol quoted')


def test_the_shard_is_learned_from_the_book_and_kept():
    assert live.remember_exchange_index(_quotes(2)) == 2
    assert live._LAST_EXCHANGE_INDEX == 2


def test_an_empty_book_does_not_erase_what_was_learned():
    live.remember_exchange_index(_quotes(2))
    assert live.remember_exchange_index({}) == 2
    assert live._LAST_EXCHANGE_INDEX == 2, (
        'a cycle where nothing quoted must not reset the shard to unknown')


def test_a_moved_shard_is_logged_loudly_and_adopted(caplog):
    """A stale shard reads the wrong balance — the failure that once had every
    order refused `insufficient_balance` while the funds sat elsewhere."""
    live.remember_exchange_index(_quotes(0))
    with caplog.at_level('WARNING'):
        assert live.remember_exchange_index(_quotes(2)) == 2
    assert any('exchange shard 0 to 2' in r.getMessage() for r in caplog.records), caplog.text
    assert live._LAST_EXCHANGE_INDEX == 2, 'the new shard is adopted, not just logged'


def test_an_unchanged_shard_is_silent(caplog):
    live.remember_exchange_index(_quotes(2))
    with caplog.at_level('WARNING'):
        live.remember_exchange_index(_quotes(2))
    assert not [r for r in caplog.records if 'exchange shard' in r.getMessage()]


def test_the_cycle_reads_the_book_after_reconciling():
    """Ordering is the whole point: everything that can precede the quote does."""
    import inspect
    source = inspect.getsource(live.run_cycle)
    assert source.index("_phase('reconcile')") < source.index("_phase('quotes')"), (
        'reconciliation must not sit between the book and the order')
    assert source.index("_phase('quotes')") < source.index("_phase('score')")
    assert source.index("_phase('bars')") < source.index("_phase('quotes')")
