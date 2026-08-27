from __future__ import annotations

import pytest

from data_collection.stream.base import BookEvent


def test_a_zero_size_level_is_a_removal_not_a_price():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='delta',
                      received=1.0, seq=7, yes=[(0.31, 0.0)], no=[])
    assert event.yes == [(0.31, 0.0)]
    assert event.is_delta and not event.is_snapshot


def test_kind_must_be_snapshot_or_delta():
    with pytest.raises(ValueError):
        BookEvent(venue='kalshi', market_ticker='K', kind='update',
                  received=1.0, seq=None, yes=[], no=[])


def test_a_snapshot_is_always_absolute_whatever_the_caller_passed():
    event = BookEvent(venue='kalshi', market_ticker='K', kind='snapshot',
                      received=1.0, seq=1, yes=[], no=[], absolute=False)
    assert event.absolute, (
        'a snapshot IS the book; folding it into what was there would double it')


def test_a_delta_keeps_the_convention_it_was_given():
    assert not BookEvent(venue='kalshi', market_ticker='K', kind='delta',
                         received=1.0, seq=2, yes=[(0.3, -5.0)], no=[],
                         absolute=False).absolute
