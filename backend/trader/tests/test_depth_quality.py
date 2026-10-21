"""Tests that `venue_depth.quality` is derived from the book.

A crossed book or a price outside [0, 1] is marked 'suspicious', not dropped. A default
`min_quality='valid'` read then excludes it.
"""

from __future__ import annotations

import pytest

from scripts.build_depth import _row


def test_a_sane_book_is_valid():
    assert _row(yes_bid=0.40, yes_ask=0.42)['quality'] == 'valid'


def test_a_crossed_book_is_flagged():
    assert _row(yes_bid=0.80, yes_ask=0.20)['quality'] == 'suspicious'


def test_a_touching_book_is_still_valid():
    """bid == ask is a zero spread, not a crossed book."""
    assert _row(yes_bid=0.50, yes_ask=0.50)['quality'] == 'valid'


@pytest.mark.parametrize('bid,ask', [(-0.1, 0.5), (0.5, 1.4), (1.2, 1.3)])
def test_a_price_outside_zero_to_one_is_flagged(bid, ask):
    assert _row(yes_bid=bid, yes_ask=ask)['quality'] == 'suspicious'


def test_a_one_sided_book_is_not_flagged():
    """A one-sided book is a real venue state, so it is not flagged."""
    assert _row(yes_bid=0.40, yes_ask=None)['quality'] == 'valid'
    assert _row(yes_bid=None, yes_ask=0.42)['quality'] == 'valid'


def test_an_explicit_quality_still_wins():
    """An explicit quality from the caller is kept."""
    assert _row(yes_bid=0.40, yes_ask=0.42, quality='suspicious')['quality'] \
        == 'suspicious'
