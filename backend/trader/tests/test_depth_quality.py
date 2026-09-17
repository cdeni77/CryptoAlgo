"""`venue_depth.quality` must be derived from the book, not asserted.

Every row this module produced carried the literal `quality='valid'`, so
`min_quality` — the mechanism that exists so flagged data cannot reach a
feature build — was inert for the dataset it matters most for.

Measured 2026-09-16 on the real store: all 1,895,986 rows read 'valid', and
**37,052 are CROSSED** (`yes_bid > yes_ask`) — 31,541 of them in 2026-04, which
is 37% of that month. `core/quotes.py` calls a crossed book "not a book" and
refuses it at feature time; `core/book_flow.py` did not until today. Nothing
upstream said so and nothing counted them.

Written as 'suspicious' rather than dropped: the rows are real evidence about
the backfill's quality, and a default `min_quality='valid'` read already
excludes them.
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
    """Absent is not wrong. A one-sided book is a real state of the venue, and
    flagging it would discard the side that does exist."""
    assert _row(yes_bid=0.40, yes_ask=None)['quality'] == 'valid'
    assert _row(yes_bid=None, yes_ask=0.42)['quality'] == 'valid'


def test_an_explicit_quality_still_wins():
    """A caller that already knows better is not overridden upward."""
    assert _row(yes_bid=0.40, yes_ask=0.42, quality='suspicious')['quality'] \
        == 'suspicious'
