"""Make a non-fill explain itself.

Five reconstructions of the 2026-09-04 fill failure each agreed with the code
and disagreed with the venue: the quote source reads correctly, the book-to-order
gap is unchanged at 0.23s, the book is thousands of contracts deep at the touch,
the orders are well-formed — and 0 of 10 filled where 165 of 224 had. Every one
of those checks was a reconstruction made minutes or hours later.

The quantity nobody records is the book we DECIDED on, and the book as it stood
the instant the venue killed the order. Without those two, a miss cannot
distinguish the only three things it can be:

  * our limit never crossed  -> a price error, ours
  * it crossed and nothing rested -> a size race, the venue's
  * it crossed with size resting  -> the order itself is malformed

`kill_diagnosis` answers that in the log line, in YES-denominated terms, because
the venue's book is two BID stacks and every previous confusion here came from
mixing denominations.
"""
from __future__ import annotations

import math

import pytest

from data_collection.kalshi_client import parse_orderbook
from scripts.live import kill_diagnosis


# --- parsing the venue's actual shape -------------------------------------

# Recorded from the live account 2026-09-04: nested under `orderbook_fp`, prices
# and sizes BOTH decimal strings, and the deci-cent grid present in the tails.
LIVE = {'orderbook': {'orderbook_fp': {
    'yes_dollars': [['0.0010', '1067552.65'], ['0.3100', '4443.00'],
                    ['0.3300', '3146.78']],
    'no_dollars':  [['0.0010', '1039419.65'], ['0.6500', '5077.80'],
                    ['0.6600', '8479.43']],
}}}


def test_it_reads_the_touch_off_the_venues_real_shape():
    book = parse_orderbook(LIVE)
    assert book['yes_bid'] == pytest.approx(0.33)
    assert book['no_bid'] == pytest.approx(0.66)
    # The YES ask is 1 - best NO bid. This is the conversion, not a convention.
    assert book['yes_ask'] == pytest.approx(0.34)
    assert book['yes_bid_size'] == pytest.approx(3146.78)
    assert book['no_bid_size'] == pytest.approx(8479.43)


def test_an_empty_book_is_nan_and_not_an_exception():
    """A settled market returns empty stacks, and a diagnostic that raises
    inside the kill path would destroy the order record it exists to explain."""
    book = parse_orderbook({'orderbook': {'orderbook_fp': {}}})
    assert math.isnan(book['yes_bid']) and math.isnan(book['yes_ask'])


# --- the three verdicts ---------------------------------------------------

def test_a_bid_under_the_ask_names_the_price_as_ours():
    """side=up sends a YES bid; it fills only at or above the YES ask."""
    book = parse_orderbook(LIVE)                      # yes_ask 0.34
    d = kill_diagnosis(side='up', sent_yes_price=0.31, book=book)
    assert d['crossed'] is False
    assert 'did not cross' in d['line']
    assert '0.34' in d['line']                        # the price it needed


def test_a_bid_that_crossed_with_size_resting_indicts_the_order():
    """Crossed, and 8479 contracts were sitting there: not price, not size."""
    book = parse_orderbook(LIVE)
    d = kill_diagnosis(side='up', sent_yes_price=0.40, book=book)
    assert d['crossed'] is True
    assert d['size_available'] == pytest.approx(8479.43)
    assert 'crossed' in d['line'] and 'resting' in d['line']


def test_the_down_side_is_judged_against_the_yes_BID():
    """Buying NO is selling YES, so it fills at or BELOW the YES bid (0.33).
    Sending 0.36 rests instead of crossing — the inverted-comparison bug."""
    book = parse_orderbook(LIVE)
    assert kill_diagnosis(side='down', sent_yes_price=0.30, book=book)['crossed'] is True
    assert kill_diagnosis(side='down', sent_yes_price=0.36, book=book)['crossed'] is False
