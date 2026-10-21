"""Tests for `kill_diagnosis`, which explains why an order did not fill.

The answers: the limit never crossed, it crossed but nothing rested, or it crossed with
size resting. Prices are YES-denominated.
"""
from __future__ import annotations

import math

import pytest

from data_collection.kalshi_client import parse_orderbook
from scripts.live import kill_diagnosis


# --- parsing the venue's actual shape -------------------------------------

# The venue's shape: nested under `orderbook_fp`, prices and sizes both decimal
# strings, and the deci-cent grid present in the tails.
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
    # The YES ask is 1 - best NO bid. This is a conversion, not a convention.
    assert book['yes_ask'] == pytest.approx(0.34)
    assert book['yes_bid_size'] == pytest.approx(3146.78)
    assert book['no_bid_size'] == pytest.approx(8479.43)


def test_an_empty_book_is_nan_and_not_an_exception():
    """An empty book gives NaN, not an exception, so the kill path keeps its record."""
    book = parse_orderbook({'orderbook': {'orderbook_fp': {}}})
    assert math.isnan(book['yes_bid']) and math.isnan(book['yes_ask'])


# --- the three verdicts ---------------------------------------------------

def test_a_bid_under_the_ask_names_the_price_as_ours():
    """side=up sends a YES bid, which fills only at or above the YES ask."""
    book = parse_orderbook(LIVE)  # yes_ask 0.34
    d = kill_diagnosis(side='up', sent_yes_price=0.31, book=book)
    assert d['crossed'] is False
    assert 'did not cross' in d['line']
    assert '0.34' in d['line']  # the price it needed


def test_a_bid_that_crossed_with_size_resting_indicts_the_order():
    """Crossed with 8479 contracts resting: neither price nor size caused the kill."""
    book = parse_orderbook(LIVE)
    d = kill_diagnosis(side='up', sent_yes_price=0.40, book=book)
    assert d['crossed'] is True
    assert d['size_available'] == pytest.approx(8479.43)
    assert 'crossed' in d['line'] and 'resting' in d['line']


def test_the_down_side_is_judged_against_the_yes_BID():
    """Buying NO is selling YES, so it fills at or below the YES bid (0.33)."""
    book = parse_orderbook(LIVE)
    assert kill_diagnosis(side='down', sent_yes_price=0.30, book=book)['crossed'] is True
    assert kill_diagnosis(side='down', sent_yes_price=0.36, book=book)['crossed'] is False
