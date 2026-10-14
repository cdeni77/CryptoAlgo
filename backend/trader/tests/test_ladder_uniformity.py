"""Tests that the Kalshi and Polymarket ladders use the same denomination.

Kalshi serves two bid stacks, so `no_levels` holds NO-side bid prices and the
YES ask is `1 - best_no_bid`. Polymarket serves bids and asks on one token, so
its asks are converted to NO-denominated levels at write time. On both venues
`1 - price` of the best `no_levels` entry is the YES ask.
"""

from __future__ import annotations

import json

from scripts.record_pm_ladder import _levels, _no_levels


BOOK = {
    # Polymarket serves bids ascending and asks descending: the touch is last.
    'bids': [{'price': '0.45', 'size': '100'}, {'price': '0.48', 'size': '200'},
             {'price': '0.49', 'size': '300'}],
    'asks': [{'price': '0.55', 'size': '150'}, {'price': '0.52', 'size': '250'},
             {'price': '0.50', 'size': '350'}],
}


def test_bids_come_back_best_first_in_yes_terms():
    """Bids are stored best first, as Kalshi does; Polymarket serves them worst first.
    """
    assert _levels(BOOK['bids']) == [[0.49, 300.0], [0.48, 200.0], [0.45, 100.0]]


def test_asks_are_stored_no_denominated_like_kalshi():
    """A YES ask of 0.50 is stored as a NO bid of 0.50; 0.55 as 0.45."""
    assert _no_levels(BOOK['asks']) == [[0.50, 350.0], [0.48, 250.0], [0.45, 150.0]]


def test_one_minus_the_best_no_level_recovers_the_yes_ask():
    """1 - the best NO level recovers the YES ask."""
    best_no = _no_levels(BOOK['asks'])[0][0]
    assert abs((1.0 - best_no) - 0.50) < 1e-9


def test_the_spread_is_one_cent_not_minus_one():
    """The spread comes out positive (one cent), not inverted."""
    best_bid = _levels(BOOK['bids'])[0][0]
    yes_ask = 1.0 - _no_levels(BOOK['asks'])[0][0]
    assert abs((yes_ask - best_bid) - 0.01) < 1e-9
