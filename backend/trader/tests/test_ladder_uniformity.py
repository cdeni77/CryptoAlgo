"""The two venues' ladders must mean the same thing, not merely share columns.

`venue_ladder` (Kalshi) and `pm_ladder` (Polymarket) carry the same column
names so the two venues can be compared on the same fifteen minutes. That is
worth nothing if the columns are denominated differently, and they were:

  * Kalshi's `/markets/{ticker}/orderbook` serves `yes_dollars` and
    `no_dollars` — two BID stacks. `no_levels` is therefore NO-side prices, and
    the YES ask is `1 - best_no_bid`. `scripts/live._cumulative` inverts it for
    exactly this reason.
  * Polymarket's CLOB serves `bids` and `asks` on ONE token. Storing `asks`
    unchanged put YES-denominated asks in the column that holds NO-denominated
    bids on the other venue.

A shared aggregate over the two would then have quietly read a 0.51 ask as a
0.51 NO bid — a 2c error at the touch and a sign error in imbalance. Same name,
different meaning, no exception raised anywhere.

So the invariant is the denomination, not the schema: on both venues
`no_levels` prices are NO-denominated and `1 - price` recovers the YES ask.
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
    """Kalshi stores best-first; Polymarket serves worst-first."""
    assert _levels(BOOK['bids']) == [[0.49, 300.0], [0.48, 200.0], [0.45, 100.0]]


def test_asks_are_stored_no_denominated_like_kalshi():
    """The best YES ask of 0.50 is a NO bid of 0.50; 0.55 is a NO bid of 0.45."""
    assert _no_levels(BOOK['asks']) == [[0.50, 350.0], [0.48, 250.0], [0.45, 150.0]]


def test_one_minus_the_best_no_level_recovers_the_yes_ask():
    """The property every shared aggregate depends on, on both venues."""
    best_no = _no_levels(BOOK['asks'])[0][0]
    assert abs((1.0 - best_no) - 0.50) < 1e-9


def test_the_spread_is_one_cent_not_minus_one():
    """Reading the raw ask as a NO price inverts the spread's sign."""
    best_bid = _levels(BOOK['bids'])[0][0]
    yes_ask = 1.0 - _no_levels(BOOK['asks'])[0][0]
    assert abs((yes_ask - best_bid) - 0.01) < 1e-9
