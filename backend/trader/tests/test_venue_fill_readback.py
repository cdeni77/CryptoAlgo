"""Tests for `filled_from_order` on Kalshi V2 order replies.

The fill comes from `fill_count_fp`, not the status: a partial fill can come back
`canceled`.
"""

from __future__ import annotations

from scripts.live import filled_from_order


def order(**over) -> dict:
    """A V2 order with the keys the venue serves."""
    base = {
        'order_id': '01a03bd6-c108-7335-84aa-4846d8aa1629',
        'client_order_id': 'BTC-USD-202608260200-12',
        'ticker': 'KXBTC15M-26AUG252215-15', 'side': 'yes', 'book_side': 'ask',
        'action': 'sell', 'outcome_side': 'no', 'status': 'canceled',
        'initial_count_fp': '5.00', 'fill_count_fp': '0.00',
        'remaining_count_fp': '0.00', 'no_price_dollars': '0.7600',
        'yes_price_dollars': '0.2400', 'taker_fill_cost_dollars': '0.000000',
        'taker_fees_dollars': '0.000000', 'exchange_index': 2,
    }
    base.update(over)
    return base


def test_a_partial_immediate_or_cancel_fill_is_booked_not_dropped():
    """Three of five taken and the rest cancelled: three are booked."""
    filled, _ = filled_from_order(
        order(status='canceled', fill_count_fp='3.00',
              remaining_count_fp='0.00',
              taker_fill_cost_dollars='2.280000'), 5)
    assert filled == 3


def test_a_genuine_kill_still_books_nothing():
    assert filled_from_order(order(fill_count_fp='0.00'), 5)[0] == 0


def test_a_full_fill_is_read_from_the_count_not_the_status():
    filled, _ = filled_from_order(
        order(status='executed', fill_count_fp='5.00'), 5)
    assert filled == 5


def test_the_venue_count_wins_over_an_optimistic_status():
    """`status='executed'` with two of five filled is two, not five."""
    filled, _ = filled_from_order(
        order(status='executed', fill_count_fp='2.00'), 5)
    assert filled == 2


def test_a_cancel_never_infers_a_fill_from_the_remaining_count():
    """A cancelled order never infers a fill from `remaining_count_fp`.

    Remaining goes to zero because the order left the book, not because it traded.
    Remaining is meaningful only while the order is still live.
    """
    served = order(status='canceled', remaining_count_fp='0.00')
    served.pop('fill_count_fp')
    assert filled_from_order(served, 5)[0] == 0


def test_remaining_count_fp_is_used_while_the_order_is_still_live():
    served = order(status='resting', remaining_count_fp='1.00')
    served.pop('fill_count_fp')
    assert filled_from_order(served, 5)[0] == 4


def test_a_reply_with_no_counts_at_all_is_still_not_a_position():
    served = {'order_id': 'x', 'status': None}
    assert filled_from_order(served, 5)[0] == 0


def test_the_fill_price_is_cost_over_count_for_a_yes_buy():
    """The fill price is `taker_fill_cost_dollars` / count: 4 YES for $0.76 is 19c each."""
    _, price = filled_from_order(
        order(status='executed', outcome_side='yes', side='yes',
              book_side='bid', action='buy', fill_count_fp='4.00',
              taker_fill_cost_dollars='0.760000'), 4)
    assert abs(price - 0.19) < 1e-9


def test_the_fill_price_is_returned_yes_denominated_for_a_no_buy():
    """A NO buy returns a YES-denominated price (`1 - paid`), since the caller inverts for DOWN."""
    _, price = filled_from_order(
        order(status='executed', outcome_side='no', fill_count_fp='1.00',
              taker_fill_cost_dollars='0.070000'), 1)
    assert abs(price - 0.93) < 1e-9
