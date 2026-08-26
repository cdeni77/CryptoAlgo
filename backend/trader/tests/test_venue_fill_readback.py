"""Reading a fill back from Kalshi V2, which does not use the documented keys.

Pulled from the live account 2026-08-26. Every order the venue has ever returned
to this system carries `fill_count_fp` / `remaining_count_fp` — fixed-point
decimal STRINGS — and none of `taker_fill_count`, `filled_count`, `fill_count`
or `remaining_count`, which is what `filled_from_order` looked for. It never
found a count and fell through to `status`.

That fallback is right only for all-or-nothing outcomes, which is all we had
while the client sent `fill_or_kill`. Under `immediate_or_cancel` a partial fill
comes back `status='canceled'` with a non-zero `fill_count_fp`, and the fallback
reads it as nothing filled: the contracts are bought and paid for, and no
position is booked. That is the one error the docstring says cannot be
reconciled later, arrived at from the opposite direction.

Fills are also genuinely fractional here — one real fill came back as 0.43 + 0.57
on the same order — so the count is a float and only the money is exact.
"""

from __future__ import annotations

from scripts.live import filled_from_order


def order(**over) -> dict:
    """A V2 order exactly as the venue serves it (keys from a live account)."""
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
    """Three of five taken, the rest cancelled. The three are ours."""
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
    """The live kill reported `remaining_count_fp='0.00'` with nothing filled.

    Remaining goes to zero because the order left the book, not because it
    traded. `requested - remaining` on a cancelled order therefore invents a
    full position out of a total miss, which is the most expensive direction to
    be wrong in. Remaining is only meaningful while the order is still live.
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
    """`taker_fill_cost_dollars` is the money actually paid for the side bought.

    Live: 4 SOL YES at a 30c limit came back `taker_fill_cost_dollars=0.76`,
    which is 19c each. Until now no price key on a V2 order matched the ones
    this function looked for, so it returned nan and the position was booked at
    the price we *expected* rather than the one we paid.
    """
    _, price = filled_from_order(
        order(status='executed', outcome_side='yes', side='yes',
              book_side='bid', action='buy', fill_count_fp='4.00',
              taker_fill_cost_dollars='0.760000'), 4)
    assert abs(price - 0.19) < 1e-9


def test_the_fill_price_is_returned_yes_denominated_for_a_no_buy():
    """The caller inverts for DOWN, so a NO buy must come back as `1 - paid`.

    Live: 1 SOL NO paid `taker_fill_cost_dollars=0.07`, and the log recorded
    7c. The cost field is already in NO terms, so returning it unchanged would
    have the caller invert a price that needs no inverting and book 93c.
    """
    _, price = filled_from_order(
        order(status='executed', outcome_side='no', fill_count_fp='1.00',
              taker_fill_cost_dollars='0.070000'), 1)
    assert abs(price - 0.93) < 1e-9
