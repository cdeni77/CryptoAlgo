"""The fill model, and the side convention it is easy to invert."""
import numpy as np
import pandas as pd
import pytest

from core.fills import (DEFAULT_FILL_CURVE, FillModel, expected_fill,
                        run_through_margin)


def test_a_selling_aggressor_fills_a_resting_yes_bid():
    # Rest a YES bid at 70c. A SELLING aggressor printing at 68c ran 2c
    # through it. `taker_book_side='ask'` is that seller, so its min_price is
    # the fill test -- the opposite of the label's natural reading.
    m = run_through_margin(['up'], [0.70], sell_low=[0.68], buy_high=[0.99])
    assert m[0] == pytest.approx(2.0)


def test_a_buying_aggressor_fills_a_resting_no_bid():
    # Buying NO at 30c rests a YES OFFER at 70c, lifted by a BUYER at 72c.
    m = run_through_margin(['down'], [0.30], sell_low=[0.01], buy_high=[0.72])
    assert m[0] == pytest.approx(2.0)


def test_the_buy_side_does_not_fill_a_bid():
    # The inversion that looks plausible: using the BUYING aggressor's high to
    # fill a resting bid. A buyer lifting the ask never hits our bid, so a
    # market that only traded upward must leave the bid unreached.
    m = run_through_margin(['up'], [0.70], sell_low=[0.99], buy_high=[0.95])
    assert m[0] < 0, 'a resting bid was filled by a buyer, so the sides are swapped'


def test_probability_is_monotone_in_margin():
    model = FillModel()
    p = model.probability([-5, 0.5, 1.5, 2.5, 4, 8, 20, 50])
    assert np.all(np.diff(p) >= 0), p
    assert p[0] < 0.6 and p[-1] > 0.99


def test_a_missing_margin_is_not_silently_a_fill():
    # No tape for the market is a coverage problem. Returning a default rate
    # would bury it inside a plausible number.
    assert np.isnan(FillModel().probability([np.nan])[0])


def test_expected_fill_keeps_the_frame_index():
    f = pd.DataFrame({'side': ['up', 'down'], 'limit_price': [0.70, 0.30],
                      'sell_low': [0.68, 0.01], 'buy_high': [0.99, 0.72]},
                     index=[7, 9])
    p = expected_fill(f)
    assert list(p.index) == [7, 9]
    assert p.notna().all()


def test_the_curve_says_shallow_touches_are_the_uncertain_ones():
    # The whole point: we get the deep run-throughs, which are the losing
    # trades. If this ever inverts, the adverse selection has gone with it.
    model = FillModel()
    shallow = model.probability([0.5])[0]
    deep = model.probability([30.0])[0]
    assert shallow < deep
    assert shallow < 0.8 <= deep


def test_curve_bands_are_ordered_and_bounded():
    uppers = [u for u, _ in DEFAULT_FILL_CURVE]
    probs = [p for _, p in DEFAULT_FILL_CURVE]
    assert uppers == sorted(uppers)
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert uppers[-1] == float('inf'), 'the top band must catch every margin'
