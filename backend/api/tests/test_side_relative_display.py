"""A DOWN row must display the probability of the side actually taken.

The stored column is P(up) and stays P(up): `scripts/promote.py` scores
`model_minus_market` by comparing `model_probability` against
`market_probability` and `outcome`, all in up terms, so flipping the stored
value would break the gate. The conversion is a DISPLAY concern.

Measured 2026-10-17 on the live table: every one of the 80 DOWN trades since
the start line satisfied `edge == (1 - model_probability) - effective_cost`,
while all 481 UP trades satisfied the direct form. So the dashboard rendered a
real trade as "paid 0.920, model probability 0.054" when the forecast was 0.946
and the edge was +2.07pp.
"""
from __future__ import annotations

import pytest

from controllers.serving import _for_side


def test_an_up_row_is_unchanged():
    assert _for_side(0.72, 'up') == pytest.approx(0.72)


def test_a_down_row_shows_the_probability_of_DOWN():
    """The case that rendered as nonsense: paid 0.920 on a 0.054 stored value."""
    assert _for_side(0.054, 'down') == pytest.approx(0.946)


def test_the_real_down_trades_become_coherent():
    """Each pair is (stored P(up), price paid) from the live table. After the
    conversion the probability must EXCEED the price, or the trade had no edge
    and should never have been taken."""
    for stored, price in ((0.054, 0.920), (0.149, 0.820),
                          (0.212, 0.750), (0.136, 0.830)):
        shown = _for_side(stored, 'down')
        assert shown > price, (
            f'a DOWN trade at {price} shows {shown:.3f}; a displayed '
            f'probability below the price paid reads as a trade with negative '
            f'edge, which is what made these look like a bug')


def test_a_refused_row_without_a_side_keeps_its_p_up():
    """Most rows are refused and never choose a side; P(up) is the only
    meaningful reading for them."""
    assert _for_side(0.31, None) == pytest.approx(0.31)


def test_a_missing_probability_stays_missing():
    """Null is a measurement — a row that could not be scored is not 0.5."""
    assert _for_side(None, 'down') is None
    assert _for_side(None, None) is None
