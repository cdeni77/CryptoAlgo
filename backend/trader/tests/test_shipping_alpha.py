"""The shrinkage that ships is the walk-forward's, not the refit's.

A deployed alpha near zero makes the model emit the market price, so every
edge is exactly minus the fee and the artifact cannot place a single trade --
while its own walk-forward reports alpha 1.000 and positive fold skill. That
happened on every promotion made on 2026-11-05.
"""
import numpy as np
import pytest

from scripts.promote import shipping_alpha


def test_the_walk_forward_wins_over_the_refit():
    # The exact numbers from 2026-11-05: six folds over 6,149 windows said
    # 0.964; the refit's one-week tail said 0.00001, and 0.00001 shipped.
    alpha, source = shipping_alpha(0.00001, 0.964)
    assert alpha == pytest.approx(0.964)
    assert source == 'walk-forward'


def test_a_walk_forward_above_the_ceiling_is_clipped_not_amplified():
    # Alpha above 1 means the split asks to AMPLIFY the correction, which is an
    # overfitting signature. The ceiling is the contract.
    alpha, _ = shipping_alpha(0.5, 1.42, 1.0)
    assert alpha == pytest.approx(1.0)


def test_the_refit_is_the_fallback_and_is_named_as_such():
    alpha, source = shipping_alpha(0.8, None)
    assert alpha == pytest.approx(0.8)
    assert source == 'refit', 'the caller must be able to warn when this happens'


def test_a_nan_walk_forward_falls_back_rather_than_shipping_nan():
    alpha, source = shipping_alpha(0.8, float('nan'))
    assert alpha == pytest.approx(0.8)
    assert source == 'refit'


def test_nothing_usable_returns_nothing_rather_than_a_default():
    # A fabricated default here would be an unfitted constant reaching a gate.
    alpha, source = shipping_alpha(None, None)
    assert alpha is None
    assert source == 'none'


def test_a_genuinely_zero_walk_forward_still_ships_zero():
    # The fix must not become "always trade". If the walk-forward itself says
    # the correction is worthless, that is evidence and it ships.
    alpha, source = shipping_alpha(0.9, 0.0)
    assert alpha == pytest.approx(0.0)
    assert source == 'walk-forward'


def test_negative_is_floored_because_it_would_invert_the_model():
    alpha, _ = shipping_alpha(0.5, -0.3)
    assert alpha == pytest.approx(0.0)
