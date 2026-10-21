"""Tests that `book_feature_row` measures depth buckets against the ladder's own touch.

A touch from a different fetch can leave the 1c bucket empty and `book_convexity` NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.live import book_feature_row


class _Quote:
    """A quote whose touch is better than any level in the ladder (a stale cache)."""
    yes_bid, yes_ask = 0.60, 0.61
    yes_bid_size, yes_ask_size = 100.0, 80.0


def test_the_one_cent_bucket_uses_the_ladder_it_is_measuring():
    # The ladder's best YES level is 0.58; the quote claims 0.60.
    yes = [[0.58, 40.0], [0.57, 60.0], [0.55, 100.0]]
    no = [[0.41, 30.0], [0.40, 50.0], [0.38, 90.0]]
    row = book_feature_row(_Quote(), yes, no, baseline_probability=0.55)
    assert np.isfinite(row['book_convexity']), (
        'book_convexity is NaN because the 1c bucket was measured against a '
        'touch the ladder does not contain')


def test_a_genuinely_empty_ladder_still_gives_nan():
    """An empty ladder still gives NaN; the fix must not invent depth."""
    row = book_feature_row(_Quote(), [], [], baseline_probability=0.55)
    assert np.isnan(row['book_convexity'])


def test_a_consistent_quote_and_ladder_are_unchanged():
    yes = [[0.60, 100.0], [0.59, 40.0], [0.56, 80.0]]
    no = [[0.39, 80.0], [0.38, 40.0], [0.35, 70.0]]
    row = book_feature_row(_Quote(), yes, no, baseline_probability=0.55)
    assert np.isfinite(row['book_convexity'])
    # 5c reaches deeper than 1c on both sides, so each ratio exceeds 1.
    assert row['book_convexity'] == pytest.approx(
        (220.0 / 140.0) - (190.0 / 120.0), abs=1e-9)
