"""The depth buckets were measured against a touch from a different sampler.

`book_feature_row` and `_record_touch` took `best` from the QUOTE (a REST
fetch) and the levels from the stream cache. When the quote's touch is a cent
better than anything in the cached ladder, `_cumulative(within=0.01)` matches
nothing and the 1c bucket reads zero while the 5c bucket is full — so
`book_convexity` divides by zero and goes NaN.

Measured on the store, this state is IMPOSSIBLE in the backfill and common live:

    1c empty but 5c populated:   backfill 0.00%   live_touch 31.24%
    both empty (real thin book): backfill 9.98%   live_touch 11.91%

The genuine thin-book rate agrees. The extra 31% is the two-sampler artifact,
and it made `book_convexity` NaN on 43% of live cycles against training's 10%.
The backfill reads the touch and the ladder from ONE snapshot, which is why it
cannot happen there.
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.live import book_feature_row


class _Quote:
    """A quote whose touch is BETTER than anything in the ladder — the stale
    -cache case, which is 31% of live cycles."""
    yes_bid, yes_ask = 0.60, 0.61
    yes_bid_size, yes_ask_size = 100.0, 80.0


def test_the_one_cent_bucket_uses_the_ladder_it_is_measuring():
    # Ladder tops out at 0.58 on the YES side; the quote claims 0.60.
    yes = [[0.58, 40.0], [0.57, 60.0], [0.55, 100.0]]
    no = [[0.41, 30.0], [0.40, 50.0], [0.38, 90.0]]
    row = book_feature_row(_Quote(), yes, no, baseline_probability=0.55)
    assert np.isfinite(row['book_convexity']), (
        'book_convexity is NaN because the 1c bucket was measured against a '
        'touch the ladder does not contain')


def test_a_genuinely_empty_ladder_still_gives_nan():
    """The fix must not manufacture depth. No levels means no measurement, and
    training carries that NaN on ~10% of rows."""
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
