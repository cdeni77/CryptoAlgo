"""`book_quote` and `book_depth` cut the book features by DATA REQUIREMENT.

`market_state` and `book_flow` each mix features that read only the quote with
features that read resting SIZES. That distinction is invisible in the names
and decides whether a feature can exist at all before 2026-08-25: the venue
serves no historical order book, and Predexon's minute-stale touch sizes are
not a substitute (ratio 1.22, correlation 0.26 against the live book).

Measured on real March 2026 candle rows: the three price-only `book_flow`
features come back 99.8% non-null, the three size-dependent ones 0.0%. So
selecting the whole group costs the entire pre-August history for the sake of
features that were never available — while dragging four recoverable ones down
with them.
"""

import pytest

from core.book_flow import BOOK_FLOW
from core.features import BOOK_GROUPS, FEATURE_GROUPS, feature_columns

PRICE_ONLY = {'spread', 'spread_change_3', 'mid_change_1', 'mid_change_3'}
NEEDS_SIZES = {'imbalance_touch', 'imbalance_5c', 'depth_ratio', 'book_convexity',
               'imbalance_change_3', 'depth_trend_3', 'imbalance_persistence'}


def test_the_split_is_exact():
    """Every feature lands on one side, none on both, none lost."""
    q = set(FEATURE_GROUPS['book_quote'])
    d = set(FEATURE_GROUPS['book_depth'])
    assert q == PRICE_ONLY
    assert d == NEEDS_SIZES
    assert not (q & d)
    assert q | d == set(FEATURE_GROUPS['market_state']) | set(FEATURE_GROUPS['book_flow'])


def test_nothing_in_book_quote_reads_a_size():
    """The whole point: these must be computable where sizes do not exist."""
    for col in FEATURE_GROUPS['book_quote']:
        assert col not in NEEDS_SIZES, col


def test_selecting_both_framings_is_refused():
    """Superseded a test asserting the two framings merely DEDUP.

    They do — and that was the defect. Deduping makes a mixed selection
    harmless, so an ablation dropping `book_depth` while `market_state` is
    still selected quietly gets every depth column back and reports no
    effect. The ablation then reads "this feature does not matter" about a
    feature it never removed. Refusing the mix is what makes the split
    testable at all, which is the only reason the split exists.
    """
    import pytest
    with pytest.raises(ValueError, match='two framings'):
        feature_columns(['market_state', 'book_flow', 'book_quote', 'book_depth'])


def test_both_are_opt_in_not_default():
    """Book groups are all-NaN for ~90% of rows in the default matrix; the
    whole BOOK_GROUPS convention exists to keep them out of it."""
    from core.features import ALL_GROUPS
    for g in ('book_quote', 'book_depth'):
        assert g in BOOK_GROUPS
        assert g not in ALL_GROUPS


def test_the_price_only_book_flow_features_are_named_in_BOOK_FLOW():
    """Guards against a rename on one side drifting from the other."""
    for col in ('spread_change_3', 'mid_change_1', 'mid_change_3'):
        assert col in BOOK_FLOW


def test_selecting_book_quote_alone_is_a_valid_group_set():
    cols = feature_columns(['book_quote'])
    assert sorted(cols) == sorted(PRICE_ONLY)
