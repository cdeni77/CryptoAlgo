"""Tests that the book feature families are registered in `FEATURE_GROUPS`.

`feature_columns()` and `scripts/evaluate.py --groups` select from that dict, so a
family outside it is invisible to the model and the ablation.
"""

from __future__ import annotations

import pytest

from core.features import (ALL_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS,
                           MEANING_FRAMING, QUOTE_DEPTH_FRAMING, feature_columns)


def test_the_three_new_families_are_selectable():
    for group in ('market_state', 'cross_venue', 'implied_vol'):
        assert group in FEATURE_GROUPS, group
        assert feature_columns([group]), f'{group} has no columns'


def test_the_calendar_control_is_still_the_only_control():
    """`time_of_day` is the only control group; `offset` is informative and is not a control."""
    assert CONTROL_GROUPS == ('time_of_day',)
    assert 'time_of_day' in ALL_GROUPS


def test_no_column_is_declared_in_two_groups():
    """No column belongs to two groups, so removing a group really removes its columns.

    Asserted WITHIN a framing. The eleven book columns are deliberately cut
    two ways — by data requirement (`book_quote`/`book_depth`) and by meaning
    (`market_state`/`book_flow`) — so the invariant cannot hold across the
    whole dict. It must still hold inside each framing, or an ablation that
    drops one group silently keeps its columns.
    """
    for framing in (QUOTE_DEPTH_FRAMING, MEANING_FRAMING):
        groups = [g for g in FEATURE_GROUPS if g not in
                  (set(QUOTE_DEPTH_FRAMING) | set(MEANING_FRAMING))] + list(framing)
        seen = {}
        for group in groups:
            for column in FEATURE_GROUPS[group]:
                assert column not in seen, (
                    f'{column} in both {seen.get(column)} and {group}')
                seen[column] = group


def test_the_two_framings_cover_exactly_the_same_columns():
    """If they ever diverge, one framing silently drops a feature."""
    assert set(feature_columns(QUOTE_DEPTH_FRAMING)) == set(
        feature_columns(MEANING_FRAMING))


def test_mixing_the_two_framings_is_refused():
    """Dedup makes mixing HARMLESS, which is why it must be refused.

    An ablation dropping `book_depth` while `market_state` is still selected
    gets every depth column back and reports no effect — "the feature does not
    matter" when the feature was never removed.
    """
    with pytest.raises(ValueError, match='two framings'):
        feature_columns(['market_state', 'book_depth'])
    with pytest.raises(ValueError, match='two framings'):
        feature_columns(['book_quote', 'book_flow'])


def test_level_counts_never_became_features():
    """Level counts are not features: they differ between backfilled and live sources."""
    for column in feature_columns():
        assert 'levels' not in column, column


def test_the_price_columns_are_isolated_so_they_can_be_ablated_alone():
    """The price columns form their own `market_price` group, so structure-only can be
    run alone.
    """
    from core.book_features import MARKET_STATE, PRICE_COLUMNS
    assert MARKET_STATE, 'structure-only must not be empty'
    assert not (set(MARKET_STATE) & set(PRICE_COLUMNS))
    assert set(PRICE_COLUMNS) <= set(feature_columns(['market_price']))


def test_selecting_an_unknown_group_still_raises():
    with pytest.raises(ValueError):
        feature_columns(['no_such_group'])


def test_book_groups_are_selectable_but_not_in_the_default_matrix():
    """Book groups are selectable but not in the default matrix, since most bar history
    has no book.
    """
    from core.features import BOOK_GROUPS
    for group in BOOK_GROUPS:
        assert group in FEATURE_GROUPS, f'{group} must stay selectable'
        assert group not in ALL_GROUPS, f'{group} must not be a default'
    default = set(feature_columns())
    for group in BOOK_GROUPS:
        assert not (set(feature_columns([group])) & default), group


def test_the_bar_only_groups_are_still_the_default():
    """The default groups are the six bar-only groups."""
    assert set(ALL_GROUPS) == {'vol_state', 'microstructure', 'cross_asset',
                               'geometry', 'offset', 'time_of_day'}


def test_market_minus_baseline_is_populated_after_the_baseline_attaches():
    """`market_minus_baseline` is computed in `attach_baseline`, after the baseline exists."""
    import pandas as pd
    from core.baseline import attach_baseline

    class _Flat:
        def probability_for(self, table):
            return pd.Series(0.40, index=table.index)

    table = pd.DataFrame({'market_prob': [0.45, 0.30], 'offset': [3, 12]})
    got = attach_baseline(table, _Flat())
    assert 'market_minus_baseline' in got.columns
    assert got['market_minus_baseline'].tolist() == pytest.approx([0.05, -0.10])


def test_a_dead_feature_is_removed_rather_than_carried():
    """`quote_intensity` is not a feature: no source can populate it."""
    from core.book_features import MARKET_STATE
    assert 'quote_intensity' not in MARKET_STATE


# --- the echo test, as a group selection -----------------------------------
#
# Given a well-calibrated quote, copying it is the cheapest way to a low log
# loss. `market_state` alone is structure only and cannot echo the price.
# Adding `market_price` puts the price back.

def test_market_state_is_structure_only():
    from core.book_features import PRICE_COLUMNS
    cols = set(FEATURE_GROUPS['market_state'])
    assert not (cols & set(PRICE_COLUMNS)), 'structure-only must exclude the price'
    assert 'spread' in cols and 'imbalance_touch' in cols


def test_market_price_holds_the_columns_that_can_echo():
    cols = set(FEATURE_GROUPS['market_price'])
    assert cols == {'market_prob', 'market_minus_baseline'}


def test_the_two_together_are_the_old_market_state():
    both = set(FEATURE_GROUPS['market_state']) | set(FEATURE_GROUPS['market_price'])
    assert both == {'market_prob', 'market_minus_baseline', 'spread',
                    'imbalance_touch', 'imbalance_5c', 'depth_ratio',
                    'book_convexity'}


def test_both_stay_out_of_the_default_matrix():
    from core.features import BOOK_GROUPS
    assert 'market_state' in BOOK_GROUPS and 'market_price' in BOOK_GROUPS
