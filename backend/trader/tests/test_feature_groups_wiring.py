"""The new families, as groups the model and the ablation runner can select.

Until they are in `FEATURE_GROUPS` none of the collected book is visible to
anything: `feature_columns()` builds the matrix from that dict, and
`scripts/evaluate.py --groups X` selects from it. Three properties matter, and
each is a rule this project already paid for.
"""

from __future__ import annotations

import pytest

from core.features import ALL_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS, feature_columns


def test_the_three_new_families_are_selectable():
    for group in ('market_state', 'cross_venue', 'implied_vol'):
        assert group in FEATURE_GROUPS, group
        assert feature_columns([group]), f'{group} has no columns'


def test_the_clock_control_is_still_the_only_control():
    """A survey that quietly loses its control is how the previous project came
    to rank `seasonality,cost` first and not notice."""
    assert CONTROL_GROUPS == ('clock',)
    assert 'clock' in ALL_GROUPS


def test_no_column_is_declared_in_two_groups():
    """An ablation that removes one group and silently keeps the column through
    another cannot measure what it claims to."""
    seen = {}
    for group, columns in FEATURE_GROUPS.items():
        for column in columns:
            assert column not in seen, f'{column} in both {seen.get(column)} and {group}'
            seen[column] = group


def test_level_counts_never_became_features():
    """Measured ratio 0.579 between the backfilled book and the live one — a
    model trained on them learns which pipe a row arrived through."""
    for column in feature_columns():
        assert 'levels' not in column, column


def test_the_price_columns_are_isolated_so_they_can_be_ablated_alone():
    """`market_minus_baseline` is the most informative column available and the
    one that invites echo. The structure-only variant has to be selectable
    without it, or the echo question cannot be answered."""
    from core.book_features import MARKET_STATE, PRICE_COLUMNS
    structural = [c for c in MARKET_STATE if c not in PRICE_COLUMNS]
    assert structural, 'structure-only must not be empty'
    assert set(PRICE_COLUMNS) <= set(feature_columns(['market_state']))


def test_selecting_an_unknown_group_still_raises():
    with pytest.raises(ValueError):
        feature_columns(['no_such_group'])


def test_book_groups_are_selectable_but_not_in_the_default_matrix():
    """The book starts 2026-01-08 against five years of bars. In the default
    matrix these would be all-NaN for ~90% of rows — exactly what
    `population_report` exists to catch — and every existing feature matrix
    would silently widen."""
    from core.features import BOOK_GROUPS
    for group in BOOK_GROUPS:
        assert group in FEATURE_GROUPS, f'{group} must stay selectable'
        assert group not in ALL_GROUPS, f'{group} must not be a default'
    default = set(feature_columns())
    for group in BOOK_GROUPS:
        assert not (set(feature_columns([group])) & default), group


def test_the_five_original_groups_are_still_the_default():
    assert set(ALL_GROUPS) == {'vol_state', 'microstructure', 'cross_asset',
                               'geometry', 'clock'}


def test_market_minus_baseline_is_populated_after_the_baseline_attaches():
    """It was all-NaN in every run. `market_minus_baseline` is mid minus
    F(x/sigma), and `baseline_probability` is attached AFTER features are built —
    so computing it in `build_features` guaranteed a dead column, and the model's
    empty-feature warning was the only thing that said so."""
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
    """`quote_intensity` needed a per-minute snapshot count that venue_depth's
    summary does not carry. A declared feature nothing can populate is worse
    than no feature: it dilutes the matrix and reads as a working column."""
    from core.book_features import MARKET_STATE
    assert 'quote_intensity' not in MARKET_STATE
