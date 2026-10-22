"""Which cells an ablation scores, and against which groups.

`available` read `g in set(FEATURE_GROUPS)` — always true — so `--groups` was
ignored and the survey scored families the model does not use. It also
selected both book framings at once, which `feature_columns` refuses.
"""

from __future__ import annotations

import pytest

from core.features import CONTROL_GROUPS, feature_columns
from scripts.ablate import trials

GROUPS = ('vol_state', 'cross_asset', 'implied_vol', 'time_of_day')


def test_the_default_survey_keeps_the_control_and_every_group_alone():
    cells = trials(GROUPS)
    assert cells['all groups'] == GROUPS
    assert f'all minus {CONTROL_GROUPS[0]}' in cells
    for g in GROUPS:
        assert any(label.startswith(f'{g} alone') for label in cells), g


def test_leave_one_out_scores_two_cells_not_one_per_group():
    cells = trials(GROUPS, ['implied_vol'])
    assert list(cells) == ['all groups', 'all minus implied_vol']
    assert 'implied_vol' not in cells['all minus implied_vol']
    assert set(cells['all minus implied_vol']) == set(GROUPS) - {'implied_vol'}


def test_dropping_a_group_that_was_not_selected_is_an_error():
    """Otherwise the cell silently equals 'all groups' and reads as zero effect."""
    with pytest.raises(SystemExit):
        trials(GROUPS, ['book_depth'])


def test_every_default_cell_is_a_selectable_set():
    """The survey must not mix the two book framings, which is refused."""
    groups = ('vol_state', 'cross_asset', 'book_quote', 'implied_vol', 'time_of_day')
    for label, cell in trials(groups).items():
        feature_columns(list(cell))          # raises if the cell is invalid
