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


def test_the_group_selection_is_read_from_args_not_config():
    """`Config` has no `groups`. Reading it there raised at run time while
    these cell tests passed, because they exercise `trials` and never `main`.
    """
    import inspect

    from core.config import DEFAULT_CONFIG
    from scripts import ablate

    assert not hasattr(DEFAULT_CONFIG, 'groups'), (
        'if Config gains a `groups` field, this guard is the wrong shape')
    src = inspect.getsource(ablate.main)
    assert 'groups_from_args(args)' in src
    assert 'config.groups' not in src


def test_main_builds_its_cells_without_touching_the_store(monkeypatch):
    """Exercises the path that broke: argument parsing through cell
    construction, stopping before the dataset load."""
    import argparse

    from scripts import ablate

    captured = {}

    def fake_load(args, config):
        captured['groups'] = ablate.groups_from_args(args)
        raise SystemExit(0)                     # stop before any I/O

    monkeypatch.setattr(ablate, 'load_dataset', fake_load)
    monkeypatch.setattr(
        'sys.argv',
        ['ablate', '--groups', 'vol_state,offset', '--drop', 'offset'])
    with pytest.raises(SystemExit):
        ablate.main()
    assert captured['groups'] == ('vol_state', 'offset')


class TestWhatTheAblationScoresAgainst:
    """A market-initialised model scored against `F(x/sigma)` reports the
    market's advantage over the barrier null -- a constant in every cell --
    plus the feature term. The constant swamps the signal and its fold
    variance sets the standard error, so every group reads as contributing
    nothing. Measured 2026-10-22 before the fix: all eight groups between
    -0.00014 and +0.00031 against se 0.00078.
    """

    def _frame(self):
        import pandas as pd
        return pd.DataFrame({
            'outcome': [1.0, 0.0], 'venue_outcome': [0.0, 1.0],
            'baseline_probability': [0.60, 0.40],
            'market_probability': [0.70, 0.30]})

    def test_a_market_init_model_is_scored_against_the_market(self):
        import dataclasses

        from core.config import DEFAULT_CONFIG
        from scripts.ablate import _null_for
        cfg = dataclasses.replace(DEFAULT_CONFIG, init_score_source='market')
        assert _null_for(self._frame(), cfg).tolist() == [0.70, 0.30]

    def test_a_baseline_init_model_is_still_scored_against_the_baseline(self):
        from core.config import DEFAULT_CONFIG
        from scripts.ablate import _null_for
        assert _null_for(self._frame(), DEFAULT_CONFIG).tolist() == [0.60, 0.40]

    def test_it_falls_back_when_the_market_column_is_absent(self):
        import dataclasses

        from core.config import DEFAULT_CONFIG
        from scripts.ablate import _null_for
        f = self._frame().drop(columns=['market_probability'])
        cfg = dataclasses.replace(DEFAULT_CONFIG, init_score_source='market')
        assert _null_for(f, cfg).tolist() == [0.60, 0.40]

    def test_the_label_honours_label_source(self):
        import dataclasses

        from core.config import DEFAULT_CONFIG
        from scripts.ablate import _graded_label
        cfg = dataclasses.replace(DEFAULT_CONFIG, label_source='venue')
        assert _graded_label(self._frame(), cfg).tolist() == [0.0, 1.0]
        assert _graded_label(self._frame(), DEFAULT_CONFIG).tolist() == [1.0, 0.0]
