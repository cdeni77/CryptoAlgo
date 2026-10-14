"""Tests that the offset features and the calendar features are separate groups.

`elapsed_fraction` and `remaining_minutes` describe the decision offset and are
informative. Hour, weekday and quarter-hour are calendar position, the real
control, since time of day cannot forecast direction.
"""

from __future__ import annotations

import pytest

from core.features import (ALL_GROUPS, CONTROL_GROUPS, FEATURE_GROUPS,
                           feature_columns)


def test_the_offset_and_the_calendar_are_separate_groups():
    assert 'offset' in FEATURE_GROUPS
    assert 'time_of_day' in FEATURE_GROUPS


def test_the_offset_group_holds_only_within_window_position():
    cols = set(FEATURE_GROUPS['offset'])
    assert cols == {'elapsed_fraction', 'remaining_minutes'}


def test_the_calendar_group_holds_the_things_that_cannot_forecast_direction():
    cols = set(FEATURE_GROUPS['time_of_day'])
    assert cols == {'quarter_of_hour', 'hour_sin', 'hour_cos',
                    'dow_sin', 'dow_cos', 'us_equity_hours'}
    assert 'remaining_minutes' not in cols, 'that is the offset, not the calendar'


def test_the_control_is_now_the_calendar_alone():
    """`control_gain_share` gates on the control group, so it must be the calendar alone.
    """
    assert CONTROL_GROUPS == ('time_of_day',)


def test_both_replace_clock_and_nothing_is_lost():
    """The split is exhaustive: every former `clock` column has a group."""
    assert 'clock' not in FEATURE_GROUPS
    moved = set(FEATURE_GROUPS['offset']) | set(FEATURE_GROUPS['time_of_day'])
    assert moved == {'elapsed_fraction', 'remaining_minutes', 'quarter_of_hour',
                     'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
                     'us_equity_hours'}


def test_both_are_in_the_default_matrix():
    defaults = set(feature_columns())
    assert set(FEATURE_GROUPS['offset']) <= defaults
    assert set(FEATURE_GROUPS['time_of_day']) <= defaults
    assert 'offset' in ALL_GROUPS and 'time_of_day' in ALL_GROUPS
