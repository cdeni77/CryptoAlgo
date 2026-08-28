"""Separating the decision offset from the calendar, so the control is honest.

`control_gain_share` failed at 0.334 — a third of the model resting on a group
that cannot forecast direction, which by this project's own rule means the
measurement is broken rather than the market. But the `clock` group was two
different things wearing one name:

  * `elapsed_fraction` and `remaining_minutes` ARE the decision offset. They are
    legitimately informative — they are why one pooled model can behave like
    four offset-specific ones, and why dropping `clock` wholesale used to cost
    the most in ablation despite clock-alone contributing nothing.
  * `quarter_of_hour`, `hour_sin/cos`, `dow_sin/cos` and `us_equity_hours` are
    calendar position. Those are the real control: time of day cannot forecast
    direction, and the previous incarnation of this project ran a 27-cell survey
    whose best cell was its own control.

Measuring them together made the control look like it carried the model when
part of what it carried was the offset.
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
    """`control_gain_share` gates on this, so what counts as the control decides
    whether the gate is measuring anything."""
    assert CONTROL_GROUPS == ('time_of_day',)


def test_both_replace_clock_and_nothing_is_lost():
    """The split must be exhaustive: every column the old group produced still
    has a home, or the matrix silently narrows."""
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
