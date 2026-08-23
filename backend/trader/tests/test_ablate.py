"""The ablation's own logic, since it is now the control test that matters.

`control_gain_share` gates LightGBM's gain share for the `clock` group, on the
reasoning that hour-of-day cannot forecast direction. The reasoning is right and
the measurement does not test it — gain share says where splits were spent, not
whether a feature forecasts. Measured on real bars the two disagreed completely:
0.279 gain share against a group that scored -0.000008 on 2 of 6 folds alone.
"""

from __future__ import annotations

import pytest

from core.features import FEATURE_GROUPS
from scripts.ablate import CONTROL, trials


def test_the_control_is_a_real_group():
    assert CONTROL in FEATURE_GROUPS, (
        f'{CONTROL!r} is not a feature group, so the ablation cannot isolate it'
    )


def test_every_group_is_tried_alone_plus_the_full_set_and_the_set_without_control():
    groups = tuple(FEATURE_GROUPS)
    built = trials(groups)

    assert built['all groups'] == groups
    assert f'all minus {CONTROL}' in built
    assert CONTROL not in built[f'all minus {CONTROL}']
    assert len(built[f'all minus {CONTROL}']) == len(groups) - 1

    for group in groups:
        alone = [v for k, v in built.items() if k.startswith(f'{group} alone')]
        assert alone == [(group,)], f'{group} is not tried alone'

    # full set + set-without-control + one per group
    assert len(built) == len(groups) + 2


def test_the_control_is_labelled_so_it_cannot_be_read_as_a_finding():
    labels = [k for k in trials(tuple(FEATURE_GROUPS)) if k.startswith(CONTROL)]
    assert labels and 'CONTROL' in labels[0], (
        'the control must be named in the output, because a reader scanning for '
        'the best row is exactly how the previous incarnation of this project '
        'reported its own control as its best cell'
    )


def test_a_single_group_universe_still_produces_a_trial():
    """No `all minus control` row when there is nothing left to remove."""
    built = trials((CONTROL,))
    assert built['all groups'] == (CONTROL,)
    assert f'all minus {CONTROL}' not in built
