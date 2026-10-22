"""An ablation without a negative control is refused.

`RESEARCH_PROTOCOL` section 5 makes negative controls mandatory. The control
WARNING in `ablate` only prints when the control is among the selected
groups, so a run that simply did not list `time_of_day` produced no check and
no complaint -- which is how the 2026-10-22 leave-one-out ablation was read,
and partly acted on, with nothing showing the measurement could return a
null. A guard a configuration can skip is not a guard.
"""

from __future__ import annotations

import argparse

import pytest

from core.features import CONTROL_GROUPS, NOISE_GROUPS
from scripts.ablate import _require_a_control


def args(**kw):
    return argparse.Namespace(**{'no_control': False, **kw})


def test_a_survey_without_any_control_is_refused():
    with pytest.raises(SystemExit) as raised:
        _require_a_control(('geometry', 'vol_state', 'cross_venue'), args())
    message = str(raised.value)
    assert 'returns a null' in message
    assert CONTROL_GROUPS[0] in message, 'it must say which group to add'


def test_the_designated_control_satisfies_it():
    _require_a_control(('geometry', CONTROL_GROUPS[0]), args())


def test_a_noise_group_also_satisfies_it():
    """The noise groups are controls too -- NC2 is the stronger one."""
    _require_a_control(('geometry', NOISE_GROUPS[0]), args())


def test_no_control_must_be_said_out_loud():
    """The escape hatch exists, for a run whose control IS the measurement."""
    _require_a_control(('geometry',), args(no_control=True))


def test_main_refuses_before_building_the_dataset():
    """Refusing AFTER a sixteen-minute build would be useless. The groups
    are on the args already, so the guard costs nothing where it is."""
    import inspect

    from scripts import ablate

    src = inspect.getsource(ablate.main)
    assert src.index('_require_a_control') < src.index('load_dataset('), (
        'the guard must precede the dataset build, not just the scoring')
