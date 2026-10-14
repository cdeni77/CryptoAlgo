"""Tests for the calibration gates.

The baseline cannot pass a tight absolute ECE bar on 7-day folds, so the absolute bar is
a sanity floor and the real gate compares the model with the baseline.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import DEFAULT_GATES, GATE_NOTES


def test_the_absolute_bar_is_a_sanity_floor_not_a_quality_bar():
    bar, direction = DEFAULT_GATES['calibration_error']
    assert direction == 'max'
    assert bar >= 0.08, (
        "the baseline's own worst fold is 0.0802; a bar below what the null "
        "achieves cannot discriminate"
    )


def test_the_relative_gate_exists_and_can_fail():
    bar, direction = DEFAULT_GATES['calibration_vs_baseline']
    assert direction == 'max'
    assert 0 < bar < 0.01, (
        'it must sit above fold noise (sd 0.00707) and well below a typical '
        'fold ECE of 0.030, or it is either unpassable or vacuous'
    )


@pytest.mark.parametrize('value,passes', [
    (0.00100, True),  # measured value
    (-0.0050, True),    # the model improves calibration
    (0.00400, True),    # inside fold noise
    (0.00600, False),   # a systematic degradation
    (0.02000, False),   # 0.02 worse than the null it corrects
])
def test_the_gate_discriminates(value, passes):
    bar, _ = DEFAULT_GATES['calibration_vs_baseline']
    assert (value <= bar) is passes


def test_both_calibration_gates_carry_a_note():
    for name in ('calibration_error', 'calibration_vs_baseline',
                 'calibration_vs_market'):
        assert GATE_NOTES.get(name), f'{name} has no operator-facing note'


def test_the_statistic_is_a_median_over_folds():
    """The statistic is a median over folds, since a max grows with the fold count."""
    from core.metrics import EvaluationReport

    src = EvaluationReport.calibration_vs_baseline.fget.__doc__ or ''
    assert 'MEDIAN' in src or 'Median' in src
    import inspect
    body = inspect.getsource(EvaluationReport.calibration_vs_baseline.fget)
    assert 'np.median' in body


def test_a_fold_with_no_calibration_does_not_vanish():
    """The median excludes non-finite folds explicitly instead of counting NaN as zero."""
    import inspect

    from core.metrics import EvaluationReport

    body = inspect.getsource(EvaluationReport.calibration_vs_baseline.fget)
    assert 'isfinite' in body, 'non-finite folds must be excluded explicitly'
