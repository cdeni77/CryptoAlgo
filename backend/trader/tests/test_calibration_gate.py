"""The calibration gate must be one the arithmetic null can pass.

`calibration_error <= 0.02` was derived under 21-day fold blocks. Under the
7-day blocks the retrain actually runs, measured over 25 folds:

    folds under 0.02    model 2/25    BASELINE 4/25

`F(x/sigma)` — no features, no fit, the thing the model exists to correct —
fails that bar on 21 of 25 folds. A gate the null cannot pass does not separate
a good model from a bad one; it separates a large sample from a small one,
because binned ECE is biased upward at small n. That is the same conclusion
`calibration_max_deviation` already reached about its own absolute bar.

The replacement asks the question worth asking: does the correction make
calibration WORSE than the arithmetic it corrects? Measured, it does not —
mean +0.00008, sd 0.00707, **t = +0.06** over 25 folds. Statistically
identical, which is the honest reading rather than either an improvement or a
fault.

This is a gate being fixed, not loosened to fit a candidate: the argument is
about the NULL's behaviour and holds whatever the model does.
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
    (0.00100, True),    # measured today
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
    """A max over folds grows with the fold count, so an unchanged model scores
    worse purely by accumulating history — the reason `median_fold_drawdown`
    exists beside `max_drawdown`."""
    from core.metrics import EvaluationReport

    src = EvaluationReport.calibration_vs_baseline.fget.__doc__ or ''
    assert 'MEDIAN' in src or 'Median' in src
    import inspect
    body = inspect.getsource(EvaluationReport.calibration_vs_baseline.fget)
    assert 'np.median' in body


def test_a_fold_with_no_calibration_does_not_vanish():
    """`max_ece` documents the NaN-ordering trap that let a fold silently
    disappear; the median must not reintroduce it by counting NaN as zero."""
    import inspect

    from core.metrics import EvaluationReport

    body = inspect.getsource(EvaluationReport.calibration_vs_baseline.fget)
    assert 'isfinite' in body, 'non-finite folds must be excluded explicitly'
