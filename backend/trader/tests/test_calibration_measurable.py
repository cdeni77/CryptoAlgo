"""A calibration gate that cannot be computed must say so, not fail silently.

`calibration_max_deviation` read NaN and failed. It is `max` over the folds of
`worst_deviation(min_count=500)`, which returns NaN when no bin holds 500 rows —
and under `--complete-cases` each fold carries roughly 3,400 rows spread across
the bins, so none does. Nothing was wrong with the model; the measurement was
under-powered and the gate reported it as a failure indistinguishable from a
real one.

Every scored row is out-of-sample whichever fold produced it, so pooling them is
a legitimate fallback that multiplies the rows per bin by the fold count. It is
a FALLBACK, not the definition: the per-fold maximum is the stricter statistic
and stays primary, because a single badly-calibrated fold is exactly what it
exists to catch.
"""

from __future__ import annotations

import numpy as np

from core.metrics import resolve_max_deviation


def test_the_per_fold_maximum_is_used_when_it_is_measurable():
    ok, why = resolve_max_deviation(per_fold=[0.021, 0.033, 0.018], pooled=0.005)
    assert ok == 0.033
    assert 'fold' in why.lower()


def test_the_pooled_value_is_used_when_no_fold_had_enough_rows():
    ok, why = resolve_max_deviation(per_fold=[np.nan, np.nan], pooled=0.027)
    assert ok == 0.027
    assert 'pool' in why.lower()


def test_a_single_measurable_fold_still_wins_over_the_pool():
    """One fold with a populated bin is a real observation; the pool is the
    weaker instrument and must not override it."""
    ok, _ = resolve_max_deviation(per_fold=[np.nan, 0.045, np.nan], pooled=0.010)
    assert ok == 0.045


def test_nothing_measurable_anywhere_stays_nan_and_says_why():
    """Not measured is not measured good — the gate must still fail, but the
    reason has to name the sample size rather than the model."""
    ok, why = resolve_max_deviation(per_fold=[np.nan], pooled=float('nan'))
    assert np.isnan(ok)
    assert 'row' in why.lower() or 'popul' in why.lower()


def test_an_empty_run_is_not_silently_fine():
    ok, _ = resolve_max_deviation(per_fold=[], pooled=float('nan'))
    assert np.isnan(ok)
