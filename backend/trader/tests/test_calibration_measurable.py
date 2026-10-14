"""Tests for `resolve_max_deviation`: an unmeasurable calibration gate says so.

The per-fold maximum is primary. When no fold has a bin with enough rows, the
pooled out-of-sample rows are the fallback. When neither is measurable the value
stays NaN and the reason names the sample size.
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
    """One measurable fold takes priority over the pooled value."""
    ok, _ = resolve_max_deviation(per_fold=[np.nan, 0.045, np.nan], pooled=0.010)
    assert ok == 0.045


def test_nothing_measurable_anywhere_stays_nan_and_says_why():
    """Nothing measurable stays NaN, and the reason names the sample size, not the model.
    """
    ok, why = resolve_max_deviation(per_fold=[np.nan], pooled=float('nan'))
    assert np.isnan(ok)
    assert 'row' in why.lower() or 'popul' in why.lower()


def test_an_empty_run_is_not_silently_fine():
    ok, _ = resolve_max_deviation(per_fold=[], pooled=float('nan'))
    assert np.isnan(ok)
