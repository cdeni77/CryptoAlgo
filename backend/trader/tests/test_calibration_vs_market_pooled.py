"""Tests that `calibration_vs_market` uses the pooled ECE difference, not the worst bin.

A maximum over bins is high-variance and can pass or fail an unchanged artifact
on sampling noise. The pooled difference is stable. The gate is at <= 0 with no
tolerance.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import _pooled_ece, _worst_populated_bin, market_gate_values


def _rows(model, market, outcomes):
    """Rows are (symbol, window_open, offset, market, baseline, model, outcome)."""
    return [('BTC-USD', f'w{i}', 12, market, 0.5, model, y)
            for i, y in enumerate(outcomes)]


def test_a_better_calibrated_model_passes():
    """Model says 0.70 and 70% happen; market says 0.50 on the same rows."""
    outcomes = [1] * 70 + [0] * 30
    v = market_gate_values(_rows(0.70, 0.50, outcomes))
    assert v['calibration_vs_market'] < 0


def test_a_worse_calibrated_model_fails():
    outcomes = [1] * 70 + [0] * 30
    v = market_gate_values(_rows(0.20, 0.70, outcomes))
    assert v['calibration_vs_market'] > 0


def test_it_is_the_pooled_statistic_not_the_worst_bin():
    """The gate reads the pooled statistic, not the worst bin."""
    import inspect

    from core import metrics

    src = inspect.getsource(metrics.market_gate_values)
    assert '_pooled_ece(' in src
    tail = src[src.index("'calibration_vs_market'"):]
    assert 'model_ece - market_ece' in tail


def test_the_worst_bin_is_still_reported():
    """The worst bin is still reported, as information about the venue."""
    v = market_gate_values(_rows(0.6, 0.6, [1] * 300 + [0] * 300))
    assert 'market_max_deviation' in v


def test_pooled_is_lower_variance_than_the_worst_bin():
    """Resampling shows the pooled statistic has lower variance than the worst bin."""
    rng = np.random.default_rng(0)
    y = np.array([1] * 500 + [0] * 500, dtype=float)
    p = np.clip(rng.normal(0.5, 0.2, y.size), 0.01, 0.99)
    pooled, worst = [], []
    for _ in range(60):
        idx = rng.integers(0, y.size, y.size)
        pooled.append(_pooled_ece(p[idx], y[idx]))
        w = _worst_populated_bin(p[idx], y[idx])
        if np.isfinite(w):
            worst.append(w)
    assert np.std(pooled) < np.std(worst), (
        'if the maximum were not noisier, there would be no reason to switch'
    )


def test_no_rows_is_unmeasured_not_passing():
    v = market_gate_values([])
    assert not np.isfinite(v['calibration_vs_market'])
