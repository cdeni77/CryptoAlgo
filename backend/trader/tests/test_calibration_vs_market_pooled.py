"""`calibration_vs_market` must be a statistic you can act on.

It compared the WORST POPULATED BIN of the model against the market's. A
maximum over bins is inherently high-variance, and on this sample it was pure
noise. Bootstrapped over 300 resamples of the same 7,608 live rows, with the
artifact held fixed (2026-09-24):

    statistic           mean       sd      passes (<=0)
    worst bin         +0.0106   0.0229        32%
    pooled ECE diff   +0.0032   0.0023        10%

The worst-bin standard deviation was TWICE the effect it measured, and its 90%
interval [-0.026, +0.050] spanned both "clearly better than the market" and
"clearly worse". It passed or failed an unchanged artifact on which rows
happened to arrive — moving 0.011 in four hours on 8% more data, and blocking
the 2026-09-20 retrain on that basis.

**Switching to the pooled difference does not make the failure go away; it makes
it legible.** The model is slightly but consistently worse calibrated than the
price — +0.0032 against a typical ECE near 0.030, with 90% of resamples above
zero. The noisy version was hiding a real finding behind an enormous error bar,
not inventing a false one.

Gated at <= 0 with no tolerance, deliberately. A noise-floor allowance would let
exactly this finding through, and reporting it is the point while the edge is
unproven.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.metrics import _pooled_ece, _worst_populated_bin, market_gate_values


def _rows(model, market, outcomes):
    """(symbol, window_open, offset, market, baseline, model, outcome)."""
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
    """The whole point of the change. These differ, and the gate must read the
    stable one."""
    import inspect

    from core import metrics

    src = inspect.getsource(metrics.market_gate_values)
    assert '_pooled_ece(' in src
    tail = src[src.index("'calibration_vs_market'"):]
    assert 'model_ece - market_ece' in tail


def test_the_worst_bin_is_still_reported():
    """It is informative about the venue even though it is too noisy to gate
    on — the market's own worst bin is ~6pp, which is why an absolute
    calibration bar was never reachable."""
    v = market_gate_values(_rows(0.6, 0.6, [1] * 300 + [0] * 300))
    assert 'market_max_deviation' in v


def test_pooled_is_lower_variance_than_the_worst_bin():
    """Demonstrated rather than asserted: resample the same data and compare
    the spread of the two statistics."""
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
