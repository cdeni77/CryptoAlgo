"""Tests for the relative calibration gate `calibration_vs_market`.

The market price itself can miss an absolute 0.04 bin bar. The gate therefore
asks whether the model is at least as well calibrated as the price. The absolute
bar stays as a loose rail for gross miscalibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.metrics import DEFAULT_GATES, market_gate_values


def _rows(model_p, market_p, actual, n=400):
    """n rows in one bin: model and market both predict, and `actual` is the share of up
    outcomes.
    """
    out = []
    for i in range(n):
        outcome = 1.0 if i < int(round(actual * n)) else 0.0
        out.append(('BTC-USD', pd.Timestamp('2026-09-03 12:00', tz='UTC'), 12,
                    market_p, 0.5, model_p, outcome, None))
    return out


def test_the_market_deviation_is_measured_on_the_same_rows():
    rows = _rows(model_p=0.68, market_p=0.65, actual=0.70)
    values = market_gate_values(rows)
    assert 'market_max_deviation' in values
    assert values['market_max_deviation'] == pytest.approx(0.05, abs=0.02)


def test_a_model_closer_than_the_market_passes_the_relative_gate():
    """A model closer to the outcome rate than the market passes."""
    rows = _rows(model_p=0.70, market_p=0.65, actual=0.71)
    values = market_gate_values(rows)
    assert values['calibration_vs_market'] <= 0.0


def test_a_model_worse_than_the_market_fails_it():
    rows = _rows(model_p=0.50, market_p=0.70, actual=0.71)
    values = market_gate_values(rows)
    assert values['calibration_vs_market'] > 0.0


def test_the_relative_gate_is_registered_and_the_absolute_one_is_a_rail():
    assert 'calibration_vs_market' in DEFAULT_GATES
    bound, direction = DEFAULT_GATES['calibration_vs_market']
    assert bound == 0.0 and direction == 'max'
    # The absolute bar remains, set loose to catch only gross miscalibration.
    assert DEFAULT_GATES['calibration_max_deviation'][0] >= 0.08


import pytest  # noqa: E402
