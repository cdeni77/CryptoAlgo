"""Tests for `skill_null_column`: skill is measured against the null the model was fitted on."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest import skill_null_column


def _table():
    return pd.DataFrame({
        'baseline_probability': [0.40, 0.60],
        'market_probability': [0.45, 0.55],
        'model_probability': [0.44, 0.56],
    })


def test_a_baseline_initialised_model_is_scored_against_the_baseline():
    col = skill_null_column(_table(), 'baseline')
    assert list(col) == [0.40, 0.60]


def test_a_market_initialised_model_is_scored_against_the_market():
    col = skill_null_column(_table(), 'market')
    assert list(col) == [0.45, 0.55]


def test_a_market_init_run_without_quotes_falls_back_and_says_so():
    """With no market column, a market-init run falls back to the baseline."""
    table = _table().drop(columns=['market_probability'])
    col = skill_null_column(table, 'market')
    assert list(col) == [0.40, 0.60]


def test_an_all_nan_market_column_is_not_used_as_a_null():
    table = _table()
    table['market_probability'] = np.nan
    col = skill_null_column(table, 'market')
    assert list(col) == [0.40, 0.60]


def test_an_unknown_source_falls_back_to_the_baseline():
    assert list(skill_null_column(_table(), 'nonsense')) == [0.40, 0.60]
