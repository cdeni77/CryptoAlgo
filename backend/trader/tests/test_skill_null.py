"""`log_loss_skill` must be measured against the null the model was FITTED on.

A market-initialised model is not trying to beat `F(x/sigma)` — it is fitted on
the price and tries to beat that. Scoring it against the baseline anyway made
two gates report a failure that was a category error:

    init_score_source=market:
      model_minus_market  +0.00078   PASS   (it does beat the price)
      log_loss_skill      -0.00016   FAIL   (measured against the wrong null)
      folds_skill_positive  3 of 6   FAIL   (same)

Four gates failed and two of them were asking the wrong question, which is worse
than a gate that fails honestly: it hides whichever failures are real.
"""

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
    """If the market column is absent the run cannot be scored against it. The
    baseline is the honest fallback — but silently swapping the null is how a
    market-init model comes to be judged as a baseline-init one."""
    table = _table().drop(columns=['market_probability'])
    col = skill_null_column(table, 'market')
    assert list(col) == [0.40, 0.60]


def test_an_all_nan_market_column_is_not_used_as_a_null():
    """A column of NaN is present and useless; scoring against it yields NaN
    skill, which reads as 'no measurement' rather than 'no skill'."""
    table = _table()
    table['market_probability'] = np.nan
    col = skill_null_column(table, 'market')
    assert list(col) == [0.40, 0.60]


def test_an_unknown_source_falls_back_to_the_baseline():
    assert list(skill_null_column(_table(), 'nonsense')) == [0.40, 0.60]
