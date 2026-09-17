"""`venue_gap_change_5` must mean the same thing in training and live.

The feature is a first difference of `venue_prob_gap` across one decision
offset. Training computed it as `shift(1)` over the rows that survived;
`scripts/live.py::gap_change` takes `offsets[i-1]` or NaN.

Those agree only if the panel always carries every offset — which the training
comment asserted and which is **false under `--complete-cases`**. That filter
runs ROW-WISE on `dataset.windows` before `build_features`, dropping individual
offsets wherever `ask_up`, `market_probability`, `bid_at_touch`,
`pm_market_probability` or `implied_sigma_per_min` is missing at that exact
offset. So at +12m — the only entry offset — training's step could be 6 or 9
minutes while live's is 3 or nothing.

Second feature by importance, in the group the config calls the only
load-bearing one. Training now uses the configured grid, because live cannot
see which rows training dropped: the grid is the only definition both sides can
compute.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.features import gap_change_column

OPEN = pd.Timestamp('2026-07-01T12:00Z')


def _table(offsets_and_gaps):
    return pd.DataFrame([
        {'symbol': 'BTC-USD', 'window_open': OPEN, 'offset': o,
         'venue_prob_gap': g, 'pm_market_probability': 0.5,
         'market_probability': 0.5, 'baseline_probability': 0.5}
        for o, g in offsets_and_gaps])


def _gap_change(offsets_and_gaps, decision_offsets=(3, 6, 9, 12)):
    table = _table(offsets_and_gaps)
    return dict(zip(table['offset'],
                    gap_change_column(table, decision_offsets)))


def test_a_complete_window_differences_one_offset():
    got = _gap_change([(3, 0.01), (6, 0.03), (9, 0.06), (12, 0.10)])
    assert np.isnan(got[3])
    assert got[6] == pytest.approx(0.02)
    assert got[9] == pytest.approx(0.03)
    assert got[12] == pytest.approx(0.04)


def test_a_missing_middle_offset_yields_NaN_not_a_longer_step():
    """The defect. With offset 9 filtered out, `shift(1)` at +12m differences
    against +6m — a six-minute change under a column named for one step — where
    live reports nothing at all."""
    got = _gap_change([(3, 0.01), (6, 0.03), (12, 0.10)])
    assert np.isnan(got[12]), (
        'the previous CONFIGURED offset is absent, so there is no one-offset '
        'change to report'
    )
    assert got[6] == pytest.approx(0.02), 'the intact step still works'


def test_the_first_offset_is_always_NaN():
    got = _gap_change([(3, 0.01), (6, 0.03)])
    assert np.isnan(got[3])


def test_a_NaN_gap_upstream_propagates():
    got = _gap_change([(3, 0.01), (6, float('nan')), (9, 0.06)])
    assert np.isnan(got[6])
    assert np.isnan(got[9]), 'differencing against a NaN is not a number'


def test_it_follows_the_configured_grid_not_the_rows_present():
    """On a two-offset policy the step is 9 minutes, and that is correct —
    what must not happen is the grid and the surviving rows disagreeing."""
    got = _gap_change([(3, 0.01), (12, 0.10)], decision_offsets=(3, 12))
    assert got[12] == pytest.approx(0.09)


def test_build_features_uses_the_helper():
    """A seam test: the arithmetic lives in one place, and the caller uses it.
    Testing the helper alone would not have caught the original bug, which was
    an inline `shift(1)` inside `build_features`."""
    import inspect

    from core.features import build_features

    src = inspect.getsource(build_features)
    assert 'gap_change_column(' in src
    assert "['venue_prob_gap'].shift(1)" not in src
