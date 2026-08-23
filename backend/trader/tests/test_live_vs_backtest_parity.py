"""PROPOSED. The live feature vector must equal the backtest feature vector.

Drop into `backend/trader/tests/`.

`core/dataset.py::score_live`'s own docstring makes this claim — "the same code
path as the backtest, deliberately" — and nothing tests it. The existing
`test_score_live_*` tests check the *shape* (one row per symbol, an `outcome`
column of NaN); neither compares a single number against what the measured path
produced for the same window.

That is the exact failure the docstring says was lived through: "the previous
incarnation of this repo had them disagree about entry price for months". A
backtested edge that the live path cannot reproduce is not an edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.dataset import Dataset, apply_fold, fit_fold, score_live
from core.features import feature_columns
from tests.conftest import make_bars

FAST = Config(n_estimators=60, early_stopping_rounds=10, n_folds=3,
              seasonality_min_days=5)


@pytest.fixture(scope='module')
def fitted():
    bars = make_bars(days=40, lead=0.3)
    dataset = Dataset.build(bars, FAST)
    index = dataset.window_index
    cut = int(len(index) * 0.75)
    fit, _ = fit_fold(dataset, index[:cut], FAST)
    measured = apply_fold(dataset, fit, index[cut:], FAST)
    return bars, dataset, fit, measured, index[cut:]


@pytest.mark.parametrize('offset', (3, 12))
def test_the_live_path_reproduces_the_measured_feature_vector(fitted, offset):
    """Same bars, same artifact, same window, same offset — same numbers.

    Compared on the feature columns the model actually consumes plus the
    baseline, because those are the inputs to the trade.
    """
    bars, dataset, fit, measured, test_index = fitted
    bundle = fit.bundle(FAST)
    window = test_index[len(test_index) // 2]

    live = score_live(bars, bundle, FAST, window_open=window, offset=offset)
    backtest = measured.loc[(measured['window_open'] == window)
                            & (measured['offset'] == offset)]
    assert not backtest.empty, 'the measured path produced no row for this window'
    assert len(live) == len(backtest)

    columns = list(feature_columns()) + [
        'strike', 'last_price', 'displacement', 'excursion_up', 'excursion_down',
        'sigma_per_min', 'sigma_remaining',
        'baseline_probability', 'baseline_probability_logit',
    ]
    a = live.set_index('symbol').sort_index()
    b = backtest.set_index('symbol').sort_index()
    assert list(a.index) == list(b.index)

    offenders = {}
    for column in columns:
        if column not in a.columns or column not in b.columns:
            offenders[column] = 'missing'
            continue
        x = a[column].to_numpy(dtype=float)
        y = b[column].to_numpy(dtype=float)
        if not np.allclose(x, y, rtol=1e-9, atol=1e-12, equal_nan=True):
            offenders[column] = (x.tolist(), y.tolist())
    assert not offenders, (
        f'offset {offset}, window {window}: the live and measured paths disagree '
        f'on {sorted(offenders)} — a backtested edge the live path cannot '
        f'reproduce is not an edge.\n{offenders}'
    )


def test_the_live_path_never_sees_the_label(fitted):
    """`score_live` must report the outcome as unknown, not as a plausible zero."""
    bars, _, fit, _, test_index = fitted
    live = score_live(bars, fit.bundle(FAST), FAST,
                      window_open=test_index[len(test_index) // 2], offset=3)
    assert live['outcome'].isna().all()
    assert live['settle_price'].isna().all()
    assert live['settle_return'].isna().all()


def test_the_live_path_refuses_a_window_the_bars_do_not_reach(fitted):
    """Rather than scoring the last window it happens to have.

    Silently scoring a stale window is how a live loop trades a settled market.
    """
    from core.dataset import DatasetError

    bars, _, fit, _, test_index = fitted
    future = test_index[-1] + pd.Timedelta(days=3)
    with pytest.raises(DatasetError, match='no window opens'):
        score_live(bars, fit.bundle(FAST), FAST, window_open=future, offset=3)


def test_a_symbol_with_no_fitted_volatility_model_is_dropped_not_guessed(fitted):
    """An artifact that does not cover a symbol must not score it off another
    symbol's model."""
    bars, _, fit, _, test_index = fitted
    bundle = fit.bundle(FAST)
    dropped = 'SOL-USD'
    assert bundle.covers(dropped)
    bundle.vol_models.pop(dropped, None)
    bundle.seasonality.pop(dropped, None)
    assert not bundle.covers(dropped)
    live = score_live(bars, bundle, FAST,
                      window_open=test_index[len(test_index) // 2], offset=3)
    assert dropped not in set(live['symbol'])
    assert len(live) == len(FAST.symbols) - 1
