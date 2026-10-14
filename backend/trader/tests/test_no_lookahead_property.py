"""Property test: nothing a decision reads may depend on a bar at or after the decision.

Scramble every bar from the decision onward; every column the decision reads must not
change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.windows import build_windows
from tests.conftest import make_bars

# Everything a decision at offset m may depend on. `settle_*` and `outcome` are
# excluded: they are the label, a future quantity.
DECISION_COLUMNS = ('strike', 'last_price', 'displacement',
                    'excursion_up', 'excursion_down')

OFFSETS = (3, 6, 9, 12)


def scramble_from(bars: pd.DataFrame, cutoff: pd.Timestamp, rng) -> pd.DataFrame:
    """Scramble every bar at or after `cutoff` by up to 5%, far outside float tolerance."""
    out = bars.copy()
    hit = (out['event_time'] >= cutoff).to_numpy()
    assert hit.any(), 'cutoff is past the end of the sample'
    shock = 1.0 + rng.uniform(-0.05, 0.05, int(hit.sum()))
    for column in ('open', 'high', 'low', 'close'):
        values = np.array(out[column].to_numpy(dtype=float), copy=True)
        values[hit] = values[hit] * shock
        out[column] = values
    out['high'] = out[['open', 'high', 'low', 'close']].max(axis=1)
    out['low'] = out[['open', 'high', 'low', 'close']].min(axis=1)
    return out


@pytest.mark.parametrize('offset', OFFSETS)
def test_no_window_column_a_decision_reads_moves_when_the_future_moves(offset):
    rng = np.random.default_rng(1000 + offset)
    bars = make_bars(days=6)['BTC-USD']
    clean, _ = build_windows(bars, 'BTC-USD', offsets=(offset,))
    clean = clean.set_index('window_open')

    targets = list(clean.index[len(clean) // 2::37])[:8]
    assert len(targets) >= 4

    for target in targets:
        cutoff = target + pd.Timedelta(minutes=offset)
        dirty, _ = build_windows(
            scramble_from(bars, cutoff, rng), 'BTC-USD', offsets=(offset,))
        dirty = dirty.set_index('window_open')
        assert target in dirty.index
        for column in DECISION_COLUMNS:
            a = float(clean.loc[target, column])
            b = float(dirty.loc[target, column])
            assert a == pytest.approx(b, rel=1e-12, abs=0.0), (
                f'offset {offset}, window {target}: `{column}` moved '
                f'{a} -> {b} when only bars at or after the decision minute '
                f'({cutoff}) were perturbed — it reads the future'
            )
        # The perturbation must change the settlement, or the check above is vacuous.
        assert float(clean.loc[target, 'settle_price']) != pytest.approx(
            float(dirty.loc[target, 'settle_price'])), 'perturbation was inert'


@pytest.mark.parametrize('offset', (3, 12))
def test_no_scored_feature_moves_when_the_future_moves(offset):
    """The same property over the whole live feature matrix.

    The fitted bundle is held fixed, so only the bars differ between the two calls.
    """
    from core.dataset import Dataset, fit_fold, score_live

    config = Config(decision_offsets=(offset,), n_estimators=40,
                    early_stopping_rounds=10, n_folds=3, seasonality_min_days=5)
    rng = np.random.default_rng(2000 + offset)
    bars = make_bars(days=30)

    dataset = Dataset.build(bars, config)
    index = dataset.window_index
    fit, _ = fit_fold(dataset, index[:int(len(index) * 0.8)], config)
    bundle = fit.bundle(config)
    target = index[-3]
    cutoff = target + pd.Timedelta(minutes=offset)

    def scored(sample):
        return score_live(sample, bundle, config,
                          window_open=target, offset=offset
                          ).set_index('symbol').sort_index()

    clean = scored(bars)
    dirty = scored({s: scramble_from(b, cutoff, rng) for s, b in bars.items()})

    excluded = {'settle_price', 'settle_return', 'outcome', 'settle_time',
                'window_open', 'decision_time', 'offset', 'minutes_missing',
                'complete'}
    numeric = [c for c in clean.columns
               if c not in excluded and pd.api.types.is_numeric_dtype(clean[c])]
    assert len(numeric) > 20, numeric
    offenders = [c for c in numeric
                 if not np.allclose(clean[c].to_numpy(dtype=float),
                                    dirty[c].to_numpy(dtype=float),
                                    rtol=1e-9, atol=0.0, equal_nan=True)]
    assert not offenders, (
        f'offset {offset}: these columns moved when only bars at or after the '
        f'decision minute ({cutoff}) were perturbed, so they read the future: '
        f'{offenders}'
    )
