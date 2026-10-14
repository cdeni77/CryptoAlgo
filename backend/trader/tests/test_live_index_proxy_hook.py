"""Live attaches the `index_proxy` group when the model was fitted on it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.index_features import INDEX_PROXY
from core.index_proxy import IndexProxyCache
from livetrade import features
from livetrade.features import _attach_book_features, _attach_index_proxy

WINDOW = pd.Timestamp('2026-10-13 12:00', tz='UTC')
DECISION = WINDOW + pd.Timedelta(minutes=3)


def _cache(strike=100.0, last=100.1, n=3) -> IndexProxyCache:
    cache = IndexProxyCache()
    start = int((WINDOW - pd.Timedelta(seconds=60)).timestamp())
    for t in range(start, int(DECISION.timestamp())):
        value = strike if t < WINDOW.timestamp() else last
        cache.record({'symbol': 'BTC-USD', 't': float(t), 'proxy': value,
                      'n_exchanges': n})
    # A row at the decision instant must not be read.
    cache.record({'symbol': 'BTC-USD', 't': float(DECISION.timestamp()),
                  'proxy': 999.0, 'n_exchanges': n})
    return cache


def _scored():
    return pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': [WINDOW], 'offset': [3],
        'decision_time': [DECISION], 'sigma_remaining': [0.002],
        'last_price': [100.05], 'baseline_probability': [0.5]})


def test_the_group_comes_from_the_recorder_cache():
    scored = _scored()
    _attach_index_proxy(scored, DEFAULT_CONFIG, cache=_cache())
    assert scored['proxy_displacement_z'].iloc[0] == pytest.approx(0.001 / 0.002)
    assert scored['proxy_minus_spot_bps'].iloc[0] == pytest.approx(
        1e4 * (100.1 / 100.05 - 1))
    assert scored['proxy_n_exchanges'].iloc[0] == 3.0


def test_no_recorder_leaves_the_group_nan(monkeypatch):
    monkeypatch.setattr(features, '_index_cache', lambda: None)
    scored = _scored()
    _attach_index_proxy(scored, DEFAULT_CONFIG)
    assert all(np.isnan(scored[c].iloc[0]) for c in INDEX_PROXY)


def test_book_features_attach_it_only_for_a_model_that_uses_it(monkeypatch):
    monkeypatch.setattr(features, '_index_cache', lambda: _cache())
    monkeypatch.setattr(features, '_venue_caches', lambda: ({}, {}))
    without = _scored()
    _attach_book_features(without, {}, groups=('cross_asset',))
    assert 'proxy_displacement_z' not in without.columns
    with_group = _scored()
    _attach_book_features(with_group, {}, groups=('cross_asset', 'index_proxy'))
    assert np.isfinite(with_group['proxy_displacement_z'].iloc[0])


def test_run_cycle_defers_the_columns():
    import inspect
    from scripts import live
    source = inspect.getsource(live.run_cycle)
    assert 'INDEX_PROXY' in source and 'groups=model.groups' in source
