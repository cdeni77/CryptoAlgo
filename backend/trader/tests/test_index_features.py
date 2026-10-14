"""The opt-in `index_proxy` feature group: point in time, parity, wiring."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.index_features import (
    INDEX_PROXY, MIN_STRIKE_SAMPLES, attach_index_proxy, index_features_row,
    load_proxy_minutes, minute_summary, proxy_displacement_z,
)

OPEN = pd.Timestamp('2026-10-14 12:15', tz='UTC')


def _seconds(start, end, value=lambda t: 100.0, n=3, symbol='BTC-USD'):
    times = pd.date_range(start, end, freq='1s', inclusive='left', tz='UTC')
    return pd.DataFrame({'symbol': symbol, 'event_time': times,
                         'proxy': [value(t) for t in times], 'n_exchanges': n})


def _window(offset=3, last_price=100.0, symbol='BTC-USD'):
    return pd.DataFrame({'symbol': [symbol], 'window_open': [OPEN],
                         'decision_time': [OPEN + pd.Timedelta(minutes=offset)],
                         'last_price': [last_price], 'offset': [offset]})


def _attach(seconds, **kw):
    return attach_index_proxy(_window(**kw), minute_summary(seconds))


def test_the_strike_is_the_mean_of_the_minute_before_the_open():
    strike = _seconds(OPEN - pd.Timedelta(minutes=1), OPEN,
                      value=lambda t: 100.0 + t.second / 59.0)
    after = _seconds(OPEN, OPEN + pd.Timedelta(minutes=3), value=lambda t: 101.0)
    out = _attach(pd.concat([strike, after]))
    assert out['proxy_strike'].iloc[0] == pytest.approx(100.5)
    assert out['proxy_last'].iloc[0] == 101.0
    assert out['proxy_displacement'].iloc[0] == pytest.approx(101.0 / 100.5 - 1)


def test_nothing_at_or_after_the_decision_is_read():
    """A spike stamped at the decision second must not reach the feature."""
    decision = OPEN + pd.Timedelta(minutes=3)
    seconds = _seconds(OPEN - pd.Timedelta(minutes=1), decision + pd.Timedelta(seconds=30),
                       value=lambda t: 200.0 if t >= decision else 100.0)
    out = _attach(seconds)
    assert out['proxy_last'].iloc[0] == 100.0
    # And the live path agrees, even handed a snapshot taken late.
    live = index_features_row(seconds, OPEN, decision, 0.001)
    assert live['proxy_displacement_z'] == 0.0


def test_a_stale_last_row_is_not_used():
    decision = OPEN + pd.Timedelta(minutes=3)
    seconds = pd.concat([
        _seconds(OPEN - pd.Timedelta(minutes=1), OPEN),
        _seconds(decision - pd.Timedelta(seconds=60), decision - pd.Timedelta(seconds=10)),
    ])
    out = attach_index_proxy(_window(), minute_summary(seconds), max_age_seconds=5)
    assert np.isnan(out['proxy_last'].iloc[0])
    out = attach_index_proxy(_window(), minute_summary(seconds), max_age_seconds=11)
    assert out['proxy_last'].iloc[0] == 100.0


def test_a_thin_strike_minute_is_nan():
    seconds = pd.concat([
        _seconds(OPEN - pd.Timedelta(seconds=MIN_STRIKE_SAMPLES - 1), OPEN),
        _seconds(OPEN, OPEN + pd.Timedelta(minutes=3)),
    ])
    assert np.isnan(_attach(seconds)['proxy_strike'].iloc[0])


def test_nan_proxy_rows_are_not_samples():
    seconds = _seconds(OPEN - pd.Timedelta(minutes=1), OPEN + pd.Timedelta(minutes=3))
    seconds.loc[seconds.index[:40], 'proxy'] = np.nan
    assert minute_summary(seconds).iloc[0]['samples'] == 20
    assert np.isnan(_attach(seconds)['proxy_strike'].iloc[0])


def test_basis_and_count():
    seconds = _seconds(OPEN - pd.Timedelta(minutes=1), OPEN + pd.Timedelta(minutes=3),
                       value=lambda t: 100.01, n=4)
    out = _attach(seconds, last_price=100.0)
    assert out['proxy_minus_spot_bps'].iloc[0] == pytest.approx(1.0)
    assert out['proxy_n_exchanges'].iloc[0] == 4.0


def test_the_join_never_changes_the_row_count():
    windows = pd.concat([_window(o) for o in (3, 6, 9, 12)], ignore_index=True)
    seconds = _seconds(OPEN - pd.Timedelta(minutes=1), OPEN + pd.Timedelta(minutes=15))
    assert len(attach_index_proxy(windows, minute_summary(seconds))) == 4
    empty = attach_index_proxy(windows, minute_summary(seconds.iloc[0:0]))
    assert len(empty) == 4 and empty['proxy_displacement'].isna().all()


def test_the_live_row_matches_the_backtest_join_exactly():
    rng = np.random.default_rng(3)
    seconds = _seconds(OPEN - pd.Timedelta(minutes=5), OPEN + pd.Timedelta(minutes=15))
    seconds['proxy'] = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-5, len(seconds))))
    seconds['n_exchanges'] = rng.integers(2, 5, len(seconds))
    for offset in (3, 6, 9, 12):
        decision = OPEN + pd.Timedelta(minutes=offset)
        sigma = 0.0012
        batch = attach_index_proxy(_window(offset, last_price=100.02),
                                   minute_summary(seconds))
        z = proxy_displacement_z(batch['proxy_displacement'], [sigma])[0]
        live = index_features_row(seconds, OPEN, decision, sigma, last_price=100.02)
        assert live['proxy_displacement_z'] == z
        assert live['proxy_minus_spot_bps'] == batch['proxy_minus_spot_bps'].iloc[0]
        assert live['proxy_n_exchanges'] == batch['proxy_n_exchanges'].iloc[0]


def test_the_live_row_reads_the_cache_snapshot_shape():
    from core.index_proxy import IndexProxyCache
    cache = IndexProxyCache()
    start = (OPEN - pd.Timedelta(minutes=1)).timestamp()
    for i in range(240):
        cache.record({'t': start + i, 'symbol': 'BTC-USD',
                      'proxy': 100.0 if i < 60 else 100.1, 'n_exchanges': 3})
    row = index_features_row(cache.snapshot('BTC-USD'), OPEN,
                             OPEN + pd.Timedelta(minutes=3), 0.001,
                             last_price=100.05)
    assert row['proxy_displacement_z'] == pytest.approx(0.001 / 0.001, rel=1e-9)
    assert row['proxy_n_exchanges'] == 3.0
    assert set(row) == set(INDEX_PROXY)


def test_an_empty_cache_is_nan_not_an_error():
    row = index_features_row(None, OPEN, OPEN + pd.Timedelta(minutes=3), 0.001,
                             symbol='BTC-USD')
    assert all(np.isnan(v) for v in row.values())


def test_duckdb_summary_matches_the_pandas_one(tmp_path):
    from core.spool import compact_table, table_spool
    rng = np.random.default_rng(5)
    seconds = pd.concat([
        _seconds(OPEN - pd.Timedelta(minutes=2), OPEN + pd.Timedelta(minutes=2),
                 symbol=s) for s in ('BTC-USD', 'ETH-USD')], ignore_index=True)
    seconds['proxy'] = rng.uniform(99, 101, len(seconds))
    seconds.loc[seconds.index[::7], 'proxy'] = np.nan
    spool = table_spool(tmp_path / 'spool', 'index_proxy_1s', 'index_proxy')
    for row in seconds.itertuples(index=False):
        t = row.event_time.timestamp()
        spool.append({'t': t, 'computed': t + 0.01, 'venue': 'index_proxy',
                      'symbol': row.symbol,
                      'proxy': None if np.isnan(row.proxy) else row.proxy,
                      'n_exchanges': int(row.n_exchanges)})
    spool.close()
    compact_table(tmp_path / 'spool', tmp_path / 'store', 'index_proxy_1s',
                  keep_days=0, now=(OPEN + pd.Timedelta(hours=3)).timestamp())
    sql = load_proxy_minutes(tmp_path / 'store').sort_values(
        ['symbol', 'minute'], ignore_index=True)
    ref = minute_summary(seconds).sort_values(['symbol', 'minute'], ignore_index=True)
    assert list(sql['minute']) == list(ref['minute'])
    np.testing.assert_allclose(sql['mean'], ref['mean'], rtol=1e-12)
    np.testing.assert_array_equal(sql['samples'], ref['samples'])
    np.testing.assert_array_equal(sql['last'], ref['last'])
    assert list(sql['last_time']) == list(ref['last_time'])
    only = load_proxy_minutes(tmp_path / 'store', symbols=['ETH-USD'])
    assert set(only['symbol']) == {'ETH-USD'}


def test_load_proxy_minutes_on_an_empty_store(tmp_path):
    assert load_proxy_minutes(tmp_path).empty


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------

def test_the_group_is_selectable_but_not_default():
    from core.features import ALL_GROUPS, FEATURE_GROUPS, FEATURE_PARITY, feature_columns
    assert FEATURE_GROUPS['index_proxy'] == INDEX_PROXY
    assert 'index_proxy' not in ALL_GROUPS
    assert not set(INDEX_PROXY) & set(feature_columns())
    assert all(c in FEATURE_PARITY for c in INDEX_PROXY)


def test_build_features_finishes_the_z_score_on_the_fitted_sigma():
    from core.features import build_features
    table = pd.DataFrame({
        'symbol': ['BTC-USD'], 'window_open': [OPEN], 'offset': [3],
        'decision_time': [OPEN + pd.Timedelta(minutes=3)],
        'displacement': [0.001], 'excursion_up': [0.002], 'excursion_down': [-0.001],
        'sigma_per_min': [0.0005], 'sigma_remaining': [0.002],
        'proxy_displacement': [0.0011], 'proxy_minus_spot_bps': [0.5],
        'proxy_n_exchanges': [3.0], 'last_price': [100.0], 'strike': [99.9],
    })
    out = build_features(table, {}, groups=('index_proxy',))
    assert out['proxy_displacement_z'].iloc[0] == pytest.approx(0.55)
    assert out['proxy_n_exchanges'].iloc[0] == 3.0


def test_the_mirror_flips_the_displacement_and_the_basis():
    from core.features import mirror_rows
    table = pd.DataFrame({'proxy_displacement_z': [1.5], 'proxy_minus_spot_bps': [-2.0],
                          'proxy_n_exchanges': [3.0], 'proxy_displacement': [0.001]})
    out = mirror_rows(table)
    assert list(out.iloc[0]) == [-1.5, 2.0, 3.0, -0.001]


def test_dataset_build_attaches_the_proxy_point_in_time():
    from core.config import Config
    from core.dataset import Dataset
    minutes = pd.date_range(OPEN - pd.Timedelta(hours=2), OPEN + pd.Timedelta(hours=1),
                            freq='1min', inclusive='left', tz='UTC')
    rng = np.random.default_rng(9)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 5e-4, len(minutes))))
    bars = pd.DataFrame({'event_time': minutes, 'open': close, 'high': close * 1.0002,
                         'low': close * 0.9998, 'close': close, 'volume': 1.0})
    seconds = _seconds(minutes[0], minutes[-1], value=lambda t: 100.0)
    ds = Dataset.build({'BTC-USD': bars}, Config(symbols=('BTC-USD',)),
                       index_proxy=minute_summary(seconds))
    assert {'proxy_displacement', 'proxy_minus_spot_bps',
            'proxy_n_exchanges'} <= set(ds.windows.columns)
    assert ds.windows['proxy_displacement'].notna().any()
    assert (ds.windows['proxy_displacement'].dropna() == 0.0).all()
