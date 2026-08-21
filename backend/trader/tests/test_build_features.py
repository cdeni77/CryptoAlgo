"""End-to-end: research store -> feature panel.

These tests use synthetic bars, which is the right tool for the properties being
checked — reproducibility, point-in-time bounding, quality filtering — because
the ground truth is known by construction. They say nothing about whether the
features predict anything; only real out-of-sample data can answer that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.datastore import ResearchStore
from core.dataset import load_dataset
from core.features import feature_columns

SYMBOLS = ('BIP', 'ETP', 'SLP', 'XPP', 'DOP')
PRICES = {'BIP': 60_000.0, 'ETP': 3_000.0, 'SLP': 150.0, 'XPP': 2.2, 'DOP': 0.35}
BARS = 1_200


def _seed_store(root, *, quality: str = 'valid', identical_venues: bool = False) -> ResearchStore:
    """Populate a store with two venues, funding and open interest."""
    store = ResearchStore(root)
    index = pd.date_range('2026-01-01', periods=BARS, freq='1h', tz='UTC')

    for i, symbol in enumerate(SYMBOLS):
        rng = np.random.default_rng(i)
        close = PRICES[symbol] * np.exp(np.cumsum(rng.normal(0.0001, 0.012, BARS)))
        open_ = np.concatenate([[close[0]], close[:-1]])

        def bars_frame(venue: str, series: np.ndarray, opens: np.ndarray) -> pd.DataFrame:
            return pd.DataFrame({
                'venue': venue, 'symbol': symbol, 'event_time': index,
                'available_time': index + pd.Timedelta(hours=1), 'quality': quality,
                'open': opens, 'high': np.maximum(opens, series) * 1.004,
                'low': np.minimum(opens, series) * 0.996, 'close': series,
                'volume': rng.lognormal(8, 0.6, BARS),
                'quote_volume': np.nan, 'trade_count': np.nan,
            })

        store.write('bars', bars_frame('coinbase', close, open_))

        # A real second venue carries its own idiosyncratic basis per instrument.
        # A constant multiple would make the basis identical across the universe,
        # which the cross-sectional guard correctly reduces to NaN.
        if identical_venues:
            ref_close = close * 1.0004
        else:
            drift = np.cumsum(rng.normal(0.0004, 0.0008, BARS)) * 0.01
            ref_close = close * (1 + drift + rng.normal(0, 0.0006, BARS))
        ref_open = np.concatenate([[ref_close[0]], ref_close[:-1]])
        store.write('bars', bars_frame('binance', ref_close, ref_open))

        store.write('funding', pd.DataFrame({
            'venue': 'coinbase', 'symbol': symbol, 'event_time': index,
            'available_time': index, 'quality': quality,
            'rate': rng.normal(1e-5, 4e-5, BARS), 'mark_price': close,
            'index_price': close, 'interval_hours': 1, 'is_settlement': 0,
        }))
        # Open interest only exists on a proxy venue: Coinbase exposes no
        # open-interest endpoint, so the builder must fall back to find it.
        store.write('open_interest', pd.DataFrame({
            'venue': 'binance', 'symbol': symbol, 'event_time': index,
            'available_time': index, 'quality': quality,
            'oi_contracts': np.abs(np.cumsum(rng.normal(0, 50, BARS)) + 5e4),
            'oi_base': np.nan, 'oi_usd': np.nan,
        }))

    return store


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(
        repo_root / 'configs/exchange/coinbase_us_perps_cde_v202602.json'
    )


@pytest.fixture
def store(tmp_path) -> ResearchStore:
    return _seed_store(tmp_path / 'research')


def _build(store: ResearchStore, config: Config, **kwargs) -> pd.DataFrame:
    """Go through `core.dataset`, the loader every script and the live path share.

    Testing the shared loader rather than one CLI's wrapper is the point: if this
    passes, `build_features`, `train`, `backtest` and `signals` all see the same
    panel, because there is only one function that can build one.
    """
    params = dict(
        venue='coinbase', reference_venue='binance',
        symbols=list(SYMBOLS), config=config,
    )
    params.update(kwargs)
    return load_dataset(store, **params).features


def test_builds_a_full_panel(store, config):
    panel = _build(store, config)

    assert panel.index.names == ['event_time', 'symbol']
    assert list(panel.columns) == feature_columns()
    assert set(panel.index.get_level_values('symbol')) == set(SYMBOLS)


def test_every_feature_populates_with_a_real_basis(store, config):
    """No group should be empty when all of its source data is present.

    A zero-coverage column means either missing inputs or a degenerate
    cross-section, and both are worth catching here rather than discovering as a
    silently absent feature during training.
    """
    panel = _build(store, config)
    coverage = 1.0 - panel.isna().mean()
    empty = coverage[coverage <= 0.0]

    assert empty.empty, f'features with no data: {list(empty.index)}'


def test_open_interest_falls_back_to_the_proxy_venue(store, config):
    """Coinbase has no open-interest endpoint, so it lives under another venue."""
    panel = _build(store, config)

    assert panel['oi_change_24h'].notna().any()
    assert panel['liquidation_cascade_24h'].notna().any()


def test_identical_venues_collapse_the_basis(tmp_path, config):
    """A constant cross-venue offset carries no cross-sectional information."""
    store = _seed_store(tmp_path / 'degenerate', identical_venues=True)
    panel = _build(store, config)

    assert panel['basis_bps'].isna().all()
    # The lead-lag correlation still varies, so it must survive.
    assert panel['lead_lag_corr_72h'].notna().any()


def test_build_is_reproducible(store, config):
    """Same inputs, same content hash — this is what a model artifact records."""
    from core.datastore import feature_hash

    first = _build(store, config)
    second = _build(store, config)

    pd.testing.assert_frame_equal(first, second)
    assert feature_hash(first) == feature_hash(second)


def test_as_of_bounds_the_panel(store, config):
    """`as_of` filters on available_time, so a past build stays reproducible."""
    cutoff = '2026-01-20T00:00:00Z'
    bounded = _build(store, config, as_of=cutoff)
    full = _build(store, config)

    assert bounded.index.get_level_values('event_time').max() <= pd.Timestamp(cutoff)
    assert len(bounded) < len(full)


def test_flagged_data_is_excluded_by_default(tmp_path, config):
    """Suspicious rows must not reach the panel unless asked for."""
    store = _seed_store(tmp_path / 'suspect', quality='suspicious')

    default_build = _build(store, config)
    permissive = _build(store, config, min_quality='suspicious')

    assert default_build.empty, 'suspicious data leaked into a default build'
    assert not permissive.empty


def test_warmup_is_trimmed(store, config):
    """Features built from a partial window are a different feature."""
    from core.features import MAX_WARMUP_BARS

    panel = _build(store, config)
    per_symbol = panel.groupby(level='symbol').size()

    assert (per_symbol == BARS - MAX_WARMUP_BARS).all()
