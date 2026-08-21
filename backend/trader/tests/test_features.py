"""Feature layer invariants.

The lookahead test is the important one. Everything else in the pipeline is
downstream of the guarantee that a feature computed at time t used only data
available at time t, and that guarantee is easy to break by accident with a
centred rolling window or a forgotten shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.costs import fee_floor, get_contract_spec
from core.features import (
    GROUPS,
    MAX_ABS_ZSCORE,
    SymbolInputs,
    build_panel,
    build_symbol_features,
    carry_features,
    cost_features,
    feature_columns,
    positioning_features,
    standardizable_columns,
)

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'
BARS = 900


def _bars(price: float, *, seed: int, n: int = BARS, drift: float = 0.0) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close,
         'volume': rng.lognormal(8, 0.6, n)},
        index=index,
    )


def _inputs(symbol: str, price: float, *, seed: int,
            market: pd.DataFrame | None = None,
            identical_reference: bool = False) -> SymbolInputs:
    bars = _bars(price, seed=seed)
    index = bars.index
    rng = np.random.default_rng(seed + 100)
    reference = bars.copy()
    reference['close'] = bars['close'] * (
        1.0004 if identical_reference else (1 + rng.normal(0.0004, 0.0006, len(index)))
    )
    return SymbolInputs(
        symbol=symbol,
        bars=bars,
        funding=pd.DataFrame({'rate': rng.normal(1e-5, 4e-5, len(index))}, index=index),
        open_interest=pd.DataFrame(
            {'oi_contracts': np.abs(np.cumsum(rng.normal(0, 50, len(index))) + 5000)},
            index=index,
        ),
        reference_bars=reference,
        market_bars=market if market is not None else bars,
    )


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


@pytest.fixture(scope='module')
def universe(config) -> list[SymbolInputs]:
    market = _bars(60_000, seed=1, drift=0.0002)
    return [
        _inputs('BIP', 60_000, seed=1, market=market),
        _inputs('ETP', 3_000, seed=2, market=market),
        _inputs('SLP', 150, seed=3, market=market),
        _inputs('XPP', 2.2, seed=4, market=market),
        _inputs('DOP', 0.35, seed=5, market=market),
    ]


def test_no_lookahead(config):
    """A feature must not change when data after its timestamp arrives.

    Build on a truncated history, build on the full history, and require the
    overlapping rows to be bit-identical. Any forward-looking window, any
    reindex that fills backwards, any normalisation fitted on the whole sample
    shows up here.
    """
    full = _inputs('BIP', 60_000, seed=7)
    cut = 700
    truncated = SymbolInputs(
        symbol='BIP',
        bars=full.bars.iloc[:cut],
        funding=full.funding.iloc[:cut],
        open_interest=full.open_interest.iloc[:cut],
        reference_bars=full.reference_bars.iloc[:cut],
        market_bars=full.market_bars.iloc[:cut],
    )

    early = build_symbol_features(truncated, config=config)
    late = build_symbol_features(full, config=config)
    overlap = early.index.intersection(late.index)

    assert len(overlap) > 100, 'not enough overlap to be a meaningful test'
    delta = (early.loc[overlap] - late.loc[overlap]).abs()
    changed = delta.columns[(delta > 1e-12).any()].tolist()
    assert not changed, f'these features saw the future: {changed}'


def test_funding_is_lagged_one_bar():
    """Funding is published after its window closes, so it must be shifted."""
    inputs = _inputs('BIP', 60_000, seed=11)
    carry = carry_features(inputs)
    expected = (inputs.funding['rate'].shift(1) * 10_000).reindex(carry.index)
    pd.testing.assert_series_equal(
        carry['carry_bps'].dropna(), expected.dropna(), check_names=False
    )


def test_open_interest_is_lagged_one_bar():
    inputs = _inputs('BIP', 60_000, seed=12)
    positioning = positioning_features(inputs)
    expected = inputs.open_interest['oi_contracts'].shift(1).pct_change().reindex(
        positioning.index
    )
    pd.testing.assert_series_equal(
        positioning['oi_change_1h'].dropna(), expected.dropna(), check_names=False
    )


def test_every_group_emits_features():
    from core.features import _group_column_names

    for group in GROUPS:
        assert _group_column_names(group), f'{group.name} emitted nothing'


def test_features_are_dense_after_warmup(config):
    market = _bars(60_000, seed=99, drift=0.0002)
    frame = build_symbol_features(
        _inputs('SLP', 150, seed=13, market=market), config=config
    )
    assert frame.shape[1] == len(feature_columns())
    worst = frame.isna().mean().max()
    assert worst < 0.05, f'sparsest feature is {worst:.1%} NaN after warmup'


def test_market_factor_omitted_for_the_market_itself(config):
    """BTC has no beta to BTC, so the group emits nothing rather than 1.0s.

    This mirrors `CoinProfile.include_btc_relative`: a coin measured against
    itself produces constants, and a constant feature is a free split for a
    tree to overfit on.
    """
    market = _bars(60_000, seed=1)
    btc = SymbolInputs(symbol='BIP', bars=market, market_bars=market)
    frame = build_symbol_features(btc, config=config)

    assert not [c for c in frame.columns if c.startswith('btc_')]

    alt = _inputs('SLP', 150, seed=2, market=market)
    assert [c for c in build_symbol_features(alt, config=config) if c.startswith('btc_')]


def test_cross_section_is_standardised(universe, config):
    panel = build_panel(universe, config=config)
    relative = standardizable_columns(panel.columns)
    grouped = panel[relative].groupby(level='event_time')

    assert grouped.mean().mean().abs().max() < 1e-9
    assert abs(grouped.std().mean().mean() - 1.0) < 0.02
    assert panel[relative].abs().max().max() <= MAX_ABS_ZSCORE + 1e-9


def test_absolute_features_are_not_standardised(universe, config):
    """The fee hurdle is 25bp whatever the other contracts cost."""
    panel = build_panel(universe, config=config)
    hurdle = panel['fee_hurdle_bps'].groupby(level='symbol').last()

    assert hurdle.nunique() == len(hurdle), 'hurdle was flattened across symbols'
    assert hurdle['ETP'] > hurdle['DOP'], 'ETH costs more per contract than DOGE'


def test_degenerate_cross_section_yields_nan(config):
    """Identical values across the universe carry no ranking information.

    Standardising them would divide float noise by float noise and hand the
    model large z-scores built from nothing.
    """
    market = _bars(60_000, seed=1)
    degenerate = [
        _inputs(symbol, price, seed=20 + i, market=market, identical_reference=True)
        for i, (symbol, price) in enumerate([('BIP', 60_000), ('ETP', 3_000), ('SLP', 150)])
    ]
    panel = build_panel(degenerate, config=config)
    assert panel['contemp_corr_72h'].isna().all()


def test_fee_hurdle_tracks_the_real_schedule(config):
    """Per-contract commission means the hurdle is set by notional per contract.

    BTC and ETH carry a $0.75 commission on a few hundred dollars of notional;
    the group-B contracts carry $0.10 on a thousand or more. That ordering is
    the whole reason the hurdle belongs in the feature set.
    """
    hurdles = {}
    for symbol, price in [('BIP', 60_000), ('ETP', 3_000), ('SLP', 150), ('DOP', 0.35)]:
        inputs = _inputs(symbol, price, seed=30)
        hurdles[symbol] = cost_features(inputs, config=config)['fee_hurdle_bps'].iloc[-1]

    assert fee_floor('BIP', config) == pytest.approx(0.75)
    assert fee_floor('DOP', config) == pytest.approx(0.10)
    assert hurdles['ETP'] > hurdles['BIP'] > hurdles['SLP']
    assert hurdles['DOP'] < hurdles['BIP']


def test_contract_notional_matches_spec(config):
    inputs = _inputs('SLP', 150, seed=31)
    features = cost_features(inputs, config=config)
    expected = get_contract_spec('SLP').units * inputs.bars['close']
    pd.testing.assert_series_equal(
        features['contract_notional_usd'], expected, check_names=False
    )
