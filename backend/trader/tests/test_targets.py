"""Target invariants.

Every one of these is exact arithmetic, so they are asserted rather than
approximated. The cost identity in particular is what makes the trading hurdle
structural instead of a tuned threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.targets import (
    TargetSpec,
    build_target_panel,
    build_targets,
    carry_return,
    price_return,
    round_trip_cost,
    summarise_targets,
    target_spec_for,
)
from core.profiles import COIN_PROFILES

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


def _flat(n: int = 12, price: float = 100.0) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = pd.Series(price, index=index, dtype=float)
    return pd.DataFrame(
        {'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1.0},
        index=index,
    )


def _ramp(n: int = 12, start: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    close = pd.Series(start + step * np.arange(n), index=index, dtype=float)
    return pd.DataFrame(
        {'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1.0},
        index=index,
    )


def _funding(index: pd.DatetimeIndex, rate: float) -> pd.DataFrame:
    return pd.DataFrame({'rate': rate}, index=index)


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def test_price_return_is_exact():
    bars = _ramp()
    returns = price_return(bars['close'], 3)

    assert returns.iloc[0] == pytest.approx(103 / 100 - 1)
    assert returns.iloc[-3:].isna().all(), 'rows without a forward window must be NaN'


def test_long_pays_positive_funding():
    """Sign convention: positive funding means longs pay shorts."""
    bars = _flat()
    carry = carry_return(_funding(bars.index, 0.0001)['rate'], bars.index, 3)

    assert carry.iloc[0] == pytest.approx(-0.0003)


def test_long_receives_negative_funding():
    bars = _flat()
    carry = carry_return(_funding(bars.index, -0.0001)['rate'], bars.index, 3)

    assert carry.iloc[0] == pytest.approx(0.0003)


def test_absent_funding_is_zero_carry_not_nan():
    bars = _flat()

    carry = carry_return(None, bars.index, 3)

    assert (carry == 0.0).all()


# ---------------------------------------------------------------------------
# The cost identity
# ---------------------------------------------------------------------------


def test_both_sides_pay_the_cost():
    """net_long + net_short == -2 * cost, exactly.

    This is the trading hurdle made structural. It is why at most one side can
    be worth taking, and why no threshold needs tuning to enforce it.
    """
    bars = _ramp()
    targets = build_targets(bars, TargetSpec(3, 0.0006), funding=_funding(bars.index, 0.0001))
    total = (targets['net_long'] + targets['net_short']).dropna()

    assert np.allclose(total, -2 * 0.0006)


def test_at_most_one_side_is_positive():
    bars = _ramp()
    targets = build_targets(bars, TargetSpec(3, 0.0006), funding=_funding(bars.index, 0.0001))

    both = (targets['net_long'] > 0) & (targets['net_short'] > 0)

    assert both.sum() == 0


def test_stands_aside_when_neither_side_clears_cost():
    bars = _flat()
    targets = build_targets(bars, TargetSpec(3, 0.0006), funding=_funding(bars.index, 0.0))

    assert (targets['best_side'].dropna() == 0).all()


def test_carry_alone_can_justify_a_position():
    """The case binary classification could not express.

    96 hours at 2bp an hour is 192bp of carry against a ~5bp round trip on a
    group-B contract. The short side collects it with no view on price at all.
    """
    index = pd.date_range('2026-01-01', periods=400, freq='1h', tz='UTC')
    close = pd.Series(100.0, index=index)
    bars = pd.DataFrame(
        {'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1.0},
        index=index,
    )

    targets = build_targets(bars, TargetSpec(96, 0.0005), funding=_funding(index, 0.0002))
    row = targets.dropna().iloc[0]

    assert row['carry'] == pytest.approx(-0.0192)
    assert row['best_side'] == -1
    assert row['net_short'] > 0.018


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_cost_varies_by_contract(config):
    eth = round_trip_cost('ETP', 3_000, config)
    doge = round_trip_cost('DOP', 0.35, config)

    assert eth > doge
    assert eth / doge > 5


def test_cost_is_size_invariant(config):
    """A per-contract commission is a fixed fraction of notional."""
    assert round_trip_cost('BIP', 60_000, config, contracts=1) == pytest.approx(
        round_trip_cost('BIP', 60_000, config, contracts=250)
    )


def test_spec_horizon_follows_the_hold_period(config):
    profile = COIN_PROFILES['SOL']
    spec = target_spec_for('SLP', profile=profile, config=config, reference_price=150.0)

    assert spec.horizon_bars == profile.max_hold_hours
    assert spec.round_trip_cost > 0


def test_spec_rejects_a_zero_horizon():
    with pytest.raises(ValueError):
        TargetSpec(0)


# ---------------------------------------------------------------------------
# No lookahead
# ---------------------------------------------------------------------------


def test_targets_do_not_change_when_the_future_arrives():
    """A resolved target must be final.

    Targets are forward-looking by definition, so the property to check is that
    a row already resolvable does not shift when later bars appear.
    """
    index = pd.date_range('2026-01-01', periods=600, freq='1h', tz='UTC')
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))), index=index)
    bars = pd.DataFrame(
        {'open': close, 'high': close * 1.002, 'low': close * 0.998,
         'close': close, 'volume': 1.0},
        index=index,
    )
    spec = TargetSpec(24, 0.0006)
    funding = _funding(index, 0.00001)

    full = build_targets(bars, spec, funding=funding)
    partial = build_targets(bars.iloc[:400], spec, funding=funding.iloc[:400])

    overlap = partial.dropna(subset=['price']).index
    pd.testing.assert_frame_equal(full.loc[overlap], partial.loc[overlap])


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def test_panel_matches_the_feature_panel_shape(config):
    index = pd.date_range('2026-01-01', periods=500, freq='1h', tz='UTC')
    bars = {}
    funding = {}
    for i, symbol in enumerate(('SLP', 'DOP', 'ETP')):
        rng = np.random.default_rng(i)
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=index)
        bars[symbol] = pd.DataFrame(
            {'open': close, 'high': close * 1.003, 'low': close * 0.997,
             'close': close, 'volume': 1.0},
            index=index,
        )
        funding[symbol] = _funding(index, 0.00002)

    panel = build_target_panel(
        bars, funding_by_symbol=funding, config=config, horizon_bars=24
    )

    assert panel.index.names == ['event_time', 'symbol']
    assert set(panel.index.get_level_values('symbol')) == set(bars)
    assert np.allclose(
        (panel['net_long'] + panel['net_short']).dropna(),
        -2 * panel['cost'].dropna(),
    )


def test_summary_reports_the_carry_share(config):
    index = pd.date_range('2026-01-01', periods=600, freq='1h', tz='UTC')
    close = pd.Series(100.0, index=index)
    bars = pd.DataFrame(
        {'open': close, 'high': close, 'low': close, 'close': close, 'volume': 1.0},
        index=index,
    )

    # Flat price, real funding: the entire outcome is carry.
    summary = summarise_targets(
        build_targets(bars, TargetSpec(24, 0.0005), funding=_funding(index, 0.0001))
    )

    assert summary.carry_share == pytest.approx(1.0)
    assert summary.mean_carry_bps < 0        # a long pays positive funding
