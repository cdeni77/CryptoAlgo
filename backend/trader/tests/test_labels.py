"""Triple-barrier labelling invariants.

The properties worth pinning are the ones that decide whether a label means what
it says: barriers set from information available at the time, ties resolved
against the trade, costs that vary by instrument, and unresolvable rows staying
unlabelled instead of defaulting to a class.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.labels import (
    LOSS,
    WIN,
    BarrierSpec,
    _first_touch,
    barrier_prices,
    barrier_spec_for,
    label_panel,
    momentum_direction,
    round_trip_cost,
    summarise_labels,
    triple_barrier_labels,
)
from core.profiles import COIN_PROFILES

CDE_CONFIG = 'configs/exchange/coinbase_us_perps_cde_v202602.json'


@pytest.fixture(scope='module')
def config(repo_root) -> Config:
    return Config().with_cost_assumptions(repo_root / CDE_CONFIG)


def _bars(n: int = 1_500, *, seed: int = 3, vol: float = 0.012) -> pd.DataFrame:
    index = pd.date_range('2026-01-01', periods=n, freq='1h', tz='UTC')
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0002, vol, n)))
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pd.DataFrame(
        {
            'open': open_,
            'high': np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n))),
            'low': np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n))),
            'close': close,
            'volume': rng.lognormal(8, 0.6, n),
        },
        index=index,
    )


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def test_round_trip_cost_varies_by_contract(config):
    """Per-contract commission means cost is set by notional per contract.

    ETH carries $0.75 on ~$300 of notional; DOGE carries $0.10 on ~$1,750. That
    ordering is the reason the cost cannot be one number for the universe.
    """
    eth = round_trip_cost('ETP', 3_000, config)
    btc = round_trip_cost('BIP', 60_000, config)
    doge = round_trip_cost('DOP', 0.35, config)

    assert eth > btc > doge
    assert eth / doge > 5


def test_round_trip_cost_is_size_invariant(config):
    """A fixed fee per contract is a fixed fraction of notional."""
    one = round_trip_cost('BIP', 60_000, config, contracts=1)
    many = round_trip_cost('BIP', 60_000, config, contracts=250)

    assert one == pytest.approx(many)


def test_cost_widens_take_profit_only():
    """A stop is a gross move; pushing it out would label losses as survivable."""
    free = BarrierSpec(72, 5.0, 3.0, round_trip_cost=0.0)
    paid = BarrierSpec(72, 5.0, 3.0, round_trip_cost=0.0050)

    for direction in (1, -1):
        tp_free, sl_free = barrier_prices(100.0, 0.01, direction, free)
        tp_paid, sl_paid = barrier_prices(100.0, 0.01, direction, paid)

        assert sl_free == sl_paid
        if direction == 1:
            assert tp_paid > tp_free
        else:
            assert tp_paid < tp_free


def test_higher_cost_lowers_the_win_rate():
    """Monotonic, and the mechanism by which a wrong fee corrupts labels."""
    bars = _bars()
    rates = [
        summarise_labels(
            triple_barrier_labels(bars, BarrierSpec(96, 5.0, 3.5, cost / 10_000))
        ).win_rate
        for cost in (0, 20, 100, 400)
    ]

    assert rates == sorted(rates, reverse=True)
    assert rates[0] > rates[-1]


# ---------------------------------------------------------------------------
# First touch
# ---------------------------------------------------------------------------


def test_take_profit_first_is_a_win():
    highs = np.array([106.0, 100.0])
    lows = np.array([100.0, 97.0])

    assert _first_touch(highs, lows, 1, 105.0, 98.0) == WIN


def test_stop_first_is_a_loss():
    highs = np.array([100.0, 106.0])
    lows = np.array([97.0, 100.0])

    assert _first_touch(highs, lows, 1, 105.0, 98.0) == LOSS


def test_same_bar_tie_resolves_against_the_trade():
    """Bar data cannot order two touches inside one bar.

    Assuming the profitable order is how a backtest quietly inflates its win
    rate, so a tie counts as a loss.
    """
    highs = np.array([106.0])
    lows = np.array([97.0])

    assert _first_touch(highs, lows, 1, 105.0, 98.0) == LOSS


def test_timeout_is_not_a_win():
    """Neither barrier touched means the trade did not achieve its target."""
    highs = np.array([101.0, 102.0])
    lows = np.array([99.0, 98.5])

    assert _first_touch(highs, lows, 1, 105.0, 98.0) == LOSS


def test_short_side_barriers_are_mirrored():
    # price falls to 94: take-profit for a short at 95
    highs = np.array([100.0])
    lows = np.array([94.0])

    assert _first_touch(highs, lows, -1, 95.0, 103.0) == WIN
    assert _first_touch(np.array([104.0]), np.array([100.0]), -1, 95.0, 103.0) == LOSS


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------


def test_labels_are_binary_or_absent():
    labels = triple_barrier_labels(_bars(), BarrierSpec(96, 5.0, 3.5, 0.0006))

    assert set(labels.dropna().unique()) <= {WIN, LOSS}


def test_trailing_horizon_is_unlabelled():
    """A bar without a full forward window has no outcome to record."""
    spec = BarrierSpec(96, 5.0, 3.5, 0.0006)
    labels = triple_barrier_labels(_bars(), spec)

    assert labels.iloc[-spec.horizon_bars:].isna().all()
    assert labels.iloc[:-spec.horizon_bars].notna().any()


def test_neutral_direction_leaves_rows_unlabelled():
    """No directional consensus is not an outcome, so it must not become a class."""
    bars = _bars()
    neutral = pd.Series(0.0, index=bars.index)

    labels = triple_barrier_labels(bars, BarrierSpec(96, 5.0, 3.5), direction=neutral)

    assert labels.isna().all()


def test_looser_direction_threshold_labels_more_bars():
    bars = _bars()
    spec = BarrierSpec(96, 5.0, 3.5, 0.0006)

    loose = summarise_labels(
        triple_barrier_labels(bars, spec, direction=momentum_direction(bars, score_threshold=1))
    )
    strict = summarise_labels(
        triple_barrier_labels(bars, spec, direction=momentum_direction(bars, score_threshold=2))
    )

    assert loose.labelled > strict.labelled


def test_labels_do_not_look_ahead():
    """Truncating the future must not change a label that was already resolvable.

    The barrier volatility is shifted one bar, so a label at t depends only on
    prices up to t plus the forward window it explicitly consumes.
    """
    bars = _bars(n=1_200)
    spec = BarrierSpec(48, 5.0, 3.5, 0.0006)

    full = triple_barrier_labels(bars, spec)
    truncated = triple_barrier_labels(bars.iloc[:900], spec)

    overlap = truncated.dropna().index
    pd.testing.assert_series_equal(full.loc[overlap], truncated.loc[overlap])


def test_spec_horizon_follows_the_hold_period(config):
    """A label must span at least as long as a position can stay open."""
    profile = COIN_PROFILES['SOL']
    spec = barrier_spec_for('SLP', profile=profile, config=config, reference_price=150.0)

    assert spec.horizon_bars == profile.max_hold_hours
    assert spec.tp_mult == profile.vol_mult_tp
    assert spec.round_trip_cost > 0


def test_spec_rejects_nonsense():
    with pytest.raises(ValueError):
        BarrierSpec(0, 5.0, 3.0)
    with pytest.raises(ValueError):
        BarrierSpec(96, 0.0, 3.0)


# ---------------------------------------------------------------------------
# Panel
# ---------------------------------------------------------------------------


def test_label_panel_matches_the_feature_panel_shape(config):
    """MultiIndexed by (event_time, symbol) so features and labels join on index."""
    bars = {'SLP': _bars(seed=1), 'DOP': _bars(seed=2), 'ETP': _bars(seed=3)}
    profiles = {'SLP': COIN_PROFILES['SOL'], 'DOP': COIN_PROFILES['DOGE'],
                'ETP': COIN_PROFILES['ETH']}

    panel = label_panel(bars, profiles=profiles, config=config)

    assert panel.index.names == ['event_time', 'symbol']
    assert set(panel.index.get_level_values('symbol')) == set(bars)
    assert set(panel.dropna().unique()) <= {WIN, LOSS}
