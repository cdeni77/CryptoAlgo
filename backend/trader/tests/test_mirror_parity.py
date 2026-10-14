"""The up/down mirror used for augmentation.

Every feature must declare how it transforms, and the declarations are checked
against the code that computes the features rather than trusted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.book_features import market_state_features
from core.book_flow import _imbalance
from core.config import Config
from core.features import (
    COMPLEMENT, EVEN, FEATURE_GROUPS, FEATURE_PARITY, ODD, ROW_PARITY,
    _geometry_features, mirror_rows,
)

ALL_FEATURES = [c for columns in FEATURE_GROUPS.values() for c in columns]


def test_every_feature_declares_its_parity():
    missing = [c for c in ALL_FEATURES if c not in FEATURE_PARITY]
    assert not missing, (
        f'{missing} have no FEATURE_PARITY entry. Decide whether each is even, '
        f'odd, a complement or a swap under an up/down mirror before adding it.')
    stale = sorted(set(FEATURE_PARITY) - set(ALL_FEATURES))
    assert not stale, f'parity declared for features that do not exist: {stale}'


@pytest.mark.parametrize('table', [FEATURE_PARITY, ROW_PARITY])
def test_every_rule_is_valid_and_every_swap_is_reciprocal(table):
    for column, rule in table.items():
        if isinstance(rule, tuple):
            kind, partner, sign = rule
            assert kind == 'swap'
            back = table[partner]
            assert isinstance(back, tuple) and back[1] == column and back[2] == sign
        else:
            assert rule in (EVEN, ODD, COMPLEMENT), (column, rule)


def _random_table(n=200, seed=0):
    rng = np.random.default_rng(seed)
    columns = sorted(set(ALL_FEATURES) | set(ROW_PARITY))
    table = pd.DataFrame(rng.normal(size=(n, len(columns))), columns=columns)
    for column, rule in {**FEATURE_PARITY, **ROW_PARITY}.items():
        if rule == COMPLEMENT:
            table[column] = rng.uniform(0, 1, n)
    table.iloc[::7, 3] = np.nan
    table['symbol'] = 'BTC-USD'
    return table


def test_mirroring_twice_is_the_identity():
    table = _random_table()
    twice = mirror_rows(mirror_rows(table))
    pd.testing.assert_frame_equal(twice, table, check_exact=False, atol=1e-12, rtol=0)


def test_the_label_and_the_null_flip():
    table = _random_table()
    out = mirror_rows(table)
    np.testing.assert_allclose(out['outcome'], 1 - table['outcome'])
    np.testing.assert_allclose(out['baseline_probability_logit'],
                               -table['baseline_probability_logit'])
    np.testing.assert_allclose(out['market_probability'], 1 - table['market_probability'])
    np.testing.assert_allclose(out['excursion_up_z'], -table['excursion_down_z'])
    np.testing.assert_allclose(out['ask_up'], table['ask_down'])
    assert (out['symbol'] == table['symbol']).all()
    assert table is not out


def _check(features, mirrored_features, columns):
    expected = mirror_rows(features)
    for column in columns:
        np.testing.assert_allclose(
            mirrored_features[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float), atol=1e-12, rtol=0,
            err_msg=f'{column} does not transform as FEATURE_PARITY says')


def test_geometry_transforms_as_declared():
    rng = np.random.default_rng(1)
    n = 500
    up = rng.uniform(0.0, 0.004, n)
    down = -rng.uniform(0.0, 0.004, n)
    raw = pd.DataFrame({
        'displacement': rng.uniform(down, up), 'excursion_up': up,
        'excursion_down': down, 'sigma_remaining': rng.uniform(0.001, 0.003, n),
        'sigma_per_min': rng.uniform(1e-4, 5e-4, n),
        'offset': rng.choice([3, 6, 9, 12], n),
    })
    _check(_geometry_features(raw, Config()),
           _geometry_features(mirror_rows(raw), Config()),
           FEATURE_GROUPS['geometry'])


def test_the_book_snapshot_transforms_as_declared():
    """Mirroring exchanges the YES bid stack with the NO bid (YES ask) stack."""
    rng = np.random.default_rng(2)
    n = 300
    bid = rng.integers(5, 60, n).astype(float)
    ask = bid + rng.integers(1, 5, n)
    sizes = {k: rng.uniform(1, 500, n) for k in
             ('bid_at_touch', 'ask_at_touch', 'bid_5c', 'ask_5c',
              'bid_1c', 'ask_1c', 'bid_vol', 'ask_vol')}
    baseline = rng.uniform(0.05, 0.95, n)
    snap = pd.DataFrame({'best_bid': bid, 'best_ask': ask,
                         'baseline_probability': baseline, **sizes})
    mirrored = mirror_rows(snap.drop(columns=['best_bid', 'best_ask']))
    mirrored['best_bid'] = 100.0 - ask
    mirrored['best_ask'] = 100.0 - bid
    columns = FEATURE_GROUPS['market_state'] + FEATURE_GROUPS['market_price']
    _check(market_state_features(snap), market_state_features(mirrored), columns)


def test_book_flow_imbalance_is_odd():
    bid, ask = np.array([3.0, 10.0, 0.0]), np.array([7.0, 2.0, 4.0])
    np.testing.assert_allclose(_imbalance(ask, bid), -_imbalance(bid, ask))
    assert FEATURE_PARITY['imbalance_change_3'] == ODD
    assert FEATURE_PARITY['imbalance_persistence'] == ODD
