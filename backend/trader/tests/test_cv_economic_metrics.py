import pandas as pd
from types import SimpleNamespace

import numpy as np

import scripts.optimize as optimize
from core.coin_profiles import COIN_PROFILES
from core.execution_sim import compute_trade_execution_pnl
from scripts.train_model import Config, calculate_pnl_exact


def test_execution_helper_matches_backtest_pnl_primitive():
    cfg = Config(fee_pct_per_side=0.001, min_fee_per_contract=0.2, slippage_bps=2.0, apply_funding=True, apply_slippage=True)
    shared = compute_trade_execution_pnl(
        entry_price=100.0,
        exit_price=102.0,
        direction=1,
        accum_funding=-0.0005,
        n_contracts=5,
        symbol="BIP-20DEC30-CDE",
        fee_pct_per_side=cfg.fee_pct_per_side,
        min_fee_per_contract=cfg.min_fee_per_contract,
        slippage_bps=cfg.slippage_bps,
        apply_funding=cfg.apply_funding,
        apply_slippage=cfg.apply_slippage,
        apply_impact=cfg.apply_impact,
        impact_bps_per_contract=cfg.impact_bps_per_contract,
        impact_max_bps_per_side=cfg.impact_max_bps_per_side,
    )
    legacy = calculate_pnl_exact(100.0, 102.0, 1, -0.0005, 5, "BIP-20DEC30-CDE", cfg)

    assert shared.net_pnl_pct == legacy[0]
    assert shared.raw_pnl_pct == legacy[1]
    assert shared.fee_pnl_pct == legacy[2]
    assert shared.pnl_dollars == legacy[6]
    assert shared.notional == legacy[7]


def test_min_fee_and_slippage_reduce_net_return_for_small_notional():
    cfg = Config(fee_pct_per_side=0.0001, min_fee_per_contract=5.0, slippage_bps=10.0, apply_funding=False, apply_slippage=True)
    pnl = compute_trade_execution_pnl(
        entry_price=100.0,
        exit_price=100.1,
        direction=1,
        accum_funding=0.0,
        n_contracts=1,
        symbol="BIP-20DEC30-CDE",
        fee_pct_per_side=cfg.fee_pct_per_side,
        min_fee_per_contract=cfg.min_fee_per_contract,
        slippage_bps=cfg.slippage_bps,
        apply_funding=cfg.apply_funding,
        apply_slippage=cfg.apply_slippage,
        apply_impact=cfg.apply_impact,
        impact_bps_per_contract=cfg.impact_bps_per_contract,
        impact_max_bps_per_side=cfg.impact_max_bps_per_side,
    )
    assert pnl.min_fee_component_dollars > 0
    assert pnl.slippage_component_dollars > 0
    assert pnl.net_pnl_pct < pnl.raw_pnl_pct


def test_funding_sign_affects_net_pnl():
    common_kwargs = dict(
        entry_price=100.0,
        exit_price=100.0,
        direction=1,
        n_contracts=10,
        symbol="BIP-20DEC30-CDE",
        fee_pct_per_side=0.0,
        min_fee_per_contract=0.0,
        slippage_bps=0.0,
        apply_funding=True,
        apply_slippage=False,
        apply_impact=False,
        impact_bps_per_contract=0.0,
        impact_max_bps_per_side=0.0,
    )
    neg = compute_trade_execution_pnl(accum_funding=-0.001, **common_kwargs)
    pos = compute_trade_execution_pnl(accum_funding=0.001, **common_kwargs)
    assert neg.net_pnl_pct < 0
    assert pos.net_pnl_pct > 0


def test_objective_stores_economic_cv_user_attrs(monkeypatch):
    fold_metric = {
        'trades': 10,
        'raw_sharpe': 1.0,
        'sharpe': 0.8,
        'return': 0.12,
        'gross_return': 0.20,
        'expectancy': 0.03,
        'equity_start': 100000.0,
        'equity_end': 112000.0,
        'total_fees': 125.0,
        'total_funding': -20.0,
        'avg_notional': 25000.0,
    }

    monkeypatch.setattr(optimize, 'create_trial_profile', lambda *_a, **_k: COIN_PROFILES['BTC'])
    monkeypatch.setattr(
        optimize,
        'evaluate_fold_with_execution_gates',
        lambda *_a, **_k: {'model_quality': {'score': 0.7, 'auc': 0.6}, 'label_quality': {'score': 0.7}, 'fold_metrics': dict(fold_metric), 'gate_counters': {}},
    )

    class _Trial:
        def __init__(self):
            self.params = {'a': 1}
            self.number = 7
            self.user_attrs = {}
            self.value = None
            self.study = SimpleNamespace(trials=[])

        def report(self, *_a, **_k):
            return None

        def should_prune(self):
            return False

        def set_user_attr(self, key, value):
            self.user_attrs[key] = value

    trial = _Trial()
    fold = optimize.CVFold(train_idx=np.array([]), test_idx=np.array([]), train_end=pd.Timestamp('2024-01-01', tz='UTC'), test_start=pd.Timestamp('2024-01-02', tz='UTC'), test_end=pd.Timestamp('2024-01-10', tz='UTC'), purge_bars=0, embargo_bars=0)
    score = optimize.objective(
        trial,
        optim_data={'BIP-20DEC30-CDE': {'features': None, 'ohlcv': None}},
        coin_prefix='BIP',
        coin_name='BTC',
        cv_splits=[fold],
        target_sym='BIP-20DEC30-CDE',
        pruned_only=True,
        base_config=Config(),
    )

    assert np.isfinite(score)
    assert trial.user_attrs['cv_net_return'] == 0.12
    assert trial.user_attrs['cv_gross_return'] == 0.2
    assert trial.user_attrs['cv_total_fees'] == 125.0
    assert trial.user_attrs['cv_total_funding'] == -20.0
    assert trial.user_attrs['cv_avg_notional'] == 25000.0


def test_absurd_cv_guardrail_detects_explosive_values():
    absurd, reason = optimize._is_absurd_cv_metrics(realized_return=20.0, realized_expectancy=0.01, realized_trades=10)
    assert absurd is True
    assert reason == 'cv_return_implausibly_high'
