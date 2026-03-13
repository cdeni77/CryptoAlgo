import pandas as pd
from types import SimpleNamespace

import numpy as np

import scripts.optimize as optimize
from core.coin_profiles import CoinProfile
from scripts.train_model import Config


class _IdentityScaler:
    def transform(self, x):
        return x


class _ProbModel:
    def predict_proba(self, x):
        rows = x.shape[0] if hasattr(x, 'shape') else 1
        return np.tile(np.array([[0.2, 0.8]], dtype=float), (rows, 1))


class _FakeMLSystem:
    def __init__(self, _config):
        pass

    def get_feature_columns(self, _available_cols, _candidates):
        return ['f1', 'f2', 'f3', 'f4']

    def build_member_features(self, _symbol, base_cols, _member_spec):
        return list(base_cols)

    def create_labels(self, _ohlcv, feat, profile=None):
        vals = np.array([(i % 2) for i in range(len(feat))], dtype=int)
        return pd.Series(vals, index=feat.index)

    def prepare_binary_training_set(self, x, y):
        return x, y

    def train(self, *_args, **_kwargs):
        model = _ProbModel()
        scaler = _IdentityScaler()
        calibrator = None
        auc = 0.7
        meta = SimpleNamespace(model=None, scaler=None, calibrator=None)
        return (model, scaler, calibrator, auc, meta)


def _patch_fold_eval_environment(monkeypatch, *, horizon: int, cooldown_hours: float = 0.0, max_positions: int = 1):
    monkeypatch.setattr(optimize, 'MLSystem', _FakeMLSystem)
    monkeypatch.setattr(optimize, 'build_ensemble_member_specs', lambda: [SimpleNamespace(train_window_days=30), SimpleNamespace(train_window_days=60)])
    monkeypatch.setattr(optimize, '_calibrator_predict', lambda _c, arr: np.asarray(arr, dtype=float))
    monkeypatch.setattr(optimize, 'primary_recall_threshold', lambda *_a, **_k: 0.5)
    monkeypatch.setattr(optimize, 'calibrator_predict', lambda _c, arr: np.asarray(arr, dtype=float))
    monkeypatch.setattr(optimize, 'resolve_label_horizon', lambda *_a, **_k: horizon)

    def _resolve_param(name, _p, _cfg, default, mode='direct'):
        if name == 'signal_threshold':
            return 0.5
        if name == 'meta_probability_threshold':
            return 0.4
        if name == 'min_directional_agreement':
            return 0.5
        if name == 'max_ensemble_std':
            return 1.0
        if name == 'min_momentum_magnitude':
            return 0.0
        return default

    monkeypatch.setattr(optimize, 'resolve_param', _resolve_param)
    monkeypatch.setattr(optimize, 'resolve_categorical_param', lambda _n, _p, _cfg, default: default)
    monkeypatch.setattr(
        optimize,
        'get_strategy_family',
        lambda _f: SimpleNamespace(evaluate=lambda *_a, **_k: SimpleNamespace(direction=1, gate_contributions={'momentum_dir_agreement': True})),
    )
    monkeypatch.setattr(optimize, '_build_strategy_context', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(optimize, '_resolve_filter_policy', lambda *_a, **_k: SimpleNamespace(reject=False, size_multiplier=1.0))
    monkeypatch.setattr(optimize, 'calculate_n_contracts', lambda equity, *_a, **_k: max(1, int(equity // 10_000)))

    sizing_equities = []

    def _fake_pnl(entry_price, exit_price, direction, accum_funding, n_contracts, symbol, config):
        pnl_dollars = float(1000.0 * n_contracts)
        notional = float(10_000.0 * n_contracts)
        return (0.1, 0.1, 0.0, 0.0, 0.0, 0.0, pnl_dollars, notional)

    def _record_sizing(equity, *args, **kwargs):
        sizing_equities.append(float(equity))
        return max(1, int(equity // 10_000))

    monkeypatch.setattr(optimize, 'calculate_pnl_exact', _fake_pnl)
    monkeypatch.setattr(optimize, 'calculate_n_contracts', _record_sizing)

    cfg = Config(min_train_samples=10, val_fraction=0.2, max_positions=max_positions)
    profile = CoinProfile(name='BTC', prefixes=['BIP'], cooldown_hours=cooldown_hours, min_vol_24h=0.0, max_vol_24h=1.0)
    return cfg, profile, sizing_equities


def _build_inputs(test_len: int = 24):
    idx = pd.date_range('2024-01-01', periods=220, freq='h', tz='UTC')
    close = pd.Series(np.linspace(100.0, 140.0, len(idx)), index=idx)
    ohlcv = pd.DataFrame({'close': close, 'sma_200': close * 0.9}, index=idx)
    features = pd.DataFrame({
        'f1': 1.0,
        'f2': 2.0,
        'f3': 3.0,
        'f4': 4.0,
        'funding_rate_bps': 0.0,
        'funding_rate_zscore': 0.0,
    }, index=idx)

    test_start_loc = 180
    test_idx = idx[test_start_loc:test_start_loc + test_len]
    train_idx = idx[:test_start_loc]
    fold = optimize.CVFold(
        train_idx=train_idx,
        test_idx=test_idx,
        train_end=train_idx.max(),
        test_start=test_idx.min(),
        test_end=test_idx.max(),
        purge_bars=0,
        embargo_bars=0,
    )
    return features, ohlcv, fold


def test_no_overlapping_trades_when_single_position(monkeypatch):
    cfg, profile, _ = _patch_fold_eval_environment(monkeypatch, horizon=4, cooldown_hours=0.0, max_positions=1)
    features, ohlcv, fold = _build_inputs(test_len=24)

    result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, cfg, 'BIP-20DEC30-CDE', pruned_only=False)

    assert result is not None
    # 24 bars, 4-bar hold, single-position sequential execution with fold-local exits => 5 trades.
    assert result['fold_metrics']['trades'] == 5


def test_fold_boundary_exit_is_enforced(monkeypatch):
    cfg, profile, _ = _patch_fold_eval_environment(monkeypatch, horizon=7, cooldown_hours=0.0, max_positions=1)
    features, ohlcv, fold = _build_inputs(test_len=24)

    result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, cfg, 'BIP-20DEC30-CDE', pruned_only=False)

    assert result is not None
    # With 24 bars and 7-bar hold: t0/t7/t14 fit, t21 is skipped as exit crosses fold end.
    assert result['fold_metrics']['trades'] == 3


def test_cooldown_starts_after_exit(monkeypatch):
    cfg, profile, _ = _patch_fold_eval_environment(monkeypatch, horizon=4, cooldown_hours=2.0, max_positions=1)
    features, ohlcv, fold = _build_inputs(test_len=24)

    result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, cfg, 'BIP-20DEC30-CDE', pruned_only=False)

    assert result is not None
    # Each trade consumes 4 bars hold + 2 bars cooldown => one trade every 6 bars.
    assert result['fold_metrics']['trades'] == 4


def test_equity_updates_only_on_close_and_no_early_compounding(monkeypatch):
    cfg, profile, sizing_equities = _patch_fold_eval_environment(monkeypatch, horizon=6, cooldown_hours=0.0, max_positions=1)
    features, ohlcv, fold = _build_inputs(test_len=24)

    result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, cfg, 'BIP-20DEC30-CDE', pruned_only=False)

    assert result is not None
    # 24 bars / 6-bar hold => 3 sequential trades with no overlap.
    assert result['fold_metrics']['trades'] == 3
    # The first valid entries use stepped equity after each close (no intra-trade reuse of unrealized PnL).
    assert sizing_equities[:3] == [100000.0, 110000.0, 121000.0]
