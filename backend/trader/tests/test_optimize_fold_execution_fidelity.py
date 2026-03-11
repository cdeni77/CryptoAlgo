from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd

import scripts.optimize as optimize
from core.coin_profiles import COIN_PROFILES


class _IdentityScaler:
    def transform(self, x):
        return x


class _IdentityCalibrator:
    def predict(self, x):
        return np.asarray(x, dtype=float)


class _FakeModel:
    def __init__(self, offset: float):
        self.offset = offset

    def predict_proba(self, x):
        v = 0.9 if self.offset > 0 else 0.1
        return np.array([[1.0 - v, v]])


class _FakeStrategyFamily:
    def __init__(self, family_name: str):
        self.family_name = family_name
        self.name = family_name

    def evaluate(self, *_args, **_kwargs):
        direction = 1 if self.family_name != "mean_reversion" else -1
        return SimpleNamespace(direction=direction, rank_modifier=0.0, gate_contributions={"momentum_dir_agreement": True})


@dataclass(frozen=True)
class _Member:
    name: str
    train_window_days: int = 90
    feature_fraction: float = 1.0
    max_depth_range: tuple[int, int] = (2, 3)
    num_leaves_range: tuple[int, int] = (15, 20)
    n_estimators_range: tuple[int, int] = (70, 80)
    min_child_samples_range: tuple[int, int] = (6, 8)


class _FakeMLSystem:
    def __init__(self, config):
        self.config = config

    def get_feature_columns(self, columns, candidates):
        col_set = set(columns)
        selected = [c for c in candidates if c in col_set]
        return selected if selected else list(columns)[:6]

    def build_member_features(self, _symbol, cols, _member):
        return list(cols)

    def create_labels(self, _ohlcv, feat, profile=None):
        y = np.where(np.arange(len(feat)) % 2 == 0, 1, 0)
        return pd.Series(y, index=feat.index)

    def prepare_binary_training_set(self, x, y):
        return x, y

    def train(self, *_args, member_spec=None, **_kwargs):
        offset = 0.1 if member_spec.name == "m1" else -0.1
        meta = SimpleNamespace(model=_FakeModel(0.05), scaler=_IdentityScaler(), calibrator=None)
        return (_FakeModel(offset), _IdentityScaler(), _IdentityCalibrator(), 0.6, meta, {}, {})


def _make_market_data():
    idx = pd.date_range("2024-01-01", periods=220, freq="h", tz="UTC")
    features = pd.DataFrame(
        {
            "f0": np.sin(np.arange(len(idx)) / 6.0),
            "f1": np.cos(np.arange(len(idx)) / 8.0),
            "fee_hurdle_pct": 0.001,
            "breakout_vs_cost": 0.2,
            "expected_cost_to_vol_ratio": 0.2,
            "vol_24h": 0.02,
            "funding_rate_zscore": 0.0,
        },
        index=idx,
    )
    close = 100 + np.arange(len(idx)) * 0.1
    ohlcv = pd.DataFrame({"close": close, "sma_200": 90.0}, index=idx)
    return features, ohlcv


def test_fold_evaluator_uses_profile_categoricals_and_nontrivial_ensemble(monkeypatch):
    features, ohlcv = _make_market_data()
    idx = features.index
    fold = optimize.CVFold(
        train_idx=idx[:150],
        test_idx=idx[150:],
        train_end=idx[149],
        test_start=idx[150],
        test_end=idx[-1],
        purge_bars=0,
        embargo_bars=0,
    )

    monkeypatch.setattr(optimize, "MLSystem", _FakeMLSystem)
    monkeypatch.setattr(optimize, "fit_transform_fold", lambda x_tr, x_val, _y: (x_tr, x_val, None))
    monkeypatch.setattr(optimize, "build_ensemble_member_specs", lambda: [_Member("m1"), _Member("m2")])
    monkeypatch.setattr(optimize, "get_strategy_family", lambda family_name: _FakeStrategyFamily(family_name))
    monkeypatch.setattr(optimize, "calculate_n_contracts", lambda *_a, **_k: 2)

    profile = COIN_PROFILES["BTC"]
    profile_breakout = profile.__class__(**{**profile.__dict__, "strategy_family": "breakout", "trade_freq_bucket": "aggressive", "cooldown_hours": 1.0})
    profile_meanrev = profile.__class__(**{**profile.__dict__, "strategy_family": "mean_reversion", "trade_freq_bucket": "conservative", "cooldown_hours": 1.0})

    cfg = optimize.Config(min_train_samples=50, val_fraction=0.2, trend_filter_mode="hard", funding_filter_mode="off")

    breakout_result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile_breakout, cfg, "BIP-20DEC30-CDE", pruned_only=False)
    meanrev_result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile_meanrev, cfg, "BIP-20DEC30-CDE", pruned_only=False)

    assert breakout_result is not None
    assert meanrev_result is not None
    assert cfg.strategy_family == "mean_reversion"
    assert cfg.trade_freq_bucket == "conservative"

    breakout_metrics = breakout_result["fold_metrics"]
    assert breakout_metrics["ensemble_agreement_mean"] < 1.0
    assert "ensemble_std_mean" in breakout_metrics


class _FakeMLSystemTernary(_FakeMLSystem):
    def create_labels(self, _ohlcv, feat, profile=None):
        pattern = np.array([-1, 0, 1], dtype=int)
        y = np.resize(pattern, len(feat))
        return pd.Series(y, index=feat.index)

    def prepare_binary_training_set(self, x, y):
        y_bin = y.loc[y != 0].map({-1: 0, 1: 1})
        x_bin = x.loc[y_bin.index]
        return x_bin, y_bin.astype(int)


def test_fold_evaluator_normalizes_ternary_labels_before_auc(monkeypatch):
    features, ohlcv = _make_market_data()
    idx = features.index
    fold = optimize.CVFold(
        train_idx=idx[:150],
        test_idx=idx[150:],
        train_end=idx[149],
        test_start=idx[150],
        test_end=idx[-1],
        purge_bars=0,
        embargo_bars=0,
    )

    monkeypatch.setattr(optimize, "MLSystem", _FakeMLSystemTernary)
    monkeypatch.setattr(optimize, "fit_transform_fold", lambda x_tr, x_val, _y: (x_tr, x_val, None))
    monkeypatch.setattr(optimize, "build_ensemble_member_specs", lambda: [_Member("m1"), _Member("m2")])
    monkeypatch.setattr(optimize, "get_strategy_family", lambda family_name: _FakeStrategyFamily(family_name))
    monkeypatch.setattr(optimize, "calculate_n_contracts", lambda *_a, **_k: 2)

    captured = {}

    original_score_model_quality = optimize._score_model_quality

    def _capturing_score_model_quality(probs, y_test):
        captured["labels"] = sorted(set(int(v) for v in y_test.tolist()))
        return original_score_model_quality(probs, y_test)

    monkeypatch.setattr(optimize, "_score_model_quality", _capturing_score_model_quality)

    profile = COIN_PROFILES["BTC"]
    cfg = optimize.Config(min_train_samples=50, val_fraction=0.2, trend_filter_mode="hard", funding_filter_mode="off")

    result = optimize.evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, cfg, "BIP-20DEC30-CDE", pruned_only=False)

    assert result is not None
    assert captured["labels"] == [0, 1]
