from __future__ import annotations

import optuna
import pandas as pd

import scripts.optimize as optimize
from core.cv_splitters import CVFold


class _TrialStub:
    def __init__(self, should_prune_after_report: bool = False):
        self._should_prune = should_prune_after_report
        self.reported: list[tuple[float, int]] = []
        self.user_attrs: dict[str, object] = {}

    def suggest_categorical(self, name, choices):
        return choices[0]

    def suggest_float(self, _name, low, high, step=None):
        if step:
            return low
        return (low + high) / 2

    def suggest_int(self, _name, low, high, step=1):
        return low

    def report(self, value, step):
        self.reported.append((float(value), int(step)))

    def should_prune(self):
        return self._should_prune

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value



def _dummy_fold() -> CVFold:
    idx = pd.date_range("2024-01-01", periods=5, freq="h", tz="UTC")
    return CVFold(
        train_idx=idx[:3],
        test_idx=idx[3:],
        train_end=idx[2],
        test_start=idx[3],
        test_end=idx[4],
        purge_bars=0,
        embargo_bars=0,
    )


def test_objective_prunes_trial_under_poor_early_fold(monkeypatch):
    trial = _TrialStub(should_prune_after_report=True)
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    optim_data = {"SYM": {"features": pd.DataFrame(index=idx), "ohlcv": pd.DataFrame(index=idx)}}

    monkeypatch.setattr(
        optimize,
        "evaluate_fold_with_execution_gates",
        lambda *_a, **_k: {
            "model_quality": {"score": -0.8, "auc": 0.49},
            "label_quality": {"score": -0.7},
            "fold_metrics": {"sharpe": -1.2, "return": -0.08, "expectancy": -0.02, "trades": 1},
            "gate_counters": {},
        },
    )

    try:
        optimize.objective(
            trial,
            optim_data=optim_data,
            coin_prefix="BIP",
            coin_name="BTC",
            cv_splits=[_dummy_fold(), _dummy_fold()],
            target_sym="SYM",
        )
    except optuna.TrialPruned:
        pass
    else:
        raise AssertionError("Expected objective to prune the trial")

    assert trial.user_attrs["reject_stage"] == "cv_fold"
    assert trial.user_attrs["prune_source"] == "optuna_pruner"
    assert trial.user_attrs["prune_fold_idx"] == 0
    assert "prune_intermediate_value" in trial.user_attrs


def test_objective_keeps_significance_invalid_when_observations_are_missing(monkeypatch):
    trial = _TrialStub(should_prune_after_report=False)
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    optim_data = {"SYM": {"features": pd.DataFrame(index=idx), "ohlcv": pd.DataFrame(index=idx)}}

    monkeypatch.setattr(
        optimize,
        "evaluate_fold_with_execution_gates",
        lambda *_a, **_k: {
            "model_quality": {"score": 0.1, "auc": 0.55},
            "label_quality": {"score": 0.1},
            "fold_metrics": {"sharpe": 0.0, "return": 0.0, "expectancy": 0.0, "trades": 0},
            "gate_counters": {},
        },
    )
    monkeypatch.setattr(optimize, "compute_psr_from_samples", lambda *_a, **_k: {"valid": False, "psr": None, "reason": None})
    monkeypatch.setattr(optimize, "compute_deflated_sharpe_metric", lambda *_a, **_k: {"valid": False, "dsr": None, "reason": None})

    _ = optimize.objective(
        trial,
        optim_data=optim_data,
        coin_prefix="BIP",
        coin_name="BTC",
        cv_splits=[_dummy_fold()],
        target_sym="SYM",
    )

    assert trial.user_attrs["psr_meta"]["valid"] is False
    assert trial.user_attrs["psr_meta"]["reason"] == "insufficient_observations"
    assert trial.user_attrs["dsr_cv_meta"]["valid"] is False
    assert trial.user_attrs["dsr_cv_meta"]["reason"] == "insufficient_observations"


def _objective_value_for_fold_metrics(monkeypatch, fold_metrics):
    trial = _TrialStub(should_prune_after_report=False)
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    optim_data = {"SYM": {"features": pd.DataFrame(index=idx), "ohlcv": pd.DataFrame(index=idx)}}

    monkeypatch.setattr(
        optimize,
        "evaluate_fold_with_execution_gates",
        lambda *_a, **_k: {
            "model_quality": {"score": 0.2, "auc": 0.58},
            "label_quality": {"score": 0.1},
            "fold_metrics": dict(fold_metrics),
            "gate_counters": {},
        },
    )

    value = optimize.objective(
        trial,
        optim_data=optim_data,
        coin_prefix="BIP",
        coin_name="BTC",
        cv_splits=[_dummy_fold(), _dummy_fold()],
        target_sym="SYM",
    )
    return float(value), trial.user_attrs


def test_objective_bounds_absurd_low_trade_sharpe(monkeypatch):
    value, attrs = _objective_value_for_fold_metrics(
        monkeypatch,
        {"raw_sharpe": 171_691_641.0, "sharpe": 0.0, "return": 0.01, "expectancy": 0.01, "trades": 1},
    )

    assert abs(value) < 1.0
    assert attrs["n_trades"] == 2
    assert attrs["realized_sharpe_term"] <= 1.0


def test_tiny_trade_count_trial_does_not_dominate_stable_trial(monkeypatch):
    tiny_value, _ = _objective_value_for_fold_metrics(
        monkeypatch,
        {"raw_sharpe": 171_691_641.0, "sharpe": 0.0, "return": 0.01, "expectancy": 0.01, "trades": 1},
    )
    stable_value, _ = _objective_value_for_fold_metrics(
        monkeypatch,
        {"sharpe": 1.2, "return": 0.02, "expectancy": 0.015, "trades": 8},
    )

    assert stable_value > tiny_value


def test_normal_trials_remain_rankable_after_sharpe_bounding(monkeypatch):
    weak_value, _ = _objective_value_for_fold_metrics(
        monkeypatch,
        {"sharpe": 0.2, "return": 0.01, "expectancy": 0.005, "trades": 8},
    )
    strong_value, _ = _objective_value_for_fold_metrics(
        monkeypatch,
        {"sharpe": 1.0, "return": 0.02, "expectancy": 0.012, "trades": 8},
    )

    assert strong_value > weak_value


def _fold_with_span(start: str, days: int) -> CVFold:
    train_idx = pd.date_range(start, periods=72, freq="h", tz="UTC")
    test_start = train_idx[-1] + pd.Timedelta(hours=1)
    test_idx = pd.date_range(test_start, periods=max(24, int(days * 24)), freq="h", tz="UTC")
    return CVFold(
        train_idx=train_idx,
        test_idx=test_idx,
        train_end=train_idx[-1],
        test_start=test_idx[0],
        test_end=test_idx[-1],
        purge_bars=0,
        embargo_bars=0,
    )


def test_objective_trade_floor_scales_with_oos_years(monkeypatch):
    trial = _TrialStub(should_prune_after_report=False)
    idx = pd.date_range("2024-01-01", periods=12, freq="h", tz="UTC")
    optim_data = {"SYM": {"features": pd.DataFrame(index=idx), "ohlcv": pd.DataFrame(index=idx)}}

    monkeypatch.setattr(
        optimize,
        "evaluate_fold_with_execution_gates",
        lambda *_a, **_k: {
            "model_quality": {"score": 0.2, "auc": 0.58},
            "label_quality": {"score": 0.1},
            "fold_metrics": {"sharpe": 0.1, "return": 0.0, "expectancy": 0.0, "trades": 4},
            "gate_counters": {},
        },
    )

    folds = [_fold_with_span("2024-01-01", 365), _fold_with_span("2025-01-01", 365)]
    optimize.objective(
        trial,
        optim_data=optim_data,
        coin_prefix="BIP",
        coin_name="BTC",
        cv_splits=folds,
        target_sym="SYM",
    )

    assert trial.user_attrs["cv_oos_years"] >= 1.9
    assert trial.user_attrs["target_trades_floor"] > len(folds)


def test_objective_activity_regime_and_starvation_penalty(monkeypatch):
    tiny_value, tiny_attrs = _objective_value_for_fold_metrics(
        monkeypatch,
        {"raw_sharpe": 10.0, "sharpe": 0.8, "return": 0.02, "expectancy": 0.01, "trades": 1},
    )
    active_value, active_attrs = _objective_value_for_fold_metrics(
        monkeypatch,
        {"raw_sharpe": 1.2, "sharpe": 0.9, "return": 0.02, "expectancy": 0.01, "trades": 16},
    )

    assert tiny_attrs["activity_regime"] in {"dead", "starved"}
    assert active_attrs["activity_regime"] in {"thin", "active"}
    assert tiny_attrs["trade_floor_penalty"] < -0.4
    assert active_value > tiny_value


def test_objective_persists_gate_blocker_diagnostics(monkeypatch):
    trial = _TrialStub(should_prune_after_report=False)
    idx = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    optim_data = {"SYM": {"features": pd.DataFrame(index=idx), "ohlcv": pd.DataFrame(index=idx)}}

    monkeypatch.setattr(
        optimize,
        "evaluate_fold_with_execution_gates",
        lambda *_a, **_k: {
            "model_quality": {"score": 0.1, "auc": 0.55},
            "label_quality": {"score": 0.1},
            "fold_metrics": {"sharpe": 0.0, "return": 0.0, "expectancy": 0.0, "trades": 0},
            "gate_counters": {
                "total_checks": 10,
                "gate_counts": {"primary_threshold": 4, "momentum_magnitude": 3},
                "gate_rates": {"primary_threshold": 0.4, "momentum_magnitude": 0.3},
            },
        },
    )

    optimize.objective(
        trial,
        optim_data=optim_data,
        coin_prefix="BIP",
        coin_name="BTC",
        cv_splits=[_dummy_fold()],
        target_sym="SYM",
    )

    assert "aggregated_gate_counters" in trial.user_attrs
    assert "main_blockers" in trial.user_attrs
    assert "starvation_signature" in trial.user_attrs
    assert "fold_blocker_summary" in trial.user_attrs
