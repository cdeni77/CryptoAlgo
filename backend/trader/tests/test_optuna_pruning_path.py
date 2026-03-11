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
