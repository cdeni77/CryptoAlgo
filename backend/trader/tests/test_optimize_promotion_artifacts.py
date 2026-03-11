from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import optuna
import pandas as pd

import scripts.optimize as optimize


@dataclass
class _FakeTrial:
    number: int = 1
    value: float = 1.1
    params: dict[str, Any] = field(default_factory=lambda: {
        "signal_threshold": 0.72,
        "strategy_family": "breakout",
        "trade_freq_bucket": "aggressive",
    })
    user_attrs: dict[str, Any] = field(default_factory=lambda: {
        "mean_sharpe": 0.35,
        "min_sharpe": 0.1,
        "std_sharpe": 0.02,
        "n_trades": 20,
        "win_rate": 0.6,
        "max_drawdown": 0.1,
        "psr_meta": {"valid": True, "psr": 0.9},
        "dsr_cv_meta": {"valid": True, "dsr": 0.4},
    })
    state: optuna.trial.TrialState = optuna.trial.TrialState.COMPLETE


class _FakeStudy:
    def __init__(self, trial: _FakeTrial):
        self.trials = [trial]
        self.best_trial = trial
        self.best_value = float(trial.value)
        self.study_name = "fake-study"

    def set_user_attr(self, *_args, **_kwargs):
        return None

    def optimize(self, *_args, **_kwargs):
        return None


def _run_optimize_coin_with_holdout(monkeypatch, holdout_metrics: dict[str, Any]):
    idx = pd.date_range("2024-01-01", periods=10, freq="h", tz="UTC")
    data = {
        "BIP-20DEC30-CDE": {
            "features": pd.DataFrame({"f": range(10)}, index=idx),
            "ohlcv": pd.DataFrame({"close": [100 + i for i in range(10)]}, index=idx),
        }
    }
    trial = _FakeTrial()
    study = _FakeStudy(trial)

    monkeypatch.setattr(optimize, "split_data_temporal", lambda *_a, **_k: (data, data))
    monkeypatch.setattr(optimize, "resolve_target_symbol", lambda *_a, **_k: "BIP-20DEC30-CDE")
    cv_idx = pd.DatetimeIndex(idx)
    cv_fold = optimize.CVFold(
        train_idx=cv_idx[:5],
        test_idx=cv_idx[5:],
        train_end=cv_idx[4],
        test_start=cv_idx[5],
        test_end=cv_idx[-1],
        purge_bars=0,
        embargo_bars=0,
    )
    monkeypatch.setattr(optimize, "create_cv_splits", lambda *_a, **_k: [cv_fold, cv_fold, cv_fold])
    monkeypatch.setattr(optimize, "_build_cost_config", lambda *_a, **_k: (optimize.Config(), {"version": "t", "source_path": None}))
    monkeypatch.setattr(optimize.optuna.storages, "RDBStorage", lambda *a, **k: object())
    monkeypatch.setattr(optimize.optuna, "create_study", lambda *a, **k: study)
    monkeypatch.setattr(optimize, "_select_best_trial", lambda *_a, **_k: trial)
    monkeypatch.setattr(optimize, "compute_deflated_sharpe_metric", lambda *_a, **_k: {"valid": True, "dsr": 0.1, "p_value": 0.4})
    monkeypatch.setattr(optimize, "append_trial_ledger_entry", lambda *_a, **_k: None)
    monkeypatch.setattr(optimize, "aggregate_cumulative_trial_counts", lambda *_a, **_k: {"coin_totals": {}, "global_total": 0})
    monkeypatch.setattr(optimize, "_candidate_trials_for_holdout", lambda *_a, **_k: [trial])
    def _eval_holdout(*_a, **_k):
        return {
            "full_sharpe": holdout_metrics.get("holdout_sharpe", 0.0),
            "full_pf": 1.1,
            "full_dd": 0.1,
            "gate_counters": {},
            "main_blockers": [],
            **dict(holdout_metrics),
        }

    monkeypatch.setattr(optimize, "evaluate_holdout", _eval_holdout)
    monkeypatch.setattr(optimize, "_compute_holdout_significance", lambda metrics, **_k: metrics)
    monkeypatch.setattr(optimize, "compose_study_significance_diagnostic", lambda **_k: {})
    monkeypatch.setattr(optimize, "assess_result_quality", lambda *_a, **_k: {"rating": "ok"})
    monkeypatch.setattr(optimize, "_persist_result_json", lambda *_a, **_k: None)

    persisted_payloads: list[dict[str, Any]] = []

    def _capture_payload(_coin: str, payload: dict[str, Any]):
        persisted_payloads.append(payload)
        return "artifact.json"

    monkeypatch.setattr(optimize, "_persist_paper_candidate_json", _capture_payload)

    result = optimize.optimize_coin(all_data={}, coin_prefix="BIP", coin_name="BTC", n_trials=1)
    return result, persisted_payloads


def test_failed_holdout_blocks_deployment_and_emits_no_promotion_artifact(monkeypatch):
    result, artifacts = _run_optimize_coin_with_holdout(
        monkeypatch,
        holdout_metrics={"holdout_sharpe": -0.2, "holdout_return": -0.1, "holdout_trades": 2},
    )

    assert result is not None
    assert result["deployment_blocked"] is True
    assert artifacts == []


def test_passing_holdout_emits_promotion_artifact_with_required_fields(monkeypatch):
    result, artifacts = _run_optimize_coin_with_holdout(
        monkeypatch,
        holdout_metrics={"holdout_sharpe": 0.3, "holdout_return": 0.04, "holdout_trades": 25},
    )

    assert result is not None
    assert result["deployment_blocked"] is False
    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["holdout_passed"] is True
    assert artifact["holdout_gate_result"] == "pass"
    assert artifact["evaluated_params"] == result["params"]
    for key in ("coin", "holdout_metrics", "run_id", "study_name", "timestamp", "gate_profile"):
        assert key in artifact

from core.coin_profiles import COIN_PROFILES
from core.paper_profile_overrides import load_paper_profile_overrides


def test_promotion_loader_overrides_static_coin_profile_defaults_deterministically(tmp_path):
    base = COIN_PROFILES["BTC"]
    artifact = tmp_path / "btc.json"
    artifact.write_text(
        '{"coin":"BTC","holdout_passed":true,"evaluated_params":{"signal_threshold":0.91,"strategy_family":"mean_reversion","trade_freq_bucket":"conservative"}}',
        encoding="utf-8",
    )

    overrides_1 = load_paper_profile_overrides(str(tmp_path))
    overrides_2 = load_paper_profile_overrides(str(tmp_path))

    assert overrides_1["BTC"].signal_threshold == 0.91
    assert overrides_1["BTC"].strategy_family == "mean_reversion"
    assert overrides_1["BTC"].trade_freq_bucket == "conservative"
    assert overrides_1["BTC"].signal_threshold != base.signal_threshold

    # deterministic loader behavior across repeated reads
    assert overrides_1["BTC"] == overrides_2["BTC"]
