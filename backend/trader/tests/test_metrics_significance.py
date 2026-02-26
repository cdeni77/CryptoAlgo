from core.metrics_significance import (
    compute_deflated_sharpe,
    compute_psr,
    compute_psr_from_samples,
    evaluate_significance_gates,
)
from scripts import optimize


def test_compute_psr_uses_fallback_moments_when_missing() -> None:
    result = compute_psr(sharpe_estimate=0.8, observations=40, benchmark_sharpe=0.0)
    assert result["valid"] is True
    assert result["fallback_moments_used"] is True
    assert 0.0 <= result["psr"] <= 1.0


def test_compute_psr_rejects_small_observation_count() -> None:
    result = compute_psr(sharpe_estimate=1.0, observations=1)
    assert result["valid"] is False
    assert result["reason"] == "insufficient_observations"


def test_compute_dsr_metadata_and_small_sample_behavior() -> None:
    invalid = compute_deflated_sharpe(observed_sharpe=1.0, observations=1, effective_test_count=10)
    assert invalid["valid"] is False
    assert invalid["reason"] == "insufficient_observations"

    valid = compute_deflated_sharpe(observed_sharpe=1.0, observations=60, effective_test_count=25)
    assert valid["valid"] is True
    assert valid["effective_test_count"] == 25
    assert "expected_max_sr" in valid


def test_compute_dsr_decreases_with_more_tests() -> None:
    few_tests = compute_deflated_sharpe(observed_sharpe=1.0, observations=80, effective_test_count=5)
    many_tests = compute_deflated_sharpe(observed_sharpe=1.0, observations=80, effective_test_count=500)
    assert few_tests["dsr"] > many_tests["dsr"]


def test_gate_helper_and_optimize_holdout_gate_use_centralized_outputs(monkeypatch) -> None:
    holdout_metrics = {
        "holdout_trades": 30,
        "holdout_sharpe": 0.4,
        "holdout_return": 0.02,
        "psr_holdout": {"valid": True, "psr": 0.99},
        "dsr_holdout": {"valid": True, "dsr": 0.8},
    }

    direct = evaluate_significance_gates(
        psr_holdout=holdout_metrics["psr_holdout"],
        dsr=holdout_metrics["dsr_holdout"],
        min_psr_holdout=0.8,
        min_dsr=0.5,
    )
    assert direct["all_passed"] is True

    called = {"count": 0}

    def fake_eval(**kwargs):
        called["count"] += 1
        return {
            "psr_cv": {"passed": True},
            "psr_holdout": {"passed": False},
            "dsr": {"passed": True},
            "all_passed": False,
        }

    monkeypatch.setattr(optimize, "evaluate_significance_gates", fake_eval)
    passed = optimize._passes_holdout_gate(
        holdout_metrics,
        min_trades=10,
        min_sharpe=0.0,
        min_return=0.0,
        min_psr_holdout=0.8,
        min_dsr=0.5,
    )
    assert called["count"] == 1
    assert passed is False


def test_compute_psr_from_samples_adds_sample_metadata() -> None:
    result = compute_psr_from_samples([0.2, 0.3, 0.4], benchmark_sharpe=0.0, effective_observations=15)
    assert result["sample_count"] == 3
    assert result["observations"] == 15
