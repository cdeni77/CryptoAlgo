from types import SimpleNamespace

from scripts.optimize import (
    _candidate_holdout_summary,
    _compute_holdout_significance,
    _derive_starvation_signature,
    _derive_top_level_holdout,
)


def test_starvation_signature_prefers_momentum_when_momentum_gates_dominate() -> None:
    gate_summary = {
        "gate_rates": {
            "momentum_magnitude": 0.40,
            "momentum_dir_agreement": 0.30,
            "vol_regime_low": 0.10,
            "primary_threshold": 0.05,
            "ensemble_agreement": 0.05,
        }
    }

    sig = _derive_starvation_signature(gate_summary)
    assert sig["label"] == "momentum_starved"
    assert sig["shares"]["momentum_starved"] > sig["shares"]["vol_starved"]


def test_starvation_signature_handles_missing_gate_data() -> None:
    sig = _derive_starvation_signature({"gate_rates": {}})
    assert sig["label"] == "insufficient_signal"
    assert sig["shares"] == {}


def test_top_level_multi_slice_holdout_keeps_selected_metrics_consistent() -> None:
    holdout_slices = {
        "recent90": {
            "holdout_trades": 6,
            "holdout_sharpe": 0.150956,
            "holdout_return": 0.015174,
            "gate_counters": {"total_checks": 10, "gate_counts": {"momentum_magnitude": 4}},
        },
        "full180": {
            "holdout_trades": 6,
            "holdout_sharpe": 0.150956,
            "holdout_return": 0.015174,
            "gate_counters": {"total_checks": 11, "gate_counts": {"primary_threshold": 3}},
        },
        "prior90": {
            "holdout_trades": 0,
            "holdout_sharpe": 0.0,
            "holdout_return": -1.0,
            "gate_counters": {"total_checks": 8, "gate_counts": {"ensemble_agreement": 5}},
        },
    }

    top_level = _derive_top_level_holdout(holdout_slices, holdout_mode="multi_slice")

    assert top_level["selected_slice"] == "median_composite"
    assert top_level["holdout_trades"] == 6
    assert top_level["holdout_sharpe"] == 0.150956
    assert top_level["holdout_return"] == 0.015174
    assert "main_blockers" not in top_level
    assert top_level["starvation_signature"]["label"] != "insufficient_signal"


def test_candidate_summary_uses_consistent_final_holdout_bundle() -> None:
    holdout_metrics = {
        "holdout_trades": 6,
        "holdout_sharpe": 0.1509567,
        "holdout_return": 0.0151744,
        "gate_counters": {"total_checks": 12, "gate_counts": {"momentum_dir_agreement": 6}},
        "main_blockers": [{"reason_code": "momentum_dir_agreement", "count": 6, "rate": 0.5}],
    }

    summary = _candidate_holdout_summary(
        SimpleNamespace(number=7),
        holdout_metrics,
        0.42,
        min_trades=10,
        min_sharpe=0.0,
        min_return=0.0,
    )

    assert summary["holdout_trades"] == 6
    assert summary["holdout_sharpe"] == 0.150957
    assert summary["holdout_return"] == 0.015174
    assert "main_blockers" not in summary
    assert summary["starvation_signature"]["label"] == "momentum_starved"


def test_holdout_significance_enriches_starvation_for_top_level_and_slices() -> None:
    enriched = _compute_holdout_significance(
        {
            "holdout_trades": 0,
            "holdout_sharpe": 0.0,
            "holdout_return": 0.0,
            "gate_counters": {"total_checks": 5, "gate_counts": {"primary_threshold": 3}},
            "holdout_slices": {
                "recent90": {
                    "holdout_trades": 0,
                    "holdout_sharpe": 0.0,
                    "holdout_return": 0.0,
                    "gate_counters": {"total_checks": 5, "gate_counts": {"primary_threshold": 4}},
                }
            },
        },
        completed_trials=20,
    )

    assert enriched["starvation_signature"]["label"] == "confidence_starved"
    assert enriched["holdout_slices"]["recent90"]["starvation_signature"]["label"] == "confidence_starved"
    assert enriched["main_blockers"]
    assert enriched["holdout_slices"]["recent90"]["main_blockers"]
