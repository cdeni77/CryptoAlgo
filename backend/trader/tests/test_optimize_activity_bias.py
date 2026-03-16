import optuna
from types import SimpleNamespace

from scripts.optimize import (
    _candidate_holdout_summary,
    _candidate_trials_for_holdout,
    _holdout_selection_score,
    _classify_activity_regime,
    _cooldown_dominance_penalty,
    _fold_balance_penalties,
    _top_level_reporting_fields,
)


def _trial(
    number: int,
    value: float,
    n_trades: int,
    cv_ratio: float,
    mean_sharpe: float = 0.4,
    realized_return: float = 0.04,
    realized_expectancy: float = 0.02,
    cooldown_penalty: float = 0.0,
    fold_balance_penalty: float = 0.0,
    zero_trade_fold_penalty: float = 0.0,
    fold_gate_counters=None,
):
    return SimpleNamespace(
        number=number,
        value=value,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={
            "n_trades": n_trades,
            "cv_trade_density_ratio": cv_ratio,
            "mean_sharpe": mean_sharpe,
            "realized_return": realized_return,
            "realized_expectancy": realized_expectancy,
            "cooldown_dominance_penalty": cooldown_penalty,
            "fold_balance_penalty": fold_balance_penalty,
            "zero_trade_fold_penalty": zero_trade_fold_penalty,
            "fold_gate_counters": fold_gate_counters or [],
            "std_sharpe": 0.05,
            "min_sharpe": 0.0,
            "max_drawdown": 0.1,
            "psr": 0.4,
        },
    )


def test_candidate_trials_for_holdout_prefers_balanced_lower_cooldown_candidates() -> None:
    cooldown_heavy = _trial(
        1,
        value=1.30,
        n_trades=28,
        cv_ratio=1.30,
        mean_sharpe=0.55,
        cooldown_penalty=0.40,
        fold_balance_penalty=0.35,
        zero_trade_fold_penalty=0.65,
        fold_gate_counters=[{"total_checks": 100, "gate_counts": {"cooldown": 76, "primary_threshold": 8}}] * 3,
    )
    balanced = _trial(
        2,
        value=1.10,
        n_trades=22,
        cv_ratio=1.05,
        mean_sharpe=0.40,
        cooldown_penalty=0.02,
        fold_balance_penalty=0.01,
        zero_trade_fold_penalty=0.0,
        fold_gate_counters=[{"total_checks": 100, "gate_counts": {"cooldown": 20, "primary_threshold": 18}}] * 3,
    )
    study = SimpleNamespace(trials=[cooldown_heavy, balanced])

    ranked = _candidate_trials_for_holdout(study, max_candidates=2, min_trades=6)

    assert ranked[0].number == 2
    assert ranked[1].number == 1


def test_holdout_selection_score_penalizes_low_trade_slices_more_aggressively() -> None:
    sparse = {
        "holdout_slices": {
            "recent90": {"holdout_sharpe": 0.6, "holdout_return": 0.08, "holdout_trades": 4},
            "prior90": {"holdout_sharpe": 0.5, "holdout_return": 0.05, "holdout_trades": 5},
        }
    }
    active = {
        "holdout_slices": {
            "recent90": {"holdout_sharpe": 0.4, "holdout_return": 0.04, "holdout_trades": 18},
            "prior90": {"holdout_sharpe": 0.35, "holdout_return": 0.03, "holdout_trades": 22},
        }
    }

    assert _holdout_selection_score(active, cv_score=0.4) > _holdout_selection_score(sparse, cv_score=0.4)


def test_candidate_summary_emits_cv_activity_fields() -> None:
    trial = SimpleNamespace(number=7, user_attrs={"n_trades": 30, "cv_trade_density": 6.0, "cv_trade_density_ratio": 1.2, "activity_regime": "active"})
    holdout_metrics = {
        "holdout_trades": 14,
        "holdout_sharpe": 0.11,
        "holdout_return": 0.02,
        "gate_counters": {"total_checks": 10, "gate_counts": {"primary_threshold": 2}},
    }

    summary = _candidate_holdout_summary(trial, holdout_metrics, 0.5, min_trades=10, min_sharpe=0.0, min_return=0.0)

    assert summary["cv_n_trades"] == 30
    assert summary["cv_trade_density"] == 6.0
    assert summary["cv_trade_density_ratio"] == 1.2
    assert summary["activity_regime"] == "active"



def test_activity_regime_classification() -> None:
    assert _classify_activity_regime(0.0, 0) == "dead"
    assert _classify_activity_regime(0.2, 2) == "starved"
    assert _classify_activity_regime(0.7, 9) == "thin"
    assert _classify_activity_regime(1.1, 20) == "active"


def test_cooldown_dominance_penalty_stronger_for_high_rate_and_fold_dominance() -> None:
    gate_summary = {
        "gate_rates": {
            "cooldown": 0.74,
            "primary_threshold": 0.11,
            "ensemble_agreement": 0.08,
        }
    }

    baseline = _cooldown_dominance_penalty(gate_summary, fold_dominant_share=0.10)
    dominated = _cooldown_dominance_penalty(gate_summary, fold_dominant_share=0.85)

    assert dominated["cooldown_rate"] == 0.74
    assert dominated["max_other_rate"] == 0.11
    assert dominated["cooldown_regime"] in {"elevated", "severe"}
    assert dominated["penalty"] > baseline["penalty"]


def test_fold_balance_penalties_penalize_zero_trade_fold_more_strongly() -> None:
    penalties = _fold_balance_penalties([18, 0, 16], [0.25, -0.22, 0.20])

    assert penalties["zero_trade_penalty"] >= 0.65
    assert penalties["total_penalty"] >= penalties["zero_trade_penalty"]


def test_top_level_reporting_fields_copies_selection_meta_and_holdout_summary() -> None:
    payload = {
        "selection_meta": {
            "research_confidence_tier": "PAPER_QUALIFIED",
            "n_candidates": 4,
            "n_passing_candidates": 1,
            "paper_eligible_promoted_candidate": {"trial": 7, "selection_score": 0.91},
        },
        "holdout_metrics": {
            "holdout_trades": 19,
            "holdout_sharpe": 0.33,
            "holdout_return": 0.07,
            "selected_slice": "recent90",
        },
        "deployment_blocked": True,
        "deployment_block_reasons": ["NO_HOLDOUT_CANDIDATE_PASSED_GATE"],
    }

    flattened = _top_level_reporting_fields(payload)

    assert flattened["tier"] == "PAPER_QUALIFIED"
    assert flattened["blocked"] is True
    assert flattened["block_reasons"] == ["NO_HOLDOUT_CANDIDATE_PASSED_GATE"]
    assert flattened["holdout_trades"] == 19
    assert flattened["holdout_sharpe"] == 0.33
    assert flattened["holdout_return"] == 0.07
    assert flattened["selected_slice"] == "recent90"
    assert flattened["n_candidates"] == 4
    assert flattened["n_passing_candidates"] == 1
    assert flattened["paper_eligible_promoted_candidate"] == {"trial": 7, "selection_score": 0.91}
