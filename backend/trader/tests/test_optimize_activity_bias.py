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


def _trial(number: int, value: float, n_trades: int, cv_ratio: float, mean_sharpe: float = 0.4):
    return SimpleNamespace(
        number=number,
        value=value,
        state=optuna.trial.TrialState.COMPLETE,
        user_attrs={
            "n_trades": n_trades,
            "cv_trade_density_ratio": cv_ratio,
            "mean_sharpe": mean_sharpe,
            "std_sharpe": 0.05,
            "min_sharpe": 0.0,
            "max_drawdown": 0.1,
            "psr": 0.4,
        },
    )


def test_candidate_trials_for_holdout_prefers_activity_first() -> None:
    inactive_but_higher_objective = _trial(1, value=1.30, n_trades=10, cv_ratio=0.40, mean_sharpe=0.8)
    active_trial = _trial(2, value=1.10, n_trades=35, cv_ratio=1.35, mean_sharpe=0.35)
    study = SimpleNamespace(trials=[inactive_but_higher_objective, active_trial])

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


def test_cooldown_dominance_penalty_triggers_when_cooldown_overwhelms_gate_mix() -> None:
    gate_summary = {
        "gate_rates": {
            "cooldown": 0.74,
            "primary_threshold": 0.11,
            "ensemble_agreement": 0.08,
        }
    }

    penalty = _cooldown_dominance_penalty(gate_summary)

    assert penalty["cooldown_rate"] == 0.74
    assert penalty["max_other_rate"] == 0.11
    assert penalty["penalty"] > 0.0


def test_fold_balance_penalties_penalize_zero_trade_fold() -> None:
    penalties = _fold_balance_penalties([18, 0, 16], [0.25, -0.22, 0.20])

    assert penalties["zero_trade_penalty"] >= 0.45
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
