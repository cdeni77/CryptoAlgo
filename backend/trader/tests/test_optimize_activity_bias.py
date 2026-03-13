import optuna
from types import SimpleNamespace

from scripts.optimize import (
    _candidate_holdout_summary,
    _candidate_trials_for_holdout,
    _holdout_selection_score,
    _classify_activity_regime,
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
