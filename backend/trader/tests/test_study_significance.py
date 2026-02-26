import numpy as np

from core.study_significance import (
    build_study_score_matrix,
    compose_study_significance_diagnostic,
    compute_study_significance,
)


class DummyTrial:
    def __init__(self, fold_sharpes, *, mean_sharpe=None, frequency_adjusted_score=None):
        self.user_attrs = {
            "fold_metrics": [{"sharpe": float(v)} for v in fold_sharpes],
            "mean_sharpe": float(mean_sharpe) if mean_sharpe is not None else float(np.mean(fold_sharpes)),
            "frequency_adjusted_score": (
                float(frequency_adjusted_score)
                if frequency_adjusted_score is not None
                else float(np.mean(fold_sharpes))
            ),
        }


def test_compute_study_significance_sanity_range():
    matrix = np.array([
        [0.30, 0.25, 0.32, 0.28, 0.31],
        [0.05, 0.02, -0.01, 0.03, 0.01],
        [0.08, 0.06, 0.10, 0.07, 0.09],
    ])
    payload = compute_study_significance(matrix, bootstrap_iterations=300, random_seed=7, score_source="fold_sharpe")
    assert payload["enabled"] is True
    assert 0.0 <= float(payload["p_value"]) <= 1.0
    assert 0.0 <= float(payload["spa_like_p_value"]) <= 1.0


def test_compute_study_significance_deterministic_seed():
    trials = [
        DummyTrial([0.20, 0.10, 0.18, 0.22]),
        DummyTrial([0.05, 0.04, 0.03, 0.02]),
        DummyTrial([0.11, 0.09, 0.10, 0.08]),
    ]
    matrix = build_study_score_matrix(trials, score_source="fold_sharpe")
    a = compute_study_significance(matrix, bootstrap_iterations=250, random_seed=123, score_source="fold_sharpe")
    b = compute_study_significance(matrix, bootstrap_iterations=250, random_seed=123, score_source="fold_sharpe")
    assert a["p_value"] == b["p_value"]
    assert a["spa_like_p_value"] == b["spa_like_p_value"]


def test_compose_study_significance_disabled_smoke():
    payload = compose_study_significance_diagnostic(enabled=False, trials=[])
    assert payload["enabled"] is False
    assert payload["p_value"] is None
    assert payload["spa_like_p_value"] is None
    assert payload["bootstrap"] is None
