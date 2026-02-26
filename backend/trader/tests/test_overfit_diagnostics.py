import json

import numpy as np

from core.overfit_diagnostics import (
    compose_robustness_diagnostics,
    compute_pbo_from_matrix,
    make_stress_costs_block,
)


def test_compute_pbo_from_matrix_synthetic_input():
    matrix = np.array([
        [0.9, 0.8, 0.1, 0.2],
        [0.4, 0.3, 0.7, 0.8],
        [0.2, 0.1, 0.5, 0.4],
    ])
    result = compute_pbo_from_matrix(matrix, score_used="fold_metrics.sharpe")
    assert result["pbo"] is not None
    assert 0.0 <= float(result["pbo"]) <= 1.0
    assert result["methodology"]["split_count"] == 4
    assert result["methodology"]["candidate_count"] == 3


def test_cost_stress_block_serializes_and_contains_deltas():
    baseline = {"holdout_sharpe": 0.45, "holdout_return": 0.12, "holdout_trades": 21}
    scenarios = {
        "fees_plus_50pct": {"holdout_sharpe": 0.30, "holdout_return": 0.07, "holdout_trades": 21},
        "slippage_x2": {"holdout_sharpe": 0.20, "holdout_return": 0.04, "holdout_trades": 20},
    }
    block = make_stress_costs_block(baseline, scenarios)
    assert abs(block["deltas"]["fees_plus_50pct"]["holdout_sharpe"] + 0.15) < 1e-9
    encoded = json.dumps(block)
    assert "fees_plus_50pct" in encoded


def test_compose_robustness_diagnostics_disabled_smoke():
    payload = compose_robustness_diagnostics(enabled=False)
    assert payload["enabled"] is False
    assert payload["pbo"] is None
    assert payload["stress_costs"] is None
