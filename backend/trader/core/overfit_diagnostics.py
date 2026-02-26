from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class PBOResult:
    pbo: float
    split_count: int
    candidate_count: int
    score_used: str
    sample_size: int
    oos_percentiles: List[float]
    logits: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pbo": float(self.pbo),
            "methodology": {
                "style": "cscv_approx_leave_one_split_out",
                "split_count": int(self.split_count),
                "candidate_count": int(self.candidate_count),
                "score_used": self.score_used,
                "sample_size": int(self.sample_size),
            },
            "oos_percentiles": [float(v) for v in self.oos_percentiles],
            "logits": [float(v) for v in self.logits],
        }


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def compute_pbo_from_matrix(score_matrix: np.ndarray, score_used: str = "sharpe") -> Dict[str, Any]:
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2:
        return {"pbo": None, "methodology": {"reason": "score_matrix_must_be_2d", "score_used": score_used}}

    n_candidates, n_splits = arr.shape
    if n_candidates < 2 or n_splits < 2:
        return {
            "pbo": None,
            "methodology": {
                "reason": "insufficient_candidates_or_splits",
                "candidate_count": int(n_candidates),
                "split_count": int(n_splits),
                "score_used": score_used,
            },
        }

    logits: List[float] = []
    percentiles: List[float] = []
    eps = 1e-9

    for holdout_idx in range(n_splits):
        in_sample_idx = [i for i in range(n_splits) if i != holdout_idx]
        ins_scores = np.nanmean(arr[:, in_sample_idx], axis=1)
        ins_scores = np.where(np.isfinite(ins_scores), ins_scores, -np.inf)
        selected_idx = int(np.argmax(ins_scores))

        oos_col = arr[:, holdout_idx]
        oos_col = np.where(np.isfinite(oos_col), oos_col, -np.inf)
        order = np.argsort(oos_col)
        rank = int(np.where(order == selected_idx)[0][0])
        percentile = float((rank + 1) / max(1, n_candidates))
        percentile = min(1.0 - eps, max(eps, percentile))

        percentiles.append(percentile)
        logits.append(float(np.log(percentile / (1.0 - percentile))))

    pbo = float(np.mean([1.0 if l < 0 else 0.0 for l in logits]))
    return PBOResult(
        pbo=pbo,
        split_count=n_splits,
        candidate_count=n_candidates,
        score_used=score_used,
        sample_size=len(logits),
        oos_percentiles=percentiles,
        logits=logits,
    ).to_dict()


def build_score_matrix_from_trials(trials: Iterable[Any], score_key: str = "sharpe") -> np.ndarray:
    rows: List[List[float]] = []
    max_len = 0
    for t in trials:
        metrics = getattr(t, "user_attrs", {}).get("fold_metrics", [])
        if not isinstance(metrics, list):
            continue
        row = [_safe_float((m or {}).get(score_key), default=np.nan) for m in metrics if isinstance(m, Mapping)]
        if not row:
            continue
        max_len = max(max_len, len(row))
        rows.append(row)

    if not rows or max_len < 2:
        return np.zeros((0, 0), dtype=float)

    norm = []
    for row in rows:
        padded = row + [np.nan] * (max_len - len(row))
        norm.append(padded)
    return np.asarray(norm, dtype=float)


def make_stress_costs_block(
    baseline_metrics: Mapping[str, Any],
    scenario_metrics: Mapping[str, Mapping[str, Any]],
    metric_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    keys = metric_keys or ["holdout_sharpe", "holdout_return", "holdout_trades", "full_dd"]
    deltas: Dict[str, Dict[str, Optional[float]]] = {}

    for scenario_name, row in scenario_metrics.items():
        scenario_delta: Dict[str, Optional[float]] = {}
        for key in keys:
            base_v = baseline_metrics.get(key)
            stress_v = row.get(key)
            if isinstance(base_v, (int, float)) and isinstance(stress_v, (int, float)):
                scenario_delta[key] = float(stress_v) - float(base_v)
            else:
                scenario_delta[key] = None
        deltas[scenario_name] = scenario_delta

    return {
        "baseline": dict(baseline_metrics),
        "stress": {k: dict(v) for k, v in scenario_metrics.items()},
        "deltas": deltas,
    }


def compose_robustness_diagnostics(
    *,
    enabled: bool,
    pbo: Optional[Mapping[str, Any]] = None,
    stress_costs: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "pbo": None,
            "stress_costs": None,
        }
    return {
        "enabled": True,
        "pbo": dict(pbo) if isinstance(pbo, Mapping) else None,
        "stress_costs": dict(stress_costs) if isinstance(stress_costs, Mapping) else None,
    }
