from __future__ import annotations

"""Study-level multiple-testing significance diagnostics.

This module provides a lightweight, bootstrap-based diagnostic inspired by
White's Reality Check / SPA family. It is intended for *post-run reporting*,
not as a hard optimization gate.

Method summary:
- Build a candidate x observation score matrix.
- Observed statistic uses the max mean score across candidates.
- Null bootstrap recenters each candidate series by its own mean, then samples
  observation indices with replacement and recomputes max mean.
- p-value estimates how often null max statistic exceeds observed max.

Limitations:
- Bootstrap here is i.i.d. index resampling (not a full block bootstrap).
- Inference quality depends on score stationarity/independence assumptions.
- Use as a diagnostic signal, not definitive proof of alpha.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np


@dataclass(frozen=True)
class StudySignificanceConfig:
    enabled: bool = False
    bootstrap_iterations: int = 500
    random_seed: int = 42
    score_source: str = "fold_sharpe"


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def _from_fold_metrics(trials: Iterable[Any], metric_key: str) -> np.ndarray:
    rows: List[List[float]] = []
    max_len = 0
    for trial in trials:
        metrics = getattr(trial, "user_attrs", {}).get("fold_metrics", [])
        if not isinstance(metrics, list):
            continue
        row = [_safe_float((m or {}).get(metric_key), default=np.nan) for m in metrics if isinstance(m, Mapping)]
        if not row:
            continue
        max_len = max(max_len, len(row))
        rows.append(row)

    if not rows or max_len <= 0:
        return np.zeros((0, 0), dtype=float)

    padded = [row + [np.nan] * (max_len - len(row)) for row in rows]
    return np.asarray(padded, dtype=float)


def _from_scalar_attr(trials: Iterable[Any], attr_key: str) -> np.ndarray:
    vals: List[float] = []
    for trial in trials:
        vals.append(_safe_float(getattr(trial, "user_attrs", {}).get(attr_key), default=np.nan))
    if not vals:
        return np.zeros((0, 0), dtype=float)
    arr = np.asarray(vals, dtype=float)
    if np.all(~np.isfinite(arr)):
        return np.zeros((0, 0), dtype=float)
    return arr[:, None]


def build_study_score_matrix(trials: Iterable[Any], score_source: str) -> np.ndarray:
    """Build candidate x observation matrix from trial attrs.

    Supported score_source values:
    - fold_sharpe (default): fold_metrics[].sharpe
    - fold_return: fold_metrics[].return
    - fold_expectancy: fold_metrics[].expectancy
    - cv_sharpe: scalar mean_sharpe repeated as 1-observation series
    - frequency_adjusted_score: scalar frequency_adjusted_score
    """
    source = (score_source or "").strip().lower()
    if source in {"fold_sharpe", "fold_metrics.sharpe"}:
        return _from_fold_metrics(trials, "sharpe")
    if source in {"fold_return", "fold_metrics.return"}:
        return _from_fold_metrics(trials, "return")
    if source in {"fold_expectancy", "fold_metrics.expectancy"}:
        return _from_fold_metrics(trials, "expectancy")
    if source in {"cv_sharpe", "mean_sharpe"}:
        return _from_scalar_attr(trials, "mean_sharpe")
    if source in {"frequency_adjusted_score", "raw_study_score"}:
        return _from_scalar_attr(trials, "frequency_adjusted_score")
    return np.zeros((0, 0), dtype=float)


def compute_study_significance(
    score_matrix: np.ndarray,
    *,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
    score_source: str = "fold_sharpe",
) -> Dict[str, Any]:
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2:
        return {
            "enabled": True,
            "p_value": None,
            "spa_like_p_value": None,
            "methodology": {"reason": "score_matrix_must_be_2d", "score_source": score_source},
        }

    n_candidates, n_obs = arr.shape
    if n_candidates < 2 or n_obs < 1:
        return {
            "enabled": True,
            "p_value": None,
            "spa_like_p_value": None,
            "methodology": {
                "reason": "insufficient_candidates_or_observations",
                "candidate_universe_size": int(n_candidates),
                "observation_count": int(n_obs),
                "score_source": score_source,
            },
        }

    clean = np.where(np.isfinite(arr), arr, np.nan)
    candidate_means = np.nanmean(clean, axis=1)
    if not np.any(np.isfinite(candidate_means)):
        return {
            "enabled": True,
            "p_value": None,
            "spa_like_p_value": None,
            "methodology": {"reason": "all_scores_non_finite", "score_source": score_source},
        }

    best_idx = int(np.nanargmax(candidate_means))
    observed_max = float(np.nanmax(candidate_means))
    observed_best = float(candidate_means[best_idx])

    recentered = clean - candidate_means[:, None]
    recentered = np.where(np.isfinite(recentered), recentered, 0.0)

    rng = np.random.default_rng(int(random_seed))
    boot_iters = max(10, int(bootstrap_iterations))
    boot_stats = np.empty(boot_iters, dtype=float)
    boot_best = np.empty(boot_iters, dtype=float)

    for i in range(boot_iters):
        idx = rng.integers(0, n_obs, size=n_obs)
        sampled = recentered[:, idx]
        boot_means = np.mean(sampled, axis=1)
        boot_stats[i] = float(np.max(boot_means))
        boot_best[i] = float(np.mean(sampled[best_idx]))

    p_value = float((1 + np.sum(boot_stats >= observed_max)) / (boot_iters + 1))
    spa_like = float((1 + np.sum(boot_best >= observed_best)) / (boot_iters + 1))

    return {
        "enabled": True,
        "p_value": p_value,
        "spa_like_p_value": spa_like,
        "best_candidate_index": best_idx,
        "observed_stat": {
            "max_mean_score": observed_max,
            "best_candidate_mean_score": observed_best,
        },
        "bootstrap": {
            "iterations": boot_iters,
            "random_seed": int(random_seed),
            "resampling": "iid_index_resample",
        },
        "methodology": {
            "style": "white_rc_spa_like_recentered_bootstrap",
            "score_definition": "candidate_mean_over_observations",
            "score_source": score_source,
            "candidate_universe_size": int(n_candidates),
            "observation_count": int(n_obs),
        },
    }


def compose_study_significance_diagnostic(
    *,
    enabled: bool,
    trials: Optional[Iterable[Any]] = None,
    bootstrap_iterations: int = 500,
    random_seed: int = 42,
    score_source: str = "fold_sharpe",
) -> Dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "p_value": None,
            "spa_like_p_value": None,
            "bootstrap": None,
            "methodology": {
                "style": "white_rc_spa_like_recentered_bootstrap",
                "reason": "disabled",
                "score_source": score_source,
            },
        }

    score_matrix = build_study_score_matrix(trials or [], score_source=score_source)
    return compute_study_significance(
        score_matrix,
        bootstrap_iterations=bootstrap_iterations,
        random_seed=random_seed,
        score_source=score_source,
    )
