"""Canonical significance metrics for Sharpe-based selection gates.

This module centralizes probabilistic Sharpe ratio (PSR) and deflated Sharpe
ratio (DSR) calculations so optimizer and launch scripts share one
implementation.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, (int, float, np.number)):
        out = float(value)
    else:
        try:
            out = float(value)
        except (TypeError, ValueError):
            return default
    if not math.isfinite(out):
        return default
    return out


def _sample_moments(samples: Sequence[float]) -> tuple[float, float, float, float]:
    arr = np.asarray([float(x) for x in samples], dtype=float)
    n = int(arr.size)
    if n <= 0:
        return 0.0, 0.0, 0.0, 3.0
    mean = float(np.mean(arr))
    centered = arr - mean
    var = float(np.mean(centered ** 2))
    if var <= 1e-12:
        return mean, var, 0.0, 3.0
    sigma = math.sqrt(var)
    m3 = float(np.mean((centered / sigma) ** 3))
    m4 = float(np.mean((centered / sigma) ** 4))
    return mean, var, m3, m4


def compute_psr(
    *,
    sharpe_estimate: float,
    observations: int,
    benchmark_sharpe: float = 0.0,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> dict[str, Any]:
    """Compute probabilistic Sharpe ratio P(SR > benchmark).

    If skewness/kurtosis are not provided, defaults are Gaussian assumptions:
    skewness=0 and kurtosis=3.
    """
    sr = _as_float(sharpe_estimate, 0.0) or 0.0
    n = int(max(0, observations or 0))
    benchmark = _as_float(benchmark_sharpe, 0.0) or 0.0
    sk = _as_float(skewness, 0.0)
    ku = _as_float(kurtosis, 3.0)
    used_fallback_moments = skewness is None or kurtosis is None

    if n < 2:
        return {
            "valid": False,
            "psr": 0.0,
            "z_score": 0.0,
            "observations": n,
            "benchmark_sharpe": benchmark,
            "sharpe_estimate": sr,
            "skewness": sk,
            "kurtosis": ku,
            "fallback_moments_used": used_fallback_moments,
            "reason": "insufficient_observations",
        }

    denom_inner = 1.0 - (sk or 0.0) * sr + ((ku or 3.0) - 1.0) * (sr ** 2) / 4.0
    if denom_inner <= 1e-12:
        return {
            "valid": False,
            "psr": 0.0,
            "z_score": 0.0,
            "observations": n,
            "benchmark_sharpe": benchmark,
            "sharpe_estimate": sr,
            "skewness": sk,
            "kurtosis": ku,
            "fallback_moments_used": used_fallback_moments,
            "reason": "invalid_denominator",
        }

    z_score = (sr - benchmark) * math.sqrt(max(1.0, n - 1.0)) / math.sqrt(denom_inner)
    psr = _NORMAL.cdf(z_score)
    return {
        "valid": True,
        "psr": float(psr),
        "z_score": float(z_score),
        "observations": n,
        "benchmark_sharpe": benchmark,
        "sharpe_estimate": sr,
        "skewness": float(sk or 0.0),
        "kurtosis": float(ku or 3.0),
        "fallback_moments_used": used_fallback_moments,
    }


def compute_psr_from_samples(
    sharpes: Sequence[float],
    *,
    benchmark_sharpe: float = 0.0,
    effective_observations: int | None = None,
) -> dict[str, Any]:
    valid = [float(s) for s in sharpes if s is not None and math.isfinite(float(s))]
    n = len(valid)
    if n <= 0:
        result = compute_psr(sharpe_estimate=0.0, observations=0, benchmark_sharpe=benchmark_sharpe)
        result["sample_count"] = 0
        return result
    mean_sr, _, skewness, kurtosis = _sample_moments(valid)
    obs = int(effective_observations) if effective_observations is not None else n
    result = compute_psr(
        sharpe_estimate=mean_sr,
        observations=obs,
        benchmark_sharpe=benchmark_sharpe,
        skewness=skewness,
        kurtosis=kurtosis,
    )
    result["sample_count"] = n
    result["sample_sharpes"] = [round(float(x), 6) for x in valid]
    return result


def compute_deflated_sharpe(
    *,
    observed_sharpe: float,
    observations: int,
    effective_test_count: int,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> dict[str, Any]:
    sr = _as_float(observed_sharpe, 0.0) or 0.0
    n = int(max(0, observations or 0))
    tests = int(max(1, effective_test_count or 1))
    sk = _as_float(skewness, 0.0)
    ku = _as_float(kurtosis, 3.0)
    used_fallback_moments = skewness is None or kurtosis is None

    if n < 2:
        return {
            "valid": False,
            "dsr": 0.0,
            "p_value": 1.0,
            "expected_max_sr": 0.0,
            "observations": n,
            "effective_test_count": tests,
            "observed_sharpe": sr,
            "skewness": sk,
            "kurtosis": ku,
            "fallback_moments_used": used_fallback_moments,
            "reason": "insufficient_observations",
        }

    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / tests)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (tests * math.e))
    max_z = (1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2
    expected_max_sr = max_z / math.sqrt(max(1.0, n))

    var_term = 1.0 + 0.5 * sr * sr - (sk or 0.0) * sr + (((ku or 3.0) - 3.0) / 4.0) * sr * sr
    sr_std = math.sqrt(max(var_term / max(1.0, n), 1e-12))
    z_score = (sr - expected_max_sr) / sr_std
    p_value = 1.0 - _NORMAL.cdf(z_score)

    return {
        "valid": True,
        "dsr": float(z_score),
        "p_value": float(p_value),
        "expected_max_sr": float(expected_max_sr),
        "significant_10pct": bool(p_value < 0.10),
        "observations": n,
        "effective_test_count": tests,
        "observed_sharpe": sr,
        "skewness": float(sk or 0.0),
        "kurtosis": float(ku or 3.0),
        "fallback_moments_used": used_fallback_moments,
    }


def evaluate_significance_gates(
    *,
    psr_cv: Mapping[str, Any] | None = None,
    psr_holdout: Mapping[str, Any] | None = None,
    dsr: Mapping[str, Any] | None = None,
    min_psr_cv: float | None = None,
    min_psr_holdout: float | None = None,
    min_dsr: float | None = None,
) -> dict[str, Any]:
    """Evaluate optional significance gates.

    Gates are disabled when thresholds are None.
    """

    def _check(metric: Mapping[str, Any] | None, threshold: float | None, key: str) -> dict[str, Any]:
        enabled = threshold is not None
        observed = _as_float((metric or {}).get(key), None)
        valid = bool((metric or {}).get("valid", False))
        passed = (not enabled) or (valid and observed is not None and observed >= float(threshold))
        reason = None
        if enabled and not passed:
            if not valid:
                reason = "invalid_metric"
            else:
                reason = f"{key}_below_threshold"
        return {
            "enabled": enabled,
            "threshold": None if threshold is None else float(threshold),
            "observed": observed,
            "valid": valid,
            "passed": bool(passed),
            "reason": reason,
        }

    checks = {
        "psr_cv": _check(psr_cv, min_psr_cv, "psr"),
        "psr_holdout": _check(psr_holdout, min_psr_holdout, "psr"),
        "dsr": _check(dsr, min_dsr, "dsr"),
    }
    checks["all_passed"] = all(check["passed"] for check in checks.values())
    return checks
