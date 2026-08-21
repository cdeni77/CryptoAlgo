"""Performance metrics and the significance tests that decide what ships.

Consolidates `metrics_significance.py`, `study_significance.py` and
`overfit_diagnostics.py`. The formulas are Bailey and López de Prado's and are
carried over unchanged; what is new is that they live together, and that the
inputs which make them honest — the effective sample size and the true trial
count — are parameters the caller must supply rather than defaults nobody sets.

Four things get measured, in increasing order of how much they hurt:

    sharpe / drawdown   What the equity curve did.
    PSR                 Is the Sharpe distinguishable from zero, given how few
                        independent observations there really are?
    DSR                 Is it distinguishable from the best of N tries?
    PBO                 Does picking the in-sample winner actually help
                        out of sample, or is selection pure noise?

The last two are the ones that reject strategies. A Sharpe of 1.2 chosen from
3,000 configurations on 40 independent observations is not a discovery, and only
DSR and PBO will say so.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Any, Mapping, Optional, Sequence

import numpy as np

_NORMAL = NormalDist()
_EULER_MASCHERONI = 0.5772156649015329

HOURS_PER_YEAR = 24 * 365


# ---------------------------------------------------------------------------
# Descriptive
# ---------------------------------------------------------------------------


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def sample_moments(samples: Sequence[float]) -> tuple[float, float, float, float]:
    """Mean, variance, skewness and kurtosis. Kurtosis is non-excess (Gaussian = 3)."""
    arr = np.asarray([float(x) for x in samples], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 3.0

    mean = float(arr.mean())
    centered = arr - mean
    variance = float((centered ** 2).mean())
    if variance <= 1e-12:
        return mean, variance, 0.0, 3.0

    sigma = math.sqrt(variance)
    return (
        mean,
        variance,
        float(((centered / sigma) ** 3).mean()),
        float(((centered / sigma) ** 4).mean()),
    )


def sharpe_ratio(
    returns: Sequence[float],
    *,
    periods_per_year: int = HOURS_PER_YEAR,
    risk_free: float = 0.0,
) -> float:
    """Annualised Sharpe of a per-period return series."""
    arr = np.asarray([float(x) for x in returns], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return 0.0
    excess = arr - (risk_free / periods_per_year)
    sigma = excess.std(ddof=1)
    if sigma <= 1e-12:
        return 0.0
    return float(excess.mean() / sigma * math.sqrt(periods_per_year))


@dataclass(frozen=True)
class DrawdownProfile:
    """Depth and duration of the worst stretch, and how long recovery took."""

    max_drawdown: float
    max_drawdown_duration: int
    time_to_recovery: Optional[int]
    calmar: float


def drawdown_profile(
    equity: Sequence[float],
    *,
    periods_per_year: int = HOURS_PER_YEAR,
) -> DrawdownProfile:
    """Drawdown statistics from an equity curve.

    `time_to_recovery` is None when the curve never regained its prior peak —
    which is the case a single max-drawdown number hides.
    """
    arr = np.asarray([float(x) for x in equity], dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return DrawdownProfile(0.0, 0, None, 0.0)

    peaks = np.maximum.accumulate(arr)
    drawdowns = np.divide(arr - peaks, peaks, out=np.zeros_like(arr), where=peaks > 0)
    trough = int(np.argmin(drawdowns))
    max_dd = float(-drawdowns[trough])

    peak_before = int(np.argmax(arr[:trough + 1])) if trough > 0 else 0
    recovered = np.flatnonzero(arr[trough:] >= peaks[trough])
    time_to_recovery = int(recovered[0]) if recovered.size else None
    duration = (trough - peak_before) + (time_to_recovery or (arr.size - trough))

    total_return = (arr[-1] / arr[0]) - 1.0 if arr[0] > 0 else 0.0
    years = max(arr.size / periods_per_year, 1e-9)
    annualised = (1.0 + total_return) ** (1.0 / years) - 1.0 if total_return > -1 else -1.0
    calmar = annualised / max_dd if max_dd > 1e-12 else 0.0

    return DrawdownProfile(max_dd, int(duration), time_to_recovery, float(calmar))


# ---------------------------------------------------------------------------
# Probabilistic Sharpe ratio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SignificanceResult:
    """A significance test outcome, with the inputs that produced it."""

    valid: bool
    statistic: float
    probability: float
    observations: int
    detail: dict[str, Any] = field(default_factory=dict)
    reason: str = ''

    def __bool__(self) -> bool:
        return self.valid


def probabilistic_sharpe(
    *,
    sharpe: float,
    observations: int,
    benchmark: float = 0.0,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> SignificanceResult:
    """P(true Sharpe > benchmark), adjusting for skew and fat tails.

    `observations` must be the **effective** count from `core.cv`, not the row
    count. Passing 2,880 hourly rows where the labels carry 40 independent
    outcomes inflates confidence by roughly the square root of 72.
    """
    sr = _finite(sharpe, 0.0) or 0.0
    n = int(max(0, observations or 0))
    bench = _finite(benchmark, 0.0) or 0.0
    skew = _finite(skewness, 0.0) or 0.0
    kurt = _finite(kurtosis, 3.0) or 3.0
    assumed_normal = skewness is None or kurtosis is None

    detail = {
        'sharpe': sr, 'benchmark': bench, 'skewness': skew, 'kurtosis': kurt,
        'assumed_normal_moments': assumed_normal,
    }
    if n < 2:
        return SignificanceResult(False, 0.0, 0.0, n, detail, 'insufficient_observations')

    variance = 1.0 - skew * sr + (kurt - 1.0) * (sr ** 2) / 4.0
    if variance <= 1e-12:
        return SignificanceResult(False, 0.0, 0.0, n, detail, 'degenerate_variance')

    z = (sr - bench) * math.sqrt(max(1.0, n - 1.0)) / math.sqrt(variance)
    return SignificanceResult(True, float(z), float(_NORMAL.cdf(z)), n, detail)


def expected_max_sharpe(trials: int, observations: int) -> float:
    """Sharpe you would expect from the luckiest of `trials` worthless strategies.

    The benchmark a real edge has to clear. It grows with the number of
    configurations tried, which is why the trial count must be recorded rather
    than guessed.
    """
    tests = int(max(1, trials))
    n = int(max(1, observations))
    if tests == 1:
        return 0.0
    z1 = _NORMAL.inv_cdf(1.0 - 1.0 / tests)
    z2 = _NORMAL.inv_cdf(1.0 - 1.0 / (tests * math.e))
    return ((1.0 - _EULER_MASCHERONI) * z1 + _EULER_MASCHERONI * z2) / math.sqrt(n)


def deflated_sharpe(
    *,
    sharpe: float,
    observations: int,
    trials: int,
    skewness: float | None = None,
    kurtosis: float | None = None,
) -> SignificanceResult:
    """Is the observed Sharpe better than the best of `trials` coin flips?

    `trials` is the total number of configurations evaluated across the whole
    search, not the number kept. Under-reporting it is the most common way a
    backtest passes a significance test it should fail.
    """
    sr = _finite(sharpe, 0.0) or 0.0
    n = int(max(0, observations or 0))
    tests = int(max(1, trials or 1))
    skew = _finite(skewness, 0.0) or 0.0
    kurt = _finite(kurtosis, 3.0) or 3.0
    assumed_normal = skewness is None or kurtosis is None

    benchmark = expected_max_sharpe(tests, n)
    detail = {
        'sharpe': sr, 'trials': tests, 'expected_max_sharpe': float(benchmark),
        'skewness': skew, 'kurtosis': kurt, 'assumed_normal_moments': assumed_normal,
    }
    if n < 2:
        return SignificanceResult(False, 0.0, 1.0, n, detail, 'insufficient_observations')

    variance = 1.0 + 0.5 * sr * sr - skew * sr + ((kurt - 3.0) / 4.0) * sr * sr
    sigma = math.sqrt(max(variance / max(1.0, n), 1e-12))
    z = (sr - benchmark) / sigma
    p_value = 1.0 - _NORMAL.cdf(z)
    detail['p_value'] = float(p_value)
    return SignificanceResult(True, float(z), float(1.0 - p_value), n, detail)


# ---------------------------------------------------------------------------
# Probability of backtest overfitting
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PBOResult:
    """Combinatorially-symmetric cross-validation estimate of overfitting.

    `pbo` is the share of splits where the configuration that looked best in
    sample landed below the median out of sample. At 0.5 selection is doing
    nothing; above it, picking the in-sample winner is actively harmful.
    """

    pbo: Optional[float]
    n_candidates: int
    n_splits: int
    logits: tuple[float, ...] = ()
    oos_percentiles: tuple[float, ...] = ()
    reason: str = ''

    @property
    def valid(self) -> bool:
        return self.pbo is not None


def probability_of_backtest_overfitting(score_matrix: np.ndarray) -> PBOResult:
    """PBO from a (candidates x splits) matrix of out-of-sample scores.

    For each split held out in turn: pick the candidate with the best mean score
    on the remaining splits, then see where that candidate ranks on the held-out
    one. A genuine edge ranks high; an overfit ranks at chance.
    """
    arr = np.asarray(score_matrix, dtype=float)
    if arr.ndim != 2:
        return PBOResult(None, 0, 0, reason='score_matrix_must_be_2d')

    n_candidates, n_splits = arr.shape
    if n_candidates < 2 or n_splits < 2:
        return PBOResult(
            None, n_candidates, n_splits, reason='need_at_least_2_candidates_and_2_splits'
        )

    epsilon = 1e-9
    logits: list[float] = []
    percentiles: list[float] = []

    for holdout in range(n_splits):
        in_sample = [i for i in range(n_splits) if i != holdout]
        with np.errstate(invalid='ignore'):
            in_sample_means = np.nanmean(arr[:, in_sample], axis=1)
        in_sample_means = np.where(np.isfinite(in_sample_means), in_sample_means, -np.inf)
        winner = int(np.argmax(in_sample_means))

        held_out = np.where(np.isfinite(arr[:, holdout]), arr[:, holdout], -np.inf)
        rank = int(np.flatnonzero(np.argsort(held_out) == winner)[0])
        percentile = min(1.0 - epsilon, max(epsilon, (rank + 1) / n_candidates))

        percentiles.append(float(percentile))
        logits.append(float(math.log(percentile / (1.0 - percentile))))

    # Share of splits whose in-sample winner fell strictly below the
    # out-of-sample median. A logit of exactly zero is the median itself, which
    # is neither above nor below it, so it does not count as a failure.
    pbo = float(sum(1 for value in logits if value < 0.0) / len(logits))

    return PBOResult(
        pbo=pbo,
        n_candidates=int(n_candidates),
        n_splits=int(n_splits),
        logits=tuple(logits),
        oos_percentiles=tuple(percentiles),
    )


# ---------------------------------------------------------------------------
# CPCV path distribution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathDistribution:
    """Sharpe across CPCV paths — the shape the promotion gates read.

    `median` and `p05` are what get gated. A high mean with a negative 5th
    percentile is a strategy that works on most cuts of the data and blows up on
    some, which a single walk-forward number would have reported as a success.
    """

    n_paths: int
    mean: float
    median: float
    p05: float
    p95: float
    std: float
    positive_fraction: float
    values: tuple[float, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            'n_paths': self.n_paths, 'mean': self.mean, 'median': self.median,
            'p05': self.p05, 'p95': self.p95, 'std': self.std,
            'positive_fraction': self.positive_fraction,
        }


def summarise_paths(path_scores: Sequence[float]) -> PathDistribution:
    """Reduce per-path scores to the distribution the gates evaluate.

    Non-finite scores are dropped before any percentile is taken. A fold whose
    IC is undefined — too few rows, or a constant forecast — contributes nothing
    rather than turning the whole distribution into NaN.
    """
    arr = np.asarray([
        float(x) for x in path_scores
        if x is not None and np.isfinite(float(x))
    ], dtype=float)
    if arr.size == 0:
        return PathDistribution(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ())

    return PathDistribution(
        n_paths=int(arr.size),
        mean=float(arr.mean()),
        median=float(np.median(arr)),
        p05=float(np.percentile(arr, 5)),
        p95=float(np.percentile(arr, 95)),
        std=float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        positive_fraction=float((arr > 0).mean()),
        values=tuple(float(x) for x in arr),
    )


# ---------------------------------------------------------------------------
# Promotion gates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One promotion criterion and whether it passed."""

    name: str
    value: Optional[float]
    threshold: float
    comparison: str        # 'min' — value must be at least; 'max' — at most
    passed: bool
    note: str = ''

    def __str__(self) -> str:
        mark = 'PASS' if self.passed else 'FAIL'
        shown = 'n/a' if self.value is None else f'{self.value:.4f}'
        symbol = '>=' if self.comparison == 'min' else '<='
        return f'[{mark}] {self.name}: {shown} {symbol} {self.threshold}'


# Defaults mirror docs/RESEARCH_PIPELINE.md section 5. They are deliberately
# strict: at ~40 independent observations the honest prior is that most apparent
# edges are noise, and a gate set that keeps confirming strategies is not
# measuring anything.
DEFAULT_GATES: dict[str, tuple[float, str]] = {
    'cpcv_median_sharpe': (0.5, 'min'),
    'cpcv_p05_sharpe': (0.0, 'min'),
    'pbo': (0.30, 'max'),
    'deflated_sharpe': (0.0, 'min'),
    'bootstrap_positive_fraction': (0.90, 'min'),
    'synthetic_positive_fraction': (0.60, 'min'),
    'stressed_median_sharpe': (0.0, 'min'),
    'parameter_plateau': (0.60, 'min'),
    'oos_trades': (100.0, 'min'),
    # Twice the entry cap. Entries are sized against a pessimistic liquidity
    # floor so they stay exitable, but the exit bar is whatever the barrier
    # lands in; if that routinely swallows a fifth of the volume, the strategy
    # has a capacity ceiling the backtest is not honouring.
    'max_exit_participation': (0.20, 'max'),
}


def evaluate_gates(
    measurements: Mapping[str, Optional[float]],
    *,
    thresholds: Optional[Mapping[str, tuple[float, str]]] = None,
    require_all: bool = True,
) -> tuple[bool, list[Gate]]:
    """Check measurements against the promotion thresholds.

    A missing measurement fails rather than passing silently: "we did not run
    that test" is not evidence of safety. Returns (promoted, gates).
    """
    thresholds = dict(thresholds or DEFAULT_GATES)
    gates: list[Gate] = []

    for name, (threshold, comparison) in thresholds.items():
        value = _finite(measurements.get(name), None)
        if value is None:
            gates.append(Gate(name, None, threshold, comparison, False, 'not measured'))
            continue
        passed = value >= threshold if comparison == 'min' else value <= threshold
        gates.append(Gate(name, value, threshold, comparison, bool(passed)))

    promoted = all(g.passed for g in gates) if require_all else any(g.passed for g in gates)
    return promoted, gates


def gate_report(gates: Sequence[Gate]) -> str:
    """Human-readable gate summary, failures first."""
    ordered = sorted(gates, key=lambda g: (g.passed, g.name))
    lines = [str(g) for g in ordered]
    failed = sum(1 for g in gates if not g.passed)
    verdict = 'PROMOTED' if failed == 0 else f'BLOCKED by {failed} gate(s)'
    return '\n'.join(lines + ['', verdict])
