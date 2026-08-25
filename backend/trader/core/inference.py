"""Significance tests for correlated samples, written before the numbers exist.

**Two established methods are unavailable here, and that is why this module is
its own file.**

`core/metrics.py` gets its standard errors from fold dispersion, and explicitly
rejects the breadth formula `N/(1+(N-1)rho)` as degenerate on this structure —
four offsets share one label and three symbols are ~0.62 correlated within a
window. That rejection stands.

But fold dispersion cannot carry an inference over a short calendar span either.
The Kalshi series are 69 days old; six expanding folds are ~11.4 days each and
adjacent in one regime. Measured by simulation at rho = 0.7 between folds,
"5 of 6 folds positive" is a **34.6%** event under the null and "6 of 6" is
**22.4%** — against 10.9% and 1.6% if they were independent. Fold agreement is
worth almost nothing on that span.

So neither method works, and the replacement has to be structural rather than
parametric:

**A circular block bootstrap over whole UTC days.** A day carries all three
symbols and all four offsets together, so cross-symbol and cross-offset
correlation are absorbed by construction instead of assumed at some rho. Intra-day
chaining — one window's strike is the previous window's settlement — survives
inside a block; only the midnight boundary is cut. Circular rather than moving,
because a moving-block scheme samples interior days more often than the endpoints
and wrapping costs nothing.

The block length is the one free parameter, so it is not free: both 1 day and 5
days are run and **the more conservative governs**, fixed in
`DECISION_RULE.md` Appendix A before any result was seen.

The second test here is the economic one the body of `DECISION_RULE.md` is
defined in terms of, which had never been written down as code — it existed only
as a description and as an ad-hoc script. Writing it now, while nobody knows the
answer, is the whole point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from core.baseline import clip_prob

logger = logging.getLogger(__name__)

DEFAULT_RESAMPLES = 10_000
DEFAULT_BLOCK_DAYS = (1, 5)
MIN_USABLE_DAYS = 30          # DECISION_RULE.md Appendix A voids the test below this


def _log_loss(outcome: np.ndarray, probability: np.ndarray) -> float:
    p = clip_prob(probability)
    y = np.asarray(outcome, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


@dataclass(frozen=True)
class BootstrapResult:
    """One block length's answer. `p` is one-sided against a null of zero."""

    block_days: int
    point: float
    p_value: float
    lo: float
    hi: float
    n_days: int
    n_rows: int
    n_resamples: int

    def line(self) -> str:
        return (f'  block {self.block_days:>2}d: {self.point:+.6f}  '
                f'90% CI [{self.lo:+.6f}, {self.hi:+.6f}]  p = {self.p_value:.4f}  '
                f'({self.n_days} days, {self.n_rows:,} rows)')


def circular_block_bootstrap(
    groups: Sequence[np.ndarray],
    statistic: Callable[[np.ndarray], float],
    *,
    block_days: int = 1,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260825,
) -> np.ndarray:
    """Resample whole day-blocks with replacement, recomputing `statistic`.

    `groups` is one array of row indices per day, in calendar order. Blocks wrap
    past the end so every day is drawn with equal probability — the reason for
    circular rather than moving.
    """
    n_days = len(groups)
    if n_days == 0:
        return np.empty(0)
    block_days = max(1, min(int(block_days), n_days))
    n_blocks = int(np.ceil(n_days / block_days))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_days, size=(n_resamples, n_blocks))
    offsets = np.arange(block_days)

    out = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        picked = (starts[i][:, None] + offsets[None, :]).ravel() % n_days
        picked = picked[:n_days]
        out[i] = statistic(np.concatenate([groups[d] for d in picked]))
    return out


def _day_groups(frame: pd.DataFrame, time_column: str) -> tuple[list[np.ndarray], pd.Index]:
    times = pd.to_datetime(frame[time_column], utc=True)
    days = times.dt.floor('D')
    order = pd.Index(sorted(days.unique()))
    positions = np.arange(len(frame))
    return [positions[(days == d).to_numpy()] for d in order], order


def model_minus_market(
    frame: pd.DataFrame,
    *,
    time_column: str = 'window_open',
    outcome_column: str = 'outcome',
    model_column: str = 'model_probability',
    market_column: str = 'market_probability',
    block_days: Sequence[int] = DEFAULT_BLOCK_DAYS,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260825,
) -> dict[int, BootstrapResult]:
    """The retroactive forecast test. Positive means the model beats the price.

    One result per block length; the caller takes the more conservative, which
    `governing()` does.
    """
    y = frame[outcome_column].to_numpy(dtype=float)
    q = frame[model_column].to_numpy(dtype=float)
    m = frame[market_column].to_numpy(dtype=float)

    def statistic(idx: np.ndarray) -> float:
        return _log_loss(y[idx], m[idx]) - _log_loss(y[idx], q[idx])

    groups, days = _day_groups(frame, time_column)
    if len(days) < MIN_USABLE_DAYS:
        logger.warning('%d usable days is under the %d minimum; DECISION_RULE.md '
                       'Appendix A voids the test below this', len(days), MIN_USABLE_DAYS)
    point = statistic(np.arange(len(frame)))
    results: dict[int, BootstrapResult] = {}
    for length in block_days:
        draws = circular_block_bootstrap(
            groups, statistic, block_days=length,
            n_resamples=n_resamples, seed=seed)
        lo, hi = np.percentile(draws, [5, 95])
        results[int(length)] = BootstrapResult(
            block_days=int(length), point=float(point),
            p_value=float(np.mean(draws <= 0.0)),
            lo=float(lo), hi=float(hi), n_days=len(days),
            n_rows=len(frame), n_resamples=n_resamples)
    return results


def governing(results: dict[int, BootstrapResult]) -> BootstrapResult:
    """The conservative one: the largest p-value. Fixed in advance so the block
    length cannot be chosen after seeing which is kinder."""
    return max(results.values(), key=lambda r: r.p_value)


@dataclass(frozen=True)
class PnlNullResult:
    actual: float
    expected: float
    sd: float
    p_value: float
    lo: float
    hi: float
    n_trades: int
    n_windows: int
    rho: float

    def line(self) -> str:
        return (f'  actual {self.actual:+.2f}  vs expected {self.expected:+.2f} '
                f'(sd {self.sd:.2f})  P(>= actual) = {self.p_value:.4f}  '
                f'[rho {self.rho}, {self.n_trades} trades, {self.n_windows} windows]')


def pnl_against_market_null(
    trades: pd.DataFrame,
    *,
    rho: float = 0.7,
    n_resamples: int = DEFAULT_RESAMPLES,
    seed: int = 20260825,
    window_column: str = 'window_open',
) -> PnlNullResult:
    """The economic test: is realised P&L unusual if the market is correctly priced?

    Each trade keeps its real price, size and fee. The market's de-spread mid is
    taken as the true probability. Outcomes are drawn from a Gaussian copula with
    a per-window common factor, so the three symbols in one window move together
    — which is what makes a naive per-trade binomial far too confident.

    `trades` needs `contracts`, `outlay`, `fee`, `pnl`, `p_win_market` and a
    window key.
    """
    from scipy.stats import norm

    c = trades['contracts'].to_numpy(dtype=float)
    cost = trades['outlay'].to_numpy(dtype=float) + trades['fee'].to_numpy(dtype=float)
    p = np.clip(trades['p_win_market'].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    win, lose = c - cost, -cost
    actual = float(trades['pnl'].to_numpy(dtype=float).sum())

    codes = pd.factorize(trades[window_column])[0]
    n_windows = int(codes.max()) + 1 if len(codes) else 0
    rng = np.random.default_rng(seed)
    threshold = norm.ppf(p)

    common = rng.standard_normal((n_resamples, n_windows))[:, codes]
    idio = rng.standard_normal((n_resamples, len(p)))
    z = np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idio
    won = z < threshold
    sims = (won * win + ~won * lose).sum(axis=1)
    lo, hi = np.percentile(sims, [5, 95])
    return PnlNullResult(
        actual=actual, expected=float(sims.mean()), sd=float(sims.std()),
        p_value=float(np.mean(sims >= actual)), lo=float(lo), hi=float(hi),
        n_trades=len(trades), n_windows=n_windows, rho=rho)
