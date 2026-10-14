"""Measurement, and the gates a candidate has to clear.

Everything is incremental against the baseline. The barrier arithmetic alone
beats a coin flip by a wide margin, so only the difference means anything.

Standard errors come from fold dispersion, not `N/(1+(N-1)rho)`: offsets share a
label and symbols are correlated within a window, so a breadth formula is
degenerate. Folds are weighted by window count, and the error bar uses the Kish
effective number of folds.

The gates read the market comparison first, then the forecast, then the money.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from core.baseline import (Reliability, brier, clip_prob, expit, log_loss,
                           logit, reliability)
from core.book import BookStats
from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightedMean:
    """A fold-size-weighted mean with its dispersion-based standard error."""

    mean: float
    standard_error: float
    effective_n: float

    @property
    def t(self) -> float:
        se = self.standard_error
        if not np.isfinite(self.mean) or not np.isfinite(se):
            return float('nan')
        if se > 0:
            return self.mean / se
        # Every fold agreeing exactly: the sign is certain, the size is not.
        return float(np.sign(self.mean) * np.inf) if self.mean else float('nan')

    @property
    def p_value(self) -> float:
        """One-sided P(T >= t) under zero mean, Student t on n_eff - 1 df."""
        from scipy import stats as sstats

        t = self.t
        if not np.isfinite(self.effective_n) or self.effective_n <= 1.0 or np.isnan(t):
            return float('nan')
        return float(sstats.t.sf(t, df=self.effective_n - 1.0))


def weighted_fold_mean(values, weights=None) -> WeightedMean:
    """Mean of per-fold values weighted by fold size, SE from weighted dispersion.

    The variance is the unbiased reliability-weights estimate, and the SE divides
    by the Kish effective count `(sum w)^2 / sum w^2`, so with equal weights this
    is exactly the ordinary `std(ddof=1) / sqrt(n)`.
    """
    v = np.asarray(values, dtype=float)
    w = (np.ones_like(v) if weights is None
         else np.asarray(weights, dtype=float))
    keep = np.isfinite(v) & np.isfinite(w) & (w > 0)
    v, w = v[keep], w[keep]
    if not len(v):
        return WeightedMean(float('nan'), float('nan'), 0.0)
    p = w / w.sum()
    mean = float(np.sum(p * v))
    sum_p2 = float(np.sum(p ** 2))
    n_eff = 1.0 / sum_p2
    if len(v) < 2 or sum_p2 >= 1.0:
        return WeightedMean(mean, float('nan'), n_eff)
    variance = float(np.sum(p * (v - mean) ** 2)) / (1.0 - sum_p2)
    return WeightedMean(mean, float(np.sqrt(variance * sum_p2)), n_eff)


# Block bootstrap settings. A fixed seed makes a gate value a function of its
# data only, so the same run always gets the same verdict.
BOOTSTRAP_RESAMPLES = 2_000
BOOTSTRAP_SEED = 0
# Two-sided confidence level; the lower bound is its lower end.
CONFIDENCE_LEVEL = 0.90


def block_bootstrap_means(values, blocks, *, resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int = BOOTSTRAP_SEED) -> np.ndarray:
    """Row means of `values` over resamples that draw whole blocks.

    Rows in one block (the symbols and offsets of one window) are not
    independent, so they are drawn together. Non-finite values are dropped.
    Empty when fewer than two blocks remain.
    """
    v = np.asarray(values, dtype=float)
    b = pd.Series(np.asarray(blocks, dtype=object))
    keep = np.isfinite(v) & b.notna().to_numpy()
    if not keep.any():
        return np.empty(0)
    codes, _ = pd.factorize(b[keep])
    sums = np.bincount(codes, weights=v[keep])
    counts = np.bincount(codes).astype(float)
    k = len(sums)
    if k < 2:
        return np.empty(0)
    rng = np.random.default_rng(seed)
    out = np.empty(int(resamples))
    # Draw in batches so a large sample does not need a resamples x k matrix.
    step = max(1, 2_000_000 // k)
    for start in range(0, len(out), step):
        n = min(step, len(out) - start)
        draw = rng.integers(0, k, size=(n, k))
        out[start:start + n] = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return out


def bootstrap_interval(draws: np.ndarray,
                       level: float = CONFIDENCE_LEVEL) -> tuple[float, float]:
    """(lower, upper) of the two-sided `level` interval, NaN without draws."""
    if not len(draws):
        return float('nan'), float('nan')
    tail = (1.0 - float(level)) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def trade_gate_values(trades: Optional[pd.DataFrame]) -> dict[str, float]:
    """`realised_edge_lcb` and `overconfidence_abs` from settled trades.

    Realised edge is the per-trade estimator of `realised_edge_pp`: win
    (payout > 0) minus effective cost per contract, in pp. `model_probability`
    on a trade is the probability of the side taken.
    """
    out = {'realised_edge_lcb': float('nan'), 'overconfidence_abs': float('nan')}
    needed = ('window_open', 'payout', 'outlay', 'contracts', 'model_probability')
    if trades is None or trades.empty or any(c not in trades.columns for c in needed):
        return out
    won = (pd.to_numeric(trades['payout'], errors='coerce') > 0).astype(float)
    cost = (pd.to_numeric(trades['outlay'], errors='coerce')
            / pd.to_numeric(trades['contracts'], errors='coerce'))
    edge_pp = ((won - cost) * 100.0).to_numpy(dtype=float)
    lower, _ = bootstrap_interval(block_bootstrap_means(edge_pp, trades['window_open']))
    out['realised_edge_lcb'] = lower
    predicted = pd.to_numeric(trades['model_probability'], errors='coerce')
    known = predicted.notna()
    if known.any():
        out['overconfidence_abs'] = float(abs(predicted[known].mean()
                                              - won[known].mean()))
    return out


def log_loss_skill(outcome: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    """Baseline log loss minus model log loss. Positive means the model helped."""
    return log_loss(outcome, baseline) - log_loss(outcome, model)


def brier_skill(outcome: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    base = brier(outcome, baseline)
    return (base - brier(outcome, model)) / base if base > 0 else float('nan')


def resolve_max_deviation(per_fold, pooled) -> tuple:
    """(value, why) for `calibration_max_deviation`, preferring the per-fold max.

    `worst_deviation` is NaN when no bin reaches its minimum row count, which
    is normal for small folds. Every scored row is out of sample, so the pooled
    rows are a valid fallback. The per-fold maximum stays preferred: it is
    stricter. With nothing measurable, the value is NaN and fails, and the
    reason names the sample size.
    """
    values = [float(v) for v in per_fold if v is not None and np.isfinite(v)]
    if values:
        return max(values), f'worst adequately-populated bin across {len(values)} fold(s)'
    if pooled is not None and np.isfinite(pooled):
        return float(pooled), ('pooled across folds: no single fold had a bin '
                               'with enough rows to measure')
    return float('nan'), ('not measurable: no calibration bin reached the '
                          'minimum row count, in any fold or pooled')


@dataclass
class FoldEvaluation:
    """One fold's out-of-sample measurement."""

    index: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_rows: int
    n_windows: int
    model_log_loss: float
    baseline_log_loss: float
    model_brier: float
    baseline_brier: float
    model_ece: float
    baseline_ece: float
    residual_scale: float
    control_gain_share: float
    reliability_table: Optional[Reliability] = None
    stats: Optional[BookStats] = None
    per_offset: Optional[pd.DataFrame] = None
    # Rows whose prediction or outcome was not finite. Counted, so a data hole
    # reads as a data hole, not as no skill.
    n_non_finite: int = 0
    # The worst deviation in any adequately-populated reliability bin. Mean ECE
    # cannot see the traded band; this can.
    model_max_deviation: float = float('nan')

    @property
    def skill(self) -> float:
        return self.baseline_log_loss - self.model_log_loss

    @property
    def brier_skill(self) -> float:
        return ((self.baseline_brier - self.model_brier) / self.baseline_brier
                if self.baseline_brier > 0 else float('nan'))

    def line(self) -> str:
        money = ''
        if self.stats is not None:
            money = (f' | {self.stats.n_trades:,} trades '
                     f'{self.stats.total_return:+.2%} Sharpe {self.stats.sharpe:+.2f}')
        return (f'  fold {self.index} [{self.test_start:%Y-%m-%d}..{self.test_end:%Y-%m-%d}] '
                f'{self.n_windows:,}w  skill {self.skill:+.5f}  '
                f'ECE {self.model_ece:.4f} (base {self.baseline_ece:.4f})  '
                f'alpha {self.residual_scale:.3f}{money}')


def evaluate_fold(
    index: int,
    test: pd.DataFrame,
    model_probability: np.ndarray,
    baseline_probability: np.ndarray,
    *,
    residual_scale: float,
    control_gain_share: float,
    stats: Optional[BookStats] = None,
) -> FoldEvaluation:
    from core.cv import effective_observations

    outcome = test['outcome'].to_numpy(dtype=float)

    # A row with no volatility estimate has no forecast (e.g. after a data
    # outage). Live refuses it as NOT_FINITE; here it is excluded and counted,
    # and the count reaches a gate as a share. Left in, NaN would poison every
    # fold statistic.
    finite = (np.isfinite(outcome) & np.isfinite(model_probability)
              & np.isfinite(baseline_probability))
    n_non_finite = int((~finite).sum())
    if n_non_finite:
        logger.info(
            'fold %d: %d of %d rows carry no forecast (a NaN sigma, usually the '
            'tail of a data outage) and are excluded from the metrics',
            index, n_non_finite, len(outcome))
    outcome = outcome[finite]
    model_probability = np.asarray(model_probability)[finite]
    baseline_probability = np.asarray(baseline_probability)[finite]
    test = test.loc[finite]

    model_reliability = reliability(outcome, model_probability)
    per_offset = None
    if 'offset' in test.columns:
        frame = test.assign(_m=model_probability, _b=baseline_probability)
        rows = []
        for offset, part in frame.groupby('offset'):
            y = part['outcome'].to_numpy(dtype=float)
            rows.append({
                'offset': int(offset), 'n': len(part),
                'skill': log_loss_skill(y, part['_m'].to_numpy(), part['_b'].to_numpy()),
                'mean_abs_correction_pp': float(
                    np.mean(np.abs(part['_m'] - part['_b'])) * 100.0),
            })
        per_offset = pd.DataFrame(rows)

    return FoldEvaluation(
        index=index,
        test_start=pd.Timestamp(test['window_open'].min()),
        test_end=pd.Timestamp(test['window_open'].max()),
        n_rows=len(test), n_windows=effective_observations(test),
        model_log_loss=log_loss(outcome, model_probability),
        baseline_log_loss=log_loss(outcome, baseline_probability),
        model_brier=brier(outcome, model_probability),
        baseline_brier=brier(outcome, baseline_probability),
        model_ece=model_reliability.expected_calibration_error,
        baseline_ece=reliability(outcome, baseline_probability).expected_calibration_error,
        residual_scale=residual_scale, control_gain_share=control_gain_share,
        reliability_table=model_reliability,
        n_non_finite=n_non_finite,
        model_max_deviation=model_reliability.worst_deviation(),
        stats=stats, per_offset=per_offset,
    )


@dataclass
class EvaluationReport:
    """Every fold, plus the aggregate and the continuous-deployment book."""

    folds: list[FoldEvaluation]
    continuous: Optional[BookStats] = None
    config_provenance: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    # How the folds were actually cut. `config_provenance['n_folds']` is the
    # configured value, which the calendar scheme does not treat as a count.
    fold_geometry: dict = field(default_factory=dict)
    # Daily P&L moments of the continuous book, for the deflated Sharpe ratio.
    sharpe_moments: dict = field(default_factory=dict)
    # `trade_gate_values` of the continuous book's trades.
    trade_metrics: dict = field(default_factory=dict)

    # ---- forecast ------------------------------------------------------
    @property
    def skills(self) -> np.ndarray:
        return np.array([f.skill for f in self.folds], dtype=float)

    @property
    def fold_weights(self) -> np.ndarray:
        """Window counts. Folds vary in size, so equal weights would let a thin
        fold move the mean as much as a thick one."""
        return np.array([max(int(getattr(f, 'n_windows', 0) or 0), 0)
                         for f in self.folds], dtype=float)

    @property
    def weighted_skill(self) -> WeightedMean:
        weights = self.fold_weights
        if not len(weights) or weights.sum() <= 0:
            weights = None
        return weighted_fold_mean(self.skills, weights)

    @property
    def unweighted_skill(self) -> WeightedMean:
        """Every fold weighted equally."""
        return weighted_fold_mean(self.skills)

    @property
    def mean_skill(self) -> float:
        return self.weighted_skill.mean if self.folds else float('nan')

    @property
    def skill_standard_error(self) -> float:
        """From weighted fold dispersion. See the module docstring."""
        return self.weighted_skill.standard_error

    @property
    def effective_folds(self) -> float:
        return self.weighted_skill.effective_n

    @property
    def skill_t(self) -> float:
        return self.weighted_skill.t

    @property
    def skill_p_value(self) -> float:
        """One-sided p of the weighted mean skill against zero.

        A point estimate `>= 0` is cleared by half of all nulls; this asks
        whether the skill is distinguishable from fold noise.
        """
        return self.weighted_skill.p_value

    @property
    def folds_positive(self) -> int:
        return int((self.skills > 0).sum())

    @property
    def folds_total(self) -> int:
        return len(self.folds)

    @property
    def fold_drawdowns(self) -> list[float]:
        """Each fold's own worst drawdown, on its own book from the same start.

        A fold is a fixed block, so this does not ratchet with history.
        """
        out = []
        for fold in self.folds:
            stats = getattr(fold, 'stats', None)
            value = getattr(stats, 'max_drawdown', None) if stats else None
            if value is not None and np.isfinite(value):
                out.append(float(value))
        return out

    @property
    def median_fold_drawdown(self) -> float:
        """The gated drawdown statistic, because it holds still.

        `max_drawdown` on the continuous curve is a running maximum and can only
        grow as days are added. The median over fixed-length fold blocks does
        not grow with the sample. The continuous figure stays as a diagnostic.
        """
        values = self.fold_drawdowns
        return float(np.median(values)) if values else float('nan')

    @property
    def worst_fold_drawdown(self) -> float:
        """Reported, not gated: a max over folds ratchets as folds accumulate."""
        values = self.fold_drawdowns
        return float(np.max(values)) if values else float('nan')

    @property
    def sign_agreement_p_value(self) -> float:
        """P(at least this many folds positive | no skill), each fold a coin flip."""
        from scipy import stats as sstats
        n, k = self.folds_total, self.folds_positive
        if n == 0:
            return float('nan')
        return float(sstats.binom.sf(k - 1, n, 0.5))

    @property
    def calibration_vs_baseline(self) -> float:
        """Median fold (model ECE - baseline ECE). Positive means worse.

        Asks whether the correction makes calibration worse than the arithmetic
        it corrects. An absolute ECE bar on short folds measures sample size,
        because binned ECE is biased upward at small n.
        Median, not max: a max over folds grows with the fold count.
        """
        pairs = [(f.model_ece, f.baseline_ece) for f in self.folds
                 if np.isfinite(f.model_ece) and np.isfinite(f.baseline_ece)]
        if not pairs:
            return float('nan')
        return float(np.median([m - b for m, b in pairs]))

    @property
    def max_ece(self) -> float:
        # `np.max`, not the builtin: builtin `max` with a NaN is order-dependent,
        # so a NaN fold could vanish. `Gate.passed` fails closed on NaN.
        if not self.folds:
            return float('nan')
        return float(np.max(np.asarray([f.model_ece for f in self.folds], dtype=float)))

    @property
    def max_calibration_deviation(self) -> float:
        """The worst adequately-populated calibration bin.

        Per fold first, pooled as a fallback. See `resolve_max_deviation`.
        """
        if not self.folds:
            return float('nan')
        value, why = resolve_max_deviation(
            [f.model_max_deviation for f in self.folds],
            self.pooled_max_deviation)
        self._max_deviation_reason = why
        return value

    @property
    def pooled_max_deviation(self) -> float:
        """The same statistic on every scored row at once.

        Valid because every row is out of sample, whichever fold produced it.
        """
        tables = [f.reliability_table for f in self.folds
                  if getattr(f, 'reliability_table', None) is not None]
        if not tables:
            return float('nan')
        counts = np.sum([t.count for t in tables], axis=0)
        populated = counts >= 500
        if not populated.any():
            return float('nan')
        # Row-weighted, so a bin's pooled figure is the average an observation saw.
        pred = np.sum([t.predicted * t.count for t in tables], axis=0)
        obs = np.sum([t.observed * t.count for t in tables], axis=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            pred = np.where(counts > 0, pred / counts, np.nan)
            obs = np.where(counts > 0, obs / counts, np.nan)
        return float(np.nanmax(np.abs(pred[populated] - obs[populated])))

    @property
    def total_non_finite(self) -> int:
        return int(sum(f.n_non_finite for f in self.folds))

    @property
    def non_finite_share(self) -> float:
        """Unscoreable rows as a fraction of all rows offered.

        A share, not a count: an outage leaves some windows without a volatility
        estimate and should pass. A large share means a lookback or embargo
        mistake.
        """
        rows = sum(f.n_rows for f in self.folds) + self.total_non_finite
        return (self.total_non_finite / rows) if rows else float('nan')

    @property
    def mean_residual_scale(self) -> float:
        """Reported, not gated. See `median_residual_scale`."""
        return float(np.mean([f.residual_scale for f in self.folds])) if self.folds else float('nan')

    @property
    def median_residual_scale(self) -> float:
        """What the `residual_scale` gate reads.

        The median, not the mean, so one runaway fold cannot clear the gate for
        the rest. A fold on a clip boundary is the optimiser giving up, not a
        fit, so it is logged. With two folds the median is the mean. This gate is
        a backstop; `log_loss_skill` and the fold tests catch a null.
        """
        if not self.folds:
            return float('nan')
        values = np.asarray([f.residual_scale for f in self.folds], dtype=float)
        at_bound = int(np.sum(np.isclose(values, 2.0) | np.isclose(values, 0.0)))
        if at_bound:
            logger.warning(
                '%d of %d folds put the shrinkage on a clip boundary (%s). A '
                'boundary is the optimiser giving up on the parameterisation, not '
                'a fitted value.', at_bound, len(values),
                ', '.join(f'{v:.3f}' for v in values))
        return float(np.median(values))

    @property
    def max_control_gain_share(self) -> float:
        if not self.folds:
            return float('nan')
        return float(np.max(np.asarray(
            [f.control_gain_share for f in self.folds], dtype=float)))

    @property
    def total_windows(self) -> int:
        return int(sum(f.n_windows for f in self.folds))

    # ---- money ---------------------------------------------------------
    @property
    def traded_folds(self) -> list[BookStats]:
        return [f.stats for f in self.folds if f.stats is not None]

    def per_offset(self) -> pd.DataFrame:
        frames = [f.per_offset.assign(fold=f.index) for f in self.folds if f.per_offset is not None]
        if not frames:
            return pd.DataFrame()
        allrows = pd.concat(frames, ignore_index=True)
        return allrows.groupby('offset').agg(
            n=('n', 'sum'), mean_skill=('skill', 'mean'),
            folds_positive=('skill', lambda s: int((s > 0).sum())),
            folds=('skill', 'size'),
            mean_abs_correction_pp=('mean_abs_correction_pp', 'mean'),
        ).reset_index()

    def gate_values(self) -> dict[str, float]:
        """Every gate value the report can supply, plus reported metrics
        (`windows_evaluated`, `total_return`, `sharpe`, `max_drawdown`)."""
        continuous = self.continuous
        return {
            'log_loss_skill': self.mean_skill,
            'log_loss_skill_p': self.skill_p_value,
            'folds_skill_positive': float(self.folds_positive),
            'sign_agreement_p': self.sign_agreement_p_value,
            'calibration_error': self.max_ece,
            'calibration_vs_baseline': self.calibration_vs_baseline,
            'calibration_max_deviation': self.max_calibration_deviation,
            'non_finite_share': self.non_finite_share,
            'residual_scale': self.median_residual_scale,
            'control_gain_share': self.max_control_gain_share,
            'windows_evaluated': float(self.total_windows),
            'trades': float(continuous.n_trades) if continuous else 0.0,
            'coverage': continuous.coverage if continuous else float('nan'),
            'realised_edge_pp': continuous.realised_edge_pp if continuous else float('nan'),
            'realised_edge_lcb': float((self.trade_metrics or {}).get(
                'realised_edge_lcb', float('nan'))),
            'overconfidence_abs': float((self.trade_metrics or {}).get(
                'overconfidence_abs', float('nan'))),
            'total_return': continuous.total_return if continuous else float('nan'),
            'sharpe': continuous.sharpe if continuous else float('nan'),
            'sharpe_implausible': (
                1.0 if (continuous and np.isfinite(continuous.sharpe)
                        and continuous.sharpe > IMPLAUSIBLE_SHARPE) else 0.0),
            'max_drawdown': continuous.max_drawdown if continuous else float('nan'),
            'median_fold_drawdown': self.median_fold_drawdown,
            'worst_fold_drawdown': self.worst_fold_drawdown,
            'halted': 1.0 if (continuous and continuous.halted) else 0.0,
        }

    def summary(self) -> str:
        plain = self.unweighted_skill
        lines = [
            f'{self.folds_total} folds, {self.total_windows:,} out-of-sample windows',
            *[f.line() for f in self.folds],
            '',
            f'  log loss skill {self.mean_skill:+.5f} +/- {self.skill_standard_error:.5f} '
            f'(t = {self.skill_t:+.2f}, one-sided p = {self.skill_p_value:.4f}, '
            f'{self.effective_folds:.1f} effective folds; window-weighted)',
            f'  unweighted     {plain.mean:+.5f} +/- {plain.standard_error:.5f} '
            f'(t = {plain.t:+.2f})',
            f'  {self.folds_positive}/{self.folds_total} folds '
            f'positive (sign p = {self.sign_agreement_p_value:.3f})',
            f'  worst-fold calibration error {self.max_ece:.4f} | '
            f'mean alpha {self.mean_residual_scale:.3f} | '
            f'worst control gain share {self.max_control_gain_share:.1%}',
        ]
        if self.continuous is not None:
            lines += ['', '  continuous book: ' + self.continuous.summary().replace('\n', '\n  ')]
        if self.notes:
            lines += [''] + [f'  note: {n}' for n in self.notes]
        return '\n'.join(lines)


# ---- the market as the benchmark -----------------------------------------

# Below this the comparison is an anecdote.
MIN_MARKET_WINDOWS = 2_000

# A quote older than this is not the price at the decision instant, and
# beating it is not skill. Shares `Config.max_quote_age_seconds` with
# `core.quotes` and the live order check, so money and forecast use one sample.
MAX_QUOTE_AGE_SECONDS = float(DEFAULT_CONFIG.max_quote_age_seconds)

# `fold` is the walk-forward fold of a backtest row. `venue_labelled` is True
# where `outcome` is the venue's own settlement.
MARKET_COLUMNS = ('symbol', 'window_open', 'offset', 'market', 'baseline',
                  'model', 'outcome', 'decision_time', 'fold', 'venue_labelled')


def market_frame(rows: Iterable[Sequence]) -> pd.DataFrame:
    """Rows from `PgWriter.scored_against_market()` or
    `market_rows_from_scored`, cleaned.

    Tolerates rows that stop before any trailing column. Rows from the serving
    store carry neither `fold` nor `venue_labelled`.
    """
    listed = list(rows)
    width = len(listed[0]) if listed else len(MARKET_COLUMNS)
    frame = pd.DataFrame(listed, columns=list(MARKET_COLUMNS[:width]))
    return frame.dropna(subset=['market', 'baseline', 'model', 'outcome'])


def market_slice(part: pd.DataFrame, label: str) -> dict:
    """Log loss and Brier for the price, the arithmetic and the model."""
    y = part['outcome'].to_numpy(dtype=float)
    out: dict = {'slice': label, 'n': len(part)}
    for name in ('market', 'baseline', 'model'):
        p = part[name].to_numpy(dtype=float)
        out[f'{name}_ll'] = log_loss(y, p)
        out[f'{name}_brier'] = brier(y, p)
    # Positive means our probability is a better forecast than the price.
    out['model_minus_market'] = out['market_ll'] - out['model_ll']
    out['baseline_minus_market'] = out['market_ll'] - out['baseline_ll']
    return out


def market_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    """The per-slice table: overall, then by symbol, then by offset."""
    if frame.empty:
        return pd.DataFrame()
    parts = [market_slice(frame, 'all')]
    for symbol, part in frame.groupby('symbol'):
        parts.append(market_slice(part, f'symbol {symbol}'))
    for offset, part in frame.groupby('offset'):
        parts.append(market_slice(part, f'offset +{int(offset)}m'))
    return pd.DataFrame(parts)


#: Observers whose quote was watched rather than reconstructed afterwards.
#: `quote_observer` carries this per row (see `core/quotes.py`); `backfill` is
#: Predexon's after-the-fact snapshot and is excluded when `recorded_only`.
LIVE_OBSERVERS = ('live_touch', 'live_ws', 'live')


def market_rows_from_scored(
        frame: pd.DataFrame, *,
        max_quote_age_seconds: float = MAX_QUOTE_AGE_SECONDS,
        entry_offsets=None,
        recorded_only: bool = False) -> list:
    """`MARKET_COLUMNS` rows from a backtest that carries recorded quotes.

    The market's forecast is the mid. Scoring the market at its ask would hand
    the model half the spread as apparent skill.
    Rows without a quote are dropped, never defaulted to the baseline.
    """
    if frame is None or not len(frame):
        return []
    needed = ('symbol', 'window_open', 'offset', 'market_probability',
              'baseline_probability', 'model_probability')
    if any(c not in frame.columns for c in needed):
        return []
    part = frame.copy()
    # Only quotes somebody watched, when asked. `backfill` is a reconstruction;
    # only recorded quotes answer "would this model have beaten the live book".
    if recorded_only:
        if 'quote_observer' not in part.columns:
            return []
        part = part[part['quote_observer'].isin(LIVE_OBSERVERS)]
        if not len(part):
            return []
    # Grade on the venue's label wherever we hold it. Our outcome and the
    # baseline both come from Coinbase bars, so our label would hand the
    # baseline its noise as free skill.
    part['_venue_labelled'] = False
    if 'venue_outcome' in part.columns:
        venue = pd.to_numeric(part['venue_outcome'], errors='coerce')
        part['_venue_labelled'] = venue.notna()
        part['outcome'] = venue.fillna(
            pd.to_numeric(part.get('outcome'), errors='coerce'))
    part = part.dropna(subset=list(needed) + ['outcome'])
    # Staleness is not skill: drop rows whose quote is older than the limit.
    # Rows with no age are kept.
    # Only the offsets that can open a position, so the gate measures the
    # policy that trades. None pools every offset.
    if entry_offsets is not None and 'offset' in part.columns:
        wanted = {int(o) for o in entry_offsets}
        part = part[pd.to_numeric(part['offset'], errors='coerce')
                    .astype('Int64').isin(wanted)]
        if part.empty:
            return []
    if 'quote_age_seconds' in part.columns:
        age = pd.to_numeric(part['quote_age_seconds'], errors='coerce').abs()
        part = part[age.isna() | (age <= max_quote_age_seconds)]
    if not len(part):
        return []
    decision = (part['decision_time'] if 'decision_time' in part.columns
                else part['window_open'])
    fold = (part['fold'] if 'fold' in part.columns
            else pd.Series(None, index=part.index, dtype=object))
    # Positional, matching MARKET_COLUMNS. Out of order would silently swap
    # the model's probability with the baseline's.
    return list(zip(part['symbol'], part['window_open'], part['offset'],
                    part['market_probability'], part['baseline_probability'],
                    part['model_probability'], part['outcome'], decision,
                    fold, part['_venue_labelled']))


# Length of the calendar block that stands in for a fold where a row has none.
MARKET_FOLD_DAYS = 7


def market_fold_test(frame: pd.DataFrame) -> tuple[float, int]:
    """(one-sided p, fold count) for `model_minus_market` above zero.

    Reads venue-labelled rows only. A row without `venue_labelled` comes from
    the serving store, whose `outcome` is the venue's settlement where one
    exists, so it counts. Per fold, `model_minus_market` is market log loss
    minus model log loss; folds are weighted by window count. A row without a
    fold takes its `MARKET_FOLD_DAYS` calendar block.
    """
    part = frame
    if 'venue_labelled' in part.columns:
        labelled = part['venue_labelled']
        part = part[labelled.isna() | labelled.fillna(False).astype(bool)]
    if part.empty:
        return float('nan'), 0
    opens = pd.to_datetime(part['window_open'], utc=True, errors='coerce',
                           format='mixed')
    block = opens.dt.floor(f'{MARKET_FOLD_DAYS}D').astype(str).where(
        opens.notna(), part['window_open'].astype(str))
    block = 'block ' + block
    fold = (part['fold'] if 'fold' in part.columns
            else pd.Series(None, index=part.index, dtype=object))
    key = fold.astype(object).where(fold.notna(), block).astype(str)
    values, weights = [], []
    for _, rows in part.groupby(key, sort=False):
        values.append(market_slice(rows, 'fold')['model_minus_market'])
        weights.append(rows.drop_duplicates(['symbol', 'window_open']).shape[0])
    return weighted_fold_mean(values, weights).p_value, len(values)


def calibration_noise_null(market, outcome, correction_sd, *,
                           draws: int = 200, seed: int = 0) -> float:
    """Mean `model_ece - market_ece` for a correction of the same size and no signal.

    ECE uses absolute deviations, so any unbiased perturbation of a calibrated
    forecast raises measured ECE. A zero bar would charge the model for moving
    at all. The gate subtracts this null, which is generated at the candidate's
    own correction magnitude.
    """
    market = pd.to_numeric(pd.Series(market), errors='coerce').to_numpy(dtype=float)
    outcome = pd.to_numeric(pd.Series(outcome), errors='coerce').to_numpy(dtype=float)
    keep = np.isfinite(market) & np.isfinite(outcome)
    market, outcome = market[keep], outcome[keep]
    if not len(market) or not np.isfinite(correction_sd) or correction_sd <= 0:
        return 0.0
    base = _pooled_ece(market, outcome)
    if not np.isfinite(base):
        return 0.0
    z = logit(clip_prob(market))
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(draws)):
        drawn = _pooled_ece(expit(z + rng.normal(0.0, float(correction_sd), z.size)),
                            outcome)
        if np.isfinite(drawn):
            out.append(drawn - base)
    return float(np.mean(out)) if out else 0.0


def _pooled_ece(pred, outcome) -> float:
    """Count-weighted expected calibration error over every populated bin.

    Far lower variance than a worst-bin statistic, so it is the stable way to
    compare two forecasters on the same rows.
    """
    from core.baseline import reliability

    pred = pd.to_numeric(pd.Series(pred), errors='coerce')
    outcome = pd.to_numeric(pd.Series(outcome), errors='coerce')
    keep = pred.notna() & outcome.notna()
    if not keep.any():
        return float('nan')
    return float(reliability(outcome[keep].to_numpy(),
                             pred[keep].to_numpy()).expected_calibration_error)


def _worst_populated_bin(pred, outcome, *, bins: Optional[int] = None,
                        min_count: int = 100) -> float:
    """Largest |actual - predicted| over adequately populated bins, or NaN.

    Uses `core.baseline.reliability`'s tail-refined edges, the same statistic as
    `calibration_max_deviation`. `min_count` is 100, not 500: live-recorded rows
    are fewer, and a 500 floor would return NaN and fail.
    Pass `bins` to force equal-width bucketing.
    """
    pred = pd.to_numeric(pd.Series(pred), errors='coerce')
    outcome = pd.to_numeric(pd.Series(outcome), errors='coerce')
    keep = pred.notna() & outcome.notna()
    if not keep.any():
        return float('nan')
    frame = pd.DataFrame({'p': pred[keep].to_numpy(),
                          'y': outcome[keep].to_numpy()})
    if bins is None:
        # `worst_deviation` is the statistic `calibration_max_deviation` reads;
        # reuse it rather than re-deriving it.
        from core.baseline import reliability as _reliability

        return float(_reliability(frame['y'].to_numpy(),
                                  frame['p'].to_numpy())
                     .worst_deviation(min_count=min_count))
    edges = np.linspace(0.0, 1.0, bins + 1)
    frame['bin'] = pd.cut(frame['p'], edges, include_lowest=True)
    grouped = frame.groupby('bin', observed=True).agg(
        n=('y', 'size'), pred=('p', 'mean'), actual=('y', 'mean'))
    grouped = grouped[grouped['n'] >= min_count]
    if grouped.empty:
        return float('nan')
    return float((grouped['actual'] - grouped['pred']).abs().max())


def market_gate_values(rows: Iterable[Sequence]) -> dict[str, float]:
    """What `DEFAULT_GATES` reads about the market, from recorded quotes.

    A backtest priced at the baseline cannot answer "beat the market"; only
    recorded quotes can. Empty input returns values that fail: `market_windows`
    is 0 and `model_minus_market` is NaN.
    """
    frame = market_frame(rows)
    if frame.empty:
        return {'market_windows': 0.0,
                'model_minus_market': float('nan'),
                'model_minus_market_p': float('nan'),
                'min_folds': 0.0,
                'baseline_minus_market': float('nan'),
                'market_max_deviation': float('nan'),
                'calibration_noise_null': float('nan'),
                'calibration_vs_market': float('nan')}
    windows = float(frame.drop_duplicates(['symbol', 'window_open']).shape[0])
    overall = market_slice(frame, 'all')
    # Calibration relative to the price, on the same rows: the model must be at
    # least as calibrated as what it trades against. An absolute bar fails the
    # market itself.
    # Pooled ECE, not the worst bin: the worst bin is too noisy to gate.
    # Gated against a matched noise null, not zero (see `calibration_noise_null`).
    model_ece = _pooled_ece(frame['model'], frame['outcome'])
    market_ece = _pooled_ece(frame['market'], frame['outcome'])
    correction_sd = float(np.std(
        logit(clip_prob(pd.to_numeric(frame['model'], errors='coerce')))
        - logit(clip_prob(pd.to_numeric(frame['market'], errors='coerce')))))
    null = calibration_noise_null(frame['market'], frame['outcome'], correction_sd)
    # Reported only: informative about the venue, too noisy to gate.
    market_dev = _worst_populated_bin(frame['market'], frame['outcome'])
    p_value, n_folds = market_fold_test(frame)
    return {'market_windows': windows,
            'model_minus_market': float(overall['model_minus_market']),
            'model_minus_market_p': float(p_value),
            'min_folds': float(n_folds),
            'baseline_minus_market': float(overall['baseline_minus_market']),
            'market_max_deviation': market_dev,
            'calibration_noise_null': float(null),
            'calibration_vs_market': float(model_ece - market_ece - null)}


# ---- gates ---------------------------------------------------------------

@dataclass(frozen=True)
class GateSpec:
    """One gate as frozen in `gate_spec.lock.json`.

    `definition_version` names the computation behind `name`. Bump it whenever
    the value's meaning changes, even if the threshold does not, so the spec
    hash changes and promotion refuses until the change is accepted.
    """

    name: str
    threshold: float
    direction: str            # 'min' passes at or above, 'max' at or below
    definition_version: int = 1


# The canonical gate list, in reading order: the market, the forecast, the money.
# A candidate that fails a forecast gate should not have its Sharpe discussed.
GATE_SPEC: tuple[GateSpec, ...] = (
    # --- the market: the price is the counterparty, not F(x/sigma) ---
    #
    # A backtest priced at the baseline cannot compute these, so they read NaN
    # and fail until enough quotes are recorded. `--force` with a written reason
    # is the way past, and the ledger records it.
    GateSpec('market_windows', float(MIN_MARKET_WINDOWS), 'min', 1),
    GateSpec('model_minus_market', 0.0, 'min', 1),
    # One-sided t-test of the window-weighted fold mean of model_minus_market
    # against zero, on venue-labelled rows. A point estimate >= 0 passes half of all nulls.
    GateSpec('model_minus_market_p', 0.05, 'max', 1),
    # Weekly folds that carry market rows. The p-value above has n_eff - 1
    # degrees of freedom, so it needs folds.
    GateSpec('min_folds', 6.0, 'min', 1),
    # --- the forecast ---
    GateSpec('log_loss_skill', 0.0, 'min', 2),
    # Significance, not just sign: one-sided Student-t on the window-weighted
    # fold mean, n_eff - 1 degrees of freedom.
    GateSpec('log_loss_skill_p', 0.05, 'max', 1),
    # Fold agreement at constant significance: the binomial probability of at
    # least this many positive folds if each were a coin flip. A fixed count
    # would weaken as folds accumulate. 0.11 equals five of six (p = 0.109).
    GateSpec('sign_agreement_p', 0.11, 'max', 1),
    # A sanity floor, not a quality bar: binned ECE on short folds is biased
    # upward, and the baseline itself fails a tight absolute bar. The relative
    # gates do the discriminating.
    GateSpec('calibration_error', 0.10, 'max', 1),
    # Does the correction make calibration worse than the arithmetic it corrects?
    # Median over folds. 0.005 sits below the fold-to-fold spread, so it catches
    # a systematic degradation without rejecting fold noise.
    GateSpec('calibration_vs_baseline', 0.005, 'max', 1),
    # The worst adequately-populated bin; mean ECE averages away the traded band.
    # A loose rail: 500 rows at p=0.9 carry a 1.3pp standard error, so it bounds
    # damage and cannot certify the edge. `calibration_vs_market` is the binding claim.
    GateSpec('calibration_max_deviation', 0.10, 'max', 1),
    GateSpec('calibration_vs_market', 0.0, 'max', 1),
    # A data hole must report as a data hole (`nan > x` is False). A share, not a
    # count: an outage is not a defect, a large share is.
    GateSpec('non_finite_share', 0.001, 'max', 1),
    GateSpec('residual_scale', 0.25, 'min', 1),
    GateSpec('control_gain_share', 0.30, 'max', 1),
    # --- the money ---
    # `windows_evaluated`, `total_return`, `sharpe` and `max_drawdown` are
    # reported, not gated.
    GateSpec('trades', 200.0, 'min', 1),
    GateSpec('coverage', 0.0005, 'min', 1),
    GateSpec('realised_edge_pp', 0.0, 'min', 1),
    # Lower end of the 90% interval of realised edge per contract on traded
    # rows, block-bootstrapped by window_open.
    GateSpec('realised_edge_lcb', 0.0, 'min', 1),
    # |mean predicted win probability - realised win rate| on traded rows.
    GateSpec('overconfidence_abs', 0.015, 'max', 1),
    GateSpec('sharpe_implausible', 0.0, 'max', 1),
    # Median over folds of each fold's own drawdown. Each fold is a fixed block
    # with its own book, so the statistic does not grow with the sample.
    GateSpec('median_fold_drawdown', 0.20, 'max', 1),
    GateSpec('halted', 0.0, 'max', 1),
)

# name -> (threshold, direction), derived so there is one list to change.
DEFAULT_GATES: dict[str, tuple[float, str]] = {
    g.name: (g.threshold, g.direction) for g in GATE_SPEC}

GATE_SPEC_LOCK = Path(__file__).with_name('gate_spec.lock.json')


def gate_spec(gates: Optional[dict[str, tuple[float, str]]] = None) -> list[dict]:
    """The spec as plain rows.

    With `gates`, the spec of those gates, taking definition versions from
    `GATE_SPEC` (0 for an unknown name)."""
    if gates is None:
        return [asdict(g) for g in GATE_SPEC]
    versions = {g.name: g.definition_version for g in GATE_SPEC}
    return [{'name': name, 'threshold': float(threshold), 'direction': direction,
             'definition_version': versions.get(name, 0)}
            for name, (threshold, direction) in gates.items()]


def gate_spec_hash(gates: Optional[dict[str, tuple[float, str]]] = None) -> str:
    rows = [{**row, 'threshold': float(row['threshold'])} for row in gate_spec(gates)]
    canonical = json.dumps(rows, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_gate_lock(path: Optional[Path] = None) -> dict:
    """The committed lock, or {} when it is missing or unreadable, which fails
    every comparison."""
    try:
        return json.loads(Path(path or GATE_SPEC_LOCK).read_text())
    except (OSError, ValueError):
        return {}


def write_gate_lock(path: Optional[Path] = None) -> dict:
    """Regenerate the lock from the code, in the same commit as the gate change:
    `python -c "from core.metrics import write_gate_lock; write_gate_lock()"`."""
    payload = {'gate_spec_hash': gate_spec_hash(), 'gates': gate_spec()}
    Path(path or GATE_SPEC_LOCK).write_text(json.dumps(payload, indent=2) + '\n')
    return payload


def gate_spec_diff(old: Sequence[dict], new: Sequence[dict]) -> dict:
    """What changed between two spec row lists, by gate name."""
    before = {row['name']: row for row in old or ()}
    after = {row['name']: row for row in new or ()}
    changed = {
        name: {'from': before[name], 'to': after[name]}
        for name in before.keys() & after.keys() if before[name] != after[name]}
    return {'added': [after[n] for n in after if n not in before],
            'removed': [before[n] for n in before if n not in after],
            'changed': dict(sorted(changed.items()))}

# Above this, a Sharpe ratio is evidence of a defect, not an edge.
IMPLAUSIBLE_SHARPE = 5.0

GATE_NOTES: dict[str, str] = {
    'market_windows': 'the market comparison needs live-recorded quotes; a '
                      'backtest has no book and stands the baseline in for one, '
                      'which answers a different question with the same number',
    'model_minus_market': 'the price is the counterparty, not F(x/sigma). Beating '
                          'the arithmetic null while losing to the quote is the '
                          'failure this whole stack is built to not make',
    'model_minus_market_p': 'one-sided p that model_minus_market is above zero: '
                            'window-weighted fold mean on venue-labelled rows, '
                            'standard error from fold dispersion',
    'min_folds': 'weekly folds that carry market rows; fewer and the fold '
                 'dispersion behind model_minus_market_p means nothing',
    'log_loss_skill': 'the model must beat F(x/sigma); a coin flip is not the benchmark. '
                      'Window-weighted mean over folds',
    'log_loss_skill_p': 'one-sided p that the weighted skill is above zero, from fold '
                        'dispersion on n_eff - 1 df. The point estimate alone clears '
                        'zero on half of all nulls',
    'folds_skill_positive': 'how many folds agreed, reported not gated — the COUNT '
                            'is only meaningful at a fixed fold count, and '
                            'anchored blocks accumulate',
    'sign_agreement_p': 'the chance this many folds agree if each were a coin flip. '
                        '0.11 is exactly the old five-of-six bar (p=0.109) and stays '
                        'that strict as folds accumulate, where ">=5" would have '
                        'decayed to a 62% event at ten folds and 81% at twelve. '
                        'Unreachable at three folds or fewer (3/3 is p=0.125), so '
                        'there it fails on sample size, not evidence',
    'calibration_error': 'a SANITY floor, not a quality bar. The old 0.02 was set '
                         'under 21-day fold blocks and the arithmetic null fails it '
                         'on 21 of 25 seven-day ones (model 2/25 under the bar, '
                         'baseline 4/25), so it separated a large sample from a '
                         'small one rather than a good model from a bad one',
    'calibration_vs_baseline': 'does the correction make calibration WORSE than the '
                               'arithmetic it corrects? Median over folds, so it does '
                               'not drift as folds accumulate. Measured at +0.00100 '
                               'with t=+0.06 over 25 folds — statistically identical, '
                               'which is the honest reading rather than either an '
                               'improvement or a fault',
    'calibration_noise_null': 'what a RANDOM correction of the same size scores on '
                              'this statistic — reported so the gate above can be read '
                              'as a margin over chance rather than an absolute bar',
    'calibration_vs_market': 'the model must be at least as calibrated as the price '
                             'it trades against. An absolute bar encodes an assumption '
                             'this venue falsifies: measured on 33,126 rows, the MARKET '
                             'is 0.063 off in its worst populated bin',
    'calibration_max_deviation': 'the mean ECE averages away the band the money is in; '
                                 'this bounds the worst populated bin',
    'non_finite_share': 'a NaN prediction is a data hole, not a forecast. Counting it '
                        'as one made "no skill" and "one missing bar" the same output',
    'residual_scale': 'how much of the claimed correction survives out of sample; near '
                      'zero means it found nothing however good the in-sample loss',
    'control_gain_share': 'hour-of-day cannot forecast direction. If the clock carries '
                          'the model, the measurement is broken, not the market',
    'trades': 'fewer than this and the money numbers are anecdote',
    'coverage': 'abstaining on everything passes every other gate trivially',
    'realised_edge_pp': 'what actually happened, against what the model claimed. The '
                        'gap between the two is the winner\'s curse',
    'realised_edge_lcb': 'lower end of the 90% interval of realised edge per '
                         'contract, resampling whole windows: the symbols of one '
                         'window are not independent trades',
    'overconfidence_abs': 'mean predicted win probability against the realised '
                          'win rate on traded rows. decide() selects rows where '
                          'the model disagrees with the price, which is where it '
                          'is most often overconfident',
    'sharpe_implausible': f'a Sharpe above {5.0} on a public venue is a bug '
                          f'signature, not an edge — every other gate asks '
                          f'whether the number is good, this one asks whether '
                          f'it is possible',
    'median_fold_drawdown': 'a typical three-week block, on a statistic that does '
                            'not grow with the sample the way a running maximum '
                            'does. The account still has to survive to compound',
    'worst_fold_drawdown': 'reported, not gated: a max over folds ratchets as '
                           'folds accumulate',
    'halted': 'the bankroll floor was breached during the run',
}


@dataclass(frozen=True)
class Gate:
    name: str
    value: float
    threshold: float
    direction: str
    note: str = ''

    @property
    def passed(self) -> bool:
        if not np.isfinite(self.value):
            return False          # not measured fails
        return (self.value >= self.threshold if self.direction == 'min'
                else self.value <= self.threshold)

    def line(self) -> str:
        mark = 'pass' if self.passed else 'FAIL'
        comparison = '>=' if self.direction == 'min' else '<='
        return (f'  [{mark}] {self.name:<24} {self.value:>10.5f} {comparison} '
                f'{self.threshold:<10.5f} {self.note}')


def evaluate_gates(
    report: EvaluationReport,
    gates: Optional[dict[str, tuple[float, str]]] = None,
    *,
    extra: Optional[dict[str, float]] = None,
) -> list[Gate]:
    """Score a report against every gate. Missing values fail.

    `extra` carries measurements the report cannot produce (the market
    comparison). Anything absent stays NaN and fails.
    """
    gates = gates or DEFAULT_GATES
    values = report.gate_values()
    if extra:
        values.update(extra)
    return [
        Gate(name=name, value=values.get(name, float('nan')), threshold=threshold,
             direction=direction, note=GATE_NOTES.get(name, ''))
        for name, (threshold, direction) in gates.items()
    ]


def gates_passed(gates: Sequence[Gate]) -> bool:
    return all(g.passed for g in gates)


def gate_report(gates: Sequence[Gate]) -> str:
    failed = [g for g in gates if not g.passed]
    header = ('all gates passed' if not failed
              else f'{len(failed)} of {len(gates)} gates failed: '
                   + ', '.join(g.name for g in failed))
    return '\n'.join([header] + [g.line() for g in gates])


# ---- the deflated Sharpe ratio ---------------------------------------------

EULER_GAMMA = 0.5772156649015329
DAYS_PER_YEAR = 365.25


def daily_pnl(trades: Optional[pd.DataFrame]) -> np.ndarray:
    """Dollars per calendar day, idle days as zero: the series `BookStats.sharpe`
    annualises, so the moments below describe the same number."""
    if trades is None or trades.empty or 'pnl' not in trades.columns:
        return np.empty(0)
    daily = (trades.set_index('settle_time')['pnl'].sort_index().resample('1D').sum())
    if len(daily) > 1:
        calendar = pd.date_range(daily.index.min(), daily.index.max(),
                                 freq='1D', tz=daily.index.tz)
        daily = daily.reindex(calendar, fill_value=0.0)
    return daily.to_numpy(dtype=float)


def sharpe_moments(trades: Optional[pd.DataFrame]) -> dict:
    """Observation count, skew and (non-excess) kurtosis of daily P&L."""
    from scipy import stats as sstats

    values = daily_pnl(trades)
    if len(values) < 4 or not np.std(values) > 0:
        return {'n_obs': int(len(values)), 'skew': float('nan'),
                'kurtosis': float('nan')}
    return {'n_obs': int(len(values)),
            'skew': float(sstats.skew(values, bias=False)),
            'kurtosis': float(sstats.kurtosis(values, fisher=False, bias=False))}


def deflated_sharpe_ratio(
    sharpe: float,
    *,
    n_obs: int,
    skew: float,
    kurtosis: float,
    trial_sharpes: Sequence[float],
    n_trials: int,
    periods_per_year: float = DAYS_PER_YEAR,
) -> dict:
    """Bailey & López de Prado (2014): P(true Sharpe > the best of N null trials).

    Sharpes arrive annualised (as `BookStats.sharpe` and the trial log carry
    them) and are de-annualised to the per-period figure the formula needs. The
    expected maximum of `n_trials` null Sharpes uses the variance ACROSS trials:

        SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))
        DSR = Phi((SR - SR0) sqrt(T - 1) / sqrt(1 - skew SR + (kurt - 1)/4 SR^2))

    With a single trial SR0 is 0 and DSR is the probabilistic Sharpe ratio
    against zero. Reported, not gated.
    """
    from scipy.stats import norm

    scale = np.sqrt(float(periods_per_year))
    finite = [v for v in map(_finite_or_nan, trial_sharpes or ()) if np.isfinite(v)]
    out = {'sharpe': _finite_or_nan(sharpe), 'n_obs': int(n_obs or 0),
           'skew': _finite_or_nan(skew), 'kurtosis': _finite_or_nan(kurtosis),
           'n_trials': int(n_trials or 0), 'trials_with_sharpe': len(finite),
           'trial_sharpe_variance': float('nan'), 'expected_max_sharpe': float('nan'),
           'probabilistic_sharpe': float('nan'), 'deflated_sharpe': float('nan')}
    if len(finite) >= 2:
        out['trial_sharpe_variance'] = float(np.var(finite, ddof=1))
    sr = out['sharpe'] / scale
    if not (np.isfinite(sr) and np.isfinite(out['skew'])
            and np.isfinite(out['kurtosis']) and out['n_obs'] >= 2):
        return out
    denominator = 1.0 - out['skew'] * sr + (out['kurtosis'] - 1.0) / 4.0 * sr ** 2
    if denominator <= 0:
        return out
    root_t = np.sqrt(out['n_obs'] - 1.0) / np.sqrt(denominator)
    out['probabilistic_sharpe'] = float(norm.cdf(sr * root_t))
    n = out['n_trials']
    if n <= 1:
        sr0 = 0.0
    elif np.isfinite(out['trial_sharpe_variance']):
        sd = np.sqrt(out['trial_sharpe_variance']) / scale
        sr0 = sd * ((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
                    + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e)))
    else:
        return out
    out['expected_max_sharpe'] = float(sr0 * scale)
    out['deflated_sharpe'] = float(norm.cdf((sr - sr0) * root_t))
    return out


def _finite_or_nan(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return value if np.isfinite(value) else float('nan')
