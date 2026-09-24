"""Measurement, and the gates a candidate has to clear.

Everything here is *incremental against the baseline*. A log loss, an accuracy
or a Brier score quoted on its own is uninterpretable in this system, because
the barrier arithmetic alone takes log loss from 0.693 to about 0.513 — a 26%
improvement over a coin flip using nothing but a clock and a volatility
estimate. Reported against 50% that reads as a large edge. It is not an edge at
all, and the only number that means anything is the difference.

**Standard errors come from fold dispersion.** Not from `N/(1+(N-1)rho)`: four
decision offsets share one label, the three symbols are ~0.7 correlated within
a window, and a breadth formula on that structure is not merely optimistic but
degenerate. Six folds give five degrees of freedom, which is few — and honestly
few, which is better than a precise-looking number from the wrong formula.

**The gates exist because a Sharpe ratio is the wrong first question.** On the
perp system a model 34x short of its cost hurdle failed every gate without any
of them saying why, because they all read simulated outcomes. Here the first
four gates read the *forecast* — skill, fold agreement, calibration, and how
much of the model's claimed correction survives out of sample — and only then
does the money get looked at. A weak forecast and an expensive venue are the
same ratio and opposite fixes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import os

import numpy as np
import pandas as pd

from core.baseline import Reliability, brier, log_loss, reliability
from core.book import BookStats
from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def log_loss_skill(outcome: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    """Baseline log loss minus model log loss. Positive means the model helped."""
    return log_loss(outcome, baseline) - log_loss(outcome, model)


def brier_skill(outcome: np.ndarray, model: np.ndarray, baseline: np.ndarray) -> float:
    base = brier(outcome, baseline)
    return (base - brier(outcome, model)) / base if base > 0 else float('nan')


def resolve_max_deviation(per_fold, pooled) -> tuple:
    """(value, why) for `calibration_max_deviation`, preferring the per-fold max.

    `worst_deviation` returns NaN when no bin holds its minimum row count, and
    `max` over folds propagates that — so a run whose folds are individually too
    small reported NaN and FAILED, indistinguishably from a model that is badly
    calibrated. Under `--complete-cases` that is the normal case, not an
    exception.

    Every scored row is out-of-sample whichever fold produced it, so the pooled
    rows are a legitimate fallback with fold-count times the rows per bin. It
    stays a FALLBACK: the per-fold maximum is the stricter statistic, and a
    single badly-calibrated fold is exactly what the gate exists to catch, so
    one measurable fold beats the pool.

    Nothing measurable anywhere still returns NaN and still fails -- not
    measured is not measured good -- but the reason names the sample size rather
    than the model.
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
    # Rows whose prediction or outcome was not finite. Reported so a data hole
    # says "data hole" instead of "no skill" — see `n_non_finite` in the gates.
    n_non_finite: int = 0
    # The worst deviation in any adequately-populated reliability bin. The mean
    # ECE cannot see the traded band; this can.
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

    # A row with no volatility estimate has no forecast, so it cannot be scored —
    # and it is not an error. Measured on real bars: a 6.5-hour Coinbase outage
    # leaves the 240-minute lookback unfillable for about two hours afterwards, so
    # ~83 rows in 53,200 come back with a NaN sigma. Live, `decide` refuses those
    # as NOT_FINITE; here they must leave the metric rather than poison it.
    #
    # They are *excluded and counted*, not dropped silently. Silence was the
    # original defect: `np.mean` propagated the NaN into every fold statistic
    # while `np.digitize` filed the rows in the 0.95-1.00 reliability bin, and one
    # of the two readers then failed open. The count reaches a gate as a share.
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

    # ---- forecast ------------------------------------------------------
    @property
    def skills(self) -> np.ndarray:
        return np.array([f.skill for f in self.folds], dtype=float)

    @property
    def mean_skill(self) -> float:
        return float(np.mean(self.skills)) if len(self.folds) else float('nan')

    @property
    def skill_standard_error(self) -> float:
        """From fold dispersion. See the module docstring for why not breadth."""
        if len(self.folds) < 2:
            return float('nan')
        return float(np.std(self.skills, ddof=1) / np.sqrt(len(self.folds)))

    @property
    def skill_t(self) -> float:
        se = self.skill_standard_error
        return self.mean_skill / se if se and np.isfinite(se) and se > 0 else float('nan')

    @property
    def folds_positive(self) -> int:
        return int((self.skills > 0).sum())

    @property
    def folds_total(self) -> int:
        return len(self.folds)

    @property
    def fold_drawdowns(self) -> list[float]:
        """Each fold's own worst drawdown, on its own book from the same start.

        A fold is a FIXED 21-day block, so this cannot ratchet with history the
        way the continuous curve does — see `median_fold_drawdown`.
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
        """**The drawdown statistic that can be gated, because it holds still.**

        `max_drawdown` on the continuous curve is a running maximum over the
        whole out-of-sample span. It can only ratchet upward: adding days can
        never lower it, so against a fixed bar it eventually fails forever
        whatever the model does. Measured on the SAME configuration three days
        apart, it went 0.182 -> 0.388 and failed a 0.35 gate it had just
        passed. That is a property of the statistic, not of the strategy.

        Any max-over-the-sample quantity has this problem, so the fix is not a
        looser bar — it is a statistic whose expectation does not grow with the
        sample. The median across folds is one: each fold is a fixed-length
        block with its own book, so it answers "what does a typical three-week
        stretch look like" and gives the same answer next week.

        The continuous figure stays in the report as a diagnostic. It is still
        the honest answer to "what is the worst this has ever been", which is
        worth knowing and not worth gating.
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
        """P(at least this many folds positive | no skill), each fold a coin flip.

        With six folds, five or more positive happens 10.9% of the time by
        chance. That is the number to hold against any "five of six" claim, and
        it is why the gate asks for five *and* a positive aggregate rather than
        treating agreement as proof.
        """
        from scipy import stats as sstats
        n, k = self.folds_total, self.folds_positive
        if n == 0:
            return float('nan')
        return float(sstats.binom.sf(k - 1, n, 0.5))

    @property
    def calibration_vs_baseline(self) -> float:
        """Median fold (model ECE - baseline ECE). Positive means worse.

        **Replaces an absolute bar that the arithmetic null itself fails.**
        `calibration_error <= 0.02` was set under 21-day fold blocks. Under the
        7-day blocks the retrain actually runs, measured over 25 folds:

            folds under 0.02    model 2/25    BASELINE 4/25

        `F(x/sigma)` — the thing the model exists to correct, with no features
        and no fit — fails that bar on 21 of 25 folds. A gate the null cannot
        pass is not measuring the model; it is measuring binned ECE on a 7-day
        sample, which is biased upward at small n. Exactly the conclusion
        `calibration_max_deviation` already reached about its own absolute bar:
        "an absolute bar encodes an assumption this venue falsifies".

        The question worth gating is whether the correction makes calibration
        WORSE than the arithmetic it corrects. Measured, it does not: the
        per-fold difference is mean +0.00008, sd 0.00707, **t = +0.06** over 25
        folds — statistically identical, neither better nor worse.

        MEDIAN, not max, for the same reason `median_fold_drawdown` exists: a
        max over folds grows with the fold count, so an unchanged model scores
        worse on it purely by accumulating history.
        """
        pairs = [(f.model_ece, f.baseline_ece) for f in self.folds
                 if np.isfinite(f.model_ece) and np.isfinite(f.baseline_ece)]
        if not pairs:
            return float('nan')
        return float(np.median([m - b for m, b in pairs]))

    @property
    def max_ece(self) -> float:
        # `np.max`, not the builtin. Builtin `max` with a NaN in the sequence is
        # order-dependent — `max([0.015, nan])` is 0.015 and `max([nan, 0.015])`
        # is nan — so a fold whose calibration could not be computed silently
        # vanished and the gate passed on the folds that worked. `Gate.passed`
        # already fails closed on a non-finite value; the aggregation has to let
        # it get there.
        if not self.folds:
            return float('nan')
        return float(np.max(np.asarray([f.model_ece for f in self.folds], dtype=float)))

    @property
    def max_calibration_deviation(self) -> float:
        """The worst adequately-populated calibration bin.

        Per-fold first, pooled as a fallback. `worst_deviation` returns NaN when
        no bin holds its minimum row count, and `max` propagated that, so a run
        with small folds reported NaN and FAILED — indistinguishable from a model
        that is genuinely badly calibrated. Under `--complete-cases` small folds
        are the normal case. See `resolve_max_deviation`.
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

        Legitimate because each row is out-of-sample whichever fold produced it,
        and it carries fold-count times the rows per bin — which is the whole
        reason the per-fold version goes un-measurable on a small sample.
        """
        tables = [f.reliability_table for f in self.folds
                  if getattr(f, 'reliability_table', None) is not None]
        if not tables:
            return float('nan')
        counts = np.sum([t.count for t in tables], axis=0)
        populated = counts >= 500
        if not populated.any():
            return float('nan')
        # Row-weighted, so a bin's pooled figure is the average an observation
        # actually saw rather than an average of fold averages.
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

        Gated as a share rather than a count. Zero is the wrong threshold: a
        venue outage leaves a couple of hours of windows without a volatility
        estimate afterwards, and refusing to evaluate at all because Coinbase went
        down for six hours in May is not a judgement about the model. A *rate*
        catches what actually matters — an embargo or lookback mistake that makes
        a large fraction unscoreable — while an outage passes and is still
        reported.
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

        The mean was the wrong aggregation for an overfitting detector. Measured
        on a provably zero-signal null across three folds: two folds correctly
        returned alpha ~0 and one ran to the 2.0 clip on a handful of trees, so
        the **mean was 0.667** and cleared the 0.25 gate while two thirds of the
        evidence said there was nothing there. The question the gate asks is
        whether the correction survives out of sample *typically*, and one
        runaway fold should not answer it.

        A fold sitting exactly on a clip boundary is also not a fit — it means the
        optimiser wanted more than the parameterisation allows, which on a null is
        noise — so those are called out.

        Two honest limits. With an even number of folds the median averages the two
        middle values, so at two folds it *is* the mean and buys nothing. And at a
        small sample alpha is noisy either way: measured on a 30-day two-fold null
        it came back `[1.305, 0.0]`, while on a 70-day slice it reads 0.0000. This
        gate is therefore a backstop and not the thing that catches a null —
        `log_loss_skill` and `folds_skill_positive` are, and they did on the same
        run. Do not treat a passing `residual_scale` as evidence of anything.
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

    @property
    def total_trades(self) -> int:
        return int(sum(s.n_trades for s in self.traded_folds))

    @property
    def mean_fold_return(self) -> float:
        stats = self.traded_folds
        return float(np.mean([s.total_return for s in stats])) if stats else float('nan')

    @property
    def folds_profitable(self) -> int:
        return int(sum(1 for s in self.traded_folds if s.total_return > 0))

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
        """Everything `DEFAULT_GATES` reads, in one dict."""
        continuous = self.continuous
        return {
            'log_loss_skill': self.mean_skill,
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
        lines = [
            f'{self.folds_total} folds, {self.total_windows:,} out-of-sample windows',
            *[f.line() for f in self.folds],
            '',
            f'  log loss skill {self.mean_skill:+.5f} +/- {self.skill_standard_error:.5f} '
            f'(t = {self.skill_t:+.2f}), {self.folds_positive}/{self.folds_total} folds '
            f'positive (p = {self.sign_agreement_p_value:.3f})',
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

# Below this the comparison is an anecdote. Two thousand windows at ~96 a day is
# roughly three weeks of one symbol, or a week of three.
MIN_MARKET_WINDOWS = 2_000

# A quote older than this is not the price at the decision instant, and beating
# it is not skill. Measured on 132,250 backtest rows: model_minus_market ran
# +0.0041 at a 5-second bar and +0.0371 at 900 seconds -- nine tenths of the
# headline was the model out-forecasting a price nobody was quoting any more.
# Both halves flatter: market_ll worsens with age while model_ll improves,
# because the rows carrying stale quotes are the easier ones.
# **Read from the same env var `core.quotes` reads, not hardcoded.** These two
# must move together: trades were once priced against quotes up to 900s old
# while this comparison counted only those under 30s, so the money and the
# forecast were measured on different samples. Pinning one to a literal while
# the other honours `QUOTE_MAX_AGE_SECONDS` re-opens exactly that desync the
# moment anyone sets it -- silently, and in the direction that manufactures
# edge.
MAX_QUOTE_AGE_SECONDS = float(os.getenv('QUOTE_MAX_AGE_SECONDS', '30'))

MARKET_COLUMNS = ('symbol', 'window_open', 'offset', 'market', 'baseline',
                  'model', 'outcome', 'decision_time')


def market_frame(rows: Iterable[Sequence]) -> pd.DataFrame:
    """Rows from `PgWriter.scored_against_market()`, cleaned.

    Tolerates rows without `decision_time` so an older store still reads.
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
    # The number that decides everything: positive means our probability is a
    # better forecast than the price we would have to pay.
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


def market_rows_from_scored(
        frame: pd.DataFrame, *,
        max_quote_age_seconds: float = MAX_QUOTE_AGE_SECONDS,
        entry_offsets=None) -> list:
    """`MARKET_COLUMNS` rows from a backtest that carries recorded quotes.

    **This is the claim `market_gate_values` used to say could not be made.**
    Its docstring reads "this cannot come from the backtest, and that is the
    point", because a backtest had no order book and `price_source` stood the
    calibrated baseline in for the market — collapsing "beat the market" into
    "beat the baseline" and answering both with one number. That was true when
    written. Eight months of book have since been collected and validated: 0.70c
    against the live recording with the clock removed, and a resting-size ratio
    of 1.000.

    **The market's forecast is the MID.** `model_minus_market` compares log
    losses — whose probability is better — while the ask is what a trade costs,
    the mid plus half the spread. Scoring the market at its ask would hand the
    model a free half-spread of apparent skill on every row, in its own favour,
    which is the exact self-flattery this gate exists to prevent. The ask belongs
    in the money numbers, and `decide()` already uses it there.

    Rows without a quote are DROPPED, never defaulted to the baseline: a
    defaulted row is the circularity this function replaces, reported as if it
    were a market.
    """
    if frame is None or not len(frame):
        return []
    needed = ('symbol', 'window_open', 'offset', 'market_probability',
              'baseline_probability', 'model_probability')
    if any(c not in frame.columns for c in needed):
        return []
    part = frame.copy()
    # **Grade on the label the MARKET was priced against, wherever we hold it.**
    # Our outcome comes from Coinbase bars and so does the baseline; the market
    # prices on CF Benchmarks BRTI. Scoring both forecasters on a label that
    # shares a source with one of them hands that one the label noise as free
    # skill. Measured, and it reversed the headline: base-mkt read +0.00382 on
    # our label and -0.00245 on the venue's, collapsing to +0.00101 on the 96.8%
    # where the two agree. The whole effect lived in the near-ties.
    if 'venue_outcome' in part.columns:
        part['outcome'] = pd.to_numeric(
            part['venue_outcome'], errors='coerce').fillna(
                pd.to_numeric(part.get('outcome'), errors='coerce'))
    part = part.dropna(subset=list(needed) + ['outcome'])
    # Staleness is not skill. A row whose quote predates the decision by minutes
    # is measuring the clock, not the market -- see MAX_QUOTE_AGE_SECONDS.
    # Rows with no age are kept: live-recorded quotes carry none, and they are
    # the one source needing no reconstruction.
    # **Only the offsets that can OPEN a position.**
    #
    # This pooled all four while `--entry-offsets 12` means one can trade.
    # Measured on 5,622 live rows:
    #
    #     offset   model - mkt      t     days+
    #     +3m       -0.00259     -1.11     2/6
    #     +6m       -0.00644     -1.63     3/6
    #     +9m       +0.00068     +0.18     3/6
    #     +12m      +0.00550     +1.09     5/6   <- the only one that trades
    #     pooled    -0.00072     -0.26     2/6   <- what the gate read
    #
    # So it rejected two candidates for losing to the market at offsets they
    # never trade, while the offset they do trade was the best of the four. Same
    # defect as the entry-offsets bug: a measurement describing a policy nobody
    # runs. None still pools, because then every offset really can enter.
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
    # Positional, matching MARKET_COLUMNS. A column out of order silently swaps
    # the model's probability with the baseline's and nothing raises.
    return list(zip(part['symbol'], part['window_open'], part['offset'],
                    part['market_probability'], part['baseline_probability'],
                    part['model_probability'], part['outcome'], decision))


def _pooled_ece(pred, outcome) -> float:
    """Count-weighted expected calibration error over every populated bin.

    The low-variance alternative to `_worst_populated_bin`. A maximum over bins
    is dominated by whichever bin got unlucky, which is why the worst-bin
    version of `calibration_vs_market` passed only 32% of bootstrap resamples of
    its OWN data with a fixed artifact. This one is ten times tighter (sd 0.0023
    against 0.0229 on 7,608 live rows) because every bin contributes in
    proportion to its count rather than one deciding the answer.

    It answers a slightly different question, and the difference is the point: a
    worst bin bounds the damage anywhere, a pooled ECE describes calibration on
    average. For comparing two forecasters on the SAME rows -- which is all
    `calibration_vs_market` does -- the average is the stable comparison and the
    maximum is a coin flip.
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

    **Uses `core.baseline.reliability`'s edges, which is what "the same shape as
    `calibration_max_deviation`" is supposed to mean.** It did not: this was ten
    EQUAL-WIDTH bins against that statistic's twenty tail-refined ones, and its
    single `(0.9, 1.0]` bucket is precisely the aggregation those edges exist to
    eliminate -- `reliability`'s own docstring records a model 5pp overconfident
    at 0.94 and 5pp underconfident at 0.86 reporting an ECE of 0.000078 across
    one wide bin. Two statistics under one description, feeding two different
    gates, with a narrative comment comparing a number from this one against a
    threshold belonging to the other.

    `min_count` stays 100 rather than that statistic's 500: this runs on
    live-recorded rows, of which there are thousands rather than tens of
    thousands, and a 500 floor would return NaN for want of data and FAIL as
    though the model were miscalibrated. Pass `bins` to force equal-width
    bucketing.
    """
    pred = pd.to_numeric(pd.Series(pred), errors='coerce')
    outcome = pd.to_numeric(pd.Series(outcome), errors='coerce')
    keep = pred.notna() & outcome.notna()
    if not keep.any():
        return float('nan')
    frame = pd.DataFrame({'p': pred[keep].to_numpy(),
                          'y': outcome[keep].to_numpy()})
    if bins is None:
        # `worst_deviation` IS the statistic `calibration_max_deviation` reads,
        # on `reliability`'s tail-refined edges. Calling it rather than
        # re-deriving it is the whole point: two implementations of one
        # description is what produced the divergence this replaces.
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
    """What `DEFAULT_GATES` reads about the market, from live-recorded quotes.

    **This cannot come from the backtest, and that is the point.** A backtest has
    no order book, so `price_source` stands the calibrated baseline in for the
    market — which makes "beat the market" and "beat the baseline" the same
    question and answers both with the same number. The comparison is only
    available from quotes the live loop actually recorded.

    Measured on the first day of live quotes, and this is why the gate exists:
    the market's log loss was 0.333 against the model's 0.430 and the baseline's
    0.429, on every symbol and every offset. A candidate can pass all twelve
    other gates — `log_loss_skill` beats `F(x/sigma)` by construction — while
    being a materially worse forecaster than the price it has to trade against.

    Empty or short input returns values that fail rather than values that pass:
    `market_windows` counts what there is and `model_minus_market` is NaN, and
    `Gate.passed` is False for both. Not measured is not the same as measured
    good, and promotion is the wrong place to blur them.
    """
    frame = market_frame(rows)
    if frame.empty:
        return {'market_windows': 0.0,
                'model_minus_market': float('nan'),
                'baseline_minus_market': float('nan'),
                'market_max_deviation': float('nan'),
                'calibration_vs_market': float('nan')}
    windows = float(frame.drop_duplicates(['symbol', 'window_open']).shape[0])
    overall = market_slice(frame, 'all')
    # **Calibration RELATIVE to the price, on the same rows.**
    #
    # The absolute bar demanded better than this market achieves. Measured on
    # the venue's own settlement, 33,126 rows at +12m, the worst populated bin
    # (0.6, 0.7] has the MARKET predicting 0.654 against an actual 0.717 — a
    # 0.063 miss, 5.4 sigma on 1,531 windows. So nothing tracking the price can
    # reach 0.040, and every candidate since 2026-08-28 failed at a stable
    # ~0.0515 regardless of features, folds or groups.
    #
    # What is worth requiring is that the model be at least as calibrated as
    # what it trades against. This one halves the market's deviation
    # (0.0326 against 0.0634), which is the clearest evidence it adds value —
    # and it was coming from the gate that kept rejecting it.
    # **POOLED ECE, not the worst bin.** A maximum over bins is inherently
    # high-variance, and on this sample it was pure noise. Bootstrapped over 300
    # resamples of the 7,608 live rows on 2026-09-24, same artifact throughout:
    #
    #     statistic              mean       sd       passes (<=0)
    #     worst bin            +0.0106   0.0229         32%
    #     pooled ECE diff      +0.0032   0.0023         10%
    #
    # The worst-bin version's standard deviation was TWICE the effect it
    # measured, and its 90% interval [-0.026, +0.050] spanned both "clearly
    # better than the market" and "clearly worse" — so it passed or failed a
    # fixed artifact on which rows happened to arrive. It moved 0.011 in four
    # hours on 8% more data, and blocked the 2026-09-20 retrain on that basis.
    #
    # The pooled difference is TEN TIMES tighter, and it does not make the
    # problem go away — it makes it legible. The model is slightly but
    # consistently WORSE calibrated than the price, +0.0032 against a typical
    # ECE near 0.030, with 90% of resamples above zero. The noisy version was
    # obscuring a real finding behind an enormous error bar rather than
    # inventing a false one.
    #
    # Gated at <= 0 with no tolerance on purpose. A noise-floor allowance would
    # let exactly this finding through, and the point of the gate is to report
    # it while the edge is still unproven.
    model_ece = _pooled_ece(frame['model'], frame['outcome'])
    market_ece = _pooled_ece(frame['market'], frame['outcome'])
    # Kept as a reported diagnostic: it is informative about the venue even
    # though it is too noisy to gate on.
    market_dev = _worst_populated_bin(frame['market'], frame['outcome'])
    return {'market_windows': windows,
            'model_minus_market': float(overall['model_minus_market']),
            'baseline_minus_market': float(overall['baseline_minus_market']),
            'market_max_deviation': market_dev,
            'calibration_vs_market': float(model_ece - market_ece)}


# ---- gates ---------------------------------------------------------------

# name -> (threshold, direction). 'min' passes at or above, 'max' at or below.
# Ordered as they should be read: the forecast first, the money second. A
# candidate that fails a forecast gate should not have its Sharpe discussed.
DEFAULT_GATES: dict[str, tuple[float, str]] = {
    # --- the benchmark that decides whether any of the rest pays ---
    #
    # These two come first because the benchmark below them is the wrong one to
    # stop at. `log_loss_skill` asks whether the model beats `F(x/sigma)`, and it
    # does — but the arithmetic null is not the counterparty. The price is.
    # Measured on the first day of live quotes: market log loss 0.333, baseline
    # 0.429, model 0.430, with the same sign on all three symbols and all four
    # offsets. Every other gate would have passed that.
    #
    # Neither can be computed from a backtest, which has no book, so both read
    # NaN and fail until the live loop has recorded enough quotes. That is the
    # honest state of the question rather than an obstacle to route around;
    # `--force` with a written reason is the documented way past it, and the
    # ledger records that it was used.
    'market_windows': (float(MIN_MARKET_WINDOWS), 'min'),
    'model_minus_market': (0.0, 'min'),
    # --- the forecast ---
    'log_loss_skill': (0.0, 'min'),
    # **A COUNT of agreeing folds is only meaningful at a fixed fold count.**
    # This was `folds_skill_positive >= 5`, written when the walk-forward always
    # produced exactly six. Anchored calendar blocks accumulate — 10 folds on a
    # 245-day span today and more as history arrives — and "5 of 10" is a 62%
    # event under the null where "5 of 6" is a 10.9% one. The bar would have
    # silently weakened every week.
    #
    # `sign_agreement_p` is the same claim held at constant significance: the
    # binomial probability of seeing at least this many positive folds if each
    # were a coin flip. 0.11 reproduces the old five-of-six exactly (p = 0.109)
    # and stays that strict at any fold count.
    'sign_agreement_p': (0.11, 'max'),
    # **A SANITY floor, not a quality bar.** 0.02 was derived under 21-day fold
    # blocks, and under the 7-day blocks the retrain actually runs the
    # ARITHMETIC NULL fails it on 21 of 25 folds (model 2/25 under the bar,
    # baseline 4/25). A gate `F(x/sigma)` cannot pass does not separate a good
    # model from a bad one — it separates a large sample from a small one,
    # because binned ECE is biased upward at small n. 0.10 still fails a
    # catastrophically miscalibrated candidate (the baseline's own worst fold is
    # 0.0802) and leaves the discrimination to the two relative gates.
    'calibration_error': (0.10, 'max'),
    # **Does the correction make calibration worse than the arithmetic it
    # corrects?** Median over folds, so it does not drift as folds accumulate.
    # 0.005 sits below the one-sd fold-to-fold spread of 0.00707 and well above
    # the standard error of the median, so it catches a systematic degradation —
    # 0.005 against a typical fold ECE of 0.030 is 17% worse — without rejecting
    # a model for fold noise. Measured 2026-09-17: +0.00100, t = +0.06.
    'calibration_vs_baseline': (0.005, 'max'),
    # The mean ECE is count-weighted over every row, and most rows sit where the
    # barrier is already decided. Measured: a model 5pp overconfident on the
    # 5% of rows it trades scores 0.0044 and passes. This bounds the worst
    # adequately-populated bin instead. It cannot resolve `min_edge_pp` (0.5pp) —
    # 500 rows at p=0.9 carry a 1.3pp standard error — so it bounds the damage
    # rather than certifying the edge.
    # A loose sanity rail now, not the binding claim — see
    # `calibration_vs_market`, which asks the question that can be met.
    'calibration_max_deviation': (0.10, 'max'),
    'calibration_vs_market': (0.0, 'max'),
    # A data hole must report as a data hole. 31 non-finite rows in 99,388 turned
    # five of six folds' metrics into NaN while `scripts/baseline.py` printed
    # "gate passed", because `nan > 0.02` is False and pandas' max skips NaN.
    # A share rather than a count, because an outage is not a defect: measured,
    # one 6.5-hour Coinbase outage accounts for 0.02% of rows. A large share means
    # a lookback or embargo mistake, which is.
    'non_finite_share': (0.001, 'max'),
    'residual_scale': (0.25, 'min'),
    'control_gain_share': (0.30, 'max'),
    'windows_evaluated': (20_000.0, 'min'),
    # --- the money ---
    'trades': (200.0, 'min'),
    'coverage': (0.0005, 'min'),
    'realised_edge_pp': (0.0, 'min'),
    'total_return': (0.0, 'min'),
    'sharpe': (0.5, 'min'),
    'sharpe_implausible': (0.0, 'max'),
    # **Two drawdown questions, and they need two statistics.**
    #
    # `max_drawdown` is the worst peak-to-trough over the whole out-of-sample
    # path, held at the SAME number as the live breaker: a drawdown that blocks
    # promotion should stop the money too, which is why
    # `Config.max_drawdown_fraction` and this bar are asserted equal.
    #
    # It looked unusable when the SAME configuration three days apart went
    # 0.182 -> 0.388 and failed a bar it had just passed. That was the sliding
    # fold boundaries re-cutting the equity path, not the strategy, and it is
    # fixed at the source. With anchored folds this moves only when a genuinely
    # worse drawdown occurs, which is worth failing on. It still ratchets — a
    # running maximum cannot do otherwise — just slowly and meaningfully.
    #
    # `median_fold_drawdown` is the same risk question asked of a statistic
    # whose expectation does not grow with the sample: each fold is a fixed
    # 21-day block with its own book, so this is "what does a typical three-week
    # stretch look like" and it gives the same answer next week.
    #
    # 0.20 against a measured median of 0.080 (folds: 0.0, 2.58, 2.59, 7.60,
    # 8.41, 12.79, 22.47, 36.11). Deliberately loose — it catches typical
    # three-week risk more than doubling, not a fine-tuning of it. The bars that
    # protect real money are the live breakers, the $250 ruin floor and the $75
    # daily-loss halt, and they are unaffected by anything here.
    'max_drawdown': (0.35, 'max'),
    'median_fold_drawdown': (0.20, 'max'),
    'halted': (0.0, 'max'),
}

# Above this, a Sharpe ratio is evidence of a defect rather than of an edge.
# Nothing trading a public venue at 30,000 trades a year earns a Sharpe of 12;
# the first run of this stack reported 12.6 and every other gate passed it,
# because they all asked whether the number was good and none asked whether it
# was possible.
IMPLAUSIBLE_SHARPE = 5.0

GATE_NOTES: dict[str, str] = {
    'market_windows': 'the market comparison needs live-recorded quotes; a '
                      'backtest has no book and stands the baseline in for one, '
                      'which answers a different question with the same number',
    'model_minus_market': 'the price is the counterparty, not F(x/sigma). Beating '
                          'the arithmetic null while losing to the quote is the '
                          'failure this whole stack is built to not make',
    'log_loss_skill': 'the model must beat F(x/sigma); a coin flip is not the benchmark',
    'folds_skill_positive': 'how many folds agreed, reported not gated — the COUNT '
                            'is only meaningful at a fixed fold count, and '
                            'anchored blocks accumulate',
    'sign_agreement_p': 'the chance this many folds agree if each were a coin flip. '
                        '0.11 is exactly the old five-of-six bar (p=0.109) and stays '
                        'that strict as folds accumulate, where ">=5" would have '
                        'decayed to a 62% event at ten folds and 81% at twelve',
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
    'windows_evaluated': 'the whole reason for this venue is sample size; without it '
                         'the standard error cannot resolve a 1pp edge',
    'trades': 'fewer than this and the money numbers are anecdote',
    'coverage': 'abstaining on everything passes every other gate trivially',
    'realised_edge_pp': 'what actually happened, against what the model claimed. The '
                        'gap between the two is the winner\'s curse',
    'total_return': 'on one continuous account across the whole out-of-sample '
                    'span, sized additively so the slope is the per-trade edge '
                    'rather than an exponential of it',
    'sharpe': 'annualised on trades actually placed, never on windows available',
    'sharpe_implausible': f'a Sharpe above {5.0} on a public venue is a bug '
                          f'signature, not an edge — every other gate asks '
                          f'whether the number is good, this one asks whether '
                          f'it is possible',
    'max_drawdown': 'the worst this has ever been, held at the live breaker\'s own '
                    'number so a drawdown that blocks promotion also stops the '
                    'money. It ratchets, which is meaningful now the folds hold '
                    'still and was not when they slid',
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
            return False          # not measured fails, like every other gate here
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

    `extra` carries measurements the report structurally cannot produce — at
    present the market comparison, which needs an order book the backtest does
    not have. Anything absent stays NaN and therefore fails, so forgetting to
    pass it cannot turn into a pass.
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
