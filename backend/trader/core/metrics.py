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
            'calibration_error': self.max_ece,
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
MAX_QUOTE_AGE_SECONDS = 30.0

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
                'baseline_minus_market': float('nan')}
    windows = float(frame.drop_duplicates(['symbol', 'window_open']).shape[0])
    overall = market_slice(frame, 'all')
    return {'market_windows': windows,
            'model_minus_market': float(overall['model_minus_market']),
            'baseline_minus_market': float(overall['baseline_minus_market'])}


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
    'folds_skill_positive': (5.0, 'min'),
    'calibration_error': (0.02, 'max'),
    # The mean ECE is count-weighted over every row, and most rows sit where the
    # barrier is already decided. Measured: a model 5pp overconfident on the
    # 5% of rows it trades scores 0.0044 and passes. This bounds the worst
    # adequately-populated bin instead. It cannot resolve `min_edge_pp` (0.5pp) —
    # 500 rows at p=0.9 carry a 1.3pp standard error — so it bounds the damage
    # rather than certifying the edge.
    'calibration_max_deviation': (0.04, 'max'),
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
    'max_drawdown': (0.35, 'max'),
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
    'folds_skill_positive': 'five of six agreeing happens 10.9% of the time by chance, '
                            'so this is necessary and not sufficient',
    'calibration_error': 'the system trades its confident predictions, so being wrong '
                         'about how confident it is matters more than the mean',
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
    'max_drawdown': 'a $100 account has to survive to compound',
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
