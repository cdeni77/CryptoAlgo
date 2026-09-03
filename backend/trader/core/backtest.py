"""Walk-forward: fit, score, trade, measure — six times, in order.

The loop is deliberately boring, because everything interesting has already
been decided elsewhere. Each fold fits its own seasonality, volatility model,
baseline and classifier on training windows only, scores the test block through
`core.dataset.apply_fold`, and runs the resulting probabilities through the same
`decide()` the live path calls. Nothing in here chooses a trade or prices one.

**Two books, and they answer different questions.** A per-fold book starts fresh
at the configured bankroll, so the six folds are comparable to each other. One
continuous book runs the whole out-of-sample span with a single bankroll, so
fold 0's losses shrink fold 1's stakes — which is what deployment actually does
to a $100 account. The per-fold numbers are the measurement; the continuous one
is the answer.

**Settlement is processed before the next window's decision.** A position opened
in the window starting at 10:00 settles at 10:15, which is the instant the next
window opens. Deciding first and settling afterwards would let the bankroll be
staked twice over, and at a $100 account with a 5% cap that is the difference
between a real constraint and none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.book import Book, BookStats, summarise
from core.config import Config, DEFAULT_CONFIG
from core.cv import WindowFold, assert_no_leakage, effective_observations, purged_walk_forward, recency_weights
from core.dataset import Dataset, apply_fold, fit_fold
from core.decide import Decision, Reason, decide_window, rejection_histogram, stateless_screen
from core.metrics import EvaluationReport, FoldEvaluation, evaluate_fold
from core.model import ForecastModel, fit_model

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything one walk-forward produced, so a script can report all of it."""

    report: EvaluationReport
    models: list[ForecastModel]
    books: list[Book]
    continuous_book: Optional[Book]
    rejections: pd.Series
    scored: pd.DataFrame

    def trades(self) -> pd.DataFrame:
        if self.continuous_book is None:
            return pd.DataFrame()
        return self.continuous_book.trades()


def run_book(
    scored: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    *,
    book: Optional[Book] = None,
) -> tuple[Book, list[Decision]]:
    """Walk scored windows in chronological order, deciding and settling.

    `scored` needs `model_probability` alongside the window columns. One pass,
    no lookahead: the only thing available when a window is decided is the row
    itself and the bankroll, and the bankroll depends only on windows that have
    already settled.
    """
    book = book or Book(config=config)
    decisions: list[Decision] = []
    # Screen the whole span once, vectorised, before touching the per-window
    # loop. On real data the state-independent gates reject the overwhelming
    # majority of rows, and grouping five years of minutes into windows to
    # iterate over rows that could never trade is where the runtime went.
    screened, stateless_rejections = stateless_screen(scored, config)
    outcomes = settlement_outcomes(scored)
    for window_open, rows in screened.groupby('window_open', sort=True):
        # Settle first: a position from the previous window matures at exactly
        # this instant. Deciding before settling would stake the same dollars
        # twice, which at a $100 account with a 5% cap is the difference between
        # a real constraint and none. Matured positions are found by settle_time
        # rather than by scanning the outcome map, which would be quadratic.
        matured = [p for p in book.open_positions if p.settle_time <= window_open]
        if matured:
            book.settle({(p.symbol, p.window_open): outcomes[(p.symbol, p.window_open)]
                         for p in matured if (p.symbol, p.window_open) in outcomes})
        if book.halted_at is not None:
            continue
        for decision in decide_window(rows, config, bankroll=book.bankroll):
            decisions.append(decision)
            book.record(decision)
    # Anything still open at the end of the span settles on its own outcome.
    if book.open_positions:
        book.settle(outcomes)
    book.stateless_rejections = stateless_rejections
    return book, decisions


def skill_null_column(table: pd.DataFrame, init_score_source: str) -> np.ndarray:
    """The forecaster `log_loss_skill` should be measured against.

    Skill means "better than the thing this model was fitted to correct". A
    market-initialised model is not trying to beat `F(x/sigma)`; it is fitted on
    the price, so scoring it against the baseline reports a failure that is a
    category error. Measured on a real run: `model_minus_market` +0.00078 (it
    does beat the price) alongside `log_loss_skill` -0.00016 and 3/6 folds — two
    of four failing gates asking the wrong question, which is worse than an
    honest failure because it hides whichever failures are real.

    Falls back to the baseline when the market column is absent or entirely
    NaN. That is the honest fallback, but it is a FALLBACK: a silent swap is how
    a market-init model comes to be judged as a baseline-init one.
    """
    if init_score_source == 'market' and 'market_probability' in table.columns:
        values = pd.to_numeric(table['market_probability'], errors='coerce')
        if values.notna().any():
            return values.to_numpy(dtype=float)
        logger.warning('init_score_source=market but every market_probability '
                       'is NaN; skill falls back to the baseline null')
    elif init_score_source == 'market':
        logger.warning('init_score_source=market but no market_probability '
                       'column; skill falls back to the baseline null')
    return table['baseline_probability'].to_numpy(dtype=float)


def settlement_outcomes(scored: pd.DataFrame) -> dict:
    """How each window settled, preferring the VENUE'S own result.

    **The leak this closes, measured.** Settling on our Coinbase-derived label
    while trading against the venue's BRTI-priced quotes, with `market_prob`
    live as a feature, let the model bet the disagreement between the two
    indices and win it by construction:

        labels agree on 96.51% of traded windows
          win rate where they AGREE  : 56.17%
          win rate where they DIFFER : 72.77%   (n=448)

    Rescoring the same 12,821 trades on the venue's settlement took the win rate
    from 56.75% to 55.16% and the edge from 8.68% to 4.99% of stake — about 43%
    of the apparent edge was the label rather than the forecast.

    Ours remains the fallback: Kalshi purges older markets, so their
    settlements do not reach the whole span, and dropping those windows would
    discard most of the sample. What must never happen is preferring ours where
    theirs exists.

    A window with NEITHER label is omitted rather than guessed. `book.settle`
    skips what it has no outcome for; inventing one would pay or charge for a
    result nobody holds.
    """
    frame = scored.drop_duplicates(['symbol', 'window_open'])
    ours = pd.to_numeric(frame['outcome'], errors='coerce')
    if 'venue_outcome' in frame.columns:
        venue = pd.to_numeric(frame['venue_outcome'], errors='coerce')
        settled = venue.where(venue.notna(), ours)
    else:
        settled = ours
    keep = settled.notna()
    return {
        (symbol, window_open): bool(value)
        for symbol, window_open, value in zip(
            frame['symbol'][keep], frame['window_open'][keep], settled[keep])
    }


def walk_forward(
    dataset: Dataset,
    config: Optional[Config] = None,
    *,
    groups: Optional[Sequence[str]] = None,
    trade: bool = True,
    folds: Optional[Sequence[WindowFold]] = None,
) -> RunResult:
    """Fit and evaluate across purged expanding folds."""
    config = config or dataset.config
    window_index = dataset.window_index
    folds = list(folds) if folds is not None else purged_walk_forward(
        window_index, n_folds=config.n_folds,
        embargo_minutes=config.embargo_minutes,
        scheme=getattr(config, 'fold_scheme', 'calendar'))

    evaluations: list[FoldEvaluation] = []
    models: list[ForecastModel] = []
    books: list[Book] = []
    all_decisions: list[Decision] = []
    stateless_totals = pd.Series(dtype=float)
    scored_parts: list[pd.DataFrame] = []
    notes: list[str] = []

    for fold in folds:
        assert_no_leakage(fold)
        logger.info(fold.label())
        fit, train_table = fit_fold(dataset, fold.train, config, groups=groups)
        weights = recency_weights(train_table['window_open'], config.recency_half_life_days)
        model = fit_model(train_table, fit.baseline, config, groups=groups,
                          weights=weights, scoring=fit.bundle(config))
        models.append(model)

        test_table = apply_fold(dataset, fit, fold.test, config, groups=groups)
        test_table = test_table.assign(
            model_probability=model.predict(test_table),
            fold=fold.index,
        )
        scored_parts.append(test_table)

        stats: Optional[BookStats] = None
        if trade:
            book, decisions = run_book(test_table, config)
            books.append(book)
            all_decisions.extend(decisions)
            stateless_totals = stateless_totals.add(
                book.stateless_rejections, fill_value=0)
            stats = summarise(book, windows_available=effective_observations(test_table))

        evaluations.append(evaluate_fold(
            fold.index, test_table,
            test_table['model_probability'].to_numpy(),
            # The null the model was FITTED on, not always the baseline.
            skill_null_column(test_table, config.init_score_source),
            residual_scale=model.residual_scale,
            control_gain_share=model.control_importance_share,
            stats=stats,
        ))
        logger.info(evaluations[-1].line())

    scored = pd.concat(scored_parts, ignore_index=True).sort_values(
        ['window_open', 'symbol', 'offset'], ignore_index=True)

    continuous_book: Optional[Book] = None
    continuous_stats: Optional[BookStats] = None
    if trade:
        continuous_book, _ = run_book(scored, config)
        continuous_stats = summarise(
            continuous_book, windows_available=effective_observations(scored))

    if len(dataset.symbols) < 3:
        notes.append(
            f'universe is {len(dataset.symbols)} symbols; the cross_asset group '
            f'is thinner than its column count suggests')

    report = EvaluationReport(
        folds=evaluations, continuous=continuous_stats,
        config_provenance=config.provenance(), notes=notes,
    )
    return RunResult(
        report=report, models=models, books=books, continuous_book=continuous_book,
        rejections=rejection_histogram(all_decisions)
                   .add(stateless_totals, fill_value=0).astype(int),
        scored=scored,
    )


def cost_stress(
    scored: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    *,
    scenarios: Optional[dict[str, dict]] = None,
) -> pd.DataFrame:
    """Re-run the book under worse cost assumptions.

    The half-spread is an assumption, not a measurement — no Kalshi order ticket
    has been read against `core/costs.py` — and it is larger than the fee at
    price above 83c, where 0.07*p(1-p) falls below a cent. So it is the parameter
    most likely to be wrong
    and the one that moves the answer most. A strategy that survives only at the
    assumed spread has not been demonstrated.
    """
    scenarios = scenarios or {
        'baseline': {},
        'spread 2x': {'half_spread_cents': config.half_spread_cents * 2},
        'spread 3x': {'half_spread_cents': config.half_spread_cents * 3},
        'fee 2x': {'fee_rate': config.fee_rate * 2},
        'both 2x': {'half_spread_cents': config.half_spread_cents * 2,
                    'fee_rate': config.fee_rate * 2},
        'maker': {'assume_maker': True, 'half_spread_cents': 0.0},
    }
    rows = []
    for name, overrides in scenarios.items():
        variant = config.with_overrides(**overrides) if overrides else config
        book, _ = run_book(scored, variant)
        stats = summarise(book, windows_available=effective_observations(scored))
        rows.append({
            'scenario': name, 'trades': stats.n_trades, 'coverage': stats.coverage,
            'total_return': stats.total_return, 'sharpe': stats.sharpe,
            'win_rate': stats.win_rate, 'realised_edge_pp': stats.realised_edge_pp,
            'fees': stats.total_fees, 'max_drawdown': stats.max_drawdown,
        })
    return pd.DataFrame(rows)


def edge_curve(
    scored: pd.DataFrame,
    config: Config = DEFAULT_CONFIG,
    *,
    gates_pp: Sequence[float] = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0),
) -> pd.DataFrame:
    """How the book behaves as the abstention gate tightens. A DIAGNOSTIC.

    **Do not choose `min_edge_pp` from this curve.** The docstring here used to say
    "the right value of `min_edge_pp` is measured, not guessed", which invites
    exactly the thing that cannot be done: every row is scored on the *same*
    out-of-sample rows, so a value picked by reading them is no longer out of
    sample. It is the selection bias the archive section of `CLAUDE.md` was written
    about, one level up.

    Measured on 326 days, the money on a run this size is not stable enough to
    choose anything with. Across six configurations, forecast skill and realised
    return came apart completely: `(3,6)` with all groups had *higher* skill than
    the shipped configuration (+0.000365 against +0.000287) and lost half the
    account, and at essentially constant skill the return spanned -50% to +303%.
    The curve below moved non-monotonically over the same rows — 2.24, 2.12, 2.12,
    2.65, 2.61, 2.24, 1.63 — which by its own reading below means the tail is
    noise, and by the same token so is the peak.

    What the shape is still good for is a sanity check on *concentration*: a curve
    that improves as the gate tightens says the forecast is real and concentrated;
    one that peaks and falls says the tail is noise. Read the shape, not the
    argmax. Narrowing a configuration on skill and on a named mechanism is
    defensible — `scripts/ablate.py` is for that — and narrowing it on return is
    not.
    """
    rows = []
    for gate in gates_pp:
        book, decisions = run_book(scored, config.with_overrides(min_edge_pp=gate))
        stats = summarise(book, windows_available=effective_observations(scored))
        rows.append({
            'min_edge_pp': gate, 'trades': stats.n_trades, 'coverage': stats.coverage,
            'total_return': stats.total_return, 'sharpe': stats.sharpe,
            'win_rate': stats.win_rate, 'mean_edge_pp': stats.mean_edge_pp,
            'realised_edge_pp': stats.realised_edge_pp,
        })
    return pd.DataFrame(rows)
