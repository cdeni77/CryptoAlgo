"""Walk-forward: fit, score, trade and measure, fold by fold, in order.

Each fold fits seasonality, the volatility model, the baseline and the
classifier on training windows only, scores the test block with
`core.dataset.apply_fold`, and trades through the same `decide()` live uses.

Two books: a per-fold book starts fresh each fold, so folds compare; one
continuous book runs the whole out-of-sample span on a single bankroll.

Settlement runs before the next window's decision. A position settles at the
instant the next window opens; deciding first would stake the bankroll twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.book import Book, BookStats, summarise
from core.config import Config, DEFAULT_CONFIG
from core.cv import WindowFold, assert_no_leakage, effective_observations, folds_for_config, recency_weights
from core.dataset import Dataset, apply_fold, fit_fold
from core.decide import Decision, Reason, decide_window, rejection_histogram, stateless_screen
from core.metrics import (EvaluationReport, FoldEvaluation, evaluate_fold, sharpe_moments,
                          trade_gate_values)
from core.model import ForecastModel, fit_for_config, table_groups

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

    `scored` needs `model_probability` alongside the window columns. No lookahead:
    the bankroll depends only on windows that have already settled.
    """
    book = book or Book(config=config)
    decisions: list[Decision] = []
    # Screen the whole span once, vectorised: the state-independent gates
    # reject most rows before the per-window loop.
    screened, stateless_rejections = stateless_screen(scored, config)
    outcomes = settlement_outcomes(scored)
    for window_open, rows in screened.groupby('window_open', sort=True):
        # Settle first: a position from the previous window matures at this
        # instant. Matured positions are found by settle_time, not by scanning outcomes.
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
    """The forecaster `log_loss_skill` is measured against.

    Skill means beating what the model was fitted to correct: the market for a
    market-initialised model, else the baseline. Falls back to the baseline,
    with a warning, when the market column is absent or all NaN.
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


def stack_folds(dataset: Dataset, config: Config) -> list[WindowFold]:
    """Folds with test blocks cut over quoted windows only, and training sets
    of every window before the embargo.

    Stage 2 scores only where a usable quote exists. Stage 1 still trains on all
    history (or `fold_train_days` of it). Unquoted rows in a tested window score NaN.
    """
    windows = dataset.windows
    market = (pd.to_numeric(windows['market_probability'], errors='coerce')
              if 'market_probability' in windows.columns
              else pd.Series(np.nan, index=windows.index))
    quoted = pd.DatetimeIndex(sorted(
        windows.loc[market.notna(), 'window_open'].unique()))
    if not len(quoted):
        raise ValueError(
            'stack mode found no window carrying a usable quote under the '
            'source/age filters, so there is nothing for stage 2 to fit or score')
    everything = dataset.window_index
    embargo = pd.Timedelta(minutes=int(config.embargo_minutes))
    train_days = getattr(config, 'fold_train_days', None)
    out = []
    for fold in folds_for_config(quoted, config):
        edge = fold.test[0] - embargo
        train = everything[everything < edge]
        if train_days is not None:
            train = train[train >= edge - pd.Timedelta(days=float(train_days))]
        out.append(WindowFold(index=fold.index, train=train, test=fold.test,
                              embargo_minutes=fold.embargo_minutes))
    logger.info('stack mode: %d fold(s) tested over %s quoted windows, trained '
                'on all bars before each', len(out), f'{len(quoted):,}')
    return out


def settlement_outcomes(scored: pd.DataFrame) -> dict:
    """How each window settled, preferring the venue's own result.

    Settling on our Coinbase label while trading venue-priced quotes lets the
    model bet the disagreement between the two indices. Our label is the
    fallback only where the venue's is missing (older markets are purged).
    A window with neither label is omitted, never guessed.
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
    supplied = folds is not None
    stacked = getattr(config, 'stack_mode', 'off') != 'off'
    if supplied:
        folds = list(folds)
    else:
        folds = (stack_folds(dataset, config) if stacked
                 else folds_for_config(window_index, config))
    # Stack mode builds stage 1's spot groups plus any stage-2 book columns.
    groups = table_groups(config, groups)

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
        model = fit_for_config(train_table, fit.baseline, config, groups=groups,
                               weights=weights, scoring=fit.bundle(config))
        models.append(model)

        test_table = apply_fold(dataset, fit, fold.test, config, groups=groups)
        test_table = test_table.assign(
            **model.score(test_table),
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
            # The null the model was fitted on, not always the baseline. Read
            # from the model: a stacked one is market-relative whatever the config says.
            skill_null_column(test_table, getattr(model, 'init_score_source',
                                                  config.init_score_source)),
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

    # The fold count actually cut. Under the calendar scheme `n_folds` is not a count.
    geometry = {
        'n_folds_actual': len(evaluations),
        'n_folds_config': int(config.n_folds),
        'scheme': getattr(config, 'fold_scheme', 'calendar'),
        'fold_block_days': getattr(config, 'fold_block_days', None),
        'min_test_windows': getattr(config, 'min_test_windows', None),
        'fold_train_days': getattr(config, 'fold_train_days', None),
        'embargo_minutes': int(config.embargo_minutes),
        'folds_supplied_by_caller': supplied,
        'test_windows': [int(e.n_windows) for e in evaluations],
    }
    report = EvaluationReport(
        folds=evaluations, continuous=continuous_stats,
        config_provenance=config.provenance(), notes=notes,
        fold_geometry=geometry,
        sharpe_moments=(sharpe_moments(continuous_book.trades())
                        if continuous_book is not None else {}),
        trade_metrics=(trade_gate_values(continuous_book.trades())
                       if continuous_book is not None else {}),
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

    The half-spread is an assumption, and above the fee crossover it is the
    larger cost. A strategy that survives only at the assumed spread is not demonstrated.
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
    """How the book behaves as the abstention gate tightens. A diagnostic.

    Do not choose `min_edge_pp` from this curve: every row is scored on the same
    out-of-sample rows, so a value picked from them is no longer out of sample.
    Read the shape, not the argmax: improvement as the gate tightens means a
    concentrated forecast; a peak then a fall means the tail is noise.
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
