"""Fold construction and account accounting — the two places arithmetic hides.

Cross-validation splits on the window, never on the row: four decision offsets
share one settlement, so a row-level split puts offset 3 in train and offset 12
in test, which is not a subtle leak but the answer itself, nine minutes closer.

The book's one trap is annualisation. A strategy trading 3% of available windows
and scored as though it traded all of them reports a Sharpe inflated by the
reciprocal of its duty cycle. That happened in this project before — 2.28 became
1.19 once the 27% duty cycle was accounted for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.book import Book, Position, SECONDS_PER_YEAR, Settlement, summarise
from core.config import Config
from core.cv import (
    LeakageError, assert_no_leakage, effective_observations, purged_walk_forward,
    recency_weights, rows_for,
)
from core.decide import Side, decide

W = pd.Timestamp('2026-01-01 00:00', tz='UTC')


def window_index(days=200):
    return pd.date_range('2025-01-01', periods=days * 96, freq='15min', tz='UTC')


# ---------------------------------------------------------------- folds

def test_folds_never_overlap_and_honour_the_embargo():
    folds = purged_walk_forward(window_index(), n_folds=6, embargo_minutes=1440)
    assert len(folds) == 6
    for fold in folds:
        assert_no_leakage(fold)
        assert fold.gap_minutes >= 1440


def test_the_embargo_covers_the_feature_lookback_not_just_the_label():
    """A fifteen-minute label needs a fifteen-minute purge.

    What needs a day is `log_rv_1440`: a training row immediately after a test
    block computes it from test-period bars. Purging for the label and forgetting
    the features is the standard version of this mistake, and it leaks in the
    direction that inflates measured skill.
    """
    folds = purged_walk_forward(window_index(), n_folds=4, embargo_minutes=1440)
    for fold in folds:
        assert (fold.test_start - fold.train_end) > pd.Timedelta(minutes=1440)


def test_a_fold_with_an_overlapping_split_is_refused():
    index = window_index(30)
    fold = purged_walk_forward(index, n_folds=3, embargo_minutes=60)[0]
    from dataclasses import replace
    broken = replace(fold, train=fold.train.append(fold.test[:5]))
    with pytest.raises(LeakageError, match='in both train and test'):
        assert_no_leakage(broken)


def test_a_fold_with_too_small_a_gap_is_refused():
    index = window_index(30)
    fold = purged_walk_forward(index, n_folds=3, embargo_minutes=15)[0]
    from dataclasses import replace
    broken = replace(fold, embargo_minutes=10_000)
    with pytest.raises(LeakageError, match='under the'):
        assert_no_leakage(broken)


def test_folds_are_expanding_not_rolling():
    folds = purged_walk_forward(window_index(), n_folds=5, embargo_minutes=1440)
    sizes = [len(f.train) for f in folds]
    assert sizes == sorted(sizes)
    assert sizes[-1] > sizes[0] * 2


def test_too_few_windows_is_an_error_not_a_degenerate_fold():
    with pytest.raises(ValueError, match='cannot support'):
        purged_walk_forward(window_index(1)[:6], n_folds=6)


def test_rows_are_selected_by_window_membership_not_a_timestamp_slice():
    """A `>=`/`<` slice on decision_time would split a window across the boundary.

    Offset 3 inside train, offset 12 inside test — the same fifteen minutes on
    both sides, which is exactly the leak the module exists to prevent.
    """
    table = pd.DataFrame({
        'window_open': [W, W, W, W, W + pd.Timedelta(minutes=15)],
        'offset': [3, 6, 9, 12, 3],
        'decision_time': [W + pd.Timedelta(minutes=m) for m in (3, 6, 9, 12, 18)],
    })
    mask = rows_for(table, pd.DatetimeIndex([W]))
    assert mask.tolist() == [True, True, True, True, False]


def test_effective_observations_counts_windows_not_rows():
    """Four offsets per window means a row count overstates the sample fourfold.

    And a standard error computed from it is half what it should be.
    """
    table = pd.DataFrame({
        'symbol': ['BTC-USD'] * 4 + ['ETH-USD'] * 4,
        'window_open': [W] * 4 + [W] * 4,
        'offset': [3, 6, 9, 12] * 2,
    })
    assert len(table) == 8
    assert effective_observations(table) == 2


def test_recency_weights_are_off_unless_asked_for():
    times = pd.Series(pd.date_range('2025-01-01', periods=100, freq='15min', tz='UTC'))
    assert recency_weights(times, None) is None
    assert recency_weights(times, 0) is None
    weights = recency_weights(times, 1.0)
    assert weights is not None
    assert weights[-1] == pytest.approx(1.0)
    assert weights[0] < weights[-1]


# ----------------------------------------------------------------- book

def make_position(side='up', contracts=3, price=0.86, outlay=2.66, fee=0.05,
                  window=W, symbol='BTC-USD'):
    return Position(symbol=symbol, window_open=window,
                    settle_time=window + pd.Timedelta(minutes=15), offset=9,
                    side=Side.UP if side == 'up' else Side.DOWN,
                    contracts=contracts, price=price, outlay=outlay, fee=fee,
                    model_probability=0.90, baseline_probability=0.86, edge=0.02)


def test_a_winning_contract_pays_a_dollar():
    position = make_position(contracts=3)
    assert position.payout(settled_up=True) == 3.0
    assert position.payout(settled_up=False) == 0.0
    short = make_position(side='down', contracts=3)
    assert short.payout(settled_up=False) == 3.0
    assert short.payout(settled_up=True) == 0.0


def test_a_position_settles_exactly_once():
    book = Book(config=Config())
    decision = decide(dict(symbol='BTC-USD', window_open=W,
                           settle_time=W + pd.Timedelta(minutes=15), offset=9,
                           baseline_probability=0.88, model_probability=0.95),
                      Config(), bankroll=100.0)
    assert book.record(decision) is not None
    first = book.settle({('BTC-USD', W): True})
    assert len(first) == 1
    second = book.settle({('BTC-USD', W): True})
    assert second == []


def test_settlement_is_keyed_on_symbol_and_window_together():
    """The three symbols settle at the same instant on different outcomes.

    Keying on the timestamp alone would settle all three against whichever one
    was looked up.
    """
    book = Book(config=Config())
    for symbol in ('BTC-USD', 'ETH-USD'):
        book.open_positions.append(make_position(symbol=symbol))
    book.bankroll -= 2 * 2.66
    settled = book.settle({('BTC-USD', W): True})
    assert len(settled) == 1
    assert settled[0].position.symbol == 'BTC-USD'
    assert len(book.open_positions) == 1


def test_equity_carries_open_positions_at_cost():
    """Marking an open binary at our own forecast books belief as profit.

    Which is how a losing system draws a rising equity curve.
    """
    book = Book(config=Config(), bankroll=90.0)
    book.open_positions.append(make_position(outlay=10.0))
    assert book.equity == pytest.approx(100.0)


def test_the_bankroll_floor_halts_the_book():
    config = Config(starting_bankroll=100.0, ruin_floor_fraction=0.9)
    book = Book(config=config, bankroll=95.0)
    position = make_position(outlay=10.0)
    book.open_positions.append(position)
    book.bankroll -= position.outlay        # as `record` would have
    assert book.halted_at is None, 'halted before anything settled'
    book.settle({('BTC-USD', W): False})    # a loss: nothing comes back
    assert book.bankroll == pytest.approx(85.0)
    assert book.halted_at is not None, 'below the floor and still trading'


def test_the_sharpe_is_annualised_on_trades_placed_not_windows_available():
    """Scaling by windows instead would inflate it by 1/coverage."""
    config = Config()
    book = Book(config=config)
    rng = np.random.default_rng(4)
    for i in range(200):
        window = W + pd.Timedelta(minutes=15 * i)
        position = make_position(window=window)
        book.open_positions.append(position)
        book.bankroll -= position.outlay
        book.settle({('BTC-USD', window): bool(rng.random() < 0.9)})

    dense = summarise(book, windows_available=200)
    sparse = summarise(book, windows_available=20_000)
    assert dense.sharpe == pytest.approx(sparse.sharpe), (
        'the Sharpe moved with the number of windows offered, so it is being '
        'scaled by opportunities rather than by trades'
    )
    assert dense.coverage == pytest.approx(1.0)
    assert sparse.coverage == pytest.approx(0.01)

    span = (book.settlements[-1].position.settle_time
            - book.settlements[0].position.window_open).total_seconds()
    assert dense.trades_per_year == pytest.approx(200 * SECONDS_PER_YEAR / span, rel=1e-6)


def test_realised_edge_is_measured_not_predicted():
    """Predicted edge is what the model claimed; realised is what happened.

    The gap between them is the winner's curse, and it comes before any Sharpe.
    """
    config = Config()
    book = Book(config=config)
    for i in range(50):
        window = W + pd.Timedelta(minutes=15 * i)
        book.open_positions.append(make_position(window=window))
        book.bankroll -= 2.66
        book.settle({('BTC-USD', window): i % 2 == 0})   # a 50% win rate
    stats = summarise(book, windows_available=50)
    assert stats.win_rate == pytest.approx(0.5)
    assert stats.mean_edge_pp == pytest.approx(2.0, abs=0.01)   # what was claimed
    assert stats.realised_edge_pp < 0                            # what happened


def test_an_empty_book_reports_no_trades_rather_than_a_nan_return():
    stats = summarise(Book(config=Config()), windows_available=1000)
    assert stats.n_trades == 0
    assert stats.coverage == 0.0
    assert stats.total_return == pytest.approx(0.0)
    assert np.isnan(stats.win_rate)


def test_the_growth_multiple_is_legible_where_a_percentage_is_not():
    book = Book(config=Config(), bankroll=400.0)
    stats = summarise(book, windows_available=10)
    assert stats.growth_multiple == pytest.approx(4.0)
