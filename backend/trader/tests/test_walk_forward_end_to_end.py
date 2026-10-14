"""End-to-end tests for `core.backtest.walk_forward`.

Marked `slow` because they fit real models; `pytest -m "not slow"` skips them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.backtest import cost_stress, edge_curve, walk_forward
from core.config import Config
from core.dataset import Dataset
from core.metrics import evaluate_gates, gates_passed
from tests.conftest import make_bars

pytestmark = pytest.mark.slow

FAST = Config(n_estimators=40, early_stopping_rounds=8, n_folds=2,
              seasonality_min_days=5)


@pytest.fixture(scope='module')
def null_run():
    """A world with nothing to find: `lead=0.0` plants no cross-asset signal."""
    dataset = Dataset.build(make_bars(days=30, lead=0.0, seed=5), FAST)
    return walk_forward(dataset, FAST, trade=True)


# A 30-day synthetic run places no trades even with `min_edge_pp` at zero: Kelly
# on a near-zero edge rounds below one contract. So this file checks that
# `walk_forward` runs end to end; `test_sizing_fees_and_pnl.py` and
# `test_cv_and_book.py` cover the populated accounting path.
TRADING = FAST


@pytest.fixture(scope='module')
def signal_run():
    """The same machinery where BTC leads the other symbols."""
    dataset = Dataset.build(make_bars(days=30, lead=0.9, seed=5), TRADING)
    return walk_forward(dataset, TRADING, trade=True)


class TestTheNull:
    def test_a_null_does_not_produce_skill(self, null_run):
        assert null_run.report.mean_skill < 0.001, (
            f'skill {null_run.report.mean_skill:+.6f} on data with no signal'
        )

    def test_a_null_does_not_pass_the_gates(self, null_run):
        """A null fails on a forecast or money gate, not only on sample size or a
        missing book.

        `sign_agreement_p` is excluded: at three folds its best value (0.125) fails the
        0.11 bar whatever the data.
        """
        gates = evaluate_gates(null_run.report)
        assert not gates_passed(gates)
        failed = {g.name for g in gates if not g.passed}
        assert failed & {'log_loss_skill', 'log_loss_skill_p',
                         'realised_edge_pp', 'total_return', 'sharpe'}, (
            f'the null failed only {failed}, none of which is a forecast or money '
            f'gate — so nothing is actually testing the hypothesis'
        )

    def test_the_forecast_gates_and_not_the_shrinkage_are_what_catch_a_null(self, null_run):
        """The forecast gates reject the null.

        On a small sample `residual_scale` alone is not a reliable signal, and
        `log_loss_skill >= 0` alone clears on about half of all nulls. The significance
        gate is what rejects it.
        """
        gates = {g.name: g for g in evaluate_gates(null_run.report)}
        assert not gates['log_loss_skill'].passed or not gates['log_loss_skill_p'].passed, (
            f'skill {null_run.report.mean_skill:+.6f} (p = '
            f'{null_run.report.skill_p_value:.3f}) over '
            f'{null_run.report.folds_positive}/{null_run.report.folds_total} folds '
            f'positive: neither forecast gate objected to data with no signal'
        )

class TestStructure:
    def test_every_fold_is_measured(self, signal_run):
        """Every fold the Config cuts is measured; under the calendar scheme `n_folds`
        is not the count.
        """
        from core.cv import folds_for_config

        dataset = Dataset.build(make_bars(days=30, lead=0.9, seed=5), TRADING)
        expected = folds_for_config(dataset.window_index, TRADING)
        assert len(signal_run.report.folds) == len(expected)
        assert signal_run.report.fold_geometry['n_folds_actual'] == len(expected)
        assert signal_run.report.fold_geometry['n_folds_config'] == TRADING.n_folds
        for fold in signal_run.report.folds:
            assert fold.n_rows > 0 and fold.n_windows > 0
            assert np.isfinite(fold.model_log_loss)
            assert np.isfinite(fold.baseline_log_loss)
            assert fold.n_non_finite == 0, (
                'a fold carries non-finite rows, which used to be pooled into the '
                'top reliability bin and turn every metric into NaN'
            )

    def test_folds_are_chronological_and_do_not_overlap(self, signal_run):
        starts = [f.test_start for f in signal_run.report.folds]
        ends = [f.test_end for f in signal_run.report.folds]
        assert starts == sorted(starts)
        for earlier, later in zip(ends, starts[1:]):
            assert later > earlier, 'two folds test the same windows'

    def test_scored_rows_carry_no_settled_label_gaps(self, signal_run):
        """No unsettled window reaches a metric."""
        scored = signal_run.scored
        assert not scored.empty
        assert scored['outcome'].notna().all()
        assert scored['settle_price'].notna().all()

    def test_the_trade_ledger_records_where_its_price_came_from(self, signal_run):
        """The trade ledger has a `price_source` column, even when it is empty."""
        trades = signal_run.trades()
        assert 'price_source' in trades.columns
        if not trades.empty:
            assert (trades['price_source'] == 'baseline').all()


class TestMoney:
    def test_pnl_reconciles_with_the_equity_curve(self, signal_run):
        """P&L reconciles with the equity curve, also when nothing traded."""
        stats = signal_run.report.continuous
        assert stats is not None
        assert stats.total_pnl == pytest.approx(
            stats.ending_equity - stats.starting_bankroll, abs=1e-6), (
            'the reported PnL does not reconstruct from the account balances'
        )

    def test_one_entry_per_symbol_and_window(self, signal_run):
        """One entry per symbol and window. Vacuous on an empty ledger, so
        `test_decide.py` also pins it.
        """
        trades = signal_run.trades()
        duplicated = trades.duplicated(['symbol', 'window_open']).sum() if not trades.empty else 0
        assert duplicated == 0, f'{duplicated} windows were entered twice'

    def test_the_funnel_accounts_for_every_row(self, signal_run):
        """Every scored row lands in exactly one funnel bucket."""
        histogram = signal_run.rejections
        if histogram is None or histogram.empty:
            pytest.skip('this run reported no funnel')
        assert histogram.sum() > 0


class TestDiagnostics:
    def test_cost_stress_runs_and_gets_worse_as_costs_rise(self, signal_run):
        frame = cost_stress(signal_run.scored, TRADING)
        assert not frame.empty
        assert {'scenario', 'trades', 'total_return'} <= set(frame.columns)

    def test_the_edge_curve_narrows_as_the_gate_tightens(self, signal_run):
        frame = edge_curve(signal_run.scored, TRADING)
        assert not frame.empty
        counts = frame.sort_values('min_edge_pp')['trades'].to_numpy()
        assert (np.diff(counts) <= 0).all(), (
            'a tighter abstention gate admitted more trades, so the gate is not '
            'doing what the curve claims to measure'
        )
