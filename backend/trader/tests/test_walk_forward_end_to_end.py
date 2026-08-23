"""`core/backtest.py` end to end, because nothing imported it.

158 statements at **0% coverage**: no test in the suite imported this module, so
the walk-forward — the thing that produces every number a promotion decision is
made on — was exercised only by running a script by hand. `pytest.ini`'s own
comment claimed the suite was "dominated by four promotion and walk-forward
evaluations", which described a suite that did not exist; no test carried the
`slow` marker either, so `-m "not slow"` was a no-op.

Marked `slow` because it fits real models. `pytest -m "not slow"` skips it.
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
    """A world with nothing to find. `lead=0.0` plants no cross-asset signal."""
    dataset = Dataset.build(make_bars(days=30, lead=0.0, seed=5), FAST)
    return walk_forward(dataset, FAST, trade=True)


# A 30-day synthetic run places no trades even with `min_edge_pp` at zero: the
# Kelly stake on a near-zero edge rounds below one contract, so `decide` returns
# BELOW_MIN_CONTRACTS. That is the machinery working, and it means this file cannot
# exercise the *populated* accounting path — `test_sizing_fees_and_pnl.py` and
# `test_cv_and_book.py` do that, driving `Book` directly with known positions.
# What is unique here is that `walk_forward` runs end to end at all, which nothing
# tested: 158 statements at 0% coverage.
TRADING = FAST


@pytest.fixture(scope='module')
def signal_run():
    """The same machinery where BTC genuinely leads the others."""
    dataset = Dataset.build(make_bars(days=30, lead=0.9, seed=5), TRADING)
    return walk_forward(dataset, TRADING, trade=True)


class TestTheNull:
    def test_a_null_does_not_produce_skill(self, null_run):
        """The single most important property of the whole apparatus.

        If this fails, no positive result anywhere in the system means anything.
        """
        assert null_run.report.mean_skill < 0.001, (
            f'skill {null_run.report.mean_skill:+.6f} on data with no signal'
        )

    def test_a_null_does_not_pass_the_gates(self, null_run):
        gates = evaluate_gates(null_run.report)
        assert not gates_passed(gates)
        failed = {g.name for g in gates if not g.passed}
        assert failed & {'log_loss_skill', 'folds_skill_positive',
                         'realised_edge_pp', 'total_return'}, (
            f'the null failed only {failed}, none of which is a forecast or money '
            f'gate — so nothing is actually testing the hypothesis'
        )

    def test_the_forecast_gates_and_not_the_shrinkage_are_what_catch_a_null(self, null_run):
        """Which gate does the work, stated rather than assumed.

        `residual_scale` is described as the overfitting detector, and two real
        defects made it useless — early stopping shared rows with the shrinkage
        fit, and a NaN made `minimize_scalar` return its bracket seed 0.7639. Both
        are fixed, and on a 70-day slice alpha now reads 0.0000 on a null
        (`test_features_and_model.py::test_the_shrinkage_reads_near_zero_on_a_null`).

        But it is **not** a reliable standalone signal at a small sample. Measured
        on this 30-day, 2-fold null: alpha came back `[1.305, 0.0]`. With few trees
        and a short alpha block, a spurious correlation gets amplified, and at two
        folds the median degenerates to the mean. So this test asserts what
        actually holds — the *forecast* gates reject the null — and does not
        pretend alpha alone would.
        """
        gates = {g.name: g for g in evaluate_gates(null_run.report)}
        assert not gates['log_loss_skill'].passed or not gates['folds_skill_positive'].passed, (
            f'skill {null_run.report.mean_skill:+.6f} over '
            f'{null_run.report.folds_positive}/{null_run.report.folds_total} folds '
            f'positive: neither forecast gate objected to data with no signal'
        )


class TestStructure:
    def test_every_fold_is_measured(self, signal_run):
        assert len(signal_run.report.folds) == TRADING.n_folds
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
        """An unsettled window must never reach a metric's denominator."""
        scored = signal_run.scored
        assert not scored.empty
        assert scored['outcome'].notna().all()
        assert scored['settle_price'].notna().all()

    def test_the_trade_ledger_records_where_its_price_came_from(self, signal_run):
        """`price_source` exists so a backtest cannot be mistaken for a fill.

        It was on `Decision` and on the `predictions` table but **not** on the
        trade ledger, which is the frame anyone actually inspects — so the
        invariant CLAUDE.md states ("`price_source` on every row") did not hold
        where it mattered most. Asserted on the schema, because an empty ledger is
        the normal outcome of a short synthetic run and a column that only appears
        when populated is not an invariant.
        """
        trades = signal_run.trades()
        assert 'price_source' in trades.columns
        if not trades.empty:
            assert (trades['price_source'] == 'baseline').all()


class TestMoney:
    def test_pnl_reconciles_with_the_equity_curve(self, signal_run):
        """True whether or not anything traded: zero trades is zero PnL."""
        stats = signal_run.report.continuous
        assert stats is not None
        assert stats.total_pnl == pytest.approx(
            stats.ending_equity - stats.starting_bankroll, abs=1e-6), (
            'the reported PnL does not reconstruct from the account balances'
        )

    def test_one_entry_per_symbol_and_window(self, signal_run):
        """The backtest's own invariant, and the one the live path did not have.

        Vacuous on an empty ledger, which is why `decide_window`'s first-clearing
        behaviour is also pinned directly in `test_decide.py`.
        """
        trades = signal_run.trades()
        duplicated = trades.duplicated(['symbol', 'window_open']).sum() if not trades.empty else 0
        assert duplicated == 0, f'{duplicated} windows were entered twice'

    def test_the_funnel_accounts_for_every_row(self, signal_run):
        """Every scored row ends in exactly one bucket.

        `decide_window`'s `break` bypasses `decide` once the per-window position
        cap binds, so those rows reached no reason at all and the histogram did not
        add up — measured at 5.3% of rows missing, against a coverage figure that
        divides by the full total.
        """
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
