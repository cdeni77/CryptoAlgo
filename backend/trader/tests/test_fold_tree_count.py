"""A fold's tree count must be visible, and a degenerate fit must say so.

Early stopping decides on a block whose log-loss noise is ~600x the effect it
is meant to resolve, so counts swing 1 to 167 across adjacent weeks of
near-identical data. A fold that keeps one or two trees ships a near-constant
correction: it cannot disagree with the market, takes zero trades, and reads
downstream as "no edge this week". That chain was followed to a conclusion on
2026-11-05 because the number was never printed.
"""
import pandas as pd
import pytest

from core.metrics import MIN_USEFUL_TREES, FoldEvaluation


def _fold(**over):
    base = dict(
        index=3, test_start=pd.Timestamp('2026-09-26', tz='UTC'),
        test_end=pd.Timestamp('2026-10-03', tz='UTC'), n_rows=7764,
        n_windows=1986, model_log_loss=0.5, baseline_log_loss=0.5,
        model_brier=0.2, baseline_brier=0.2, model_ece=0.0095,
        baseline_ece=0.0100, residual_scale=1.0, control_gain_share=0.0)
    base.update(over)
    return FoldEvaluation(**base)


def test_the_tree_count_appears_on_the_fold_line():
    assert 'trees 36-41' in _fold(n_trees=(38, 41, 36, 40, 39)).line()


def test_an_ensemble_of_stubs_is_called_degenerate():
    # One tree at learning_rate 0.03 moves the forecast a fraction of a
    # percentage point. That is a constant, not a weak forecast.
    assert 'DEGENERATE FIT' in _fold(n_trees=(1, 2, 1, 2, 1)).line()


def test_one_stunted_member_does_not_condemn_the_ensemble():
    """`best_iteration` is member 0's alone.

    Judging five members by the first would call a healthy ensemble degenerate
    whenever member 0 happened to stop early -- and they are all judged against
    the same inner block, so the first is not representative by construction.
    """
    line = _fold(n_trees=(1, 44, 51, 48, 47)).line()
    assert 'trees 1-51' in line
    assert 'DEGENERATE' not in line


def test_the_threshold_is_on_the_median_and_inclusive():
    assert 'DEGENERATE FIT' in _fold(n_trees=(MIN_USEFUL_TREES,) * 5).line()
    assert 'DEGENERATE FIT' not in _fold(n_trees=(MIN_USEFUL_TREES + 1,) * 5).line()


def test_an_unknown_count_prints_nothing_rather_than_a_fake_zero():
    # A zero here would read as the worst possible fit. Absent is absent.
    line = _fold(n_trees=None).line()
    assert 'trees' not in line
    assert 'DEGENERATE' not in line


def test_the_backtest_actually_passes_the_count():
    import inspect
    from core import backtest
    src = inspect.getsource(backtest.walk_forward)
    assert 'n_trees=_member_trees(model)' in src, (
        'the fold line must receive EVERY member, not just best_iteration')


def test_a_booster_without_early_stopping_reports_its_real_size():
    """`best_iteration` is 0 unless early stopping ran.

    Under `early_stopping_mode` 'cv' or 'off' no callback runs, so every
    booster reports 0 and a fully trained 400-tree ensemble was printed as
    "trees 0 <- DEGENERATE FIT" while placing 1,319 trades.
    """
    from core.model import ForecastModel

    class Booster:
        def __init__(self, stopped, size):
            self.best_iteration = stopped
            self._size = size
        def num_trees(self):
            return self._size

    m = ForecastModel.__new__(ForecastModel)
    object.__setattr__(m, 'members', [Booster(0, 400), Booster(0, 400)])
    object.__setattr__(m, 'booster', None)
    assert m.member_trees == (400, 400)

    object.__setattr__(m, 'members', [Booster(17, 400)])
    assert m.member_trees == (17,), 'early stopping wins when it ran'
