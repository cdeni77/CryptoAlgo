from types import SimpleNamespace

from core.reason_codes import ReasonCode
from scripts.optimize import (
    _normalize_early_trade_feasibility_mode,
    _trade_frequency_attrs,
    _trade_frequency_penalty,
    _reject_trial,
    build_trade_frequency_diagnostics,
)


class DummyTrial:
    def __init__(self):
        self.user_attrs = {}
        self.number = 7
        self.study = SimpleNamespace(study_name='optimize_BTC_t', user_attrs={'coin_name': 'BTC'})

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def test_mode_normalization():
    assert _normalize_early_trade_feasibility_mode('reject') == 'reject'
    assert _normalize_early_trade_feasibility_mode('PENALIZE') == 'penalize'
    assert _normalize_early_trade_feasibility_mode('invalid-mode') == 'penalize'


def test_trade_frequency_attrs_complete():
    attrs = _trade_frequency_attrs(
        folds_observed=3,
        observed_tpy=9.0,
        min_feasible_tpy=24.0,
        target_tpy=52.0,
        target_trades_per_week=1.0,
        min_trade_frequency_ratio=0.46,
        strategy_family='momentum_trend',
        trade_freq_bucket='balanced',
        fold_trade_counts=[1, 2, 3],
        coin_name='BTC',
    )
    for key in (
        'observed_folds', 'projected_trades_per_year_partial', 'projected_trades_per_year_threshold',
        'target_trades_per_year', 'target_trades_per_week', 'min_trade_frequency_ratio',
        'strategy_family', 'trade_freq_bucket', 'fold_trade_counts', 'coin_name',
    ):
        assert key in attrs


def test_trade_frequency_penalty_monotonic():
    penalty_bad = _trade_frequency_penalty(observed_tpy=5.0, min_feasible_tpy=25.0, penalty_scale=3.0)
    penalty_less_bad = _trade_frequency_penalty(observed_tpy=15.0, min_feasible_tpy=25.0, penalty_scale=3.0)
    assert penalty_bad > penalty_less_bad > 0.0


def test_reject_trial_preserves_code_and_attrs():
    trial = DummyTrial()
    score = _reject_trial(
        trial,
        code=ReasonCode.UNLIKELY_TRADE_FREQUENCY,
        reason='unlikely_trade_frequency:8<20',
        stage='early_feasibility',
        observed=8.0,
        threshold=20.0,
        fold_results=[{'n_trades': 1, 'sharpe': 0.1}],
        extra_attrs={'target_trades_per_week': 1.0},
    )
    assert score < 0.0
    assert trial.user_attrs['reject_code'] == str(ReasonCode.UNLIKELY_TRADE_FREQUENCY)
    assert trial.user_attrs['target_trades_per_week'] == 1.0


def test_trade_frequency_diagnostics_summary():
    trials = [
        SimpleNamespace(user_attrs={'reject_code': str(ReasonCode.UNLIKELY_TRADE_FREQUENCY), 'projected_trades_per_year_partial': 8.0, 'projected_trades_per_year_threshold': 20.0, 'strategy_family': 'breakout', 'trade_freq_bucket': 'aggressive'}),
        SimpleNamespace(user_attrs={'reject_code': str(ReasonCode.UNLIKELY_TRADE_FREQUENCY), 'projected_trades_per_year_partial': 10.0, 'projected_trades_per_year_threshold': 21.0, 'strategy_family': 'breakout', 'trade_freq_bucket': 'aggressive'}),
        SimpleNamespace(user_attrs={'reject_code': 'OTHER'}),
    ]
    diag = build_trade_frequency_diagnostics(trials)
    assert diag['early_reject_count'] == 2
    assert diag['recent_projected_tpy_stats']['median'] == 9.0
    assert diag['dominant_strategy_families']['breakout'] == 2
