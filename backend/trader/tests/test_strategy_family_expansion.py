import scripts.optimize as optimize
from core.coin_profiles import COIN_PROFILES
from core.strategies import get_strategy_family


class _StubTrial:
    def __init__(self, strategy_family: str):
        self.strategy_family = strategy_family
        self.params = {}

    def suggest_categorical(self, name, choices):
        if name == 'strategy_family':
            value = self.strategy_family
        elif name == 'trade_freq_bucket':
            value = 'balanced'
        else:
            value = choices[0]
        self.params[name] = value
        return value

    def suggest_float(self, name, low, high, step=None):
        value = round((low + high) / 2.0, 6)
        self.params[name] = value
        return value

    def suggest_int(self, name, low, high, step=1):
        value = low
        self.params[name] = value
        return value


def test_new_strategy_families_registered_and_evaluable() -> None:
    for family in ('trend_pullback', 'breakout_expansion'):
        strategy = get_strategy_family(family)
        assert strategy.name == family


def test_create_trial_profile_samples_family_specific_knobs() -> None:
    trend_trial = _StubTrial('trend_pullback')
    breakout_trial = _StubTrial('breakout_expansion')

    trend_profile = optimize.create_trial_profile(trend_trial, 'SOL')
    breakout_profile = optimize.create_trial_profile(breakout_trial, 'SOL')

    assert trend_profile.strategy_family == 'trend_pullback'
    assert 'pullback_depth_threshold' in trend_trial.params
    assert 'rebound_confirmation_threshold' in trend_trial.params
    assert 'trend_strength_min' in trend_trial.params
    assert 'pullback_lookback' in trend_trial.params

    assert breakout_profile.strategy_family == 'breakout_expansion'
    assert 'breakout_lookback' in breakout_trial.params
    assert 'breakout_buffer' in breakout_trial.params
    assert 'expansion_confirm_threshold' in breakout_trial.params


def test_profile_from_params_supports_new_families_on_all_coins() -> None:
    for coin_name in COIN_PROFILES:
        trend_profile = optimize.profile_from_params(
            {
                'strategy_family': 'trend_pullback',
                'trade_freq_bucket': 'balanced',
                'pullback_depth_threshold': 0.02,
                'rebound_confirmation_threshold': 0.005,
                'trend_strength_min': 0.003,
                'pullback_lookback': 24,
                'cooldown_hours': 4.0,
                'max_hold_hours': 72,
                'vol_mult_tp': 5.0,
                'vol_mult_sl': 3.0,
            },
            coin_name,
        )
        breakout_profile = optimize.profile_from_params(
            {
                'strategy_family': 'breakout_expansion',
                'trade_freq_bucket': 'balanced',
                'breakout_lookback': 48,
                'breakout_buffer': 0.004,
                'expansion_confirm_threshold': 0.006,
                'cooldown_hours': 4.0,
                'max_hold_hours': 72,
                'vol_mult_tp': 5.0,
                'vol_mult_sl': 3.0,
            },
            coin_name,
        )

        assert trend_profile.name == coin_name
        assert trend_profile.strategy_family == 'trend_pullback'
        assert trend_profile.pullback_depth_threshold == 0.02
        assert breakout_profile.name == coin_name
        assert breakout_profile.strategy_family == 'breakout_expansion'
        assert breakout_profile.breakout_buffer == 0.004


def test_family_param_fingerprints_persist_distinct_payloads() -> None:
    params = {
        'strategy_family': 'trend_pullback',
        'pullback_depth_threshold': 0.02,
        'rebound_confirmation_threshold': 0.005,
        'trend_strength_min': 0.003,
        'pullback_lookback': 24,
        'cooldown_hours': 4.0,
        'max_hold_hours': 72,
        'vol_mult_tp': 5.0,
        'vol_mult_sl': 3.0,
    }
    fp1 = optimize._family_param_fingerprint(params, 'trend_pullback')
    fp2 = optimize._family_param_fingerprint({**params, 'pullback_depth_threshold': 0.03}, 'trend_pullback')
    assert fp1 != fp2


def test_each_family_evaluates_without_crashing() -> None:
    from core.strategies.base import StrategyContext

    ctx = StrategyContext(
        ret_24h=0.01,
        ret_72h=0.02,
        price=101.0,
        sma_50=100.0,
        sma_200=98.0,
        vol_24h=0.02,
        features={
            'trend_strength_24h': 0.01,
            'range_position_24h': 0.45,
            'breakout_strength_24h': 0.02,
            'range_expansion': 0.01,
            'volume_ratio_24h': 1.2,
        },
    )

    family_params = {
        'pullback_depth_threshold': 0.02,
        'rebound_confirmation_threshold': 0.004,
        'trend_strength_min': 0.002,
        'pullback_lookback': 24,
        'breakout_lookback': 48,
        'breakout_buffer': 0.003,
        'expansion_confirm_threshold': 0.004,
    }

    for family in optimize.STRATEGY_FAMILIES:
        decision = get_strategy_family(family).evaluate(
            ctx,
            min_momentum_magnitude=0.005,
            score_threshold=1.0,
            strict_mode=False,
            family_params=family_params,
        )
        assert hasattr(decision, 'direction')
        assert hasattr(decision, 'gate_contributions')
