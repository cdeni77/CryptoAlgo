from scripts.optimize import COIN_OPTIMIZATION_PRIORS, profile_from_params
from core.coin_profiles import COIN_PROFILES


def test_coin_optimization_priors_are_ordered_and_nontrivial() -> None:
    expected_ranges = {
        "BTC": {
            "min_momentum_magnitude": (0.005, 0.030),
            "min_vol_24h": (0.0001, 0.003),
            "min_directional_agreement": (0.49, 0.69),
            "meta_probability_threshold": (0.42, 0.62),
            "cooldown": (6.0, 28.0),
            "min_trade_frequency_ratio": 0.55,
        },
        "ETH": {
            "min_momentum_magnitude": (0.003, 0.028),
            "min_vol_24h": (0.0004, 0.005),
            "min_directional_agreement": (0.46, 0.66),
            "meta_probability_threshold": (0.41, 0.61),
            "cooldown": (5.0, 24.0),
            "min_trade_frequency_ratio": 0.55,
        },
        "SOL": {
            "min_momentum_magnitude": (0.003, 0.032),
            "min_vol_24h": (0.0002, 0.005),
            "min_directional_agreement": (0.44, 0.64),
            "meta_probability_threshold": (0.40, 0.60),
            "cooldown": (3.0, 16.0),
            "min_trade_frequency_ratio": 0.58,
        },
        "XRP": {
            "min_momentum_magnitude": (0.002, 0.028),
            "min_vol_24h": (0.0004, 0.005),
            "min_directional_agreement": (0.48, 0.68),
            "meta_probability_threshold": (0.42, 0.62),
            "cooldown": (4.0, 20.0),
            "min_trade_frequency_ratio": 0.58,
        },
        "DOGE": {
            "min_momentum_magnitude": (0.003, 0.032),
            "min_vol_24h": (0.0002, 0.0045),
            "min_directional_agreement": (0.44, 0.64),
            "meta_probability_threshold": (0.40, 0.60),
            "cooldown": (2.0, 12.0),
            "min_trade_frequency_ratio": 0.60,
        },
    }

    for coin, checks in expected_ranges.items():
        priors = COIN_OPTIMIZATION_PRIORS[coin]
        for key in ("min_momentum_magnitude", "min_vol_24h", "min_directional_agreement", "meta_probability_threshold"):
            lo, hi = priors[key]
            assert lo < hi
            assert (lo, hi) == checks[key]
        assert (priors["cooldown_min"], priors["cooldown_max"]) == checks["cooldown"]
        assert priors["min_trade_frequency_ratio"] == checks["min_trade_frequency_ratio"]


def test_profile_defaults_shifted_for_trade_starvation_issues() -> None:
    assert COIN_PROFILES["BTC"].signal_threshold == 0.53
    assert COIN_PROFILES["BTC"].min_vol_24h == 0.0005
    assert COIN_PROFILES["BTC"].min_momentum_magnitude == 0.007
    assert COIN_PROFILES["BTC"].min_directional_agreement == 0.50
    assert COIN_PROFILES["BTC"].meta_probability_threshold == 0.45
    assert COIN_PROFILES["BTC"].cooldown_hours == 6.0

    assert COIN_PROFILES["ETH"].signal_threshold == 0.53
    assert COIN_PROFILES["ETH"].min_momentum_magnitude == 0.006
    assert COIN_PROFILES["ETH"].min_directional_agreement == 0.47
    assert COIN_PROFILES["ETH"].meta_probability_threshold == 0.44
    assert COIN_PROFILES["ETH"].cooldown_hours == 5.0

    assert COIN_PROFILES["XRP"].signal_threshold == 0.53
    assert COIN_PROFILES["XRP"].min_momentum_magnitude == 0.004
    assert COIN_PROFILES["XRP"].min_directional_agreement == 0.50
    assert COIN_PROFILES["XRP"].meta_probability_threshold == 0.45
    assert COIN_PROFILES["XRP"].cooldown_hours == 5.0

    assert COIN_PROFILES["SOL"].signal_threshold == 0.53
    assert COIN_PROFILES["SOL"].min_momentum_magnitude == 0.007
    assert COIN_PROFILES["SOL"].min_directional_agreement == 0.45
    assert COIN_PROFILES["SOL"].meta_probability_threshold == 0.44
    assert COIN_PROFILES["SOL"].min_vol_24h == 0.0008
    assert COIN_PROFILES["SOL"].cooldown_hours == 3.0

    assert COIN_PROFILES["DOGE"].signal_threshold == 0.52
    assert COIN_PROFILES["DOGE"].min_momentum_magnitude == 0.007
    assert COIN_PROFILES["DOGE"].min_directional_agreement == 0.45
    assert COIN_PROFILES["DOGE"].meta_probability_threshold == 0.44
    assert COIN_PROFILES["DOGE"].min_vol_24h == 0.0007
    assert COIN_PROFILES["DOGE"].cooldown_hours == 2.5


def test_profile_from_params_still_builds_valid_profile_with_new_ranges() -> None:
    params = {
        "signal_threshold": 0.55,
        "min_momentum_magnitude": 0.006,
        "min_vol_24h": 0.001,
        "min_directional_agreement": 0.52,
        "meta_probability_threshold": 0.47,
        "cooldown_hours": 7.0,
        "strategy_family": "momentum_trend",
        "trade_freq_bucket": "aggressive",
    }
    profile = profile_from_params(params, "XRP")

    assert profile.name == "XRP"
    assert profile.signal_threshold == 0.55
    assert profile.min_momentum_magnitude == 0.006
    assert profile.min_vol_24h == 0.001
    assert profile.min_directional_agreement == 0.52
    assert profile.meta_probability_threshold == 0.47
    assert profile.cooldown_hours == 7.0
