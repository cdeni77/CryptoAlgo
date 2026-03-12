from scripts.optimize import COIN_OPTIMIZATION_PRIORS, profile_from_params
from core.coin_profiles import COIN_PROFILES


def test_coin_optimization_priors_are_ordered_and_nontrivial() -> None:
    expected_ranges = {
        "BTC": {
            "min_momentum_magnitude": (0.010, 0.045),
            "min_vol_24h": (0.0005, 0.0045),
            "min_directional_agreement": (0.58, 0.76),
        },
        "ETH": {
            "min_momentum_magnitude": (0.009, 0.045),
            "min_vol_24h": (0.002, 0.008),
            "min_directional_agreement": (0.57, 0.75),
        },
        "SOL": {
            "min_momentum_magnitude": (0.012, 0.052),
            "min_vol_24h": (0.0015, 0.0085),
            "min_directional_agreement": (0.55, 0.74),
        },
        "XRP": {
            "min_momentum_magnitude": (0.006, 0.040),
            "min_vol_24h": (0.002, 0.008),
            "min_directional_agreement": (0.59, 0.77),
        },
        "DOGE": {
            "min_momentum_magnitude": (0.011, 0.052),
            "min_vol_24h": (0.001, 0.0075),
            "min_directional_agreement": (0.55, 0.74),
        },
    }

    for coin, checks in expected_ranges.items():
        priors = COIN_OPTIMIZATION_PRIORS[coin]
        for key, expected in checks.items():
            lo, hi = priors[key]
            assert lo < hi
            assert (lo, hi) == expected


def test_profile_defaults_shifted_for_trade_starvation_issues() -> None:
    assert COIN_PROFILES["BTC"].min_vol_24h == 0.002
    assert COIN_PROFILES["BTC"].min_momentum_magnitude == 0.017

    assert COIN_PROFILES["ETH"].min_momentum_magnitude == 0.016
    assert COIN_PROFILES["ETH"].min_directional_agreement == 0.53

    assert COIN_PROFILES["XRP"].min_momentum_magnitude == 0.012

    assert COIN_PROFILES["SOL"].min_directional_agreement == 0.52
    assert COIN_PROFILES["SOL"].min_vol_24h == 0.004

    assert COIN_PROFILES["DOGE"].min_directional_agreement == 0.52
    assert COIN_PROFILES["DOGE"].min_vol_24h == 0.0035


def test_profile_from_params_still_builds_valid_profile_with_new_ranges() -> None:
    params = {
        "min_momentum_magnitude": 0.008,
        "min_vol_24h": 0.001,
        "min_directional_agreement": 0.56,
        "meta_probability_threshold": 0.50,
        "strategy_family": "momentum_trend",
        "trade_freq_bucket": "balanced",
    }
    profile = profile_from_params(params, "XRP")

    assert profile.name == "XRP"
    assert profile.min_momentum_magnitude == 0.008
    assert profile.min_vol_24h == 0.001
    assert profile.min_directional_agreement == 0.56
    assert profile.meta_probability_threshold == 0.50
