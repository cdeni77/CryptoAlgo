from scripts.optimize import build_effective_params, profile_from_params
from scripts.train_model import Config, resolve_categorical_param


def test_strategy_family_and_trade_freq_bucket_round_trip_to_runtime_resolution() -> None:
    params = {
        "signal_threshold": 0.77,
        "strategy_family": "breakout",
        "trade_freq_bucket": "aggressive",
    }

    effective = build_effective_params(params, "BTC")
    profile = profile_from_params(params, "BTC")
    cfg = Config()

    assert effective["strategy_family"] == "breakout"
    assert effective["trade_freq_bucket"] == "aggressive"
    assert profile.strategy_family == "breakout"
    assert profile.trade_freq_bucket == "aggressive"
    assert resolve_categorical_param("strategy_family", profile, cfg, Config.strategy_family) == "breakout"
    assert resolve_categorical_param("trade_freq_bucket", profile, cfg, Config.trade_freq_bucket) == "aggressive"


def test_invalid_categorical_values_fall_back_to_deterministic_defaults() -> None:
    params = {
        "strategy_family": "not-a-family",
        "trade_freq_bucket": "not-a-bucket",
    }

    effective = build_effective_params(params, "BTC")
    profile = profile_from_params(params, "BTC")

    assert effective["strategy_family"] == "momentum_trend"
    assert effective["trade_freq_bucket"] == "balanced"
    assert profile.strategy_family == "momentum_trend"
    assert profile.trade_freq_bucket == "balanced"
