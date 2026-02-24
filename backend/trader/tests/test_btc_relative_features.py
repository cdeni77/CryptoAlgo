import pandas as pd

from core.coin_profiles import BTC_RELATIVE_FEATURES, COIN_PROFILES, FEATURE_SCHEMA_VERSION, save_model
from features.engineering import BTCRelativeFeatures, FeatureConfig, FeaturePipeline


def _make_ohlcv(rows: int = 220, seed: int = 42) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    rnd = pd.Series(range(rows), index=idx, dtype=float)
    close = 100 + rnd * 0.2 + (seed % 7) * 0.01
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": 1000 + rnd,
        },
        index=idx,
    )


def test_btc_relative_features_present_for_altcoins() -> None:
    btc = _make_ohlcv(seed=1)
    eth = _make_ohlcv(seed=2)
    pipeline = FeaturePipeline(FeatureConfig(compute_funding=False, compute_oi=False))

    features = pipeline.compute_features(
        ohlcv_data={"BIP-20DEC30-CDE": btc, "ETP-20DEC30-CDE": eth},
        reference_symbol="BIP-20DEC30-CDE",
    )

    assert "ETP-20DEC30-CDE" in features
    assert all(col in features["ETP-20DEC30-CDE"].columns for col in BTC_RELATIVE_FEATURES)
    assert all(col not in features["BIP-20DEC30-CDE"].columns for col in BTC_RELATIVE_FEATURES)


def test_btc_relative_feature_set_is_exposed_in_altcoin_profiles() -> None:
    for coin in ("ETH", "SOL", "XRP", "DOGE"):
        cols = COIN_PROFILES[coin].feature_columns
        assert all(col in cols for col in BTC_RELATIVE_FEATURES)


def test_model_artifact_contains_feature_schema_version(tmp_path) -> None:
    out = save_model(
        symbol="ETP-20DEC30-CDE",
        model=None,
        scaler=None,
        calibrator=None,
        feature_columns=COIN_PROFILES["ETH"].feature_columns,
        auc=0.61,
        profile_name="ETH",
        target_dir=tmp_path,
    )

    import joblib

    payload = joblib.load(out)
    assert payload["meta"]["feature_schema_version"] == FEATURE_SCHEMA_VERSION
