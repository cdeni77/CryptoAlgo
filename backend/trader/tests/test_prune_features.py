import json
from pathlib import Path

import numpy as np
import pandas as pd

from core.coin_profiles import COIN_PROFILES
from scripts.prune_features import PruneConfig, load_ml_datasets, prune_coin_features


def _build_dataset(rows: int = 900, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=rows, freq="h", tz="UTC")

    profile = COIN_PROFILES["BTC"]
    cols = profile.feature_columns
    data = {col: rng.normal(0, 1, size=rows) for col in cols}

    signal = 1.8 * data[cols[0]] - 1.2 * data[cols[1]] + rng.normal(0, 0.2, size=rows)
    target = (signal > np.quantile(signal, 0.55)).astype(int)

    df = pd.DataFrame(data, index=index)
    df["target_tb"] = target
    return df


def test_pruned_features_are_non_empty_and_subset_columns(tmp_path: Path) -> None:
    df = _build_dataset()
    out = tmp_path / "BIP_20DEC30_CDE_ml_dataset.csv"
    df.to_csv(out)

    datasets = load_ml_datasets(tmp_path)
    cfg = PruneConfig(seed=77, top_n=16)
    result = prune_coin_features(COIN_PROFILES["BTC"], datasets, cfg)

    assert result is not None
    selected = result["selected_features"]
    assert selected
    assert set(selected).issubset(set(df.columns))
    assert len(selected) <= cfg.top_n


def test_pruned_features_are_deterministic_with_seed(tmp_path: Path) -> None:
    df = _build_dataset(seed=7)
    out = tmp_path / "BIP_20DEC30_CDE_ml_dataset.csv"
    df.to_csv(out)

    datasets = load_ml_datasets(tmp_path)
    cfg = PruneConfig(seed=2024, top_n=15)

    res1 = prune_coin_features(COIN_PROFILES["BTC"], datasets, cfg)
    res2 = prune_coin_features(COIN_PROFILES["BTC"], datasets, cfg)

    assert res1 is not None and res2 is not None
    assert json.dumps(res1["selected_features"]) == json.dumps(res2["selected_features"])
