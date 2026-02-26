#!/usr/bin/env python3
"""Feature pruning using validation-only SHAP ranking."""

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.coin_profiles import COIN_PROFILES, CoinProfile

FEATURES_DIR = Path("./data/features")


@dataclass
class PruneConfig:
    # Keep a broader feature slate to reduce regime-specific underfitting.
    top_n: int = 30
    val_fraction: float = 0.20
    n_splits: int = 4
    seed: int = 42
    min_shap_ratio: float = 0.01
    min_sign_consistency: float = 0.50


def _symbol_from_dataset_name(path: Path) -> str:
    stem = path.stem.replace("_ml_dataset", "")
    return stem.replace("_", "-")


def _normalize_target(y: pd.Series) -> Optional[pd.Series]:
    """Normalize labels into binary class targets for SHAP ranking.

    Supported label spaces:
      - {0, 1}: already binary.
      - {-1, 0, 1}: mapped to (target_tb > 0) -> 1 else 0.
    """
    clean = pd.to_numeric(y, errors="coerce").dropna()
    unique = set(clean.unique().tolist())
    if not unique:
        return None
    if unique.issubset({0, 1}):
        return clean.astype(int)
    if unique.issubset({-1, 0, 1}):
        return (clean > 0).astype(int)
    return None


def load_ml_datasets(features_dir: Path = FEATURES_DIR) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for path in sorted(features_dir.glob("*_ml_dataset.csv")):
        symbol = _symbol_from_dataset_name(path)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        datasets[symbol] = df
    return datasets


def _profile_datasets(profile: CoinProfile, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    matched = {}
    for symbol, df in datasets.items():
        prefix = symbol.split("-")[0].upper()
        if prefix in profile.prefixes:
            matched[symbol] = df
    return matched


def prune_coin_features(
    profile: CoinProfile,
    datasets: Dict[str, pd.DataFrame],
    config: PruneConfig,
) -> Optional[Dict]:
    matched = _profile_datasets(profile, datasets)
    if not matched:
        return None

    df = pd.concat(matched.values(), axis=0).sort_index()
    target_col = "target_tb" if "target_tb" in df.columns else "target"
    if target_col not in df.columns:
        return None

    y = _normalize_target(df[target_col])
    if y is None:
        return None

    feature_candidates = [
        c for c in profile.feature_columns
        if c in df.columns and c != target_col
    ]
    if len(feature_candidates) < 4:
        return None

    Xy = pd.concat([df[feature_candidates], y.rename("target")], axis=1).dropna(subset=["target"])
    if Xy.empty:
        return None
    X = Xy[feature_candidates]
    y = Xy["target"].astype(int)
    split_idx = int(len(X) * (1 - config.val_fraction))
    if split_idx <= 200:
        return None

    X_train = X.iloc[:split_idx]
    y_train = y.iloc[:split_idx]

    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    fold_signed_shap: List[np.ndarray] = []
    fold_abs_shap: List[np.ndarray] = []

    for fold_idx, (tr_idx, vl_idx) in enumerate(tscv.split(X_train), start=1):
        X_tr, X_vl = X_train.iloc[tr_idx], X_train.iloc[vl_idx]
        y_tr, y_vl = y_train.iloc[tr_idx], y_train.iloc[vl_idx]

        if len(X_vl) < 50 or y_vl.nunique() < 2:
            continue

        model = lgb.LGBMClassifier(
            n_estimators=profile.n_estimators,
            max_depth=profile.max_depth,
            learning_rate=profile.learning_rate,
            min_child_samples=profile.min_child_samples,
            class_weight='balanced',
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=config.seed + fold_idx,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(X_tr, y_tr)

        shap_values = np.asarray(model.predict(X_vl, pred_contrib=True))
        if shap_values.ndim != 2:
            continue
        if shap_values.shape[1] == len(feature_candidates) + 1:
            shap_values = shap_values[:, :-1]
        if shap_values.shape[1] != len(feature_candidates):
            continue

        fold_signed_shap.append(np.mean(shap_values, axis=0))
        fold_abs_shap.append(np.mean(np.abs(shap_values), axis=0))

    if not fold_abs_shap:
        return None

    abs_matrix = np.vstack(fold_abs_shap)
    signed_matrix = np.vstack(fold_signed_shap)

    mean_abs = abs_matrix.mean(axis=0)
    signs = np.sign(signed_matrix)
    sign_consistency = np.abs(signs.mean(axis=0))

    max_abs = float(np.max(mean_abs)) if len(mean_abs) else 0.0
    min_abs_threshold = max_abs * config.min_shap_ratio

    stable_idx = [
        i for i in range(len(feature_candidates))
        if mean_abs[i] > min_abs_threshold and sign_consistency[i] >= config.min_sign_consistency
    ]

    ranked_idx = sorted(stable_idx, key=lambda i: mean_abs[i], reverse=True)
    selected_idx = ranked_idx[: config.top_n]

    if not selected_idx:
        ranked_all = sorted(range(len(feature_candidates)), key=lambda i: mean_abs[i], reverse=True)
        selected_idx = ranked_all[: min(config.top_n, len(ranked_all))]

    selected_features = [feature_candidates[i] for i in selected_idx]

    return {
        "coin": profile.name,
        "symbols": sorted(matched.keys()),
        "seed": config.seed,
        "top_n": config.top_n,
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X) - len(X_train)),
        "selected_features": selected_features,
        "feature_stats": {
            feature_candidates[i]: {
                "mean_abs_shap": float(mean_abs[i]),
                "sign_consistency": float(sign_consistency[i]),
            }
            for i in selected_idx
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def run_pruning(config: PruneConfig, features_dir: Path = FEATURES_DIR) -> Dict[str, Dict]:
    datasets = load_ml_datasets(features_dir)
    outputs: Dict[str, Dict] = {}

    print(f"Loaded {len(datasets)} ML datasets from {features_dir}")

    for coin, profile in COIN_PROFILES.items():
        result = prune_coin_features(profile, datasets, config)
        if not result:
            print(f"[{coin}] skipped (insufficient data)")
            continue

        out_path = features_dir / f"pruned_features_{coin.lower()}.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        outputs[coin] = result
        print(f"[{coin}] wrote {len(result['selected_features'])} features → {out_path}")

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune features per coin via validation SHAP")
    parser.add_argument("--top-n", type=int, default=30, help="Number of features to keep per coin")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-shap-ratio", type=float, default=0.01)
    parser.add_argument("--min-sign-consistency", type=float, default=0.50)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PruneConfig(
        top_n=args.top_n,
        val_fraction=args.val_fraction,
        n_splits=args.n_splits,
        seed=args.seed,
        min_shap_ratio=args.min_shap_ratio,
        min_sign_consistency=args.min_sign_consistency,
    )
    run_pruning(config=config, features_dir=args.features_dir)


if __name__ == "__main__":
    main()
