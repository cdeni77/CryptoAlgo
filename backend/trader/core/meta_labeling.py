"""Meta-labeling utilities for two-stage signal filtering."""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_score
from sklearn.preprocessing import RobustScaler


@dataclass
class MetaArtifacts:
    model: Optional[lgb.LGBMClassifier]
    scaler: Optional[RobustScaler]
    calibrator: Optional[IsotonicRegression]
    primary_threshold: float
    meta_threshold: float
    metrics: Dict[str, float]


def primary_recall_threshold(profile_threshold: float, min_signal_edge: float) -> float:
    """Lower primary threshold slightly to increase recall before meta filter."""
    return float(np.clip(profile_threshold - min_signal_edge, 0.50, 0.95))


def build_meta_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    primary_probs: np.ndarray,
    primary_threshold: float,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Create dataset only where primary model emits a tradable signal."""
    signal_mask = pd.Series(primary_probs >= primary_threshold, index=X.index)
    if signal_mask.sum() == 0:
        return pd.DataFrame(), pd.Series(dtype=float), signal_mask

    X_meta = X.loc[signal_mask]
    y_meta = y.loc[signal_mask].astype(float)
    return X_meta, y_meta, signal_mask


def train_meta_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    primary_val_mask: pd.Series,
    *,
    primary_threshold: float,
    meta_threshold: float,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    min_child_samples: int,
) -> MetaArtifacts:
    """Train secondary classifier for profitability conditioned on primary signal."""
    X_meta_tr, y_meta_tr, _ = build_meta_dataset(X_train, y_train, np.ones(len(X_train)), 0.5)

    # Training is already conditioned by caller; keep guard rails.
    if len(X_meta_tr) < 50 or y_meta_tr.nunique() < 2:
        return MetaArtifacts(None, None, None, primary_threshold, meta_threshold, {
            'meta_train_samples': float(len(X_meta_tr)),
            'meta_val_samples': 0.0,
            'meta_precision': 0.0,
            'final_trade_precision': 0.0,
        })

    scaler = RobustScaler()
    X_tr_scaled = scaler.fit_transform(X_meta_tr)

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
        reg_alpha=0.1,
        reg_lambda=0.1,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(X_tr_scaled, y_meta_tr)

    val_signal_idx = primary_val_mask[primary_val_mask].index
    if len(val_signal_idx) == 0:
        return MetaArtifacts(model, scaler, None, primary_threshold, meta_threshold, {
            'meta_train_samples': float(len(X_meta_tr)),
            'meta_val_samples': 0.0,
            'meta_precision': 0.0,
            'final_trade_precision': 0.0,
        })

    X_meta_val = X_val.loc[val_signal_idx]
    y_meta_val = y_val.loc[val_signal_idx]
    if len(X_meta_val) < 20 or y_meta_val.nunique() < 2:
        return MetaArtifacts(model, scaler, None, primary_threshold, meta_threshold, {
            'meta_train_samples': float(len(X_meta_tr)),
            'meta_val_samples': float(len(X_meta_val)),
            'meta_precision': 0.0,
            'final_trade_precision': 0.0,
        })

    raw_meta_val = model.predict_proba(scaler.transform(X_meta_val))[:, 1]
    calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
    calibrator.fit(raw_meta_val, y_meta_val)
    cal_meta_val = calibrator.predict(raw_meta_val)

    meta_pred = (cal_meta_val >= meta_threshold).astype(int)
    meta_precision = float(precision_score(y_meta_val, meta_pred, zero_division=0))

    return MetaArtifacts(model, scaler, calibrator, primary_threshold, meta_threshold, {
        'meta_train_samples': float(len(X_meta_tr)),
        'meta_val_samples': float(len(X_meta_val)),
        'meta_precision': meta_precision,
        'final_trade_precision': meta_precision,
    })
