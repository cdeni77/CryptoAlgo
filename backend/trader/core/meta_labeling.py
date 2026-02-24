"""Meta-labeling utilities for two-stage signal filtering."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, precision_score
from sklearn.preprocessing import RobustScaler

try:
    from betacal import BetaCalibration  # type: ignore
except Exception:
    BetaCalibration = None


@dataclass
class MetaArtifacts:
    model: Optional[lgb.LGBMClassifier]
    scaler: Optional[RobustScaler]
    calibrator: Optional[object]
    calibrator_type: str
    calibration_metrics: Dict[str, object]
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


def fit_calibrator(strategy: str, scores: np.ndarray, labels: pd.Series, isotonic_min_samples: int = 200):
    mode = (strategy or 'platt').lower()
    x = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, dtype=int)

    if mode == 'isotonic' and len(x) >= isotonic_min_samples:
        calibrator = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        calibrator.fit(x, y)
        return calibrator, 'isotonic'

    if mode == 'beta' and BetaCalibration is not None:
        calibrator = BetaCalibration(parameters='abm')
        calibrator.fit(x.reshape(-1, 1), y)
        return calibrator, 'beta'

    calibrator = LogisticRegression(solver='lbfgs', max_iter=2000)
    calibrator.fit(x.reshape(-1, 1), y)
    return calibrator, 'platt'


def calibrator_predict(calibrator, scores: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(x.reshape(-1, 1))[:, 1]
    if BetaCalibration is not None and isinstance(calibrator, BetaCalibration):
        return calibrator.predict(x.reshape(-1, 1))
    return calibrator.predict(x)


def reliability_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (probs >= lo) & (probs <= hi) if i == n_bins - 1 else (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        out.append({
            'bin_start': float(lo),
            'bin_end': float(hi),
            'count': count,
            'avg_confidence': float(np.mean(probs[mask])),
            'event_rate': float(np.mean(labels[mask])),
        })
    return out


def calibration_report(labels: np.ndarray, raw_probs: np.ndarray, calibrated_probs: np.ndarray) -> Dict[str, object]:
    if len(labels) < 2:
        return {'samples': int(len(labels)), 'raw_brier': None, 'calibrated_brier': None, 'reliability_bins': []}
    return {
        'samples': int(len(labels)),
        'raw_brier': float(brier_score_loss(labels, raw_probs)),
        'calibrated_brier': float(brier_score_loss(labels, calibrated_probs)),
        'reliability_bins': reliability_bins(labels, calibrated_probs),
    }


def calibrator_params(calibrator) -> Dict[str, object]:
    if calibrator is None:
        return {}
    if isinstance(calibrator, LogisticRegression):
        return {
            'coef': [float(v) for v in calibrator.coef_.ravel().tolist()],
            'intercept': [float(v) for v in calibrator.intercept_.ravel().tolist()],
        }
    if isinstance(calibrator, IsotonicRegression):
        return {
            'x_thresholds': [float(v) for v in np.asarray(calibrator.X_thresholds_).tolist()],
            'y_thresholds': [float(v) for v in np.asarray(calibrator.y_thresholds_).tolist()],
        }
    if BetaCalibration is not None and isinstance(calibrator, BetaCalibration):
        params = {}
        for name in ('a_', 'b_', 'm_'):
            if hasattr(calibrator, name):
                params[name] = float(getattr(calibrator, name))
        return params
    return {'type': calibrator.__class__.__name__}


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
    calibration_strategy: str = 'platt',
) -> MetaArtifacts:
    """Train secondary classifier for profitability conditioned on primary signal."""
    X_meta_tr, y_meta_tr, _ = build_meta_dataset(X_train, y_train, np.ones(len(X_train)), 0.5)

    if len(X_meta_tr) < 50 or y_meta_tr.nunique() < 2:
        return MetaArtifacts(None, None, None, 'none', {}, primary_threshold, meta_threshold, {
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
        return MetaArtifacts(model, scaler, None, 'none', {}, primary_threshold, meta_threshold, {
            'meta_train_samples': float(len(X_meta_tr)),
            'meta_val_samples': 0.0,
            'meta_precision': 0.0,
            'final_trade_precision': 0.0,
        })

    X_meta_val = X_val.loc[val_signal_idx]
    y_meta_val = y_val.loc[val_signal_idx]
    if len(X_meta_val) < 20 or y_meta_val.nunique() < 2:
        return MetaArtifacts(model, scaler, None, 'none', {}, primary_threshold, meta_threshold, {
            'meta_train_samples': float(len(X_meta_tr)),
            'meta_val_samples': float(len(X_meta_val)),
            'meta_precision': 0.0,
            'final_trade_precision': 0.0,
        })

    raw_meta_val = model.predict_proba(scaler.transform(X_meta_val))[:, 1]
    holdout_size = max(10, int(len(raw_meta_val) * 0.25)) if len(raw_meta_val) >= 60 else 0
    fit_scores = raw_meta_val[:-holdout_size] if holdout_size > 0 else raw_meta_val
    fit_labels = y_meta_val.iloc[:-holdout_size] if holdout_size > 0 else y_meta_val

    calibrator, calibrator_type = fit_calibrator(calibration_strategy, fit_scores, fit_labels)
    cal_meta_val = calibrator_predict(calibrator, raw_meta_val)

    metrics_report = {
        'validation': calibration_report(y_meta_val.to_numpy(dtype=int), raw_meta_val, cal_meta_val),
        'holdout': None,
    }
    if holdout_size > 0:
        metrics_report['holdout'] = calibration_report(
            y_meta_val.iloc[-holdout_size:].to_numpy(dtype=int),
            raw_meta_val[-holdout_size:],
            cal_meta_val[-holdout_size:],
        )

    meta_pred = (cal_meta_val >= meta_threshold).astype(int)
    meta_precision = float(precision_score(y_meta_val, meta_pred, zero_division=0))

    return MetaArtifacts(model, scaler, calibrator, calibrator_type, metrics_report, primary_threshold, meta_threshold, {
        'meta_train_samples': float(len(X_meta_tr)),
        'meta_val_samples': float(len(X_meta_val)),
        'meta_precision': meta_precision,
        'final_trade_precision': meta_precision,
    })
