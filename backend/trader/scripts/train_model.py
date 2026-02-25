"""
Crypto ML Trading System v8 — Per-Coin Profiles + Model Persistence

KEY CHANGES from v7:
  - Per-coin profiles with tuned thresholds, exits, hyperparameters
  - All 5 coins trade (no exclusions by default)
  - Coin-specific extra features feed into ML model
  - Model saving to disk after training (joblib)
  - All coins use momentum strategy with per-coin parameter tuning

Coinbase CDE fee model: 0.10% per side, $0.20 minimum per contract.
Funding: Normalized hourly funding data (Coinbase native preferred, CCXT fallback).
"""
import argparse
import hashlib
import joblib
import os
import sqlite3
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.preprocessing import RobustScaler
import lightgbm as lgb

from core.coin_profiles import (
    get_coin_profile, save_model, load_model, list_saved_models,
    COIN_PROFILES, BASE_FEATURES, CoinProfile, MODELS_DIR,
)
from core.trading_costs import get_contract_spec
from core.labeling import (
    TripleBarrierSpec,
    compute_labels_from_feature_index,
    momentum_direction_series,
    resolve_profile_label_horizon,
)
from core.meta_labeling import (
    MetaArtifacts,
    build_meta_dataset,
    primary_recall_threshold,
    train_meta_classifier,
    calibrator_predict as meta_calibrator_predict,
    calibrator_params as meta_calibrator_params,
)

warnings.filterwarnings('ignore')

try:
    from betacal import BetaCalibration  # type: ignore
except Exception:
    BetaCalibration = None


# --- Paths ---
FEATURES_DIR = Path("./data/features")
DB_PATH = "./data/trading.db"

# BASE FEATURE LIST (fallback — coin_profiles provides per-coin lists)
FEATURE_COLUMNS = BASE_FEATURES


COST_AWARE_FEATURES = {
    'fee_hurdle_pct',
    'breakout_vs_cost',
    'expected_cost_to_vol_ratio',
}


def summarize_feature_importance(model, feature_columns: List[str]) -> Dict[str, object]:
    importances = getattr(model, 'feature_importances_', None)
    if importances is None or len(importances) != len(feature_columns):
        return {}

    importance_map = {str(c): float(v) for c, v in zip(feature_columns, importances)}
    total = sum(importance_map.values())
    if total <= 0:
        return {'cost_aware_block': {'share': 0.0, 'top_features': []}}

    cost_items = sorted(
        [(name, val) for name, val in importance_map.items() if name in COST_AWARE_FEATURES],
        key=lambda x: x[1],
        reverse=True,
    )
    cost_total = sum(v for _, v in cost_items)
    top_cost_features = [
        {'feature': name, 'importance_share': float(val / total)}
        for name, val in cost_items[:3]
    ]

    top_overall = sorted(importance_map.items(), key=lambda x: x[1], reverse=True)[:10]
    return {
        'cost_aware_block': {
            'share': float(cost_total / total),
            'top_features': top_cost_features,
        },
        'top_features': [
            {'feature': name, 'importance_share': float(val / total)}
            for name, val in top_overall
        ],
    }


@dataclass
class Trade:
    symbol: str
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    net_pnl: float
    raw_pnl: float
    funding_pnl: float
    fee_pnl: float
    fee_pct_pnl: float
    min_fee_pnl: float
    slippage_pnl: float
    pnl_dollars: float
    n_contracts: int
    notional: float
    exit_reason: str


@dataclass
class Config:
    # Walk-Forward Windows
    train_lookback_days: int = 120
    retrain_frequency_days: int = 7
    min_train_samples: int = 400

    # Signal Filters (defaults — overridden per-coin by profiles)
    signal_threshold: float = 0.80
    min_funding_z: float = 0.0
    min_momentum_magnitude: float = 0.07

    # Exit Strategy (defaults — overridden per-coin by profiles)
    vol_mult_tp: float = 5.5
    vol_mult_sl: float = 3.0
    breakeven_trigger: float = 999.0
    trailing_active: bool = False
    trailing_mult: float = 999.0
    max_hold_hours: int = 96

    # Symbol selection — v8: no exclusions by default
    excluded_symbols: Optional[List[str]] = None

    # Risk
    max_positions: int = 5
    position_size: float = 0.15
    leverage: int = 4
    vol_sizing_target: float = 0.025

    # Fees — EXACT COINBASE US CDE
    fee_pct_per_side: float = 0.0010
    min_fee_per_contract: float = 0.20
    slippage_bps: float = 2.0

    # Regime Filter (defaults — overridden per-coin)
    min_vol_24h: float = 0.008
    max_vol_24h: float = 0.06

    # Safety
    min_equity: float = 1000.0
    cooldown_hours: float = 24.0

    # Validation
    val_fraction: float = 0.20
    min_val_auc: float = 0.54

    # Correlation filter
    max_portfolio_correlation: float = 0.75
    correlation_lookback_hours: int = 72

    # Label
    label_forward_hours: int = 24
    label_vol_target: float = 1.8

    # Position sizing control
    max_weekly_equity_growth: float = 0.03

    # Entry quality filters
    min_signal_edge: float = 0.02
    max_ensemble_std: float = 0.12
    min_directional_agreement: float = 0.67
    disagreement_confidence_cap: float = 0.86
    meta_probability_threshold: float = 0.57

    # Walk-forward leakage protection / evaluation
    train_embargo_hours: int = 24
    oos_eval_days: int = 60

    # Feature controls
    enforce_pruned_features: bool = False

    # Recency weighting
    recency_half_life_days: float = 50.0

    # Calibration
    calibration_strategy: str = 'platt'


def _fit_calibrator(
    strategy: str,
    train_scores: np.ndarray,
    train_labels: pd.Series,
    *,
    isotonic_min_samples: int = 200,
):
    strategy_normalized = (strategy or 'platt').lower()
    x = np.clip(np.asarray(train_scores, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(train_labels, dtype=int)

    if strategy_normalized == 'isotonic':
        if len(x) < isotonic_min_samples:
            strategy_normalized = 'platt'
        else:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso.fit(x, y)
            return iso, 'isotonic'

    if strategy_normalized == 'beta':
        if BetaCalibration is not None:
            beta_cal = BetaCalibration(parameters='abm')
            beta_cal.fit(x.reshape(-1, 1), y)
            return beta_cal, 'beta'
        strategy_normalized = 'platt'

    platt = LogisticRegression(solver='lbfgs', max_iter=2000)
    platt.fit(x.reshape(-1, 1), y)
    return platt, 'platt'


def _calibrator_predict(calibrator, scores: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    if isinstance(calibrator, LogisticRegression):
        return calibrator.predict_proba(x.reshape(-1, 1))[:, 1]
    if BetaCalibration is not None and isinstance(calibrator, BetaCalibration):
        return calibrator.predict(x.reshape(-1, 1))
    return calibrator.predict(x)


def _reliability_bins(labels: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        bins.append({
            'bin_start': float(lo),
            'bin_end': float(hi),
            'count': count,
            'avg_confidence': float(np.mean(probs[mask])),
            'event_rate': float(np.mean(labels[mask])),
        })
    return bins


def _calibration_report(labels: np.ndarray, raw_probs: np.ndarray, cal_probs: np.ndarray) -> Dict[str, object]:
    if len(labels) < 2:
        return {'samples': int(len(labels)), 'raw_brier': None, 'calibrated_brier': None, 'reliability_bins': []}
    return {
        'samples': int(len(labels)),
        'raw_brier': float(brier_score_loss(labels, raw_probs)),
        'calibrated_brier': float(brier_score_loss(labels, cal_probs)),
        'reliability_bins': _reliability_bins(labels, cal_probs),
    }


def _calibrator_params(calibrator) -> Dict[str, object]:
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
        attrs = {}
        for name in ('a_', 'b_', 'm_'):
            if hasattr(calibrator, name):
                attrs[name] = float(getattr(calibrator, name))
        return attrs
    return {'type': calibrator.__class__.__name__}


@dataclass(frozen=True)
class EnsembleMemberSpec:
    name: str
    train_window_days: int
    feature_fraction: float
    max_depth_range: Tuple[int, int]
    num_leaves_range: Tuple[int, int]
    n_estimators_range: Tuple[int, int]
    min_child_samples_range: Tuple[int, int]


CORE_ENSEMBLE_FEATURES = {
    'fee_hurdle_pct',
    'breakout_vs_cost',
    'expected_cost_to_vol_ratio',
    'vol_24h',
    'funding_rate_zscore',
}


def build_ensemble_member_specs() -> List[EnsembleMemberSpec]:
    return [
        EnsembleMemberSpec("fast_90d", 90, 0.70, (2, 4), (15, 45), (70, 130), (12, 24)),
        EnsembleMemberSpec("base_120d", 120, 0.78, (3, 5), (25, 70), (90, 170), (16, 30)),
        EnsembleMemberSpec("slow_150d", 150, 0.85, (4, 6), (45, 100), (120, 220), (20, 40)),
        EnsembleMemberSpec("robust_180d", 180, 0.82, (3, 6), (30, 90), (110, 210), (18, 36)),
    ]


def calculate_coinbase_fee(n_contracts: int, price: float, symbol: str,
                           config: Config) -> float:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * price
    pct_fee = n_contracts * notional_per_contract * config.fee_pct_per_side
    min_fee = n_contracts * config.min_fee_per_contract
    return max(pct_fee, min_fee)


def calculate_execution_costs(n_contracts: int, entry_price: float, exit_price: float,
                              symbol: str, config: Config) -> Tuple[float, float, float, float]:
    """Return round-trip cost components in dollars.

    Returns: (total_cost, pct_fee_component, min_fee_component, slippage_component)
    """
    spec = get_contract_spec(symbol)
    entry_notional = n_contracts * spec['units'] * entry_price
    exit_notional = n_contracts * spec['units'] * exit_price

    entry_pct_fee = entry_notional * config.fee_pct_per_side
    exit_pct_fee = exit_notional * config.fee_pct_per_side
    entry_min_fee = n_contracts * config.min_fee_per_contract
    exit_min_fee = n_contracts * config.min_fee_per_contract

    entry_fee = max(entry_pct_fee, entry_min_fee)
    exit_fee = max(exit_pct_fee, exit_min_fee)

    pct_component = min(entry_pct_fee, entry_fee) + min(exit_pct_fee, exit_fee)
    min_component = max(entry_fee - entry_pct_fee, 0.0) + max(exit_fee - exit_pct_fee, 0.0)
    slippage_component = (entry_notional + exit_notional) * (config.slippage_bps / 10000.0)
    total_cost = entry_fee + exit_fee + slippage_component
    return total_cost, pct_component, min_component, slippage_component


def calculate_pnl_exact(entry_price: float, exit_price: float, direction: int,
                         accum_funding: float, n_contracts: int, symbol: str,
                         config: Config) -> Tuple[float, float, float, float, float, float, float, float]:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * entry_price
    total_notional = n_contracts * notional_per_contract
    if total_notional == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    raw_pnl_pct = (exit_price - entry_price) / entry_price * direction
    raw_pnl_dollars = total_notional * raw_pnl_pct
    total_fee_dollars, pct_fee_component, min_fee_component, slippage_component = calculate_execution_costs(
        n_contracts, entry_price, exit_price, symbol, config
    )
    total_fee_pct = total_fee_dollars / total_notional
    pct_fee_pnl = -(pct_fee_component / total_notional)
    min_fee_pnl = -(min_fee_component / total_notional)
    slippage_pnl = -(slippage_component / total_notional)
    funding_dollars = accum_funding * total_notional
    net_pnl_dollars = raw_pnl_dollars - total_fee_dollars + funding_dollars
    net_pnl_pct = net_pnl_dollars / total_notional
    return net_pnl_pct, raw_pnl_pct, -total_fee_pct, pct_fee_pnl, min_fee_pnl, slippage_pnl, net_pnl_dollars, total_notional


def calculate_n_contracts(equity: float, price: float, symbol: str,
                           config: Config, vol_24h: float = 0.0,
                           profile: Optional[CoinProfile] = None) -> int:
    spec = get_contract_spec(symbol)
    notional_per_contract = spec['units'] * price
    if notional_per_contract <= 0:
        return 0
    pos_size = profile.position_size if profile else config.position_size
    vol_target = profile.vol_sizing_target if profile else config.vol_sizing_target
    if vol_24h > 0 and vol_target > 0:
        vol_ratio = vol_target / vol_24h
        vol_ratio = min(vol_ratio, 1.5)
        vol_ratio = max(vol_ratio, 0.3)
        pos_size = pos_size * vol_ratio
    target_notional = equity * pos_size * config.leverage
    n_contracts = int(target_notional / notional_per_contract)
    return max(n_contracts, 0)


class MLSystem:
    def __init__(self, config: Config):
        self.config = config

    def get_feature_columns(self, available_columns: pd.Index,
                            coin_features: Optional[List[str]] = None) -> List[str]:
        feature_list = coin_features if coin_features else FEATURE_COLUMNS
        cols = [c for c in feature_list if c in available_columns]
        return cols if len(cols) >= 4 else []

    def create_labels(self, ohlcv: pd.DataFrame, features: pd.DataFrame,
                      profile: Optional[CoinProfile] = None) -> pd.Series:
        """Momentum-aware triple-barrier labels with explicit timeout class.

        Label values:
          1  -> TP first touch (in momentum direction)
          -1 -> SL first touch
          0  -> timeout within horizon (neutral)

        Neutral-direction rows are left as NaN and excluded before binary mapping.
        """
        spec = TripleBarrierSpec(
            horizon_hours=resolve_label_horizon(profile, self.config),
            tp_mult=profile.vol_mult_tp if profile else self.config.vol_mult_tp,
            sl_mult=profile.vol_mult_sl if profile else self.config.vol_mult_sl,
        )
        direction = momentum_direction_series(ohlcv)
        return compute_labels_from_feature_index(ohlcv, features.index, spec, direction)

    def prepare_binary_training_set(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Explicitly remove neutral timeout labels, then map SL/TP to binary 0/1."""
        valid = y.dropna()
        if valid.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        non_neutral_idx = valid[valid != 0].index
        if len(non_neutral_idx) == 0:
            return pd.DataFrame(), pd.Series(dtype=float)

        X_out = X.loc[non_neutral_idx]
        y_out = valid.loc[non_neutral_idx].map({-1.0: 0.0, 1.0: 1.0})
        return X_out, y_out

    @staticmethod
    def _compute_average_uniqueness(index: pd.DatetimeIndex, label_horizon_hours: int) -> np.ndarray:
        """Compute per-sample average uniqueness from overlapping label intervals."""
        if len(index) == 0:
            return np.array([], dtype=float)

        ordered_index = pd.DatetimeIndex(index)
        times_ns = ordered_index.view("int64")
        horizon_ns = int(max(label_horizon_hours, 1) * 3600 * 1_000_000_000)
        end_ns = times_ns + horizon_ns

        start_pos = np.arange(len(times_ns), dtype=np.int64)
        end_pos = np.searchsorted(times_ns, end_ns, side="right") - 1
        end_pos = np.maximum(end_pos, start_pos)

        # Sweep-line concurrency on sample timestamps.
        delta = np.zeros(len(times_ns) + 1, dtype=np.float64)
        np.add.at(delta, start_pos, 1.0)
        valid_end = end_pos + 1
        valid_end = np.minimum(valid_end, len(times_ns))
        np.add.at(delta, valid_end, -1.0)
        concurrency = np.cumsum(delta[:-1])
        concurrency = np.maximum(concurrency, 1.0)

        inv_concurrency = 1.0 / concurrency
        inv_cumsum = np.concatenate(([0.0], np.cumsum(inv_concurrency)))
        interval_lengths = (end_pos - start_pos + 1).astype(np.float64)
        avg_uniqueness = (inv_cumsum[end_pos + 1] - inv_cumsum[start_pos]) / interval_lengths
        return avg_uniqueness

    def _build_sample_weights(
        self,
        y: pd.Series,
        label_horizon_hours: int,
        symbol: str,
    ) -> np.ndarray:
        """Combine uniqueness and class-balance weights for LightGBM training."""
        debug_weights = str(os.getenv("TRAIN_DEBUG_WEIGHTS", "0")).lower() in {"1", "true", "yes", "on"}
        symbol_label = symbol or "UNKNOWN"
        if y.empty:
            return np.array([], dtype=float)

        uniqueness_weights = self._compute_average_uniqueness(y.index, label_horizon_hours)

        counts = y.value_counts()
        n_samples = float(len(y))
        n_classes = float(max(len(counts), 1))
        class_weights = {
            cls: n_samples / (n_classes * cnt)
            for cls, cnt in counts.items()
            if cnt > 0
        }
        class_weight_vec = y.map(class_weights).to_numpy(dtype=float)

        half_life_hours = max(float(self.config.recency_half_life_days) * 24.0, 1e-8)
        index = y.index
        if isinstance(index, pd.DatetimeIndex):
            index_ts = index
        else:
            try:
                index_ts = pd.to_datetime(index, utc=True)
            except (TypeError, ValueError):
                index_ts = None

        if isinstance(index_ts, pd.DatetimeIndex) and not index_ts.isna().any():
            if index_ts.tz is None:
                index_ts = index_ts.tz_localize("UTC")
            most_recent_ts = index_ts.max()
            age_hours = (most_recent_ts - index_ts).total_seconds() / 3600.0
            age_hours = np.maximum(np.asarray(age_hours, dtype=float), 0.0)
        else:
            age_hours = np.arange(len(y) - 1, -1, -1, dtype=float)
            if debug_weights:
                print(
                    f"[{symbol_label}] non-datetime index detected for sample weights; "
                    "using positional recency decay fallback."
                )

        recency_weight_vec = np.asarray(
            np.exp(-np.log(2.0) * age_hours / half_life_hours),
            dtype=float,
        )

        base_weights = uniqueness_weights * class_weight_vec
        sample_weights = np.asarray(base_weights * recency_weight_vec, dtype=float)
        sample_weights = np.clip(sample_weights, 1e-8, None)

        base_weights = np.clip(base_weights, 1e-8, None)
        ess_before = (base_weights.sum() ** 2) / np.square(base_weights).sum()
        ess_after = (sample_weights.sum() ** 2) / np.square(sample_weights).sum()
        if debug_weights:
            print(
                f"[{symbol_label}] recency_weight stats: min={recency_weight_vec.min():.6f} "
                f"median={np.median(recency_weight_vec):.6f} max={recency_weight_vec.max():.6f} "
                f"half_life_days={self.config.recency_half_life_days:.2f}"
            )
            print(
                f"[{symbol_label}] sample_weight stats: min={sample_weights.min():.6f} "
                f"median={np.median(sample_weights):.6f} max={sample_weights.max():.6f} "
                f"ESS(before={ess_before:.1f}, after={ess_after:.1f})/{len(sample_weights)}"
            )
        return sample_weights

    @staticmethod
    def _stable_seed(*parts: str) -> int:
        joined = "|".join(parts).encode("utf-8")
        return int(hashlib.sha256(joined).hexdigest()[:8], 16)

    def build_member_features(self, symbol: str, cols: List[str], member: EnsembleMemberSpec) -> List[str]:
        if not cols:
            return []
        mandatory = [c for c in cols if c in CORE_ENSEMBLE_FEATURES]
        optional = [c for c in cols if c not in CORE_ENSEMBLE_FEATURES]
        target = max(int(round(len(cols) * member.feature_fraction)), len(mandatory), 4)
        optional_needed = max(target - len(mandatory), 0)
        if optional_needed >= len(optional):
            selected_optional = optional
        else:
            rng = np.random.default_rng(self._stable_seed(symbol, member.name, "feature_subset"))
            idx = rng.choice(len(optional), size=optional_needed, replace=False)
            selected_optional = [optional[i] for i in np.sort(idx)]
        return mandatory + selected_optional

    def sample_member_hyperparams(
        self,
        symbol: str,
        member: EnsembleMemberSpec,
        profile: Optional[CoinProfile],
    ) -> Dict[str, float]:
        rng = np.random.default_rng(self._stable_seed(symbol, member.name, "hparams"))
        profile_lr = profile.learning_rate if profile else 0.05
        return {
            'n_estimators': int(rng.integers(member.n_estimators_range[0], member.n_estimators_range[1] + 1)),
            'max_depth': int(rng.integers(member.max_depth_range[0], member.max_depth_range[1] + 1)),
            'num_leaves': int(rng.integers(member.num_leaves_range[0], member.num_leaves_range[1] + 1)),
            'min_child_samples': int(rng.integers(member.min_child_samples_range[0], member.min_child_samples_range[1] + 1)),
            'learning_rate': float(profile_lr * rng.uniform(0.85, 1.15)),
        }

    @staticmethod
    def log_member_pairwise_correlation(symbol: str, member_outputs: List[Dict[str, object]]) -> None:
        if len(member_outputs) < 2:
            return
        print(f"[{symbol}] Ensemble pairwise correlation (validation probs):")
        for i in range(len(member_outputs)):
            for j in range(i + 1, len(member_outputs)):
                a = np.asarray(member_outputs[i].get('val_probs', []), dtype=float)
                b = np.asarray(member_outputs[j].get('val_probs', []), dtype=float)

                corr_text = "n/a"
                if len(a) == len(b) and len(a) >= 2:
                    finite_mask = np.isfinite(a) & np.isfinite(b)
                    a_f = a[finite_mask]
                    b_f = b[finite_mask]
                    if len(a_f) >= 2 and np.std(a_f) > 0 and np.std(b_f) > 0:
                        corr = float(np.corrcoef(a_f, b_f)[0, 1])
                        if np.isfinite(corr):
                            corr_text = f"{corr:.4f}"

                print(
                    f"  {member_outputs[i].get('member_name', f'm{i}'):<14} vs "
                    f"{member_outputs[j].get('member_name', f'm{j}'):<14}: corr={corr_text}"
                )

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame, y_val: pd.Series,
              profile: Optional[CoinProfile] = None,
              symbol: str = "UNKNOWN",
              member_spec: Optional[EnsembleMemberSpec] = None,
              member_features: Optional[List[str]] = None) -> Optional[Tuple]:
        if len(X_train) < self.config.min_train_samples:
            return None
        if y_train.sum() < 10 or (1 - y_train).sum() < 10:
            return None
        if len(X_val) < 20 or y_val.sum() < 3:
            return None

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        member_name = member_spec.name if member_spec else "single"
        if member_spec:
            sampled = self.sample_member_hyperparams(symbol, member_spec, profile)
            n_est = int(sampled['n_estimators'])
            depth = int(sampled['max_depth'])
            num_leaves = int(sampled['num_leaves'])
            lr = float(sampled['learning_rate'])
            min_child = int(sampled['min_child_samples'])
        else:
            n_est = profile.n_estimators if profile else 100
            depth = profile.max_depth if profile else 3
            num_leaves = max(15, min(2 ** max(depth, 1), 63))
            lr = profile.learning_rate if profile else 0.05
            min_child = profile.min_child_samples if profile else 20
        label_horizon = resolve_label_horizon(profile, self.config)

        train_sample_weights = self._build_sample_weights(
            y_train,
            label_horizon_hours=label_horizon,
            symbol=f"{symbol}/train",
        )

        base_model = lgb.LGBMClassifier(
            n_estimators=n_est,
            max_depth=depth,
            num_leaves=num_leaves,
            learning_rate=lr,
            verbose=-1,
            min_child_samples=min_child,
            reg_alpha=0.1,
            reg_lambda=0.1,
            n_jobs=1
        )
        base_model.fit(X_train_scaled, y_train, sample_weight=train_sample_weights)

        val_probs = base_model.predict_proba(X_val_scaled)[:, 1]
        try:
            auc = roc_auc_score(y_val, val_probs)
        except ValueError:
            return None

        min_auc = min(profile.min_val_auc, self.config.min_val_auc) if profile else self.config.min_val_auc
        if auc < min_auc:
            return None

        holdout_size = max(20, int(len(X_val) * 0.25)) if len(X_val) >= 80 else 0
        cal_fit_probs = val_probs[:-holdout_size] if holdout_size > 0 else val_probs
        cal_fit_y = y_val.iloc[:-holdout_size] if holdout_size > 0 else y_val

        calibrator, calibrator_type = _fit_calibrator(
            self.config.calibration_strategy,
            cal_fit_probs,
            cal_fit_y,
        )

        effective_threshold = min(profile.signal_threshold, self.config.signal_threshold) if profile else self.config.signal_threshold
        primary_threshold = primary_recall_threshold(effective_threshold, self.config.min_signal_edge)
        primary_train_probs = _calibrator_predict(calibrator, base_model.predict_proba(X_train_scaled)[:, 1])
        primary_val_probs = _calibrator_predict(calibrator, val_probs)

        val_report = _calibration_report(y_val.to_numpy(dtype=int), val_probs, primary_val_probs)
        holdout_report = None
        if holdout_size > 0:
            holdout_report = _calibration_report(
                y_val.iloc[-holdout_size:].to_numpy(dtype=int),
                val_probs[-holdout_size:],
                primary_val_probs[-holdout_size:],
            )

        X_meta_tr, y_meta_tr, _ = build_meta_dataset(X_train, y_train, primary_train_probs, primary_threshold)
        X_meta_vl, y_meta_vl, _ = build_meta_dataset(X_val, y_val, primary_val_probs, primary_threshold)

        meta_artifacts: MetaArtifacts = train_meta_classifier(
            X_meta_tr,
            y_meta_tr,
            X_meta_vl,
            y_meta_vl,
            pd.Series(np.ones(len(X_meta_vl), dtype=bool), index=X_meta_vl.index),
            primary_threshold=primary_threshold,
            meta_threshold=self.config.meta_probability_threshold,
            n_estimators=n_est,
            max_depth=depth,
            learning_rate=lr,
            min_child_samples=min_child,
            calibration_strategy=self.config.calibration_strategy,
        )

        X_full = np.vstack([X_train_scaled, X_val_scaled])
        y_full = pd.concat([y_train, y_val])
        full_sample_weights = self._build_sample_weights(
            y_full,
            label_horizon_hours=label_horizon,
            symbol=f"{symbol}/full",
        )
        base_model.fit(X_full, y_full, sample_weight=full_sample_weights)

        primary_val_mask = primary_val_probs >= primary_threshold
        primary_recall = float(((primary_val_mask) & (y_val.to_numpy() == 1)).sum() / max((y_val == 1).sum(), 1))
        stage_metrics = {
            'primary_recall': primary_recall,
            'meta_precision': float(meta_artifacts.metrics.get('meta_precision', 0.0)),
            'final_trade_precision': float(meta_artifacts.metrics.get('final_trade_precision', 0.0)),
            'primary_threshold': float(primary_threshold),
            'meta_threshold': float(self.config.meta_probability_threshold),
            'calibration_strategy': calibrator_type,
            'val_calibrated_brier': val_report.get('calibrated_brier'),
            'holdout_calibrated_brier': holdout_report.get('calibrated_brier') if holdout_report else None,
        }

        member_meta = {
            'member_name': member_name,
            'member_spec': {
                'train_window_days': member_spec.train_window_days if member_spec else None,
                'feature_fraction': member_spec.feature_fraction if member_spec else None,
            },
            'feature_count': int(len(member_features) if member_features is not None else X_train.shape[1]),
            'features': list(member_features) if member_features is not None else list(X_train.columns),
            'hyperparameters': {
                'n_estimators': n_est,
                'max_depth': depth,
                'num_leaves': num_leaves,
                'learning_rate': lr,
                'min_child_samples': min_child,
            },
            'val_probs': [float(x) for x in primary_val_probs],
            'val_samples': int(len(y_val)),
            'calibration': {
                'strategy_requested': self.config.calibration_strategy,
                'strategy_used': calibrator_type,
                'primary_params': _calibrator_params(calibrator),
                'meta_strategy_used': meta_artifacts.calibrator_type,
                'meta_params': meta_calibrator_params(meta_artifacts.calibrator),
                'validation': val_report,
                'holdout': holdout_report,
                'meta_validation': meta_artifacts.calibration_metrics,
            },
        }

        return (base_model, scaler, calibrator, auc, meta_artifacts, stage_metrics, member_meta)


def resolve_label_horizon(profile: Optional[CoinProfile], config: Config) -> int:
    max_hold = profile.max_hold_hours if profile is not None else config.max_hold_hours
    forward = profile.label_forward_hours if profile is not None else config.label_forward_hours
    return resolve_profile_label_horizon(max_hold, forward)


def validate_label_execution_config(symbol: str, profile: CoinProfile, config: Config) -> None:
    label_cfg = {
        'tp_mult': profile.vol_mult_tp,
        'sl_mult': profile.vol_mult_sl,
        'horizon_h': resolve_label_horizon(profile, config),
        'profile_label_forward_h': profile.label_forward_hours,
    }
    exec_cfg = {
        'tp_mult': profile.vol_mult_tp,
        'sl_mult': profile.vol_mult_sl,
        'horizon_h': profile.max_hold_hours,
        'config_tp_default': config.vol_mult_tp,
        'config_sl_default': config.vol_mult_sl,
        'config_hold_default': config.max_hold_hours,
    }
    print(f"[{symbol}] Label config: {label_cfg} | Execution config: {exec_cfg}")
    if (
        label_cfg['tp_mult'] != exec_cfg['tp_mult']
        or label_cfg['sl_mult'] != exec_cfg['sl_mult']
        or label_cfg['horizon_h'] != exec_cfg['horizon_h']
    ):
        raise ValueError(f"Label/execution configuration mismatch for {symbol}.")


def validate_all_symbol_configs(all_data: Dict, config: Config,
                                profile_overrides: Optional[Dict[str, CoinProfile]] = None) -> None:
    for sym in all_data:
        profile = _get_profile(sym, profile_overrides)
        validate_label_execution_config(sym, profile, config)


def load_data():
    data = {}
    print("⏳ Loading data from features and database...")
    feature_files = list(FEATURES_DIR.glob("*_features.csv"))

    for f in feature_files:
        sym = f.stem.replace("_features", "").replace("_", "-")
        try:
            conn = sqlite3.connect(DB_PATH)
            ohlcv = pd.read_sql_query(
                f"SELECT event_time, open, high, low, close FROM ohlcv "
                f"WHERE symbol='{sym}' AND timeframe='1h'",
                conn, parse_dates=['event_time']
            ).set_index('event_time').sort_index()
            conn.close()
            ohlcv.index = pd.to_datetime(ohlcv.index, utc=True)
            ohlcv['sma_200'] = ohlcv['close'].rolling(200).mean()

            feat = pd.read_csv(f, index_col=0, parse_dates=True)
            feat = feat.replace([np.inf, -np.inf], 0).ffill().fillna(0)
            feat.index = pd.to_datetime(feat.index, utc=True)

            common = feat.index.intersection(ohlcv.index)
            if len(common) > 1000:
                data[sym] = {'features': feat.loc[common], 'ohlcv': ohlcv.loc[common]}
        except Exception as e:
            print(f"Skipping {sym}: {e}")
    print(f"✅ Loaded {len(data)} symbols.")

    for sym in data:
        spec = get_contract_spec(sym)
        price = data[sym]['ohlcv']['close'].iloc[-1]
        notional = spec['units'] * price
        eff_fee = max(0.20, notional * 0.0010) / notional * 100
        profile = get_coin_profile(sym)
        print(f"  {sym}: {spec['units']} units/contract, "
              f"~${notional:.2f}/contract, "
              f"effective fee: {eff_fee:.3f}% per side | "
              f"profile: {profile.name} momentum")

    return data


def check_correlation(sym: str, direction: int, active_positions: Dict,
                      all_data: Dict, ts, config: Config) -> bool:
    if not active_positions:
        return True
    lookback = config.correlation_lookback_hours
    new_returns = all_data[sym]['ohlcv']['close'].pct_change()

    for existing_sym in active_positions:
        existing_returns = all_data[existing_sym]['ohlcv']['close'].pct_change()
        combined = pd.DataFrame({
            'new': new_returns, 'existing': existing_returns
        }).dropna().loc[:ts].tail(lookback)
        if len(combined) < 20:
            continue
        corr = combined['new'].corr(combined['existing'])
        if abs(corr) > config.max_portfolio_correlation:
            existing_dir = active_positions[existing_sym]['dir']
            if corr > 0 and direction == existing_dir:
                return False
            if corr < -config.max_portfolio_correlation and direction != existing_dir:
                return False
    return True


# BACKTEST — v8 PER-COIN PROFILES
def _get_profile(symbol: str, profile_overrides: Optional[Dict[str, CoinProfile]] = None) -> CoinProfile:
    if profile_overrides:
        prefix = symbol.split('-')[0].upper()
        for profile in profile_overrides.values():
            if prefix in profile.prefixes:
                return profile
    return get_coin_profile(symbol)


def run_backtest(all_data: Dict, config: Config,
                 profile_overrides: Optional[Dict[str, CoinProfile]] = None):
    system = MLSystem(config)
    validate_all_symbol_configs(all_data, config, profile_overrides=profile_overrides)

    all_ts = [ts for d in all_data.values() for ts in d['ohlcv'].index]
    if not all_ts:
        return
    current_date = min(all_ts) + timedelta(days=config.train_lookback_days)
    end_date = max(all_ts)

    print(f"\n⏩ STARTING BACKTEST (v8 — Per-Coin Profiles)")
    print(f"   Period: {current_date.date()} to {end_date.date()}")
    print(f"   Leverage: {config.leverage}x | Fee: 0.10%/side (0.20% round-trip)")
    print(f"   Coin profiles:")
    for sym in all_data:
        p = _get_profile(sym, profile_overrides)
        print(f"     {sym}: {p.name} (momentum) | thresh={p.signal_threshold} | "
              f"TP={p.vol_mult_tp}x SL={p.vol_mult_sl}x | hold={p.max_hold_hours}h")

    equity = 100_000.0
    peak_equity = equity
    max_drawdown = 0.0
    completed_trades = []
    active_positions = {}
    last_exit_time = {}
    models_rejected = 0
    models_accepted = 0
    regime_filtered = 0
    no_contracts = 0
    stage_metric_history: List[Dict[str, float]] = []

    weekly_equity_base = equity

    while current_date < end_date:
        if equity < config.min_equity:
            print(f"\n🛑 EQUITY BELOW MINIMUM (${equity:,.2f}). Stopping.")
            break

        week_end = current_date + timedelta(days=config.retrain_frequency_days)

        weekly_equity_base = min(equity, weekly_equity_base * (1 + config.max_weekly_equity_growth))
        weekly_equity_base = max(weekly_equity_base, equity * 0.9)

        # --- TRAINING (ENSEMBLE: heterogeneous members) ---
        ensemble_members = build_ensemble_member_specs()
        all_ensemble_models = {}

        for member_spec in ensemble_members:
            for sym, d in all_data.items():
                profile = _get_profile(sym, profile_overrides)
                feat, ohlc = d['features'], d['ohlcv']
                coin_feature_list = profile.resolve_feature_columns(
                    use_pruned_features=config.enforce_pruned_features,
                    strict_pruned=config.enforce_pruned_features,
                )
                base_cols = system.get_feature_columns(feat.columns, coin_feature_list)
                cols = system.build_member_features(sym, base_cols, member_spec)
                print(
                    f"[{sym}/{member_spec.name}] Features selected: {len(cols)} / {len(base_cols)} "
                    f"(window={member_spec.train_window_days}d, pruned_only={config.enforce_pruned_features})"
                )
                if not cols:
                    continue

                label_horizon = resolve_label_horizon(profile, config)
                embargo_hours = max(config.train_embargo_hours, label_horizon, 1)
                train_end = current_date - timedelta(hours=embargo_hours)
                train_start = train_end - timedelta(days=member_spec.train_window_days)
                if train_end <= train_start:
                    continue

                train_feat = feat.loc[train_start:train_end]
                train_ohlc = ohlc.loc[train_start:train_end + timedelta(hours=label_horizon)]

                if len(train_feat) < config.min_train_samples:
                    continue

                y = system.create_labels(train_ohlc, train_feat, profile=profile)
                valid_idx = y.dropna().index
                X_all = train_feat.loc[valid_idx, cols]
                y_all = y.loc[valid_idx]
                X_all, y_all = system.prepare_binary_training_set(X_all, y_all)

                if len(X_all) < config.min_train_samples:
                    continue

                split_idx = int(len(X_all) * (1 - config.val_fraction))
                X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
                y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

                result = system.train(
                    X_tr,
                    y_tr,
                    X_vl,
                    y_vl,
                    profile=profile,
                    symbol=sym,
                    member_spec=member_spec,
                    member_features=cols,
                )
                if result:
                    model, scaler, iso, auc, meta_artifacts, stage_metrics, member_meta = result
                    if sym not in all_ensemble_models:
                        all_ensemble_models[sym] = []
                    all_ensemble_models[sym].append((
                        model, scaler, cols, iso, auc, meta_artifacts, stage_metrics, member_meta
                    ))
                    stage_metric_history.append(stage_metrics)
                    models_accepted += 1
                else:
                    models_rejected += 1

        models = {}
        for sym, ensemble_list in all_ensemble_models.items():
            if ensemble_list:
                models[sym] = ensemble_list
                member_outputs = [m[7] for m in ensemble_list if len(m) > 7]
                system.log_member_pairwise_correlation(sym, member_outputs)

        # --- TRADING ---
        test_start = current_date + timedelta(hours=1)
        test_hours = pd.date_range(test_start, week_end, freq='1h')

        for ts in test_hours:
            # 1. MANAGE EXITS
            to_close = []
            for sym, pos in active_positions.items():
                if ts not in all_data[sym]['ohlcv'].index:
                    continue

                pos_profile = _get_profile(sym, profile_overrides)

                funding_hourly_bps = all_data[sym]['features'].loc[ts].get('funding_rate_bps', 0.0)
                if pd.isna(funding_hourly_bps):
                    funding_hourly_bps = 0.0
                pos['accum_funding'] += -(funding_hourly_bps / 10000.0) * pos['dir']

                bar = all_data[sym]['ohlcv'].loc[ts]
                direction = pos['dir']

                if direction == 1:
                    pos['peak_price'] = max(pos['peak_price'], bar['high'])
                else:
                    pos['peak_price'] = min(pos['peak_price'], bar['low'])

                peak_move = (pos['peak_price'] - pos['entry']) / pos['entry'] * direction

                if not pos['at_breakeven'] and peak_move >= config.breakeven_trigger * pos['vol']:
                    effective_fee_pct = pos['effective_fee_pct']
                    pos['sl'] = pos['entry'] * (1 + effective_fee_pct * direction)
                    pos['at_breakeven'] = True

                if pos['at_breakeven'] and config.trailing_active:
                    if direction == 1:
                        trail_sl = pos['peak_price'] * (1 - config.trailing_mult * pos['vol'])
                        pos['sl'] = max(pos['sl'], trail_sl)
                    else:
                        trail_sl = pos['peak_price'] * (1 + config.trailing_mult * pos['vol'])
                        pos['sl'] = min(pos['sl'], trail_sl)

                exit_price, reason = None, None

                tp_hit = (direction == 1 and bar['high'] >= pos['tp']) or \
                         (direction == -1 and bar['low'] <= pos['tp'])
                sl_hit = (direction == 1 and bar['low'] <= pos['sl']) or \
                         (direction == -1 and bar['high'] >= pos['sl'])

                if tp_hit and sl_hit:
                    dist_tp = abs(bar['open'] - pos['tp'])
                    dist_sl = abs(bar['open'] - pos['sl'])
                    if dist_sl <= dist_tp:
                        exit_price = pos['sl']
                        reason = 'stop_loss' if not pos['at_breakeven'] else \
                                 ('trailing_stop' if pos['sl'] != pos['entry'] and pos['at_breakeven'] else 'breakeven')
                    else:
                        exit_price, reason = pos['tp'], 'take_profit'
                elif tp_hit:
                    exit_price, reason = pos['tp'], 'take_profit'
                elif sl_hit:
                    exit_price = pos['sl']
                    if pos['at_breakeven']:
                        raw_ret = (exit_price - pos['entry']) / pos['entry'] * direction
                        reason = 'trailing_stop' if raw_ret > 0.001 else 'breakeven'
                    else:
                        reason = 'stop_loss'

                # Max hold — use per-coin profile
                if not exit_price and (ts - pos['time']).total_seconds() / 3600 >= pos_profile.max_hold_hours:
                    exit_price, reason = bar['close'], 'max_hold'

                if exit_price:
                    net_pnl, raw_pnl, fee_pnl, fee_pct_pnl, min_fee_pnl, slippage_pnl, pnl_dollars, notional = calculate_pnl_exact(
                        pos['entry'], exit_price, direction,
                        pos['accum_funding'], pos['n_contracts'], sym, config
                    )
                    pnl_dollars = max(pnl_dollars, -notional)
                    equity += pnl_dollars
                    peak_equity = max(peak_equity, equity)
                    drawdown = (peak_equity - equity) / peak_equity
                    max_drawdown = max(max_drawdown, drawdown)
                    completed_trades.append(Trade(
                        sym, pos['time'], ts, direction, pos['entry'], exit_price,
                        net_pnl, raw_pnl, pos['accum_funding'], fee_pnl,
                        fee_pct_pnl, min_fee_pnl, slippage_pnl,
                        pnl_dollars, pos['n_contracts'], notional, reason
                    ))
                    to_close.append(sym)
                    last_exit_time[sym] = ts

            for sym in to_close:
                del active_positions[sym]

            # 2. ENTRIES
            if len(active_positions) < config.max_positions and equity >= config.min_equity:
                candidates = []
                for sym, ensemble_models in models.items():
                    if sym in active_positions or ts not in all_data[sym]['features'].index:
                        continue
                    if ts not in all_data[sym]['ohlcv'].index:
                        continue

                    profile = _get_profile(sym, profile_overrides)

                    # Cooldown — use per-coin profile
                    if sym in last_exit_time and \
                       (ts - last_exit_time[sym]).total_seconds() < profile.cooldown_hours * 3600:
                        continue

                    # Symbol exclusion (CLI override)
                    if config.excluded_symbols:
                        sym_prefix = sym.split('-')[0] if '-' in sym else sym
                        if sym_prefix in config.excluded_symbols:
                            continue

                    row = all_data[sym]['features'].loc[ts]
                    ohlcv_row = all_data[sym]['ohlcv'].loc[ts]
                    price = ohlcv_row['close']
                    sma_200 = ohlcv_row['sma_200']
                    if pd.isna(sma_200):
                        continue

                    # Regime filter — per-coin thresholds
                    vol_24h = all_data[sym]['ohlcv']['close'].pct_change().rolling(24).std().get(ts, None)
                    if vol_24h is None or pd.isna(vol_24h):
                        continue
                    if vol_24h < profile.min_vol_24h or vol_24h > profile.max_vol_24h:
                        regime_filtered += 1
                        continue

                    # ── Momentum direction (all coins) ──
                    ts_loc = all_data[sym]['ohlcv'].index.get_loc(ts)

                    # v7 guard: need at least 72 bars of history
                    if ts_loc < 72:
                        continue

                    ohlcv_ts = all_data[sym]['ohlcv']
                    ret_24h = (price / ohlcv_ts['close'].iloc[ts_loc - 24] - 1)
                    ret_72h = (price / ohlcv_ts['close'].iloc[ts_loc - 72] - 1)
                    sma_50 = ohlcv_ts['close'].iloc[max(0, ts_loc - 50):ts_loc].mean()

                    # v7.3: Minimum momentum magnitude
                    effective_momentum = min(profile.min_momentum_magnitude, config.min_momentum_magnitude)
                    if abs(ret_72h) < effective_momentum:
                        continue

                    # v7.3: Require 24h and 72h returns to agree on direction
                    if ret_24h * ret_72h < 0:
                        continue

                    momentum_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)

                    if momentum_score >= 2:
                        direction = 1
                    elif momentum_score <= -2:
                        direction = -1
                    else:
                        continue

                    # Trend filter: long only above SMA200, short only below
                    if direction == 1 and price < sma_200:
                        continue
                    if direction == -1 and price > sma_200:
                        continue

                    # Funding carry filter
                    f_z = row.get('funding_rate_zscore', 0)
                    if pd.isna(f_z):
                        f_z = 0
                    if direction == 1 and f_z > 2.5:
                        continue
                    if direction == -1 and f_z < -2.5:
                        continue

                    # Correlation filter
                    if not check_correlation(sym, direction, active_positions, all_data, ts, config):
                        continue

                    probs = []
                    meta_probs = []
                    directional_votes = []
                    effective_threshold = min(profile.signal_threshold, config.signal_threshold)
                    primary_cutoff = primary_recall_threshold(effective_threshold, config.min_signal_edge)
                    for (model, scaler, cols, iso, auc, meta_artifacts, stage_metrics, member_meta) in models[sym]:
                        x_in = np.nan_to_num(
                            np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0
                        )
                        raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
                        cal_prob = float(_calibrator_predict(iso, np.array([raw_prob]))[0])
                        probs.append(cal_prob)
                        directional_votes.append(1 if (cal_prob >= 0.5 and direction == 1) or (cal_prob < 0.5 and direction == -1) else 0)

                        if cal_prob >= primary_cutoff and meta_artifacts.model is not None and meta_artifacts.scaler is not None:
                            meta_raw = meta_artifacts.model.predict_proba(meta_artifacts.scaler.transform(x_in))[0, 1]
                            if meta_artifacts.calibrator is not None:
                                meta_cal = float(meta_calibrator_predict(meta_artifacts.calibrator, np.array([meta_raw]))[0])
                            else:
                                meta_cal = float(meta_raw)
                            meta_probs.append(meta_cal)

                    if not probs:
                        continue
                    agreement = float(np.mean(directional_votes)) if directional_votes else 0.0
                    if agreement < config.min_directional_agreement:
                        continue

                    prob = float(np.mean(probs))
                    prob_std = float(np.std(probs))
                    if agreement < 0.999:
                        prob = min(prob, config.disagreement_confidence_cap)
                    if prob < primary_cutoff or prob_std > config.max_ensemble_std:
                        continue

                    if not meta_probs:
                        continue
                    meta_prob = float(np.mean(meta_probs))
                    if meta_prob < config.meta_probability_threshold:
                        continue

                    edge_score = (meta_prob - config.meta_probability_threshold) / max(0.01, prob_std + 0.01)
                    momentum_strength = abs(ret_72h)
                    rank_score = edge_score + (0.2 * momentum_strength)
                    candidates.append((rank_score, sym, profile, price, vol_24h, direction))

                slots = max(config.max_positions - len(active_positions), 0)
                for _, sym, profile, price, vol_24h, direction in sorted(candidates, key=lambda x: x[0], reverse=True)[:slots]:
                    vol = vol_24h if vol_24h > 0 else 0.02

                    n_contracts = calculate_n_contracts(
                        weekly_equity_base, price, sym, config,
                        vol_24h=vol_24h, profile=profile
                    )
                    if n_contracts < 1:
                        no_contracts += 1
                        continue

                    spec = get_contract_spec(sym)
                    notional_per_contract = spec['units'] * price
                    total_notional = n_contracts * notional_per_contract
                    est_roundtrip_cost, _, _, _ = calculate_execution_costs(
                        n_contracts, price, price, sym, config
                    )
                    effective_fee_pct = est_roundtrip_cost / total_notional

                    # Per-coin TP/SL
                    active_positions[sym] = {
                        'time': ts,
                        'entry': price,
                        'dir': direction,
                        'vol': vol,
                        'tp': price * (1 + profile.vol_mult_tp * vol * direction),
                        'sl': price * (1 - profile.vol_mult_sl * vol * direction),
                        'accum_funding': 0.0,
                        'n_contracts': n_contracts,
                        'peak_price': price,
                        'at_breakeven': False,
                        'effective_fee_pct': effective_fee_pct,
                    }

        current_date = week_end

    # --- REPORT ---
    if not completed_trades:
        print("\n⚠️ No trades executed.")
        print(f"   Models accepted: {models_accepted}, rejected: {models_rejected}")
        print(f"   Regime filtered: {regime_filtered}")
        print(f"   No contracts (too small): {no_contracts}")
        return {'n_trades': 0, 'sharpe_annual': -99, 'total_return': -1, 'max_drawdown': 1.0,
                'profit_factor': 0, 'ann_return': -1, 'final_equity': equity, 'trade_pnls': []}

    df = pd.DataFrame([t.__dict__ for t in completed_trades])
    win_rate = (df['net_pnl'] > 0).mean()
    total_ret = (equity / 100000) - 1
    reason_counts = df['exit_reason'].value_counts()

    avg_pnl = df['net_pnl'].mean()
    std_pnl = df['net_pnl'].std()
    sharpe_per_trade = avg_pnl / std_pnl if std_pnl > 0 else 0

    gross_wins = df.loc[df['net_pnl'] > 0, 'pnl_dollars'].sum()
    gross_losses = abs(df.loc[df['net_pnl'] < 0, 'pnl_dollars'].sum())
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    total_pnl_dollars = df['pnl_dollars'].sum()

    days = (end_date - (min(all_ts) + timedelta(days=config.train_lookback_days))).days
    years = days / 365.25
    ann_return = (1 + total_ret) ** (1 / years) - 1 if years > 0 and total_ret > -1 else -1

    trades_per_year = len(df) / years if years > 0 else 0
    ann_sharpe = sharpe_per_trade * np.sqrt(trades_per_year) if trades_per_year > 0 else 0

    avg_fee = df['fee_pnl'].mean()
    avg_fee_pct = df['fee_pct_pnl'].mean()
    avg_min_fee = df['min_fee_pnl'].mean()
    avg_slippage = df['slippage_pnl'].mean()
    avg_raw = df['raw_pnl'].mean()
    avg_funding = df['funding_pnl'].mean()

    print(f"\n{'=' * 70}")
    print(f"📊 BACKTEST RESULTS (v8 — Per-Coin Profiles)")
    print(f"{'=' * 70}")
    print(f"Total Trades:           {len(df)}")
    print(f"Win Rate:               {win_rate:.1%}")
    print(f"Profit Factor:          {profit_factor:.2f}")
    print(f"Total Return:           {total_ret:.2%}")
    print(f"Annualized Return:      {ann_return:.2%}")
    print(f"Final Equity:           ${equity:,.2f}")
    print(f"Max Drawdown:           {max_drawdown:.2%}")
    print(f"Total PnL:              ${total_pnl_dollars:,.2f}")
    print(f"Avg PnL/Trade:          ${df['pnl_dollars'].mean():,.2f}")
    print(f"")
    print(f"Sharpe (per-trade):     {sharpe_per_trade:.3f}")
    print(f"Sharpe (annualized):    {ann_sharpe:.2f}")
    print(f"Trades/Year:            {trades_per_year:.0f}")
    print(f"")
    print(f"Edge Decomposition (% of notional):")
    print(f"  Avg Raw PnL:          {avg_raw:.4%}")
    print(f"  Avg Funding:          {avg_funding:.4%}")
    print(f"  Avg Fee % Component:  {avg_fee_pct:.4%}")
    print(f"  Avg Min-Fee Component:{avg_min_fee:.4%}")
    print(f"  Avg Slippage:         {avg_slippage:.4%}")
    print(f"  Avg Fees (Total):     {avg_fee:.4%}")
    print(f"  Avg Net PnL:          {avg_pnl:.4%}")
    if avg_raw != 0:
        print(f"  Fee/Edge Ratio:       {abs(avg_fee/avg_raw)*100:.0f}% of raw edge consumed by fees")
    print(f"")
    print(f"Avg Contracts/Trade:    {df['n_contracts'].mean():.1f}")
    print(f"Avg Notional/Trade:     ${df['notional'].mean():,.0f}")
    print(f"")
    print(f"Model Stats:")
    print(f"  Accepted: {models_accepted}  |  Rejected: {models_rejected}")
    print(f"  Regime filtered: {regime_filtered}")
    print(f"  No contracts (sizing): {no_contracts}")
    if stage_metric_history:
        avg_primary_recall = np.mean([m.get('primary_recall', 0.0) for m in stage_metric_history])
        avg_meta_precision = np.mean([m.get('meta_precision', 0.0) for m in stage_metric_history])
        avg_final_precision = np.mean([m.get('final_trade_precision', 0.0) for m in stage_metric_history])
        print(f"  Stage Metrics: primary_recall={avg_primary_recall:.3f} | meta_precision={avg_meta_precision:.3f} | final_trade_precision={avg_final_precision:.3f}")
    print(f"")
    print(f"Exit Reasons:")
    for reason, count in reason_counts.items():
        pct = count / len(df)
        avg = df[df['exit_reason'] == reason]['net_pnl'].mean()
        avg_d = df[df['exit_reason'] == reason]['pnl_dollars'].mean()
        print(f"  {reason:20s}: {count:4d} ({pct:.1%}) | Avg: {avg:.4%} | ${avg_d:,.2f}")

    print(f"\nPer-Symbol Performance:")
    for sym in df['symbol'].unique():
        sym_df = df[df['symbol'] == sym]
        sym_wr = (sym_df['net_pnl'] > 0).mean()
        sym_pnl = sym_df['pnl_dollars'].sum()
        sym_avg = sym_df['net_pnl'].mean()
        sym_fee = sym_df['fee_pnl'].mean()
        p = _get_profile(sym, profile_overrides)
        print(f"  {sym:20s}: {len(sym_df):3d} trades | WR: {sym_wr:.0%} | "
              f"PnL: ${sym_pnl:+,.0f} | Avg: {sym_avg:.4%} | Fee: {sym_fee:.4%} | "
              f"[{p.name}/momentum]")
    print(f"{'=' * 70}")

    # ── Save final models from last walk-forward window ──
    print(f"\n💾 Saving final models...")
    saved_count = 0
    for sym, ensemble_list in models.items():
        if ensemble_list:
            best = max(ensemble_list, key=lambda x: x[4])
            model, scaler, cols, iso, auc, meta_artifacts, stage_metrics, member_meta = best
            importance_summary = summarize_feature_importance(model, cols)
            profile = get_coin_profile(sym)
            save_model(
                symbol=sym,
                model=model,
                scaler=scaler,
                calibrator=iso,
                feature_columns=cols,
                auc=auc,
                profile_name=profile.name,
                extra_meta={
                    'strategy': 'momentum',
                    'signal_threshold': profile.signal_threshold,
                    'n_ensemble': len(ensemble_list),
                    'backtest_trades': len(df[df['symbol'] == sym]) if sym in df['symbol'].values else 0,
                    'stage_metrics': stage_metrics,
                    'calibration': member_meta.get('calibration', {}),
                    'secondary_calibration': {
                        'strategy_used': meta_artifacts.calibrator_type,
                        'params': meta_calibrator_params(meta_artifacts.calibrator),
                        'metrics': meta_artifacts.calibration_metrics,
                    },
                    'feature_importance': importance_summary,
                    'ensemble_member': member_meta,
                    'ensemble_members': [m[7] for m in ensemble_list if len(m) > 7],
                },
                secondary_model=meta_artifacts.model,
                secondary_scaler=meta_artifacts.scaler,
                secondary_calibrator=meta_artifacts.calibrator,
            )
            saved_count += 1
    print(f"✅ Saved {saved_count} models to {MODELS_DIR}/")

    oos_cutoff = end_date - timedelta(days=max(config.oos_eval_days, 1))
    oos_df = df[df['exit_time'] >= oos_cutoff]
    oos_sharpe = -99.0
    oos_return = -1.0
    if len(oos_df) >= 5:
        oos_avg = oos_df['net_pnl'].mean()
        oos_std = oos_df['net_pnl'].std()
        oos_sharpe = oos_avg / oos_std if oos_std > 0 else 0.0
        oos_return = oos_df['pnl_dollars'].sum() / 100000.0

    return {
        'total_return': total_ret,
        'ann_return': ann_return,
        'sharpe_annual': ann_sharpe,
        'sharpe_per_trade': sharpe_per_trade,
        'max_drawdown': max_drawdown,
        'profit_factor': profit_factor,
        'win_rate': win_rate,
        'n_trades': len(df),
        'trades_per_year': trades_per_year,
        'avg_net_pnl': avg_pnl,
        'avg_raw_pnl': avg_raw,
        'avg_fee_pnl': avg_fee,
        'avg_fee_pct_component': avg_fee_pct,
        'avg_min_fee_component': avg_min_fee,
        'avg_slippage_component': avg_slippage,
        'final_equity': equity,
        'oos_trades': int(len(oos_df)),
        'oos_sharpe': oos_sharpe,
        'oos_return': oos_return,
        'trade_pnls': [float(x) for x in df['pnl_dollars'].tolist()],
    }


def retrain_models(all_data: Dict, config: Config, target_dir: Optional[Path] = None, train_window_days: int = 90) -> Dict:
    system = MLSystem(config)
    validate_all_symbol_configs(all_data, config)
    metrics = {}
    symbols_trained = 0
    train_end_global = pd.Timestamp.now(tz='UTC')
    train_start_global = train_end_global - pd.Timedelta(days=train_window_days)

    for sym, d in all_data.items():
        profile = get_coin_profile(sym)
        feat, ohlc = d['features'], d['ohlcv']
        coin_feature_list = profile.resolve_feature_columns(
            use_pruned_features=config.enforce_pruned_features,
            strict_pruned=config.enforce_pruned_features,
        )
        cols = system.get_feature_columns(feat.columns, coin_feature_list)
        print(f"[{sym}] Features selected: {len(cols)} (pruned_only={config.enforce_pruned_features})")
        if not cols:
            continue

        train_feat = feat.loc[feat.index >= train_start_global]
        train_ohlc = ohlc.loc[ohlc.index >= train_start_global]

        y = system.create_labels(train_ohlc, train_feat, profile=profile)
        valid = y.dropna().index
        X_all = train_feat.loc[valid, cols]
        y_all = y.loc[valid]
        X_all, y_all = system.prepare_binary_training_set(X_all, y_all)
        if len(X_all) < config.min_train_samples:
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

        result = system.train(X_tr, y_tr, X_vl, y_vl, profile=profile, symbol=sym)
        if not result:
            continue

        model, scaler, iso, auc, meta_artifacts, stage_metrics, member_meta = result
        importance_summary = summarize_feature_importance(model, cols)
        symbol_train_start = X_all.index.min()
        symbol_train_end = X_all.index.max()
        model_metrics = {
            'auc': float(auc),
            'train_samples': int(len(X_all)),
            'val_samples': int(len(X_vl)),
            'primary_recall': float(stage_metrics.get('primary_recall', 0.0)),
            'meta_precision': float(stage_metrics.get('meta_precision', 0.0)),
            'final_trade_precision': float(stage_metrics.get('final_trade_precision', 0.0)),
            'cost_aware_importance_share': float(importance_summary.get('cost_aware_block', {}).get('share', 0.0)),
        }
        save_model(
            symbol=sym,
            model=model,
            scaler=scaler,
            calibrator=iso,
            feature_columns=cols,
            auc=auc,
            profile_name=profile.name,
            target_dir=target_dir,
            extra_meta={
                'strategy': 'momentum',
                'signal_threshold': profile.signal_threshold,
                'train_start': symbol_train_start.isoformat() if symbol_train_start is not None else None,
                'train_end': symbol_train_end.isoformat() if symbol_train_end is not None else None,
                'metrics': model_metrics,
                'stage_metrics': stage_metrics,
                'calibration': member_meta.get('calibration', {}),
                'secondary_calibration': {
                    'strategy_used': meta_artifacts.calibrator_type,
                    'params': meta_calibrator_params(meta_artifacts.calibrator),
                    'metrics': meta_artifacts.calibration_metrics,
                },
                'feature_importance': importance_summary,
                'ensemble_member': member_meta,
            },
            secondary_model=meta_artifacts.model,
            secondary_scaler=meta_artifacts.scaler,
            secondary_calibrator=meta_artifacts.calibrator,
        )
        symbols_trained += 1
        metrics[sym] = model_metrics

    return {
        'symbols_total': len(all_data),
        'symbols_trained': symbols_trained,
        'train_start': train_start_global.isoformat(),
        'train_end': train_end_global.isoformat(),
        'metrics': metrics,
    }


# LIVE SIGNALS — v8
def run_signals(all_data: Dict, config: Config, debug: bool = False):
    system = MLSystem(config)
    validate_all_symbol_configs(all_data, config)
    print(f"\n🔍 ANALYZING LIVE MARKETS (v8 — Per-Coin Profiles)...")

    for sym, d in all_data.items():
        profile = get_coin_profile(sym)
        feat, ohlc = d['features'], d['ohlcv']
        coin_feature_list = profile.resolve_feature_columns(
            use_pruned_features=config.enforce_pruned_features,
            strict_pruned=config.enforce_pruned_features,
        )
        cols = system.get_feature_columns(feat.columns, coin_feature_list)
        print(f"[{sym}] Features selected: {len(cols)} (pruned_only={config.enforce_pruned_features})")
        if not cols:
            if debug:
                print(f"\n[{sym}] ❌ Not enough features (need ≥4, got {len(cols)})")
            continue

        train_end = feat.index[-1]
        train_start = train_end - timedelta(days=config.train_lookback_days)
        train_feat = feat.loc[train_start:train_end]
        train_ohlc = ohlc.loc[train_start:train_end]

        y = system.create_labels(train_ohlc, train_feat, profile=profile)
        valid = y.dropna().index
        X_all = train_feat.loc[valid, cols]
        y_all = y.loc[valid]
        X_all, y_all = system.prepare_binary_training_set(X_all, y_all)

        if len(X_all) < config.min_train_samples:
            if debug:
                print(f"\n[{sym}] ❌ Insufficient samples ({len(X_all)} < {config.min_train_samples})")
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        X_tr, X_vl = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
        y_tr, y_vl = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

        result = system.train(X_tr, y_tr, X_vl, y_vl, profile=profile, symbol=sym)
        if not result:
            if debug:
                print(f"\n[{sym}] ❌ MODEL REJECTED (low AUC < {profile.min_val_auc})")
            continue

        model, scaler, iso, auc, meta_artifacts, stage_metrics, member_meta = result
        importance_summary = summarize_feature_importance(model, cols)

        save_model(
            symbol=sym,
            model=model,
            scaler=scaler,
            calibrator=iso,
            feature_columns=cols,
            auc=auc,
            profile_name=profile.name,
            extra_meta={
                'strategy': 'momentum',
                'signal_threshold': profile.signal_threshold,
                'train_samples': len(X_all),
                'stage_metrics': stage_metrics,
                'calibration': member_meta.get('calibration', {}),
                'secondary_calibration': {
                    'strategy_used': meta_artifacts.calibrator_type,
                    'params': meta_calibrator_params(meta_artifacts.calibrator),
                    'metrics': meta_artifacts.calibration_metrics,
                },
                'feature_importance': importance_summary,
                'ensemble_member': member_meta,
            },
            secondary_model=meta_artifacts.model,
            secondary_scaler=meta_artifacts.scaler,
            secondary_calibrator=meta_artifacts.calibrator,
        )

        row = feat.iloc[-1]
        price = ohlc.iloc[-1]['close']
        sma_200 = ohlc.iloc[-1]['sma_200']

        f_z = row.get('funding_rate_zscore', 0)
        if pd.isna(f_z):
            f_z = 0

        ts_loc = len(ohlc) - 1

        ret_24h = (price / ohlc['close'].iloc[ts_loc - 24] - 1) if ts_loc >= 24 else 0
        ret_72h = (price / ohlc['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
        sma_50 = ohlc['close'].iloc[max(0, ts_loc - 50):ts_loc].mean() if ts_loc >= 10 else price
        momentum_score = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)
        if momentum_score >= 2:
            direction = 1
        elif momentum_score <= -2:
            direction = -1
        else:
            if debug:
                print(f"\n[{sym}] ⏸️  No momentum consensus (score={momentum_score})")
            continue

        if direction == 1 and price < sma_200 and not pd.isna(sma_200):
            if debug:
                print(f"\n[{sym}] ⏸️  Long rejected: price < SMA200")
            continue
        if direction == -1 and price > sma_200 and not pd.isna(sma_200):
            if debug:
                print(f"\n[{sym}] ⏸️  Short rejected: price > SMA200")
            continue

        if direction == 1 and f_z > 2.5:
            continue
        if direction == -1 and f_z < -2.5:
            continue

        x_in = np.nan_to_num(
            np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0
        )
        raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
        prob = float(_calibrator_predict(iso, np.array([raw_prob]))[0])
        effective_threshold = min(profile.signal_threshold, config.signal_threshold)
        primary_cutoff = primary_recall_threshold(effective_threshold, config.min_signal_edge)
        ml_pass = prob >= primary_cutoff
        meta_prob = 0.0
        meta_pass = False
        if ml_pass and meta_artifacts.model is not None and meta_artifacts.scaler is not None:
            meta_raw = meta_artifacts.model.predict_proba(meta_artifacts.scaler.transform(x_in))[0, 1]
            if meta_artifacts.calibrator is not None:
                meta_prob = float(meta_calibrator_predict(meta_artifacts.calibrator, np.array([meta_raw]))[0])
            else:
                meta_prob = float(meta_raw)
            meta_pass = meta_prob >= config.meta_probability_threshold

        vol_24h = ohlc['close'].pct_change().rolling(24).std().iloc[-1]
        regime_pass = profile.min_vol_24h <= vol_24h <= profile.max_vol_24h if not pd.isna(vol_24h) else False

        ret_72h = (price / ohlc['close'].iloc[ts_loc - 72] - 1) if ts_loc >= 72 else 0
        effective_momentum = min(profile.min_momentum_magnitude, config.min_momentum_magnitude)
        momentum_pass = abs(ret_72h) >= effective_momentum

        if debug:
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            print(f"\n[{sym}] ({profile.name})")
            print(f"  Price: ${price:,.2f} | SMA200: ${sma_200:,.2f}" if not pd.isna(sma_200) else f"  Price: ${price:,.2f}")
            print(f"  Direction: {dir_str}")
            print(f"  Primary prob: {raw_prob:.3f} → Calibrated: {prob:.3f} (thresh: {primary_cutoff:.3f})")
            print(f"  AUC: {auc:.3f}")
            print(f"  Gates: Primary={'✅' if ml_pass else '❌'} | Meta={'✅' if meta_pass else '❌'} | Regime={'✅' if regime_pass else '❌'} | Mom={'✅' if momentum_pass else '❌'}")
            print(f"  Funding z-score: {f_z:.2f}")
            print(f"  24h Vol: {vol_24h:.4f}" if not pd.isna(vol_24h) else "  24h Vol: N/A")

        if ml_pass and meta_pass and regime_pass and momentum_pass:
            n_contracts = calculate_n_contracts(
                100_000, price, sym, config,
                vol_24h=vol_24h if not pd.isna(vol_24h) else 0.02,
                profile=profile
            )
            spec = get_contract_spec(sym)
            dir_str = 'LONG' if direction == 1 else 'SHORT'
            notional = n_contracts * spec['units'] * price
            print(f"🎯 {sym} [{profile.name}]: {dir_str} | "
                  f"{n_contracts} contracts | ${notional:,.0f} notional | "
                  f"Primary: {prob:.1%} | Meta: {meta_prob:.1%} | AUC: {auc:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crypto ML Trading System v8 — Per-Coin Profiles"
    )
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--signals", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.68,
                        help="Default signal threshold (overridden by per-coin profiles)")
    parser.add_argument("--min-auc", type=float, default=0.51)
    parser.add_argument("--leverage", type=int, default=4)
    parser.add_argument("--tp", type=float, default=5.5, help="Default TP vol multiplier")
    parser.add_argument("--sl", type=float, default=3.0, help="Default SL vol multiplier")
    parser.add_argument("--momentum", type=float, default=0.04, help="Default min 72h momentum magnitude")
    parser.add_argument("--hold", type=int, default=96, help="Default max hold hours")
    parser.add_argument("--cooldown", type=float, default=24, help="Default hours cooldown after exit")
    parser.add_argument("--min-edge", type=float, default=0.01, help="Require prob >= threshold + min-edge")
    parser.add_argument("--max-ensemble-std", type=float, default=0.18, help="Max std across ensemble probs")
    parser.add_argument("--min-directional-agreement", type=float, default=0.50, help="Minimum fraction of members agreeing with momentum direction")
    parser.add_argument("--disagreement-confidence-cap", type=float, default=0.86, help="Cap primary confidence when ensemble is not unanimous")
    parser.add_argument("--meta-threshold", type=float, default=0.52, help="Secondary meta-model probability threshold")
    parser.add_argument("--calibration", choices=['isotonic', 'platt', 'beta'], default='platt',
                        help="Calibration strategy for primary+meta models")
    parser.add_argument("--exclude", type=str, default="",
                        help="Comma-separated symbol prefixes to exclude (default: none)")
    parser.add_argument("--pruned-only", action="store_true",
                        help="Require pruned feature artifacts from prune_features.py")
    parser.add_argument("--recency-half-life-days", type=float, default=50.0,
                        help="Half-life in days for exponential recency sample weighting")
    args = parser.parse_args()

    excluded = [s.strip() for s in args.exclude.split(',') if s.strip()] if args.exclude else None
    config = Config(
        signal_threshold=args.threshold,
        min_val_auc=args.min_auc,
        leverage=args.leverage,
        vol_mult_tp=args.tp,
        vol_mult_sl=args.sl,
        min_momentum_magnitude=args.momentum,
        max_hold_hours=args.hold,
        cooldown_hours=args.cooldown,
        min_signal_edge=args.min_edge,
        max_ensemble_std=args.max_ensemble_std,
        min_directional_agreement=args.min_directional_agreement,
        disagreement_confidence_cap=args.disagreement_confidence_cap,
        meta_probability_threshold=args.meta_threshold,
        calibration_strategy=args.calibration,
        excluded_symbols=excluded,
        enforce_pruned_features=args.pruned_only,
        recency_half_life_days=args.recency_half_life_days,
    )
    data = load_data()

    if args.backtest:
        run_backtest(data, config)
    elif args.signals or args.debug:
        run_signals(data, config, debug=args.debug)
    else:
        parser.print_help()
