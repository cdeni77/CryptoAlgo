#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (v11: Anti-Overfit).

v11 CHANGES (from v10):
  ════════════════════════════════════════════════════════════════════
  1. WALK-FORWARD CROSS-VALIDATION in objective (3-fold temporal CV)
     - Score = average OOS Sharpe across folds, NOT in-sample Sharpe
     - Optimizer literally cannot overfit to a single lucky period
  2. REDUCED PARAMETER SPACE (18 → 10 tunable params)
     - Fixed: min_val_auc, n_estimators, max_depth, learning_rate,
       min_child_samples, position_size, vol_sizing_target, cooldown_hours
     - These are either ML internals (shouldn't be tuned per-coin via
       outer optimizer) or risk params (should be set by policy)
  3. TIGHTER RANGES with coarser steps
     - Fewer unique combinations = less room to find noise patterns
  4. SCORING = OOS-FIRST
     - Primary score is mean OOS Sharpe across CV folds
     - In-sample metrics only used as tiebreakers/penalties
  5. HARD MINIMUM TRADE FILTER
     - Trials with <30 total trades across all folds → rejected
     - Trials with <5 OOS trades per fold → heavily penalized
  6. STRONGER OVERFIT PENALTIES
     - IS/OOS Sharpe ratio penalty (if IS >> OOS, penalize)
     - Win rate > 75% with few trades → suspicious
  7. REDUCED DEFAULT TRIALS (200 → 100)
     - With tighter space, 100 trials is plenty
  ════════════════════════════════════════════════════════════════════

Usage:
    python optimize.py --coin BTC --trials 100 --jobs 4
    python optimize.py --all --trials 100 --jobs 16
    python optimize.py --show
"""
import argparse
import json
import warnings
import sys
import os
import logging
import sqlite3
import functools
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from scripts.train_model import Config, load_data, run_backtest
from core.coin_profiles import (
    CoinProfile, COIN_PROFILES,
    BTC_EXTRA_FEATURES, SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
PREFIX_TO_SYMBOL: Dict[str, str] = {}
DEBUG_TRIALS = False


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS (unchanged from v10)
# ═══════════════════════════════════════════════════════════════════════════

def _to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def init_db_wal(db_name="optuna_trading.db"):
    try:
        conn = sqlite3.connect(db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")
        conn.close()
    except Exception as e:
        print(f"⚠️ Warning: Could not set WAL mode: {e}")


def get_extra_features(coin_name: str):
    mapping = {
        'BTC': BTC_EXTRA_FEATURES,
        'SOL': SOL_EXTRA_FEATURES,
        'DOGE': DOGE_EXTRA_FEATURES,
    }
    return mapping.get(coin_name, [])


def _as_number(value, default=None):
    if value is None: return default
    if isinstance(value, (int, float)): return float(value)
    try: return float(value)
    except (TypeError, ValueError): return default


def _finite_metric(value, default=0.0):
    n = _as_number(value, default=default)
    if n is None or not np.isfinite(n): return default
    return float(n)


def _fmt_pct(value, decimals=1, fallback="?"):
    n = _as_number(value)
    return f"{n:.{decimals}%}" if n is not None else fallback


def _fmt_float(value, decimals=3, fallback="?"):
    n = _as_number(value)
    return f"{n:.{decimals}f}" if n is not None else fallback


def _is_invalid_holdout_metric(holdout_sharpe, holdout_return, holdout_trades):
    if holdout_sharpe <= -90: return True
    if holdout_trades <= 0: return True
    if holdout_return <= -0.99: return True
    return False


def _set_reject_reason(trial, reason):
    trial.set_user_attr('reject_reason', reason)


# ═══════════════════════════════════════════════════════════════════════════
# DEFLATED SHARPE RATIO (unchanged from v10)
# ═══════════════════════════════════════════════════════════════════════════

def compute_deflated_sharpe(observed_sharpe, n_trades, n_trials=200,
                            skewness=0.0, kurtosis=3.0):
    from scipy import stats
    if n_trades < 10 or n_trials < 2:
        return {'dsr': 0.0, 'p_value': 1.0, 'expected_max_sr': 0.0,
                'significant_10pct': False, 'valid': True}

    euler_mascheroni = 0.5772156649
    max_z = ((1 - euler_mascheroni) * stats.norm.ppf(1 - 1.0 / n_trials) +
             euler_mascheroni * stats.norm.ppf(1 - 1.0 / (n_trials * np.e)))
    expected_max_sr = max_z * np.sqrt(1.0 / n_trades)

    sr_std = np.sqrt((1 + 0.5 * observed_sharpe**2 -
                      skewness * observed_sharpe +
                      ((kurtosis - 3) / 4.0) * observed_sharpe**2) / max(n_trades, 1))
    if sr_std <= 0: sr_std = 0.001

    dsr_z = (observed_sharpe - expected_max_sr) / sr_std
    p_value = 1 - stats.norm.cdf(dsr_z)

    return {
        'dsr': round(float(dsr_z), 4),
        'p_value': round(float(p_value), 4),
        'expected_max_sr': round(float(expected_max_sr), 4),
        'significant_10pct': p_value < 0.10,
        'valid': True,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

def resolve_target_symbol(all_data, coin_prefix, coin_name):
    for sym in all_data:
        parts = sym.upper().split('-')
        if parts[0] in (coin_prefix.upper(), coin_name.upper()):
            return sym
    for sym in all_data:
        if coin_name.upper() in sym.upper() or coin_prefix.upper() in sym.upper():
            return sym
    return None


def split_data_temporal(all_data, holdout_days=180):
    """Split into optimization window + final holdout (never seen during optimization)."""
    all_ends = [d['ohlcv'].index.max() for d in all_data.values() if len(d['ohlcv']) > 0]
    if not all_ends or holdout_days <= 0:
        return all_data, {}
    global_end = max(all_ends)
    holdout_start = global_end - pd.Timedelta(days=holdout_days)

    optim_data, holdout_data = {}, {}
    for sym, d in all_data.items():
        feat, ohlcv = d['features'], d['ohlcv']
        optim_feat = feat[feat.index < holdout_start]
        optim_ohlcv = ohlcv[ohlcv.index < holdout_start]
        if len(optim_feat) > 500:
            optim_data[sym] = {'features': optim_feat, 'ohlcv': optim_ohlcv}
        if len(feat) > 500:
            holdout_data[sym] = {'features': feat.copy(), 'ohlcv': ohlcv.copy()}
    return optim_data, holdout_data


def split_data_cv_folds(all_data, n_folds=3, min_fold_days=90):
    """
    v11: Create temporal CV folds from the optimization data.
    
    Each fold uses all data BEFORE the fold as training, and the fold itself as OOS.
    This is anchored walk-forward: train always starts from the beginning.
    
    Returns list of (train_data, test_data) tuples.
    """
    all_ends = [d['ohlcv'].index.max() for d in all_data.values() if len(d['ohlcv']) > 0]
    all_starts = [d['ohlcv'].index.min() for d in all_data.values() if len(d['ohlcv']) > 0]
    if not all_ends or not all_starts:
        return []

    global_start = min(all_starts)
    global_end = max(all_ends)
    total_days = (global_end - global_start).days

    # We need at least min_train_days + n_folds * min_fold_days
    min_train_days = 120  # minimum initial training window
    required_days = min_train_days + n_folds * min_fold_days
    
    if total_days < required_days:
        # Fall back to 2 folds with shorter windows
        n_folds = max(2, total_days // (min_train_days + 30))
        if n_folds < 2:
            # Not enough data even for 2 folds — return single split
            mid = global_start + pd.Timedelta(days=total_days * 0.7)
            train_data, test_data = {}, {}
            for sym, d in all_data.items():
                feat, ohlcv = d['features'], d['ohlcv']
                train_data[sym] = {
                    'features': feat[feat.index < mid],
                    'ohlcv': ohlcv[ohlcv.index < mid],
                }
                test_data[sym] = {
                    'features': feat[feat.index >= mid],
                    'ohlcv': ohlcv[ohlcv.index >= mid],
                }
            return [(train_data, test_data)]

    # Calculate fold boundaries
    # Reserve first min_train_days for initial training window
    test_zone_start = global_start + pd.Timedelta(days=min_train_days)
    test_zone_days = (global_end - test_zone_start).days
    fold_days = test_zone_days // n_folds

    folds = []
    for i in range(n_folds):
        fold_start = test_zone_start + pd.Timedelta(days=i * fold_days)
        fold_end = fold_start + pd.Timedelta(days=fold_days)
        if i == n_folds - 1:
            fold_end = global_end  # last fold goes to end

        train_data, test_data = {}, {}
        for sym, d in all_data.items():
            feat, ohlcv = d['features'], d['ohlcv']
            # Train: everything before this fold
            tf = feat[feat.index < fold_start]
            to = ohlcv[ohlcv.index < fold_start]
            if len(tf) > 200:
                train_data[sym] = {'features': tf, 'ohlcv': to}

            # Test: this fold only
            sf = feat[(feat.index >= fold_start) & (feat.index < fold_end)]
            so = ohlcv[(ohlcv.index >= fold_start) & (ohlcv.index < fold_end)]
            if len(sf) > 50:
                test_data[sym] = {'features': sf, 'ohlcv': so}

        if train_data and test_data:
            folds.append((train_data, test_data))

    return folds


# ═══════════════════════════════════════════════════════════════════════════
# v11: REDUCED TRIAL PROFILE (18 → 10 tunable parameters)
# ═══════════════════════════════════════════════════════════════════════════

# These are FIXED and not tuned by Optuna — reduces search space dramatically
FIXED_ML_PARAMS = {
    'n_estimators': 100,
    'max_depth': 3,
    'learning_rate': 0.05,
    'min_child_samples': 20,
}

FIXED_RISK_PARAMS = {
    'position_size': 0.12,
    'vol_sizing_target': 0.025,
    'cooldown_hours': 24.0,
    'min_val_auc': 0.53,
}


def create_trial_profile(trial, coin_name):
    """
    v11: Only 10 tunable parameters (down from 18).
    
    TUNABLE (strategy-critical, coin-dependent):
      signal_threshold, label_forward_hours, label_vol_target,
      min_momentum_magnitude, vol_mult_tp, vol_mult_sl,
      max_hold_hours, min_vol_24h, max_vol_24h
    
    FIXED (ML internals + risk policy — same for all coins):
      n_estimators, max_depth, learning_rate, min_child_samples,
      position_size, vol_sizing_target, cooldown_hours, min_val_auc
    """
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]

    return CoinProfile(
        name=coin_name, prefixes=prefixes,
        extra_features=get_extra_features(coin_name),
        # ── TUNABLE (10 params, tighter ranges, coarser steps) ──
        signal_threshold=trial.suggest_float('signal_threshold', 0.65, 0.85, step=0.02),
        label_forward_hours=trial.suggest_int('label_forward_hours', 12, 48, step=12),
        label_vol_target=trial.suggest_float('label_vol_target', 1.2, 2.4, step=0.3),
        min_momentum_magnitude=trial.suggest_float('min_momentum_magnitude', 0.02, 0.10, step=0.02),
        vol_mult_tp=trial.suggest_float('vol_mult_tp', 3.0, 7.0, step=1.0),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', 2.0, 4.0, step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', 36, 96, step=12),
        min_vol_24h=trial.suggest_float('min_vol_24h', 0.005, 0.015, step=0.005),
        max_vol_24h=trial.suggest_float('max_vol_24h', 0.04, 0.08, step=0.02),
        # ── FIXED ──
        cooldown_hours=FIXED_RISK_PARAMS['cooldown_hours'],
        position_size=FIXED_RISK_PARAMS['position_size'],
        vol_sizing_target=FIXED_RISK_PARAMS['vol_sizing_target'],
        min_val_auc=FIXED_RISK_PARAMS['min_val_auc'],
        n_estimators=FIXED_ML_PARAMS['n_estimators'],
        max_depth=FIXED_ML_PARAMS['max_depth'],
        learning_rate=FIXED_ML_PARAMS['learning_rate'],
        min_child_samples=FIXED_ML_PARAMS['min_child_samples'],
    )


def profile_from_params(params, coin_name):
    """Reconstruct CoinProfile from saved params dict (for validation/holdout)."""
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]
    return CoinProfile(
        name=coin_name, prefixes=prefixes,
        extra_features=get_extra_features(coin_name),
        signal_threshold=params.get('signal_threshold', 0.75),
        min_val_auc=params.get('min_val_auc', FIXED_RISK_PARAMS['min_val_auc']),
        label_forward_hours=params.get('label_forward_hours', 24),
        label_vol_target=params.get('label_vol_target', 1.8),
        min_momentum_magnitude=params.get('min_momentum_magnitude', 0.06),
        vol_mult_tp=params.get('vol_mult_tp', 5.0),
        vol_mult_sl=params.get('vol_mult_sl', 3.0),
        max_hold_hours=params.get('max_hold_hours', 72),
        cooldown_hours=params.get('cooldown_hours', FIXED_RISK_PARAMS['cooldown_hours']),
        min_vol_24h=params.get('min_vol_24h', 0.008),
        max_vol_24h=params.get('max_vol_24h', 0.06),
        position_size=params.get('position_size', FIXED_RISK_PARAMS['position_size']),
        vol_sizing_target=params.get('vol_sizing_target', FIXED_RISK_PARAMS['vol_sizing_target']),
        n_estimators=params.get('n_estimators', FIXED_ML_PARAMS['n_estimators']),
        max_depth=params.get('max_depth', FIXED_ML_PARAMS['max_depth']),
        learning_rate=params.get('learning_rate', FIXED_ML_PARAMS['learning_rate']),
        min_child_samples=params.get('min_child_samples', FIXED_ML_PARAMS['min_child_samples']),
    )


# ═══════════════════════════════════════════════════════════════════════════
# v11: WALK-FORWARD CV OBJECTIVE
# ═══════════════════════════════════════════════════════════════════════════

def _run_single_fold(fold_train, fold_test, profile, coin_name, coin_prefix):
    """Run backtest on a single CV fold, return OOS metrics."""
    target_sym = resolve_target_symbol(fold_test, coin_prefix, coin_name)
    if not target_sym:
        return None

    # Merge train+test data for run_backtest (it does its own walk-forward internally)
    # But we only care about performance in the TEST period
    merged = {}
    for sym in set(list(fold_train.keys()) + list(fold_test.keys())):
        train_d = fold_train.get(sym, {})
        test_d = fold_test.get(sym, {})
        
        feat_parts = [d['features'] for d in [train_d, test_d] if 'features' in d and len(d['features']) > 0]
        ohlcv_parts = [d['ohlcv'] for d in [train_d, test_d] if 'ohlcv' in d and len(d['ohlcv']) > 0]
        
        if feat_parts and ohlcv_parts:
            merged[sym] = {
                'features': pd.concat(feat_parts).sort_index(),
                'ohlcv': pd.concat(ohlcv_parts).sort_index(),
            }
            # Deduplicate
            merged[sym]['features'] = merged[sym]['features'][~merged[sym]['features'].index.duplicated(keep='last')]
            merged[sym]['ohlcv'] = merged[sym]['ohlcv'][~merged[sym]['ohlcv'].index.duplicated(keep='last')]

    if not merged:
        return None

    # Get test period boundaries
    test_start = min(d['features'].index.min() for d in fold_test.values() if len(d['features']) > 0)
    test_end = max(d['features'].index.max() for d in fold_test.values() if len(d['features']) > 0)
    test_days = (test_end - test_start).days

    config = Config(
        max_positions=1, leverage=4, min_signal_edge=0.00,
        max_ensemble_std=0.10, train_embargo_hours=24,
        oos_eval_days=max(30, test_days),
    )

    try:
        result = run_backtest(merged, config, profile_overrides={coin_name: profile})
    except Exception:
        return None

    if result is None:
        return None

    return {
        'sharpe': _finite_metric(result.get('sharpe_annual', 0)),
        'oos_sharpe': _finite_metric(result.get('oos_sharpe', 0)),
        'oos_return': _finite_metric(result.get('oos_return', 0)),
        'oos_trades': int(result.get('oos_trades', 0) or 0),
        'n_trades': int(result.get('n_trades', 0) or 0),
        'profit_factor': _finite_metric(result.get('profit_factor', 0)),
        'win_rate': _finite_metric(result.get('win_rate', 0)),
        'max_drawdown': _finite_metric(result.get('max_drawdown', 1.0), 1.0),
        'ann_return': _finite_metric(result.get('ann_return', -1.0), -1.0),
        'trades_per_year': _finite_metric(result.get('trades_per_year', 0)),
    }


def objective(trial, all_data, coin_prefix, coin_name, cv_folds=None):
    """
    v11: Walk-forward CV objective.
    
    Score = weighted average of OOS performance across temporal folds.
    This is fundamentally different from v10 which scored on in-sample Sharpe.
    """
    profile = create_trial_profile(trial, coin_name)
    
    if not cv_folds:
        # Fallback to single-fold if CV folds not provided
        return _objective_single_fold(trial, all_data, coin_prefix, coin_name, profile)

    # ── Run across all CV folds ──
    fold_results = []
    total_trades = 0
    total_oos_trades = 0

    for fold_idx, (fold_train, fold_test) in enumerate(cv_folds):
        result = _run_single_fold(fold_train, fold_test, profile, coin_name, coin_prefix)
        if result is not None:
            fold_results.append(result)
            total_trades += result['n_trades']
            total_oos_trades += result['oos_trades']

    # ── Reject if too few folds produced results ──
    if len(fold_results) < max(1, len(cv_folds) // 2):
        _set_reject_reason(trial, f'too_few_folds:{len(fold_results)}/{len(cv_folds)}')
        return -99.0

    # ── Reject if total trades across all folds too low ──
    if total_trades < 30:
        _set_reject_reason(trial, f'too_few_total_trades:{total_trades}')
        return -99.0

    # ── Compute cross-fold OOS metrics ──
    oos_sharpes = [r['oos_sharpe'] if r['oos_sharpe'] > -90 else 0.0 for r in fold_results]
    oos_returns = [r['oos_return'] for r in fold_results]
    oos_trade_counts = [r['oos_trades'] for r in fold_results]
    is_sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in fold_results]
    win_rates = [r['win_rate'] for r in fold_results]
    max_dds = [r['max_drawdown'] for r in fold_results]
    pfs = [r['profit_factor'] for r in fold_results]

    mean_oos_sharpe = np.mean(oos_sharpes)
    min_oos_sharpe = np.min(oos_sharpes)
    mean_oos_return = np.mean(oos_returns)
    mean_is_sharpe = np.mean(is_sharpes)
    mean_wr = np.mean(win_rates)
    mean_dd = np.mean(max_dds)
    mean_pf = np.mean(pfs)
    std_oos_sharpe = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0.0

    # ── v11 SCORING: OOS-first ──
    # Primary component: mean OOS Sharpe (this IS the score)
    score = mean_oos_sharpe

    # Bonus for consistency across folds (low variance)
    if std_oos_sharpe < 0.3 and len(oos_sharpes) >= 2:
        score += 0.15  # consistency bonus
    elif std_oos_sharpe > 0.8:
        score -= 0.25  # inconsistency penalty

    # Bonus for minimum OOS Sharpe being positive
    if min_oos_sharpe > 0:
        score += 0.10
    elif min_oos_sharpe < -0.5:
        score -= 0.30  # at least one fold was bad

    # Profit factor bonus (mild)
    mean_pf_capped = min(max(0, mean_pf), 5.0)
    if mean_pf_capped > 1.2:
        score += min(0.2, (mean_pf_capped - 1.0) * 0.15)

    # Drawdown penalty
    if mean_dd > 0.25:
        score -= (mean_dd - 0.25) * 2.0

    # Trade activity penalty
    avg_oos_trades = np.mean(oos_trade_counts)
    if avg_oos_trades < 5:
        score -= 0.5
    elif avg_oos_trades < 10:
        score -= 0.2

    # Overfit detection: IS >> OOS
    if mean_is_sharpe > 0 and mean_oos_sharpe > -90:
        overfit_ratio = mean_is_sharpe / max(mean_oos_sharpe, 0.01) if mean_oos_sharpe > 0 else 10.0
        if overfit_ratio > 3.0:
            score -= min(1.0, (overfit_ratio - 3.0) * 0.3)

    # Suspicious patterns
    if mean_wr > 0.75 and total_trades < 40:
        score -= 0.3  # suspicious high WR with few trades

    # Store metrics for trial selection
    trial.set_user_attr('n_trades', total_trades)
    trial.set_user_attr('oos_trades', total_oos_trades)
    trial.set_user_attr('n_folds', len(fold_results))
    trial.set_user_attr('mean_oos_sharpe', round(mean_oos_sharpe, 3))
    trial.set_user_attr('min_oos_sharpe', round(min_oos_sharpe, 3))
    trial.set_user_attr('std_oos_sharpe', round(std_oos_sharpe, 3))
    trial.set_user_attr('mean_oos_return', round(mean_oos_return, 4))
    trial.set_user_attr('mean_is_sharpe', round(mean_is_sharpe, 3))
    trial.set_user_attr('sharpe', round(mean_is_sharpe, 3))  # compat with v10
    trial.set_user_attr('oos_sharpe', round(mean_oos_sharpe, 3))  # compat
    trial.set_user_attr('oos_return', round(mean_oos_return, 4))  # compat
    trial.set_user_attr('win_rate', round(mean_wr, 3))
    trial.set_user_attr('profit_factor', round(mean_pf_capped, 3))
    trial.set_user_attr('max_drawdown', round(mean_dd, 4))
    trial.set_user_attr('ann_return', round(np.mean([r['ann_return'] for r in fold_results]), 4))
    tpy = np.mean([r['trades_per_year'] for r in fold_results])
    trial.set_user_attr('trades_per_month', round(tpy / 12.0, 1))
    calmar = min(np.mean([r['ann_return'] for r in fold_results]) / mean_dd if mean_dd > 0.01 else 0, 10.0)
    trial.set_user_attr('calmar', round(calmar, 3))

    if DEBUG_TRIALS:
        print(f"\n  Trial {trial.number}: score={score:.3f} | "
              f"OOS_SR={mean_oos_sharpe:.3f}±{std_oos_sharpe:.3f} | "
              f"IS_SR={mean_is_sharpe:.3f} | trades={total_trades} | "
              f"folds={len(fold_results)}")

    return score


def _objective_single_fold(trial, all_data, coin_prefix, coin_name, profile):
    """Fallback: single-fold objective (similar to v10 but with stronger penalties)."""
    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        _set_reject_reason(trial, f'missing_symbol:{coin_prefix}/{coin_name}')
        return -99.0

    single_data = {target_sym: all_data[target_sym]}
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24,
                    oos_eval_days=60)
    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        _set_reject_reason(trial, f'backtest_error:{type(e).__name__}')
        return -99.0

    if result is None:
        _set_reject_reason(trial, 'result_none')
        return -99.0

    n_trades = int(result.get('n_trades', 0) or 0)
    oos_sharpe = _finite_metric(result.get('oos_sharpe', 0))
    oos_return = _finite_metric(result.get('oos_return', 0))
    oos_trades = int(result.get('oos_trades', 0) or 0)
    sharpe = _finite_metric(result.get('sharpe_annual', 0))
    dd = _finite_metric(result.get('max_drawdown', 1.0), 1.0)
    wr = _finite_metric(result.get('win_rate', 0))
    pf = _finite_metric(result.get('profit_factor', 0))

    if sharpe <= -90: sharpe = 0.0
    if oos_sharpe <= -90: oos_sharpe = 0.0

    if n_trades < 30:
        _set_reject_reason(trial, f'too_few_trades:{n_trades}')
        return -99.0

    # v11: Score = OOS Sharpe primarily
    score = oos_sharpe * 0.7 + min(sharpe, 2.0) * 0.3  # OOS-weighted

    if dd > 0.25: score -= (dd - 0.25) * 2.0
    if oos_trades < 5: score -= 0.5
    if oos_return < -0.05: score -= min(1.0, abs(oos_return) * 3)
    if wr > 0.75 and n_trades < 40: score -= 0.3

    trial.set_user_attr('n_trades', n_trades)
    trial.set_user_attr('oos_trades', oos_trades)
    trial.set_user_attr('sharpe', round(sharpe, 3))
    trial.set_user_attr('oos_sharpe', round(oos_sharpe, 3))
    trial.set_user_attr('oos_return', round(oos_return, 4))
    trial.set_user_attr('win_rate', round(wr, 3))
    trial.set_user_attr('profit_factor', round(min(pf, 5.0), 3))
    trial.set_user_attr('max_drawdown', round(dd, 4))
    trial.set_user_attr('ann_return', round(_finite_metric(result.get('ann_return', 0)), 4))
    tpy = _finite_metric(result.get('trades_per_year', 0))
    trial.set_user_attr('trades_per_month', round(tpy / 12.0, 1))
    calmar = min(_finite_metric(result.get('ann_return', 0)) / dd if dd > 0.01 else 0, 10.0)
    trial.set_user_attr('calmar', round(calmar, 3))

    return score


# ═══════════════════════════════════════════════════════════════════════════
# PLATEAU STOPPER (tuned for v11)
# ═══════════════════════════════════════════════════════════════════════════

class PlateauStopper:
    def __init__(self, patience=60, min_delta=0.02, warmup_trials=30):
        self.patience = max(1, patience)
        self.min_delta = max(0.0, min_delta)
        self.warmup_trials = max(0, warmup_trials)
        self.best_value = None
        self.best_trial_number = None

    def __call__(self, study, trial):
        completed = [t for t in study.trials
                     if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(completed) < self.warmup_trials: return
        if self.best_value is None:
            self.best_value = study.best_value
            self.best_trial_number = study.best_trial.number
            return
        current_best = study.best_value
        if current_best > (self.best_value + self.min_delta):
            self.best_value = current_best
            self.best_trial_number = study.best_trial.number
            return
        best_num = self.best_trial_number if self.best_trial_number is not None else trial.number
        since_best = sum(1 for t in completed if t.number > best_num)
        if since_best >= self.patience:
            print(f"\n🛑 Plateau stop: no improvement > {self.min_delta:.4f} "
                  f"for {self.patience} trials (best={self.best_value:.4f} @ trial {self.best_trial_number}).")
            study.stop()


# ═══════════════════════════════════════════════════════════════════════════
# TRIAL SELECTION (v11 — OOS-consistency ranked)
# ═══════════════════════════════════════════════════════════════════════════

def _select_best_trial(study, min_trades=30, min_oos_trades=8):
    """Select best trial prioritizing OOS consistency over raw score."""
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        return study.best_trial

    robust = []
    acceptable = []

    for t in completed:
        nt = int(t.user_attrs.get('n_trades', 0) or 0)
        ot = int(t.user_attrs.get('oos_trades', 0) or 0)

        if nt < min_trades or ot < min_oos_trades:
            continue

        acceptable.append(t)

        oos_sr = _as_number(t.user_attrs.get('mean_oos_sharpe',
                            t.user_attrs.get('oos_sharpe')), 0.0) or 0.0
        min_oos = _as_number(t.user_attrs.get('min_oos_sharpe'), None)
        std_oos = _as_number(t.user_attrs.get('std_oos_sharpe'), 1.0)
        is_sr = _as_number(t.user_attrs.get('sharpe'), 0.0) or 0.0
        wr = _as_number(t.user_attrs.get('win_rate'), 0.0) or 0.0
        dd = _as_number(t.user_attrs.get('max_drawdown'), 1.0) or 1.0

        if oos_sr <= -90: oos_sr = 0.0
        if is_sr <= -90: is_sr = 0.0

        # Reject clearly bad
        if oos_sr < -0.3: continue
        if dd > 0.35: continue
        if wr > 0.80 and nt < 40: continue

        # Check consistency (if CV was used)
        if min_oos is not None and min_oos < -0.5: continue
        if std_oos is not None and std_oos > 1.0: continue

        # Overfit check
        if is_sr > 0 and oos_sr > 0:
            decay = 1.0 - (oos_sr / is_sr) if is_sr > 0 else 0
            if decay > 0.75: continue

        robust.append(t)

    if robust:
        # Rank by OOS Sharpe primarily
        def _rank(t):
            oos = _as_number(t.user_attrs.get('mean_oos_sharpe',
                             t.user_attrs.get('oos_sharpe')), 0.0) or 0.0
            if oos <= -90: oos = 0.0
            # Small bonus for consistency
            std = _as_number(t.user_attrs.get('std_oos_sharpe'), 0.5) or 0.5
            consistency_bonus = max(0, 0.1 - std * 0.1)
            return oos + consistency_bonus
        return max(robust, key=_rank)

    if acceptable:
        return max(acceptable, key=lambda t: t.value)
    return study.best_trial


# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT EVALUATION (unchanged logic from v10)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_holdout(holdout_data, params, coin_name, coin_prefix, holdout_days):
    """Run backtest on the true holdout data."""
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym:
        return None

    profile = profile_from_params(params, coin_name)
    single_data = {target_sym: holdout_data[target_sym]}
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24,
                    oos_eval_days=holdout_days)

    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        print(f"  ❌ Holdout error: {e}")
        return None

    if result is None:
        return None

    oos_sharpe = _finite_metric(result.get('oos_sharpe', 0))
    oos_return = _finite_metric(result.get('oos_return', 0))
    oos_trades = int(result.get('oos_trades', 0) or 0)
    full_sharpe = _finite_metric(result.get('sharpe_annual', 0))
    full_pf = _finite_metric(result.get('profit_factor', 0))
    full_dd = _finite_metric(result.get('max_drawdown', 1.0), 1.0)

    return {
        'holdout_sharpe': oos_sharpe if oos_sharpe > -90 else 0.0,
        'holdout_return': oos_return,
        'holdout_trades': oos_trades,
        'full_sharpe': full_sharpe if full_sharpe > -90 else 0.0,
        'full_pf': full_pf,
        'full_dd': full_dd,
    }


# ═══════════════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT (enhanced for v11)
# ═══════════════════════════════════════════════════════════════════════════

def assess_result_quality(result_data):
    issues = []
    warnings_list = []
    m = result_data.get('optim_metrics', {})
    h = result_data.get('holdout_metrics', {})
    dsr = result_data.get('deflated_sharpe', {})

    nt = int(m.get('n_trades', 0) or 0)
    is_sr = _finite_metric(m.get('sharpe', m.get('mean_is_sharpe', 0)))
    oos_sr = _finite_metric(m.get('mean_oos_sharpe', m.get('oos_sharpe', 0)))
    wr = _finite_metric(m.get('win_rate', 0))
    dd = _finite_metric(m.get('max_drawdown', 1.0), 1.0)
    ho_sr = _finite_metric(h.get('holdout_sharpe', 0))
    ho_tr = int(h.get('holdout_trades', 0) or 0)

    # Issues (fatal)
    if nt < 30: issues.append(f'low_trades:{nt}')
    if oos_sr < 0: issues.append(f'negative_oos_sharpe:{oos_sr:.3f}')
    if dd > 0.30: issues.append(f'high_drawdown:{dd:.1%}')
    if ho_tr > 0 and ho_sr < -0.2: issues.append(f'bad_holdout:{ho_sr:.3f}')

    # Warnings (concerning but not fatal)
    if is_sr > 0 and oos_sr > 0:
        decay = 1.0 - (oos_sr / is_sr) if is_sr > 0 else 0
        if decay > 0.50: warnings_list.append(f'sharpe_decay:{decay:.0%}')
    if wr > 0.75 and nt < 50: warnings_list.append('suspicious_high_wr')
    if dsr.get('p_value', 1) > 0.10: warnings_list.append('dsr_not_significant')

    # v11: CV consistency warnings
    std_oos = _finite_metric(m.get('std_oos_sharpe', 0))
    if std_oos > 0.5: warnings_list.append(f'unstable_cv:{std_oos:.3f}')
    min_oos = _as_number(m.get('min_oos_sharpe'))
    if min_oos is not None and min_oos < -0.3: warnings_list.append(f'worst_fold:{min_oos:.3f}')

    if not issues and not warnings_list:
        rating = 'EXCELLENT'
    elif not issues and len(warnings_list) <= 1:
        rating = 'GOOD'
    elif not issues:
        rating = 'ACCEPTABLE'
    elif len(issues) <= 1:
        rating = 'MARGINAL'
    else:
        rating = 'POOR'

    return {'rating': rating, 'issues': issues, 'warnings': warnings_list}


# ═══════════════════════════════════════════════════════════════════════════
# PATHS AND PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def _db_path():
    return SCRIPT_DIR / "optuna_trading.db"

def _sqlite_url(path):
    return f"sqlite:///{path.resolve()}"

def _candidate_results_dirs():
    return [SCRIPT_DIR / "optimization_results", Path.cwd() / "optimization_results"]

def _persist_result_json(coin_name, result_data):
    last_error = None
    for d in _candidate_results_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{coin_name}_optimization.json"
            with open(p, 'w') as f:
                json.dump(_to_json_safe(result_data), f, indent=2)
            return p
        except (PermissionError, OSError) as e:
            last_error = e
    print(f"\n  ❌ Failed to save result for {coin_name}: {last_error}")
    return None


# ═══════════════════════════════════════════════════════════════════════════
# OPTIMIZE COIN (v11 main logic)
# ═══════════════════════════════════════════════════════════════════════════

def optimize_coin(all_data, coin_prefix, coin_name, n_trials=100, n_jobs=1,
                  plateau_patience=60, plateau_min_delta=0.02, plateau_warmup=30,
                  study_suffix="", resume_study=False, holdout_days=180,
                  min_internal_oos_trades=0, min_total_trades=0,
                  n_cv_folds=3):

    # STEP 0: Split data — holdout first, then CV folds from optimization data
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"❌ {coin_name}: not enough data after holdout split")
        return None

    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start, optim_end = ohlcv.index.min(), ohlcv.index.max()
    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    holdout_end = holdout_data[holdout_sym]['ohlcv'].index.max() if holdout_sym else optim_end

    # Create CV folds from optimization data
    cv_folds = split_data_cv_folds(optim_data, n_folds=n_cv_folds)

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} ({coin_prefix}) — v11 WALK-FORWARD CV")
    print(f"   Optimization window: {optim_start.date()} → {optim_end.date()}")
    print(f"   Holdout window:      last {holdout_days} days (→ {holdout_end.date()})")
    print(f"   CV folds:            {len(cv_folds)} (walk-forward)")
    print(f"   Tunable params:      9 (reduced from 18)")
    print(f"   Trials: {n_trials} | Cores: {n_jobs}")
    print(f"   Plateau: patience={plateau_patience}, delta={plateau_min_delta}, warmup={plateau_warmup}")
    print(f"   ⚠️  Score = mean OOS Sharpe across CV folds (NOT in-sample)")
    print(f"{'='*60}")

    # STEP 1: Optuna study
    storage_url = _sqlite_url(_db_path())
    study_name = f"optimize_{coin_name}{'_' + study_suffix if study_suffix else ''}"
    sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))

    study = None
    for attempt in range(3):
        try:
            study = optuna.create_study(direction='maximize', sampler=sampler,
                                        study_name=study_name, storage=storage_url,
                                        load_if_exists=resume_study)
            break
        except Exception as e:
            msg = str(e)
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in msg:
                study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)
                break
            if "database is locked" in msg and attempt < 2:
                time.sleep(0.25 * (attempt + 1))
                continue
            raise

    if study is None:
        print(f"❌ Could not create/load study '{study_name}'.")
        return None

    # STEP 2: Optimize with CV objective
    obj = functools.partial(objective, all_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, cv_folds=cv_folds)
    stopper = PlateauStopper(patience=plateau_patience, min_delta=plateau_min_delta,
                             warmup_trials=plateau_warmup)
    try:
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs,
                       show_progress_bar=sys.stderr.isatty(), callbacks=[stopper])
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return None

    if not study.trials:
        print("No trials completed.")
        return None

    # STEP 3: Select best
    mtt = min_total_trades or 30
    miot = min_internal_oos_trades or 8
    best = _select_best_trial(study, min_trades=mtt, min_oos_trades=miot)
    if best.number != study.best_trial.number:
        print(f"\n🛡️ Selected trial #{best.number} over raw best #{study.best_trial.number} for robustness.")

    if _as_number(best.value) == -99.0:
        rc = {}
        for t in study.trials:
            r = t.user_attrs.get('reject_reason', 'unknown')
            rc[r] = rc.get(r, 0) + 1
        for r, c in sorted(rc.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {r}: {c}")

    print(f"\n✅ BEST for {coin_name} (v11 CV):")
    print(f"  Score={_fmt_float(best.value,3)} | Mean OOS SR={_fmt_float(best.user_attrs.get('mean_oos_sharpe', best.user_attrs.get('oos_sharpe')),3)} "
          f"| Min OOS SR={_fmt_float(best.user_attrs.get('min_oos_sharpe'),3)} "
          f"| Std OOS SR={_fmt_float(best.user_attrs.get('std_oos_sharpe'),3)}")
    print(f"  Trades={best.user_attrs.get('n_trades','?')} | "
          f"WR={_fmt_pct(best.user_attrs.get('win_rate'),1)} | "
          f"PF={_fmt_float(best.user_attrs.get('profit_factor'),3)} | "
          f"DD={_fmt_pct(best.user_attrs.get('max_drawdown'),2)} | "
          f"Calmar={_fmt_float(best.user_attrs.get('calmar'),3)}")

    # v11: DSR
    optim_sharpe = _finite_metric(best.user_attrs.get('mean_oos_sharpe',
                                  best.user_attrs.get('sharpe', 0)))
    nc = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None])
    nt_best = int(best.user_attrs.get('n_trades', 0) or 0)
    dsr_result = compute_deflated_sharpe(optim_sharpe, nt_best, nc)
    print(f"\n  📐 DSR: {dsr_result['dsr']:.3f}  p={dsr_result['p_value']:.3f}  "
          f"ExpMaxSR={dsr_result['expected_max_sr']:.3f}  Sig@10%={dsr_result['significant_10pct']}")
    if dsr_result['p_value'] > 0.10:
        print(f"     ⚠️  Sharpe NOT significant after correcting for {nc} trials.")

    # STEP 4: Holdout
    holdout_result = None
    if holdout_data and holdout_sym:
        print(f"\n🔬 HOLDOUT EVALUATION (last {holdout_days} days)...")
        holdout_result = evaluate_holdout(holdout_data, best.params, coin_name, coin_prefix, holdout_days)
        if holdout_result:
            h = holdout_result
            ho_s = h['holdout_sharpe'] if h['holdout_sharpe'] > -90 else 0.0
            print(f"  Holdout: Sharpe={_fmt_float(ho_s,3)} Return={_fmt_pct(h['holdout_return'],2)} "
                  f"Trades={h['holdout_trades']} | Full: Sharpe={_fmt_float(h['full_sharpe'],3)} "
                  f"PF={_fmt_float(h['full_pf'],3)} DD={_fmt_pct(h['full_dd'],2)}")
            if optim_sharpe and ho_s < optim_sharpe * 0.3:
                print(f"  ⚠️  Holdout Sharpe ({ho_s:.3f}) << optim ({optim_sharpe:.3f}). Possible overfit!")
            elif ho_s > 0.3:
                print(f"  ✅ Holdout looks healthy.")
        else:
            print(f"  ⚠️ Holdout eval failed.")
    else:
        print(f"\n  ⚠️ Skipping holdout — insufficient data.")

    # STEP 5: Save
    result_data = {
        'coin': coin_name, 'prefix': coin_prefix,
        'optim_score': best.value,
        'optim_metrics': dict(best.user_attrs),
        'holdout_metrics': holdout_result or {},
        'params': best.params,
        'n_trials': len(study.trials),
        'n_cv_folds': len(cv_folds),
        'holdout_days': holdout_days,
        'deflated_sharpe': dsr_result,
        'version': 'v11',
        'timestamp': datetime.now().isoformat(),
    }
    result_data['quality'] = assess_result_quality(result_data)
    q = result_data['quality']
    print(f"\n  🧪 Quality: {q['rating']} | Issues: {', '.join(q['issues']) or 'none'} | "
          f"Warnings: {', '.join(q.get('warnings',[])) or 'none'}")
    p = _persist_result_json(coin_name, result_data)
    if p:
        print(f"  💾 Saved to {p}")

    # STEP 6: Profile snippet
    print(f"\n  📝 Suggested CoinProfile:")
    print(f"    '{coin_name}': CoinProfile(")
    print(f"        name='{coin_name}', prefixes={COIN_PROFILES[coin_name].prefixes},")
    extras = get_extra_features(coin_name)
    print(f"        extra_features={coin_name}_EXTRA_FEATURES," if extras else "        extra_features=[],")
    for k, v in sorted(best.params.items()):
        pretty = f"{v:.4f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
        print(f"        {k}={pretty},")
    # Also print fixed params for completeness
    for k, v in sorted({**FIXED_ML_PARAMS, **FIXED_RISK_PARAMS}.items()):
        pretty = f"{v:.4f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
        print(f"        {k}={pretty},  # fixed")
    print(f"    ),")

    return result_data


# ═══════════════════════════════════════════════════════════════════════════
# SHOW RESULTS
# ═══════════════════════════════════════════════════════════════════════════

def show_results():
    all_results = []
    seen = set()
    for d in _candidate_results_dirs():
        for p in d.glob("*_optimization.json"):
            rp = str(p.resolve())
            if rp not in seen:
                seen.add(rp)
                all_results.append(p)
    if not all_results:
        print("No optimization results found.")
        return
    print(f"\n{'='*80}\n📊 OPTIMIZATION RESULTS (v11)\n{'='*80}")
    for rpath in sorted(all_results):
        with open(rpath) as f:
            r = json.load(f)
        m = r.get('optim_metrics', {})
        h = r.get('holdout_metrics', {})
        dsr = r.get('deflated_sharpe', {})
        ver = r.get('version', 'v10')
        print(f"\n{r['coin']} ({r.get('prefix','?')}) — {r['n_trials']} trials — {ver} — {r['timestamp'][:16]}")
        print(f"  OOS SR: mean={_fmt_float(m.get('mean_oos_sharpe', m.get('oos_sharpe')),3)} "
              f"min={_fmt_float(m.get('min_oos_sharpe'),3)} std={_fmt_float(m.get('std_oos_sharpe'),3)}")
        print(f"  IS SR={_fmt_float(m.get('sharpe', m.get('mean_is_sharpe')),3)} "
              f"WR={_fmt_pct(m.get('win_rate'),1)} PF={_fmt_float(m.get('profit_factor'),3)} "
              f"DD={_fmt_pct(m.get('max_drawdown'),1)} Trades={m.get('n_trades','?')}")
        if dsr:
            print(f"  DSR: {_fmt_float(dsr.get('dsr'),3)} p={_fmt_float(dsr.get('p_value'),3)} Sig@10%={dsr.get('significant_10pct','?')}")
        if h:
            hs = _finite_metric(h.get('holdout_sharpe', 0))
            hr = _finite_metric(h.get('holdout_return', 0))
            ht = int(h.get('holdout_trades', 0) or 0)
            if _is_invalid_holdout_metric(hs, hr, ht):
                hs, hr, ht = 0.0, 0.0, 0
            print(f"  Holdout: Sharpe={_fmt_float(hs,3)} Return={_fmt_pct(hr,2)} Trades={ht}")
        q = r.get('quality') or assess_result_quality(r)
        print(f"  Quality: {q.get('rating','?')} | Issues={', '.join(q.get('issues',[])[:3]) or 'none'} | "
              f"Warnings={', '.join(q.get('warnings',[])[:3]) or 'none'}")


# ═══════════════════════════════════════════════════════════════════════════
# PRESETS (v11 — tighter defaults)
# ═══════════════════════════════════════════════════════════════════════════

def apply_runtime_preset(args):
    presets = {
        'robust180': {
            'plateau_patience': 60, 'plateau_warmup': 30,
            'plateau_min_delta': 0.02, 'holdout_days': 180,
            'min_internal_oos_trades': 8, 'min_total_trades': 30,
            'n_cv_folds': 3,
        },
        'robust120': {
            'plateau_patience': 50, 'plateau_warmup': 25,
            'plateau_min_delta': 0.02, 'holdout_days': 120,
            'min_internal_oos_trades': 6, 'min_total_trades': 25,
            'n_cv_folds': 3,
        },
        'quick': {
            'plateau_patience': 30, 'plateau_warmup': 15,
            'plateau_min_delta': 0.03, 'holdout_days': 90,
            'min_internal_oos_trades': 5, 'min_total_trades': 20,
            'n_cv_folds': 2,
        },
    }
    name = getattr(args, 'preset', 'none')
    if name in (None, '', 'none'):
        return args
    cfg = presets.get(name)
    if not cfg:
        return args
    for k, v in cfg.items():
        setattr(args, k, v)
    print(f"🧭 Applied preset '{name}': " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    return args


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

COIN_MAP = {'BIP': 'BTC', 'BTC': 'BTC', 'ETP': 'ETH', 'ETH': 'ETH',
            'XPP': 'XRP', 'XRP': 'XRP', 'SLP': 'SOL', 'SOL': 'SOL',
            'DOP': 'DOGE', 'DOGE': 'DOGE'}
PREFIX_FOR_COIN = {'BTC': 'BIP', 'ETH': 'ETP', 'XRP': 'XPP', 'SOL': 'SLP', 'DOGE': 'DOP'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-coin Optuna optimization (v11 — Walk-Forward CV)")
    parser.add_argument("--coin", type=str, help="Coin prefix or name")
    parser.add_argument("--all", action="store_true", help="Optimize all coins")
    parser.add_argument("--show", action="store_true", help="Show saved results")
    parser.add_argument("--trials", type=int, default=100, help="Trials (default: 100, was 200)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--plateau-patience", type=int, default=60)
    parser.add_argument("--plateau-min-delta", type=float, default=0.02)
    parser.add_argument("--plateau-warmup", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="robust180",
                        choices=["none", "robust120", "robust180", "quick"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0)
    parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=3, help="Number of walk-forward CV folds")
    parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--debug-trials", action="store_true")
    args = parser.parse_args()
    args = apply_runtime_preset(args)

    if args.debug_trials:
        DEBUG_TRIALS = True

    if args.show:
        show_results()
        sys.exit(0)

    init_db_wal(str(_db_path()))

    all_data = load_data()
    if not all_data:
        print("❌ No data. Run pipeline + compute_features first.")
        sys.exit(1)

    coins = []
    if args.all:
        coins = list(COIN_MAP.values())
        coins = list(dict.fromkeys(coins))  # dedupe
    elif args.coin:
        c = COIN_MAP.get(args.coin.upper(), args.coin.upper())
        coins = [c]
    else:
        parser.print_help()
        sys.exit(1)

    for coin_name in coins:
        prefix = PREFIX_FOR_COIN.get(coin_name, coin_name)
        optimize_coin(
            all_data, prefix, coin_name,
            n_trials=args.trials, n_jobs=args.jobs,
            plateau_patience=args.plateau_patience,
            plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup,
            study_suffix=args.study_suffix,
            resume_study=args.resume,
            holdout_days=args.holdout_days,
            min_internal_oos_trades=args.min_internal_oos_trades,
            min_total_trades=args.min_total_trades,
            n_cv_folds=args.n_cv_folds,
        )