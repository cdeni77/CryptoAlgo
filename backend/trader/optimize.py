#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (Parallel Enabled).

Runs the existing backtest filtered to a single coin at a time,
optimizing threshold, exits, label params, and ML hyperparams.

v9: TRUE HOLDOUT SCORING
    - Data is split into optimization window + holdout window BEFORE Optuna runs
    - Optuna only sees the optimization window when scoring trials
    - After optimization completes, best params are evaluated on the holdout
    - This prevents indirect overfitting to the OOS period through hyperparameter search

Usage:
    python optimize.py --coin BIP --trials 50 --jobs 4
    python optimize.py --all --trials 200 --jobs 16
    python optimize.py --show                        # Show saved results
"""
import argparse
import json
import warnings
import sys
import os
import logging
import sqlite3
import functools  # <--- Critical for multiprocessing
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

# Force single-threaded linear algebra BEFORE importing numpy/pandas/sklearn
# This prevents 16 workers x 20 threads = 320 threads crashing the CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Import your existing logic
from train_model import Config, load_data, run_backtest
from coin_profiles import (
    CoinProfile, COIN_PROFILES,
    BTC_EXTRA_FEATURES, SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)

warnings.filterwarnings('ignore')
# Turn off Optuna/LightGBM logging to keep console clean during parallel runs
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent

# Map coin prefix to symbol (filled at runtime)
PREFIX_TO_SYMBOL: Dict[str, str] = {}
DEBUG_TRIALS = False


def init_db_wal(db_name="optuna_trading.db"):
    """Enable Write-Ahead Logging and set a long timeout for concurrency."""
    try:
        conn = sqlite3.connect(db_name)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout = 30000;")  # Wait up to 30s if DB is locked
        conn.close()
    except Exception as e:
        print(f"⚠️ Warning: Could not set WAL mode: {e}")

def get_extra_features(coin_name: str):
    """Get the extra features list for a coin."""
    mapping = {
        'BTC': BTC_EXTRA_FEATURES,
        'SOL': SOL_EXTRA_FEATURES,
        'DOGE': DOGE_EXTRA_FEATURES,
    }
    return mapping.get(coin_name, [])


def _as_number(value, default: Optional[float] = None) -> Optional[float]:
    """Safely coerce values from Optuna attrs/JSON to float for formatting."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_metric(value, default: float = 0.0) -> float:
    """Return finite float metrics, replacing NaN/inf with a safe default."""
    n = _as_number(value, default=default)
    if n is None or not np.isfinite(n):
        return default
    return float(n)


def _fmt_pct(value, decimals: int = 1, fallback: str = "?") -> str:
    n = _as_number(value)
    return f"{n:.{decimals}%}" if n is not None else fallback


def _fmt_float(value, decimals: int = 3, fallback: str = "?") -> str:
    n = _as_number(value)
    return f"{n:.{decimals}f}" if n is not None else fallback


def _is_invalid_holdout_metric(holdout_sharpe: float, holdout_return: float, holdout_trades: int) -> bool:
    """Detect sentinel / unusable holdout outputs produced when no real OOS test happened."""
    if holdout_sharpe <= -90:
        return True
    if holdout_trades <= 0:
        return True
    if holdout_return <= -0.99:
        return True
    return False


def assess_result_quality(result_data: Dict) -> Dict:
    """Attach interpretable quality diagnostics so weak runs are easy to filter out."""
    optim = result_data.get('optim_metrics', {}) or {}
    holdout = result_data.get('holdout_metrics', {}) or {}

    n_trials = int(result_data.get('n_trials', 0) or 0)
    n_trades = int(optim.get('n_trades', 0) or 0)
    pf = _finite_metric(optim.get('profit_factor', 0.0), default=0.0)
    sharpe = _finite_metric(optim.get('sharpe', 0.0), default=0.0)

    holdout_sharpe = _finite_metric(holdout.get('holdout_sharpe', 0.0), default=0.0)
    holdout_return = _finite_metric(holdout.get('holdout_return', 0.0), default=0.0)
    holdout_trades = int(holdout.get('holdout_trades', 0) or 0)
    holdout_valid = not _is_invalid_holdout_metric(holdout_sharpe, holdout_return, holdout_trades)

    issues = []
    if n_trials < 50:
        issues.append(f'low_trials:{n_trials}')
    if n_trades < 30:
        issues.append(f'low_optim_trades:{n_trades}')
    if pf < 1.15:
        issues.append(f'weak_profit_factor:{pf:.3f}')
    if sharpe < 0.75:
        issues.append(f'weak_sharpe:{sharpe:.3f}')

    if not holdout_valid:
        issues.append('invalid_holdout')
    else:
        if holdout_trades < 12:
            issues.append(f'low_holdout_trades:{holdout_trades}')
        if holdout_sharpe < 0.10:
            issues.append(f'weak_holdout_sharpe:{holdout_sharpe:.3f}')
        if holdout_return < 0.0:
            issues.append(f'negative_holdout_return:{holdout_return:.4f}')

    if not issues:
        rating = 'promising'
    elif len(issues) <= 2:
        rating = 'watchlist'
    else:
        rating = 'reject'

    return {
        'rating': rating,
        'holdout_valid': holdout_valid,
        'issues': issues,
    }


def _set_reject_reason(trial: optuna.Trial, reason: str) -> None:
    trial.set_user_attr('reject_reason', reason)


def resolve_target_symbol(all_data: Dict, coin_prefix: str, coin_name: str) -> Optional[str]:
    """Resolve symbol robustly for legacy/new prefix styles (e.g. DOP vs DOGE)."""
    # 1) Direct prefix map from loaded dataset keys
    target = PREFIX_TO_SYMBOL.get(coin_prefix)
    if target:
        return target

    # 2) Common aliases used in this repo
    aliases = {
        'BIP': 'BTC', 'ETP': 'ETH', 'XPP': 'XRP', 'SLP': 'SOL', 'DOP': 'DOGE',
        'BTC': 'BTC', 'ETH': 'ETH', 'XRP': 'XRP', 'SOL': 'SOL', 'DOGE': 'DOGE',
    }
    candidates = [coin_prefix, coin_name, aliases.get(coin_prefix), aliases.get(coin_name)]
    for c in candidates:
        if not c:
            continue
        # Try direct prefix lookup
        direct = PREFIX_TO_SYMBOL.get(c)
        if direct:
            return direct
        # Try scanning loaded symbols by prefix/base substring
        c_up = str(c).upper()
        for sym in all_data:
            sym_up = sym.upper()
            sym_prefix = sym_up.split('-')[0] if '-' in sym_up else sym_up
            if sym_prefix == c_up or c_up in sym_prefix or c_up in sym_up:
                return sym
    return None


# DATA SPLITTING — TRUE HOLDOUT

def split_data_temporal(all_data: Dict, holdout_days: int = 120) -> Tuple[Dict, Dict]:
    """
    Split data into optimization window and holdout window.
    
    The holdout window is the LAST `holdout_days` of data.
    Optuna only ever sees the optimization window.
    After optimization, the best params are evaluated on holdout.
    
    Returns:
        (optim_data, holdout_data) — same structure as all_data
    """
    # Find the global end date across all symbols
    all_ends = []
    for sym, d in all_data.items():
        if len(d['ohlcv']) > 0:
            all_ends.append(d['ohlcv'].index.max())
    
    if not all_ends:
        return all_data, {}
    
    global_end = max(all_ends)
    holdout_start = global_end - pd.Timedelta(days=holdout_days)
    
    optim_data = {}
    holdout_data = {}
    
    for sym, d in all_data.items():
        feat = d['features']
        ohlcv = d['ohlcv']
        
        # Optimization window: everything before holdout_start
        optim_feat = feat[feat.index < holdout_start]
        optim_ohlcv = ohlcv[ohlcv.index < holdout_start]
        
        # Holdout window: everything (model still trains walk-forward from beginning,
        # but the holdout trades occur in the holdout period)
        # We give holdout the FULL data so the walk-forward can train on earlier data
        # and trade in the holdout period
        holdout_feat = feat.copy()
        holdout_ohlcv = ohlcv.copy()
        
        if len(optim_feat) > 500:  # Need enough data for walk-forward
            optim_data[sym] = {'features': optim_feat, 'ohlcv': optim_ohlcv}
        
        if len(holdout_feat) > 500:
            holdout_data[sym] = {'features': holdout_feat, 'ohlcv': holdout_ohlcv}
    
    return optim_data, holdout_data


def create_trial_profile(trial: optuna.Trial, coin_name: str) -> CoinProfile:
    """Create a CoinProfile from Optuna trial suggestions."""
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]

    return CoinProfile(
        name=coin_name,
        prefixes=prefixes,
        extra_features=get_extra_features(coin_name),

        # Signal threshold
        signal_threshold=trial.suggest_float('signal_threshold', 0.58, 0.86, step=0.01),

        # Model quality gate
        min_val_auc=trial.suggest_float('min_val_auc', 0.50, 0.58, step=0.01),

        # Labeling
        label_forward_hours=trial.suggest_int('label_forward_hours', 12, 48, step=6),
        label_vol_target=trial.suggest_float('label_vol_target', 1.2, 2.4, step=0.2),
        min_momentum_magnitude=trial.suggest_float('min_momentum_magnitude', 0.01, 0.12, step=0.01),

        # Exits
        vol_mult_tp=trial.suggest_float('vol_mult_tp', 3.0, 8.0, step=0.5),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', 2.0, 5.0, step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', 36, 120, step=12),
        cooldown_hours=trial.suggest_float('cooldown_hours', 12.0, 48.0, step=6.0),

        # Regime filter
        min_vol_24h=trial.suggest_float('min_vol_24h', 0.004, 0.015, step=0.001),
        max_vol_24h=trial.suggest_float('max_vol_24h', 0.04, 0.10, step=0.01),

        # Sizing
        position_size=trial.suggest_float('position_size', 0.06, 0.20, step=0.02),
        vol_sizing_target=trial.suggest_float('vol_sizing_target', 0.015, 0.035, step=0.005),

        # ML hyperparameters
        n_estimators=trial.suggest_int('n_estimators', 60, 200, step=20),
        max_depth=trial.suggest_int('max_depth', 2, 5),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.10, step=0.01),
        min_child_samples=trial.suggest_int('min_child_samples', 15, 40, step=5),
    )


def profile_from_params(params: Dict, coin_name: str) -> CoinProfile:
    """Reconstruct a CoinProfile from saved Optuna params dict (for holdout eval)."""
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]

    return CoinProfile(
        name=coin_name,
        prefixes=prefixes,
        extra_features=get_extra_features(coin_name),
        signal_threshold=params['signal_threshold'],
        min_val_auc=params['min_val_auc'],
        label_forward_hours=params['label_forward_hours'],
        label_vol_target=params['label_vol_target'],
        min_momentum_magnitude=params['min_momentum_magnitude'],
        vol_mult_tp=params['vol_mult_tp'],
        vol_mult_sl=params['vol_mult_sl'],
        max_hold_hours=params['max_hold_hours'],
        cooldown_hours=params['cooldown_hours'],
        min_vol_24h=params['min_vol_24h'],
        max_vol_24h=params['max_vol_24h'],
        position_size=params['position_size'],
        vol_sizing_target=params['vol_sizing_target'],
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        min_child_samples=params['min_child_samples'],
    )


def objective(trial: optuna.Trial, all_data: Dict, coin_prefix: str, coin_name: str, internal_oos_days: int = 90) -> float:
    """
    Optuna objective: run single-coin backtest on OPTIMIZATION WINDOW ONLY.
    
    Key change from v8: This function only sees truncated data (holdout excluded).
    Score is based on walk-forward metrics within the optimization window.
    No OOS component in the score — that's reserved for the true holdout eval.
    """
    profile = create_trial_profile(trial, coin_name)

    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        _set_reject_reason(trial, f'missing_symbol:{coin_prefix}/{coin_name}')
        return -99.0

    # Filter data to only this coin
    single_data = {target_sym: all_data[target_sym]}

    config = Config(
        max_positions=1,
        leverage=4,
        min_signal_edge=0.00,
        max_ensemble_std=0.10,
        train_embargo_hours=24,
        oos_eval_days=max(30, int(internal_oos_days)),  # penalty-only OOS diagnostic window
    )

    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        err_name = type(e).__name__
        err_msg = str(e).strip() or '<no-message>'
        tb_last = traceback.format_exc().strip().splitlines()[-1]
        trial.set_user_attr('error_type', err_name)
        trial.set_user_attr('error_message', err_msg[:300])
        trial.set_user_attr('error_tail', tb_last[:300])
        _set_reject_reason(trial, f'run_backtest_error:{err_name}')
        if DEBUG_TRIALS:
            print(f"\n❌ Trial {trial.number} backtest exception: {err_name}: {err_msg}")
            print(traceback.format_exc())
        return -99.0

    if result is None:
        _set_reject_reason(trial, 'result_none')
        return -99.0

    # Extract metrics
    n_trades = int(result.get('n_trades', 0) or 0)
    sharpe = _finite_metric(result.get('sharpe_annual', 0.0), default=0.0)
    pf_raw = _as_number(result.get('profit_factor', 0.0), default=0.0) or 0.0
    pf = float(pf_raw) if np.isfinite(pf_raw) else 5.0
    dd = _finite_metric(result.get('max_drawdown', 1.0), default=1.0)
    wr = _finite_metric(result.get('win_rate', 0.0), default=0.0)
    ann_ret = _finite_metric(result.get('ann_return', -1.0), default=-1.0)
    trades_per_year = _finite_metric(result.get('trades_per_year', 0.0), default=0.0)
    oos_sharpe = _finite_metric(result.get('oos_sharpe', 0.0), default=0.0)
    oos_return = _finite_metric(result.get('oos_return', 0.0), default=0.0)
    oos_trades = int(result.get('oos_trades', 0) or 0)

    # Treat sentinel values as missing data
    if sharpe <= -90:
        sharpe = 0.0
    if oos_sharpe <= -90:
        oos_sharpe = 0.0

    # SCORING v9 — NO OOS IN OPTIMIZATION SCORE
    # 
    # The optimization score uses ONLY the walk-forward backtest metrics from
    # the optimization window. The walk-forward itself provides some protection
    # (model never sees future data), but we no longer let Optuna optimize
    # toward a specific OOS window — that's reserved for the holdout eval.
    #
    # We still use the internal OOS (penalty-only diagnostic window) as a
    # PENALTY signal to catch severe overfit, but it doesn't contribute
    # positively to the score.

    # Cap PF contribution to prevent infinite/unstable trial values from dominating.
    pf_capped = min(max(0.0, pf), 5.0)
    pf_bonus = max(0, (pf_capped - 1.0)) * 0.5 if pf_capped > 0 else 0.0
    dd_penalty = max(0.0, dd - 0.30) * 3.0

    # Trade frequency penalties
    trade_penalty = 0.0
    if n_trades < 15:
        _set_reject_reason(trial, f'too_few_trades:{n_trades}')
        trade_penalty += min(3.0, (15 - n_trades) * 0.2)

    if trades_per_year < 10:
        trade_penalty += 0.5 + max(0.0, (5 - trades_per_year) * 0.05)
    elif trades_per_year < 25:
        trade_penalty += 0.25
    elif trades_per_year > 100:
        trade_penalty += 0.5

    # Bonus for ideal selectivity
    if 30 <= trades_per_year <= 80:
        trade_penalty -= 0.4

    # Base score: primarily walk-forward Sharpe + PF, penalized by DD and trades
    score = sharpe + pf_bonus - dd_penalty - trade_penalty

    # Penalize if internal OOS is terrible (catch severe overfit within optim window)
    # But do NOT reward good OOS — that would leak the optimization target
    if oos_trades >= 5:
        if oos_sharpe < -0.5:
            score -= min(1.5, abs(oos_sharpe) * 0.5)
        if oos_return < -0.05:
            score -= min(1.0, abs(oos_return) * 5)

    if oos_trades < 3 and n_trades >= 15:
        # Very few OOS trades suggests extreme regime-fitting
        score -= 0.3

    if ann_ret < -0.05:
        score = min(score, -1.0)

    # Store all metrics in trial for later analysis
    trial.set_user_attr('n_trades', n_trades)
    trial.set_user_attr('win_rate', round(wr, 3))
    trial.set_user_attr('ann_return', round(ann_ret, 4))
    trial.set_user_attr('sharpe', round(sharpe, 3))
    trial.set_user_attr('profit_factor', round(pf_capped, 3))
    trial.set_user_attr('max_drawdown', round(dd, 4))
    trial.set_user_attr('oos_trades', int(oos_trades))
    trial.set_user_attr('oos_sharpe', round(oos_sharpe, 3))
    trial.set_user_attr('oos_return', round(oos_return, 4))

    return score


class PlateauStopper:
    """Stop a study if no best-score improvement is observed for N completed trials."""

    def __init__(self, patience: int = 80, min_delta: float = 0.02, warmup_trials: int = 40):
        self.patience = max(1, patience)
        self.min_delta = max(0.0, min_delta)
        self.warmup_trials = max(0, warmup_trials)
        self.best_value = None
        self.best_trial_number = None

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(completed) < self.warmup_trials:
            return

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
            print(
                f"\n🛑 Plateau stop: no improvement > {self.min_delta:.4f} "
                f"for {self.patience} trials (best={self.best_value:.4f} @ trial {self.best_trial_number})."
            )
            study.stop()


def evaluate_holdout(holdout_data: Dict, best_params: Dict, coin_name: str,
                     coin_prefix: str, holdout_days: int = 120) -> Optional[Dict]:
    """
    Run the best parameters on the FULL dataset (including holdout period).
    
    Since the backtest is walk-forward, the model trains on earlier data and
    trades into the holdout period. We then extract ONLY the trades from the
    holdout window to get a true out-of-sample estimate.
    """
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"  ⚠️ Could not resolve {coin_name} for holdout eval")
        return None

    single_data = {target_sym: holdout_data[target_sym]}
    profile = profile_from_params(best_params, coin_name)

    config = Config(
        max_positions=1,
        leverage=4,
        min_signal_edge=0.00,
        max_ensemble_std=0.10,
        train_embargo_hours=24,
        oos_eval_days=holdout_days,  # This now matches our true holdout
    )

    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        print(f"  ❌ Holdout backtest error: {e}")
        return None

    if result is None:
        print(f"  ⚠️ Holdout backtest returned None")
        return None

    holdout_sharpe = _finite_metric(result.get('oos_sharpe', 0.0), default=0.0)
    holdout_return = _finite_metric(result.get('oos_return', 0.0), default=0.0)
    holdout_trades = int(result.get('oos_trades', 0) or 0)

    if _is_invalid_holdout_metric(holdout_sharpe, holdout_return, holdout_trades):
        holdout_sharpe = 0.0
        holdout_return = 0.0
        holdout_trades = 0

    return {
        'holdout_sharpe': holdout_sharpe,
        'holdout_return': holdout_return,
        'holdout_trades': holdout_trades,
        'full_sharpe': float(result.get('sharpe_annual', 0.0) or 0.0),
        'full_trades': int(result.get('n_trades', 0) or 0),
        'full_return': float(result.get('ann_return', 0.0) or 0.0),
        'full_pf': float(result.get('profit_factor', 0.0) or 0.0),
        'full_dd': float(result.get('max_drawdown', 0.0) or 0.0),
        'full_wr': float(result.get('win_rate', 0.0) or 0.0),
    }


def optimize_coin(all_data: Dict, coin_prefix: str, coin_name: str,
                  n_trials: int = 50, n_jobs: int = 1,
                  plateau_patience: int = 100, plateau_min_delta: float = 0.02,
                  plateau_warmup: int = 60,
                  study_suffix: str = "",
                  resume_study: bool = False,
                  holdout_days: int = 120,
                  min_internal_oos_trades: int = 0,
                  min_total_trades: int = 0):
    """Run Optuna optimization for a single coin with true holdout evaluation."""

    # STEP 0: Split data into optimization + holdout windows
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)

    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"❌ {coin_name}: not enough data in optimization window after holdout split")
        return None

    # Report the split
    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start = ohlcv.index.min()
    optim_end = ohlcv.index.max()

    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if holdout_sym:
        h_ohlcv = holdout_data[holdout_sym]['ohlcv']
        holdout_end = h_ohlcv.index.max()
    else:
        holdout_end = optim_end

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} ({coin_prefix}) — v9 TRUE HOLDOUT")
    print(f"   Optimization window: {optim_start.date()} → {optim_end.date()}")
    print(f"   Holdout window:      last {holdout_days} days (→ {holdout_end.date()})")
    print(f"   Trials: {n_trials} | Cores: {n_jobs}")
    print(f"   Plateau stop: patience={plateau_patience}, min_delta={plateau_min_delta}, warmup={plateau_warmup}")
    print(f"   ⚠️  Optuna NEVER sees holdout data during optimization")
    print(f"{'='*60}")

    # STEP 1: Setup Optuna study
    storage_url = _sqlite_url(_db_path())
    study_name = f"optimize_{coin_name}{'_' + study_suffix if study_suffix else ''}"

    sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = None
    for attempt in range(3):
        try:
            study = optuna.create_study(
                direction='maximize',
                sampler=sampler,
                study_name=study_name,
                storage=storage_url,
                load_if_exists=resume_study,
            )
            break
        except Exception as e:
            msg = str(e)
            duplicate_name = (
                isinstance(e, optuna.exceptions.DuplicatedStudyError)
                or "already exists" in msg
                or "UNIQUE constraint failed: studies.study_name" in msg
            )
            if duplicate_name:
                print(f"ℹ️ Study '{study_name}' already exists; loading existing study.")
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_url,
                    sampler=sampler,
                )
                break

            if "database is locked" in msg and attempt < 2:
                sleep_s = 0.25 * (attempt + 1)
                print(f"⚠️ Study DB locked while creating '{study_name}'. Retrying in {sleep_s:.2f}s...")
                time.sleep(sleep_s)
                continue
            raise

    if study is None:
        print(f"❌ Could not create or load study '{study_name}'.")
        return None

    # STEP 2: Run Optimization on OPTIMIZATION WINDOW ONLY
    internal_oos_days = min(120, max(60, holdout_days // 2))
    objective_func = functools.partial(
        objective,
        all_data=optim_data,  # <-- KEY: only optimization data, no holdout
        coin_prefix=coin_prefix,
        coin_name=coin_name,
        internal_oos_days=internal_oos_days,
    )

    stopper = PlateauStopper(
        patience=plateau_patience,
        min_delta=plateau_min_delta,
        warmup_trials=plateau_warmup,
    )

    try:
        study.optimize(
            objective_func,
            n_trials=n_trials,
            n_jobs=n_jobs,
            show_progress_bar=sys.stderr.isatty(),
            callbacks=[stopper],
        )
    except KeyboardInterrupt:
        print("\n🛑 Optimization stopped by user.")
    except Exception as e:
        print(f"\n❌ Error during optimization: {e}")
        return None

    if len(study.trials) == 0:
        print("No trials completed.")
        return None

    # STEP 3: Report optimization results
    # Increase minimum internal-OOS trade requirement for longer holdouts.
    # This encourages selecting parameter sets with enough signal cadence to
    # actually produce trades in a 120-180d true holdout.
    min_internal_oos_trades = min_internal_oos_trades or max(8, int(round(internal_oos_days / 8)))
    min_total_trades = min_total_trades or max(20, int(round(internal_oos_days / 3)))
    best = _select_best_trial(
        study,
        min_trades=min_total_trades,
        min_oos_trades=min_internal_oos_trades,
        min_oos_return=-0.01,
    )

    if best.number != study.best_trial.number:
        print(
            f"\n🛡️ Selected trial #{best.number} over raw best #{study.best_trial.number} "
            "for better OOS robustness constraints."
        )

    print(
        f"  Robust trial gates: min_trades={min_total_trades}, "
        f"min_internal_oos_trades={min_internal_oos_trades}, min_internal_oos_return=-1.00%"
    )

    if _as_number(best.value) == -99.0:
        reason_counts = {}
        for t in study.trials:
            reason = t.user_attrs.get('reject_reason', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        top_reasons = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:5]
        if top_reasons:
            print("  ⚠️ Top reject reasons:")
            for reason, count in top_reasons:
                print(f"    - {reason}: {count}")

    print(f"\n✅ BEST OPTIMIZATION RESULT for {coin_name}:")
    print(f"  Optim Score:  {_fmt_float(best.value, 3)}")
    print(f"  Trades:       {best.user_attrs.get('n_trades', '?')}")
    print(f"  Win Rate:     {_fmt_pct(best.user_attrs.get('win_rate'), 1)}")
    print(f"  Ann Return:   {_fmt_pct(best.user_attrs.get('ann_return'), 2)}")
    print(f"  Sharpe:       {_fmt_float(best.user_attrs.get('sharpe'), 3)}")
    print(f"  Profit Factor:{_fmt_float(best.user_attrs.get('profit_factor'), 3)}")
    print(f"  Max Drawdown: {_fmt_pct(best.user_attrs.get('max_drawdown'), 2)}")

    # STEP 4: TRUE HOLDOUT EVALUATION
    holdout_result = None
    if holdout_data and holdout_sym:
        print(f"\n🔬 HOLDOUT EVALUATION (last {holdout_days} days — never seen by Optuna)...")
        holdout_result = evaluate_holdout(
            holdout_data, best.params, coin_name, coin_prefix,
            holdout_days=holdout_days
        )

        if holdout_result:
            h = holdout_result
            ho_sharpe = h['holdout_sharpe']
            if ho_sharpe <= -90:
                ho_sharpe = 0.0

            print(f"\n  📊 HOLDOUT RESULTS (TRUE OUT-OF-SAMPLE):")
            print(f"     Holdout Sharpe:  {_fmt_float(ho_sharpe, 3)}")
            print(f"     Holdout Return:  {_fmt_pct(h['holdout_return'], 2)}")
            print(f"     Holdout Trades:  {h['holdout_trades']}")
            print(f"     Full Sharpe:     {_fmt_float(h['full_sharpe'], 3)}")
            print(f"     Full Trades:     {h['full_trades']}")
            print(f"     Full PF:         {_fmt_float(h['full_pf'], 3)}")
            print(f"     Full DD:         {_fmt_pct(h['full_dd'], 2)}")

            # Flag potential overfit
            optim_sharpe = _as_number(best.user_attrs.get('sharpe'), 0.0)
            if optim_sharpe and ho_sharpe < optim_sharpe * 0.3:
                print(f"\n  ⚠️  WARNING: Holdout Sharpe ({ho_sharpe:.3f}) is much lower than")
                print(f"     optimization Sharpe ({optim_sharpe:.3f}). Possible overfit!")
            elif ho_sharpe > 0.5:
                print(f"\n  ✅ Holdout Sharpe looks healthy — params likely generalize.")
        else:
            print(f"  ⚠️ Holdout evaluation failed or returned no results.")
    else:
        print(f"\n  ⚠️ Skipping holdout eval — insufficient holdout data.")

    # STEP 5: Save results
    result_data = {
        'coin': coin_name,
        'prefix': coin_prefix,
        'optim_score': best.value,
        'optim_metrics': dict(best.user_attrs),
        'holdout_metrics': holdout_result if holdout_result else {},
        'params': best.params,
        'n_trials': len(study.trials),
        'holdout_days': holdout_days,
        'timestamp': datetime.now().isoformat(),
    }
    result_data['quality'] = assess_result_quality(result_data)

    quality = result_data['quality']
    quality_issues = ', '.join(quality.get('issues', [])) if quality.get('issues') else 'none'
    print(f"\n  🧪 Quality rating: {quality.get('rating', 'unknown')}")
    print(f"     Holdout valid: {quality.get('holdout_valid', False)}")
    print(f"     Issues:        {quality_issues}")
    result_path = _persist_result_json(coin_name, result_data)
    if result_path:
        print(f"\n  💾 Saved to {result_path}")

    # STEP 6: Generate code snippet
    print(f"\n  📝 Suggested CoinProfile:")
    print(f"    '{coin_name}': CoinProfile(")
    print(f"        name='{coin_name}',")
    print(f"        prefixes={COIN_PROFILES[coin_name].prefixes},")
    extras = get_extra_features(coin_name)
    if extras:
        print(f"        extra_features={coin_name}_EXTRA_FEATURES,")
    else:
        print(f"        extra_features=[],")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            pretty = f"{v:.4f}".rstrip('0').rstrip('.')
            print(f"        {k}={pretty},")
        else:
            print(f"        {k}={v},")
    print(f"    ),")

    return result_data


def show_results():
    """Display all saved optimization results."""
    all_results = []
    seen = set()
    for result_dir in _candidate_results_dirs():
        for p in result_dir.glob("*_optimization.json"):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            all_results.append(p)

    results = sorted(all_results)
    if not results:
        print("No optimization results found.")
        return

    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*80}")

    for rpath in results:
        with open(rpath) as f:
            r = json.load(f)
        m = r.get('optim_metrics', r.get('metrics', {}))
        h = r.get('holdout_metrics', {})
        print(f"\n{r['coin']} ({r.get('prefix','?')}) — {r['n_trials']} trials — {r['timestamp'][:16]}")
        print(
            f"  Optim: Score={_fmt_float(r.get('optim_score', r.get('score')), 3)} | "
            f"Sharpe={_fmt_float(m.get('sharpe'), 3)} | "
            f"WR={_fmt_pct(m.get('win_rate'), 1)} | "
            f"PF={_fmt_float(m.get('profit_factor'), 3)} | "
            f"DD={_fmt_pct(m.get('max_drawdown'), 1)} | "
            f"Trades={m.get('n_trades', '?')}"
        )
        if h:
            ho_sharpe = _finite_metric(h.get('holdout_sharpe', 0.0), default=0.0)
            ho_return = _finite_metric(h.get('holdout_return', 0.0), default=0.0)
            ho_trades = int(h.get('holdout_trades', 0) or 0)
            if _is_invalid_holdout_metric(ho_sharpe, ho_return, ho_trades):
                ho_sharpe = 0.0
                ho_return = 0.0
                ho_trades = 0
            print(
                f"  Holdout: Sharpe={_fmt_float(ho_sharpe, 3)} | "
                f"Return={_fmt_pct(ho_return, 2)} | "
                f"Trades={ho_trades}"
            )

        quality = r.get('quality') or assess_result_quality(r)
        issues = quality.get('issues', [])
        issue_text = ', '.join(issues[:3]) if issues else 'none'
        print(
            f"  Quality: Rating={quality.get('rating', 'unknown')} | "
            f"HoldoutValid={quality.get('holdout_valid', False)} | "
            f"Issues={issue_text}"
        )


def _db_path() -> Path:
    """Return the Optuna DB path anchored to this script's directory."""
    return SCRIPT_DIR / "optuna_trading.db"


def _sqlite_url(path: Path) -> str:
    """Build a sqlite URL that is independent of the current working directory."""
    return f"sqlite:///{path.resolve()}"


def _candidate_results_dirs() -> list[Path]:
    """Candidate locations for optimization result JSON files.

    We prefer script-local paths so optimize.py and parallel_launch.py agree, but
    we keep fallbacks for environments where script directories are read-only.
    """
    return [
        SCRIPT_DIR / "optimization_results",
        Path.cwd() / "optimization_results",
    ]


def _persist_result_json(coin_name: str, result_data: Dict) -> Optional[Path]:
    """Persist optimization results with writable-path fallbacks."""
    last_error = None
    for candidate_dir in _candidate_results_dirs():
        try:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            result_path = candidate_dir / f"{coin_name}_optimization.json"
            with open(result_path, 'w') as f:
                json.dump(result_data, f, indent=2)
            return result_path
        except PermissionError as e:
            last_error = e
            continue
        except OSError as e:
            last_error = e
            continue

    print(f"\n  ❌ Failed to save optimization result for {coin_name}: {last_error}")
    print("     Tried paths:")
    for p in _candidate_results_dirs():
        print(f"       - {p}")
    return None


def _select_best_trial(
    study: optuna.Study,
    min_trades: int = 20,
    min_oos_trades: int = 10,
    min_oos_return: float = -0.02,
) -> optuna.trial.FrozenTrial:
    """Prefer robust trials over raw best score to reduce overfit selection.

    Selection rank (strict to relaxed):
      1) Meets all robustness filters.
      2) Meets trade-count filters only.
      3) Raw Optuna best trial.
    """
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        return study.best_trial

    robust = []
    trade_only = []

    for t in completed:
        n_trades = int(t.user_attrs.get('n_trades', 0) or 0)
        oos_trades = int(t.user_attrs.get('oos_trades', 0) or 0)
        if n_trades < min_trades or oos_trades < min_oos_trades:
            continue

        trade_only.append(t)

        oos_sharpe = _as_number(t.user_attrs.get('oos_sharpe'), 0.0) or 0.0
        oos_return = _as_number(t.user_attrs.get('oos_return'), 0.0) or 0.0
        if oos_sharpe <= -0.2:
            continue
        if oos_return < min_oos_return:
            continue
        robust.append(t)

    if robust:
        return max(robust, key=lambda t: t.value)
    if trade_only:
        return max(trade_only, key=lambda t: t.value)

    return study.best_trial



COIN_MAP = {
    'BIP': 'BTC', 'BTC': 'BTC',
    'ETP': 'ETH', 'ETH': 'ETH',
    'XPP': 'XRP', 'XRP': 'XRP',
    'SLP': 'SOL', 'SOL': 'SOL',
    'DOP': 'DOGE', 'DOGE': 'DOGE',
}

PREFIX_FOR_COIN = {
    'BTC': 'BIP', 'ETH': 'ETP', 'XRP': 'XPP', 'SOL': 'SLP', 'DOGE': 'DOP',
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-coin Optuna parameter optimization (v9 — True Holdout)")
    parser.add_argument("--coin", type=str, help="Coin prefix or name (e.g. BIP, BTC)")
    parser.add_argument("--all", action="store_true", help="Optimize all coins")
    parser.add_argument("--trials", type=int, default=150, help="Number of trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--show", action="store_true", help="Show saved results")
    parser.add_argument("--study-suffix", type=str, default="",
                        help="Optional suffix to isolate studies per launch (useful for parallel runs)")
    parser.add_argument("--plateau-patience", type=int, default=100,
                        help="Stop if best score does not improve for this many trials")
    parser.add_argument("--plateau-min-delta", type=float, default=0.02,
                        help="Minimum best-score improvement to reset plateau counter")
    parser.add_argument("--plateau-warmup", type=int, default=60,
                        help="Minimum completed trials before plateau checks start")
    parser.add_argument("--resume-study", action="store_true",
                        help="Resume existing study name instead of starting a fresh one")
    parser.add_argument("--holdout-days", type=int, default=180,
                        help="Days of data to reserve as true holdout (never seen by Optuna)")
    parser.add_argument("--min-internal-oos-trades", type=int, default=0,
                        help="Override minimum internal OOS trades required when selecting best trial (0=auto)")
    parser.add_argument("--min-total-trades", type=int, default=0,
                        help="Override minimum total trades required when selecting best trial (0=auto)")
    parser.add_argument("--debug-trials", action="store_true",
                        help="Enable verbose per-trial output")
    args = parser.parse_args()

    if args.debug_trials:
        DEBUG_TRIALS = True

    # Initialize SQLite WAL mode BEFORE running anything else
    init_db_wal(str(_db_path()))

    if args.show:
        show_results()
        sys.exit(0)

    # Default to fresh studies per run to avoid reusing stale/plateaued trials.
    effective_study_suffix = args.study_suffix
    if not effective_study_suffix:
        effective_study_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        print(f"🆕 Using fresh study suffix: {effective_study_suffix}")

    if not args.coin and not args.all:
        parser.print_help()
        sys.exit(1)

    # Load data once in the main process
    print("⏳ Loading data...")
    all_data = load_data()

    # Build prefix→symbol mapping
    for sym in all_data:
        prefix = sym.split('-')[0] if '-' in sym else sym
        PREFIX_TO_SYMBOL[prefix] = sym

    if args.all:
        for coin_name in ['ETH', 'BTC', 'SOL', 'XRP', 'DOGE']:
            prefix = PREFIX_FOR_COIN.get(coin_name)
            if prefix and prefix in PREFIX_TO_SYMBOL:
                optimize_coin(
                    all_data,
                    prefix,
                    coin_name,
                    n_trials=args.trials,
                    n_jobs=args.jobs,
                    plateau_patience=args.plateau_patience,
                    plateau_min_delta=args.plateau_min_delta,
                    plateau_warmup=args.plateau_warmup,
                    study_suffix=effective_study_suffix,
                    resume_study=args.resume_study,
                    holdout_days=args.holdout_days,
                    min_internal_oos_trades=args.min_internal_oos_trades,
                    min_total_trades=args.min_total_trades,
                )
    else:
        coin_input = args.coin.upper()
        coin_name = COIN_MAP.get(coin_input, coin_input)
        prefix = PREFIX_FOR_COIN.get(coin_name, coin_input)

        if prefix not in PREFIX_TO_SYMBOL:
            if coin_input in PREFIX_TO_SYMBOL:
                prefix = coin_input
                coin_name = COIN_MAP.get(prefix, prefix)
            else:
                print(f"❌ Coin '{args.coin}' not found. Available: {list(PREFIX_TO_SYMBOL.keys())}")
                sys.exit(1)

        optimize_coin(
            all_data,
            prefix,
            coin_name,
            n_trials=args.trials,
            n_jobs=args.jobs,
            plateau_patience=args.plateau_patience,
            plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup,
            study_suffix=effective_study_suffix,
            resume_study=args.resume_study,
            holdout_days=args.holdout_days,
            min_internal_oos_trades=args.min_internal_oos_trades,
            min_total_trades=args.min_total_trades,
        )
