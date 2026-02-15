#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (Parallel Enabled).

v10: TRUE HOLDOUT + ROBUSTNESS VALIDATION PIPELINE
    - Data is split into optimization window + holdout window BEFORE Optuna runs
    - Optuna only sees the optimization window when scoring trials
    - After optimization completes, best params are evaluated on the holdout
    - Deflated Sharpe Ratio tracked to flag multiple-testing inflation
    - Enhanced quality assessment with Sharpe decay, suspicion flags, warnings
    - Better trial selection: consistency-ranked, overfit-resistant
    - Calmar ratio and trades_per_month tracked per trial
    - Integrates with validate_robustness.py for paper-trade readiness scoring

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
import functools
import traceback
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

# Force single-threaded linear algebra BEFORE importing numpy/pandas/sklearn
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from train_model import Config, load_data, run_backtest
from coin_profiles import (
    CoinProfile, COIN_PROFILES,
    BTC_EXTRA_FEATURES, SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
PREFIX_TO_SYMBOL: Dict[str, str] = {}
DEBUG_TRIALS = False


# ---------------------------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------------------------

def init_db_wal(db_name="optuna_trading.db"):
    """Enable Write-Ahead Logging and set a long timeout for concurrency."""
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


def _as_number(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _finite_metric(value, default: float = 0.0) -> float:
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


def _is_invalid_holdout_metric(holdout_sharpe, holdout_return, holdout_trades):
    if holdout_sharpe <= -90:
        return True
    if holdout_trades <= 0:
        return True
    if holdout_return <= -0.99:
        return True
    return False


def _set_reject_reason(trial, reason):
    trial.set_user_attr('reject_reason', reason)


# ---------------------------------------------------------------------------
# v10: DEFLATED SHARPE RATIO  (Bailey & Lopez de Prado, 2014)
# ---------------------------------------------------------------------------

def compute_deflated_sharpe(observed_sharpe: float, n_trades: int,
                            n_trials: int = 200, skewness: float = 0.0,
                            kurtosis: float = 3.0) -> Dict:
    """Correct Sharpe for multiple testing, non-normality, short samples."""
    if n_trades < 10 or observed_sharpe <= 0 or n_trials < 2:
        return {'dsr': 0.0, 'p_value': 1.0, 'significant_5pct': False,
                'significant_10pct': False, 'expected_max_sr': 0.0,
                'z_stat': 0.0, 'sr_std': 0.0}
    try:
        from scipy import stats
        euler_mascheroni = 0.5772156649
        log_n = np.log(max(n_trials, 2))
        expected_max_sr = np.sqrt(2 * log_n) - \
            (np.log(np.pi) + euler_mascheroni) / (2 * np.sqrt(2 * log_n))
        excess_kurtosis = kurtosis - 3.0
        sr_var = (1.0 + 0.5 * observed_sharpe**2
                  - skewness * observed_sharpe
                  + (excess_kurtosis / 4.0) * observed_sharpe**2) / max(n_trades, 1)
        sr_std = np.sqrt(max(sr_var, 1e-10))
        z_stat = (observed_sharpe - expected_max_sr) / sr_std
        p_value = 1.0 - stats.norm.cdf(z_stat)
        dsr = observed_sharpe * (1.0 - p_value) if p_value < 1.0 else 0.0
        return {'dsr': round(dsr, 4), 'p_value': round(p_value, 4),
                'z_stat': round(z_stat, 4), 'expected_max_sr': round(expected_max_sr, 4),
                'sr_std': round(sr_std, 4), 'significant_5pct': p_value < 0.05,
                'significant_10pct': p_value < 0.10}
    except ImportError:
        return {'dsr': observed_sharpe * 0.5, 'p_value': 0.5,
                'significant_5pct': False, 'significant_10pct': False,
                'expected_max_sr': 0.0, 'z_stat': 0.0, 'sr_std': 0.0}


# ---------------------------------------------------------------------------
# QUALITY ASSESSMENT  (v10 — enhanced with decay + suspicion flags)
# ---------------------------------------------------------------------------

def assess_result_quality(result_data: Dict) -> Dict:
    optim = result_data.get('optim_metrics', {}) or {}
    holdout = result_data.get('holdout_metrics', {}) or {}

    n_trials = int(result_data.get('n_trials', 0) or 0)
    n_trades = int(optim.get('n_trades', 0) or 0)
    pf = _finite_metric(optim.get('profit_factor', 0.0))
    sharpe = _finite_metric(optim.get('sharpe', 0.0))
    win_rate = _finite_metric(optim.get('win_rate', 0.0))
    max_dd = _finite_metric(optim.get('max_drawdown', 1.0))

    holdout_sharpe = _finite_metric(holdout.get('holdout_sharpe', 0.0))
    holdout_return = _finite_metric(holdout.get('holdout_return', 0.0))
    holdout_trades = int(holdout.get('holdout_trades', 0) or 0)
    holdout_valid = not _is_invalid_holdout_metric(holdout_sharpe, holdout_return, holdout_trades)

    issues = []
    warns = []

    if n_trials < 50:          issues.append(f'low_trials:{n_trials}')
    elif n_trials < 100:       warns.append(f'moderate_trials:{n_trials}')
    if n_trades < 30:          issues.append(f'low_optim_trades:{n_trades}')
    elif n_trades < 50:        warns.append(f'moderate_optim_trades:{n_trades}')
    if pf < 1.10:              issues.append(f'weak_profit_factor:{pf:.3f}')
    elif pf < 1.25:            warns.append(f'moderate_profit_factor:{pf:.3f}')
    if sharpe < 0.50:          issues.append(f'weak_sharpe:{sharpe:.3f}')
    elif sharpe < 0.75:        warns.append(f'moderate_sharpe:{sharpe:.3f}')

    # v10: suspicion flags
    if sharpe > 3.0:           issues.append(f'suspicious_sharpe:{sharpe:.3f}')
    if pf > 4.0 and n_trades < 50:
                               issues.append(f'suspicious_pf_low_trades:{pf:.3f}/{n_trades}')
    if win_rate > 0.75 and n_trades < 50:
                               warns.append(f'high_wr_low_trades:{win_rate:.1%}/{n_trades}')
    if win_rate < 0.30:        issues.append(f'low_win_rate:{win_rate:.1%}')
    if max_dd > 0.35:          issues.append(f'high_drawdown:{max_dd:.1%}')
    elif max_dd > 0.25:        warns.append(f'moderate_drawdown:{max_dd:.1%}')

    if not holdout_valid:
        issues.append('invalid_holdout')
    else:
        if holdout_trades < 8:         issues.append(f'low_holdout_trades:{holdout_trades}')
        elif holdout_trades < 15:      warns.append(f'moderate_holdout_trades:{holdout_trades}')
        if holdout_sharpe < 0.0:       issues.append(f'negative_holdout_sharpe:{holdout_sharpe:.3f}')
        elif holdout_sharpe < 0.20:    warns.append(f'weak_holdout_sharpe:{holdout_sharpe:.3f}')
        if holdout_return < -0.05:     issues.append(f'negative_holdout_return:{holdout_return:.4f}')
        elif holdout_return < 0.0:     warns.append(f'flat_holdout_return:{holdout_return:.4f}')
        # v10: Sharpe decay
        if sharpe > 0 and holdout_sharpe >= 0:
            decay = 1.0 - (holdout_sharpe / sharpe) if sharpe > 0 else 0
            if decay > 0.70:           issues.append(f'severe_sharpe_decay:{decay:.0%}')
            elif decay > 0.50:         warns.append(f'moderate_sharpe_decay:{decay:.0%}')

    dsr_info = result_data.get('deflated_sharpe', {})
    if dsr_info and dsr_info.get('p_value') is not None:
        if dsr_info['p_value'] > 0.20:
            warns.append(f'dsr_not_significant:p={dsr_info["p_value"]:.3f}')

    if not issues and len(warns) <= 1:      rating = 'promising'
    elif not issues and len(warns) <= 3:    rating = 'watchlist_mild'
    elif len(issues) <= 1 and len(warns) <= 2: rating = 'watchlist'
    elif len(issues) <= 2:                  rating = 'weak'
    else:                                   rating = 'reject'

    return {'rating': rating, 'holdout_valid': holdout_valid,
            'issues': issues, 'warnings': warns,
            'n_issues': len(issues), 'n_warnings': len(warns)}


# ---------------------------------------------------------------------------
# SYMBOL RESOLUTION
# ---------------------------------------------------------------------------

def resolve_target_symbol(all_data, coin_prefix, coin_name):
    target = PREFIX_TO_SYMBOL.get(coin_prefix)
    if target:
        return target
    aliases = {'BIP':'BTC','ETP':'ETH','XPP':'XRP','SLP':'SOL','DOP':'DOGE',
               'BTC':'BTC','ETH':'ETH','XRP':'XRP','SOL':'SOL','DOGE':'DOGE'}
    candidates = [coin_prefix, coin_name, aliases.get(coin_prefix), aliases.get(coin_name)]
    for c in candidates:
        if not c: continue
        direct = PREFIX_TO_SYMBOL.get(c)
        if direct: return direct
        c_up = str(c).upper()
        for sym in all_data:
            sym_up = sym.upper()
            sym_prefix = sym_up.split('-')[0] if '-' in sym_up else sym_up
            if sym_prefix == c_up or c_up in sym_prefix or c_up in sym_up:
                return sym
    return None


# ---------------------------------------------------------------------------
# DATA SPLITTING — TRUE HOLDOUT
# ---------------------------------------------------------------------------

def split_data_temporal(all_data, holdout_days=120):
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
        feat, ohlcv = d['features'], d['ohlcv']
        optim_feat = feat[feat.index < holdout_start]
        optim_ohlcv = ohlcv[ohlcv.index < holdout_start]
        if len(optim_feat) > 500:
            optim_data[sym] = {'features': optim_feat, 'ohlcv': optim_ohlcv}
        if len(feat) > 500:
            holdout_data[sym] = {'features': feat.copy(), 'ohlcv': ohlcv.copy()}
    return optim_data, holdout_data


# ---------------------------------------------------------------------------
# TRIAL PROFILE
# ---------------------------------------------------------------------------

def create_trial_profile(trial, coin_name):
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]
    return CoinProfile(
        name=coin_name, prefixes=prefixes,
        extra_features=get_extra_features(coin_name),
        signal_threshold=trial.suggest_float('signal_threshold', 0.58, 0.86, step=0.01),
        min_val_auc=trial.suggest_float('min_val_auc', 0.50, 0.58, step=0.01),
        label_forward_hours=trial.suggest_int('label_forward_hours', 12, 48, step=6),
        label_vol_target=trial.suggest_float('label_vol_target', 1.2, 2.4, step=0.2),
        min_momentum_magnitude=trial.suggest_float('min_momentum_magnitude', 0.01, 0.12, step=0.01),
        vol_mult_tp=trial.suggest_float('vol_mult_tp', 3.0, 8.0, step=0.5),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', 2.0, 5.0, step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', 36, 120, step=12),
        cooldown_hours=trial.suggest_float('cooldown_hours', 12.0, 48.0, step=6.0),
        min_vol_24h=trial.suggest_float('min_vol_24h', 0.004, 0.015, step=0.001),
        max_vol_24h=trial.suggest_float('max_vol_24h', 0.04, 0.10, step=0.01),
        position_size=trial.suggest_float('position_size', 0.06, 0.20, step=0.02),
        vol_sizing_target=trial.suggest_float('vol_sizing_target', 0.015, 0.035, step=0.005),
        n_estimators=trial.suggest_int('n_estimators', 60, 200, step=20),
        max_depth=trial.suggest_int('max_depth', 2, 5),
        learning_rate=trial.suggest_float('learning_rate', 0.03, 0.10, step=0.01),
        min_child_samples=trial.suggest_int('min_child_samples', 15, 40, step=5),
    )


def profile_from_params(params, coin_name):
    base_profile = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    prefixes = base_profile.prefixes if base_profile else [coin_name]
    return CoinProfile(
        name=coin_name, prefixes=prefixes,
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


# ---------------------------------------------------------------------------
# OBJECTIVE (v10 — Calmar, trades/month, suspicion penalty)
# ---------------------------------------------------------------------------

def objective(trial, all_data, coin_prefix, coin_name, internal_oos_days=90):
    profile = create_trial_profile(trial, coin_name)
    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        _set_reject_reason(trial, f'missing_symbol:{coin_prefix}/{coin_name}')
        return -99.0
    single_data = {target_sym: all_data[target_sym]}
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24,
                    oos_eval_days=max(30, int(internal_oos_days)))
    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        err_name = type(e).__name__
        trial.set_user_attr('error_type', err_name)
        trial.set_user_attr('error_message', str(e)[:300])
        trial.set_user_attr('error_tail', traceback.format_exc().strip().splitlines()[-1][:300])
        _set_reject_reason(trial, f'run_backtest_error:{err_name}')
        if DEBUG_TRIALS:
            print(f"\n❌ Trial {trial.number}: {err_name}: {e}")
        return -99.0
    if result is None:
        _set_reject_reason(trial, 'result_none')
        return -99.0

    n_trades = int(result.get('n_trades', 0) or 0)
    sharpe = _finite_metric(result.get('sharpe_annual', 0.0))
    pf_raw = _as_number(result.get('profit_factor', 0.0), 0.0) or 0.0
    pf = float(pf_raw) if np.isfinite(pf_raw) else 5.0
    dd = _finite_metric(result.get('max_drawdown', 1.0), 1.0)
    wr = _finite_metric(result.get('win_rate', 0.0))
    ann_ret = _finite_metric(result.get('ann_return', -1.0), -1.0)
    trades_per_year = _finite_metric(result.get('trades_per_year', 0.0))
    oos_sharpe = _finite_metric(result.get('oos_sharpe', 0.0))
    oos_return = _finite_metric(result.get('oos_return', 0.0))
    oos_trades = int(result.get('oos_trades', 0) or 0)
    if sharpe <= -90: sharpe = 0.0
    if oos_sharpe <= -90: oos_sharpe = 0.0

    # SCORING v10
    pf_capped = min(max(0.0, pf), 5.0)
    pf_bonus = max(0, (pf_capped - 1.0)) * 0.5 if pf_capped > 0 else 0.0
    dd_penalty = max(0.0, dd - 0.30) * 3.0
    trade_penalty = 0.0
    if n_trades < 15:
        _set_reject_reason(trial, f'too_few_trades:{n_trades}')
        trade_penalty += min(3.0, (15 - n_trades) * 0.2)
    if trades_per_year < 10:    trade_penalty += 0.5 + max(0.0, (5 - trades_per_year) * 0.05)
    elif trades_per_year < 25:  trade_penalty += 0.25
    elif trades_per_year > 100: trade_penalty += 0.5
    if 30 <= trades_per_year <= 80: trade_penalty -= 0.4

    score = sharpe + pf_bonus - dd_penalty - trade_penalty

    if oos_trades >= 5:
        if oos_sharpe < -0.5:  score -= min(1.5, abs(oos_sharpe) * 0.5)
        if oos_return < -0.05: score -= min(1.0, abs(oos_return) * 5)
    if oos_trades < 3 and n_trades >= 15: score -= 0.3
    if ann_ret < -0.05: score = min(score, -1.0)
    # v10: penalise suspiciously high Sharpe
    if sharpe > 3.0: score -= (sharpe - 3.0) * 0.3

    trial.set_user_attr('n_trades', n_trades)
    trial.set_user_attr('win_rate', round(wr, 3))
    trial.set_user_attr('ann_return', round(ann_ret, 4))
    trial.set_user_attr('sharpe', round(sharpe, 3))
    trial.set_user_attr('profit_factor', round(pf_capped, 3))
    trial.set_user_attr('max_drawdown', round(dd, 4))
    trial.set_user_attr('oos_trades', int(oos_trades))
    trial.set_user_attr('oos_sharpe', round(oos_sharpe, 3))
    trial.set_user_attr('oos_return', round(oos_return, 4))
    # v10 extras
    tpm = trades_per_year / 12.0 if trades_per_year > 0 else 0
    calmar = min(ann_ret / dd if dd > 0.01 else 0.0, 10.0)
    trial.set_user_attr('trades_per_month', round(tpm, 1))
    trial.set_user_attr('calmar', round(calmar, 3))
    return score


# ---------------------------------------------------------------------------
# PLATEAU STOPPER
# ---------------------------------------------------------------------------

class PlateauStopper:
    def __init__(self, patience=80, min_delta=0.02, warmup_trials=40):
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


# ---------------------------------------------------------------------------
# TRIAL SELECTION (v10 — consistency-ranked)
# ---------------------------------------------------------------------------

def _select_best_trial(study, min_trades=20, min_oos_trades=10, min_oos_return=-0.02):
    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed: return study.best_trial
    robust, trade_only = [], []
    for t in completed:
        nt = int(t.user_attrs.get('n_trades', 0) or 0)
        ot = int(t.user_attrs.get('oos_trades', 0) or 0)
        if nt < min_trades or ot < min_oos_trades: continue
        trade_only.append(t)
        os_ = _as_number(t.user_attrs.get('oos_sharpe'), 0.0) or 0.0
        or_ = _as_number(t.user_attrs.get('oos_return'), 0.0) or 0.0
        is_ = _as_number(t.user_attrs.get('sharpe'), 0.0) or 0.0
        wr_ = _as_number(t.user_attrs.get('win_rate'), 0.0) or 0.0
        if os_ <= -90: os_ = 0.0
        if is_ <= -90: is_ = 0.0
        if os_ <= -0.2: continue
        if or_ < min_oos_return: continue
        # v10: consistency — reject >80% Sharpe decay
        if is_ > 0 and os_ >= 0:
            decay = 1.0 - (os_ / is_) if is_ > 0 else 0
            if decay > 0.80: continue
        if wr_ > 0.80 and nt < 40: continue
        robust.append(t)
    if robust:
        def _rank(t):
            oos_s = _as_number(t.user_attrs.get('oos_sharpe'), 0.0) or 0.0
            if oos_s <= -90: oos_s = 0.0
            return t.value + min(0.2, max(0, oos_s * 0.1))
        return max(robust, key=_rank)
    if trade_only: return max(trade_only, key=lambda t: t.value)
    return study.best_trial


# ---------------------------------------------------------------------------
# HOLDOUT EVALUATION
# ---------------------------------------------------------------------------

def evaluate_holdout(holdout_data, best_params, coin_name, coin_prefix, holdout_days=120):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"  ⚠️ Could not resolve {coin_name} for holdout eval")
        return None
    single_data = {target_sym: holdout_data[target_sym]}
    profile = profile_from_params(best_params, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24,
                    oos_eval_days=holdout_days)
    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        print(f"  ❌ Holdout backtest error: {e}")
        return None
    if result is None:
        print(f"  ⚠️ Holdout backtest returned None")
        return None
    hs = _finite_metric(result.get('oos_sharpe', 0.0))
    hr = _finite_metric(result.get('oos_return', 0.0))
    ht = int(result.get('oos_trades', 0) or 0)
    if _is_invalid_holdout_metric(hs, hr, ht): hs, hr, ht = 0.0, 0.0, 0
    return {'holdout_sharpe': hs, 'holdout_return': hr, 'holdout_trades': ht,
            'full_sharpe': float(result.get('sharpe_annual', 0) or 0),
            'full_trades': int(result.get('n_trades', 0) or 0),
            'full_return': float(result.get('ann_return', 0) or 0),
            'full_pf': float(result.get('profit_factor', 0) or 0),
            'full_dd': float(result.get('max_drawdown', 0) or 0),
            'full_wr': float(result.get('win_rate', 0) or 0)}


# ---------------------------------------------------------------------------
# PATHS AND PERSISTENCE
# ---------------------------------------------------------------------------

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
            with open(p, 'w') as f: json.dump(result_data, f, indent=2)
            return p
        except (PermissionError, OSError) as e:
            last_error = e
    print(f"\n  ❌ Failed to save result for {coin_name}: {last_error}")
    return None


# ---------------------------------------------------------------------------
# OPTIMIZE COIN (main logic)
# ---------------------------------------------------------------------------

def optimize_coin(all_data, coin_prefix, coin_name, n_trials=50, n_jobs=1,
                  plateau_patience=100, plateau_min_delta=0.02, plateau_warmup=60,
                  study_suffix="", resume_study=False, holdout_days=120,
                  min_internal_oos_trades=0, min_total_trades=0):
    # STEP 0: Split data
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"❌ {coin_name}: not enough data after holdout split")
        return None
    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start, optim_end = ohlcv.index.min(), ohlcv.index.max()
    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    holdout_end = holdout_data[holdout_sym]['ohlcv'].index.max() if holdout_sym else optim_end

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} ({coin_prefix}) — v10 TRUE HOLDOUT + DSR")
    print(f"   Optimization window: {optim_start.date()} → {optim_end.date()}")
    print(f"   Holdout window:      last {holdout_days} days (→ {holdout_end.date()})")
    print(f"   Trials: {n_trials} | Cores: {n_jobs}")
    print(f"   Plateau: patience={plateau_patience}, delta={plateau_min_delta}, warmup={plateau_warmup}")
    print(f"   ⚠️  Optuna NEVER sees holdout data during optimization")
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
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in msg or "UNIQUE constraint" in msg:
                study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)
                break
            if "database is locked" in msg and attempt < 2:
                time.sleep(0.25 * (attempt + 1)); continue
            raise
    if study is None:
        print(f"❌ Could not create/load study '{study_name}'."); return None

    # STEP 2: Optimize
    internal_oos_days = min(120, max(60, holdout_days // 2))
    obj = functools.partial(objective, all_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, internal_oos_days=internal_oos_days)
    stopper = PlateauStopper(patience=plateau_patience, min_delta=plateau_min_delta,
                             warmup_trials=plateau_warmup)
    try:
        study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs,
                       show_progress_bar=sys.stderr.isatty(), callbacks=[stopper])
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}"); return None
    if not study.trials:
        print("No trials completed."); return None

    # STEP 3: Select best
    miot = min_internal_oos_trades or max(8, int(round(internal_oos_days / 8)))
    mtt = min_total_trades or max(20, int(round(internal_oos_days / 3)))
    best = _select_best_trial(study, min_trades=mtt, min_oos_trades=miot, min_oos_return=-0.01)
    if best.number != study.best_trial.number:
        print(f"\n🛡️ Selected trial #{best.number} over raw best #{study.best_trial.number} for robustness.")

    if _as_number(best.value) == -99.0:
        rc = {}
        for t in study.trials:
            r = t.user_attrs.get('reject_reason', 'unknown'); rc[r] = rc.get(r, 0) + 1
        for r, c in sorted(rc.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"    - {r}: {c}")

    print(f"\n✅ BEST for {coin_name}:")
    print(f"  Score={_fmt_float(best.value,3)} Trades={best.user_attrs.get('n_trades','?')} "
          f"WR={_fmt_pct(best.user_attrs.get('win_rate'),1)} Sharpe={_fmt_float(best.user_attrs.get('sharpe'),3)} "
          f"PF={_fmt_float(best.user_attrs.get('profit_factor'),3)} DD={_fmt_pct(best.user_attrs.get('max_drawdown'),2)} "
          f"Calmar={_fmt_float(best.user_attrs.get('calmar'),3)} T/Mo={_fmt_float(best.user_attrs.get('trades_per_month'),1)}")

    # v10: DSR
    optim_sharpe = _finite_metric(best.user_attrs.get('sharpe', 0.0))
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
            elif ho_s > 0.5:
                print(f"  ✅ Holdout looks healthy.")
        else:
            print(f"  ⚠️ Holdout eval failed.")
    else:
        print(f"\n  ⚠️ Skipping holdout — insufficient data.")

    # STEP 5: Save
    result_data = {
        'coin': coin_name, 'prefix': coin_prefix,
        'optim_score': best.value, 'optim_metrics': dict(best.user_attrs),
        'holdout_metrics': holdout_result or {},
        'params': best.params, 'n_trials': len(study.trials),
        'holdout_days': holdout_days, 'deflated_sharpe': dsr_result,
        'timestamp': datetime.now().isoformat(),
    }
    result_data['quality'] = assess_result_quality(result_data)
    q = result_data['quality']
    print(f"\n  🧪 Quality: {q['rating']} | Issues: {', '.join(q['issues']) or 'none'} | "
          f"Warnings: {', '.join(q.get('warnings',[])) or 'none'}")
    p = _persist_result_json(coin_name, result_data)
    if p: print(f"  💾 Saved to {p}")

    # STEP 6: Snippet
    print(f"\n  📝 Suggested CoinProfile:")
    print(f"    '{coin_name}': CoinProfile(")
    print(f"        name='{coin_name}', prefixes={COIN_PROFILES[coin_name].prefixes},")
    extras = get_extra_features(coin_name)
    print(f"        extra_features={coin_name}_EXTRA_FEATURES," if extras else "        extra_features=[],")
    for k, v in sorted(best.params.items()):
        pretty = f"{v:.4f}".rstrip('0').rstrip('.') if isinstance(v, float) else str(v)
        print(f"        {k}={pretty},")
    print(f"    ),")
    return result_data


# ---------------------------------------------------------------------------
# SHOW RESULTS
# ---------------------------------------------------------------------------

def show_results():
    all_results = []
    seen = set()
    for d in _candidate_results_dirs():
        for p in d.glob("*_optimization.json"):
            rp = str(p.resolve())
            if rp not in seen: seen.add(rp); all_results.append(p)
    if not all_results:
        print("No optimization results found."); return
    print(f"\n{'='*80}\n📊 OPTIMIZATION RESULTS (v10)\n{'='*80}")
    for rpath in sorted(all_results):
        with open(rpath) as f: r = json.load(f)
        m = r.get('optim_metrics', {})
        h = r.get('holdout_metrics', {})
        dsr = r.get('deflated_sharpe', {})
        print(f"\n{r['coin']} ({r.get('prefix','?')}) — {r['n_trials']} trials — {r['timestamp'][:16]}")
        print(f"  Optim: Score={_fmt_float(r.get('optim_score'),3)} Sharpe={_fmt_float(m.get('sharpe'),3)} "
              f"WR={_fmt_pct(m.get('win_rate'),1)} PF={_fmt_float(m.get('profit_factor'),3)} "
              f"DD={_fmt_pct(m.get('max_drawdown'),1)} Trades={m.get('n_trades','?')} Calmar={_fmt_float(m.get('calmar'),3)}")
        if dsr:
            print(f"  DSR: {_fmt_float(dsr.get('dsr'),3)} p={_fmt_float(dsr.get('p_value'),3)} Sig@10%={dsr.get('significant_10pct','?')}")
        if h:
            hs = _finite_metric(h.get('holdout_sharpe', 0.0))
            hr = _finite_metric(h.get('holdout_return', 0.0))
            ht = int(h.get('holdout_trades', 0) or 0)
            if _is_invalid_holdout_metric(hs, hr, ht): hs, hr, ht = 0.0, 0.0, 0
            print(f"  Holdout: Sharpe={_fmt_float(hs,3)} Return={_fmt_pct(hr,2)} Trades={ht}")
        q = r.get('quality') or assess_result_quality(r)
        print(f"  Quality: {q.get('rating','?')} | Issues={', '.join(q.get('issues',[])[:3]) or 'none'} | "
              f"Warnings={', '.join(q.get('warnings',[])[:3]) or 'none'}")


# ---------------------------------------------------------------------------
# PRESETS
# ---------------------------------------------------------------------------

def apply_runtime_preset(args):
    presets = {
        'robust180': {'trials': 350, 'plateau_patience': 140, 'plateau_warmup': 80,
                      'plateau_min_delta': 0.015, 'holdout_days': 180,
                      'min_internal_oos_trades': 14, 'min_total_trades': 45},
        'robust120': {'trials': 280, 'plateau_patience': 120, 'plateau_warmup': 70,
                      'plateau_min_delta': 0.02, 'holdout_days': 120,
                      'min_internal_oos_trades': 10, 'min_total_trades': 35},
    }
    name = getattr(args, 'preset', 'none')
    if name in (None, '', 'none'): return args
    cfg = presets.get(name)
    if not cfg: return args
    for k, v in cfg.items(): setattr(args, k, v)
    print(f"🧭 Applied preset '{name}': " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    return args


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

COIN_MAP = {'BIP':'BTC','BTC':'BTC','ETP':'ETH','ETH':'ETH',
            'XPP':'XRP','XRP':'XRP','SLP':'SOL','SOL':'SOL','DOP':'DOGE','DOGE':'DOGE'}
PREFIX_FOR_COIN = {'BTC':'BIP','ETH':'ETP','XRP':'XPP','SOL':'SLP','DOGE':'DOP'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-coin Optuna optimization (v10 — True Holdout + DSR)")
    parser.add_argument("--coin", type=str, help="Coin prefix or name (e.g. BIP, BTC)")
    parser.add_argument("--all", action="store_true", help="Optimize all coins")
    parser.add_argument("--trials", type=int, default=150)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--plateau-patience", type=int, default=100)
    parser.add_argument("--plateau-min-delta", type=float, default=0.02)
    parser.add_argument("--plateau-warmup", type=int, default=60)
    parser.add_argument("--resume-study", action="store_true")
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="robust180", choices=["none","robust120","robust180"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0)
    parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--debug-trials", action="store_true")
    args = parser.parse_args()
    args = apply_runtime_preset(args)
    if args.debug_trials: DEBUG_TRIALS = True
    init_db_wal(str(_db_path()))
    if args.show: show_results(); sys.exit(0)

    effective_study_suffix = args.study_suffix
    if not effective_study_suffix:
        effective_study_suffix = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        print(f"🆕 Fresh study suffix: {effective_study_suffix}")
    if not args.coin and not args.all: parser.print_help(); sys.exit(1)

    print("⏳ Loading data...")
    all_data = load_data()
    for sym in all_data:
        prefix = sym.split('-')[0] if '-' in sym else sym
        PREFIX_TO_SYMBOL[prefix] = sym

    if args.all:
        for cn in ['ETH','BTC','SOL','XRP','DOGE']:
            px = PREFIX_FOR_COIN.get(cn)
            if px and px in PREFIX_TO_SYMBOL:
                optimize_coin(all_data, px, cn, n_trials=args.trials, n_jobs=args.jobs,
                              plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta,
                              plateau_warmup=args.plateau_warmup, study_suffix=effective_study_suffix,
                              resume_study=args.resume_study, holdout_days=args.holdout_days,
                              min_internal_oos_trades=args.min_internal_oos_trades,
                              min_total_trades=args.min_total_trades)
    else:
        ci = args.coin.upper()
        cn = COIN_MAP.get(ci, ci)
        px = PREFIX_FOR_COIN.get(cn, ci)
        if px not in PREFIX_TO_SYMBOL:
            if ci in PREFIX_TO_SYMBOL: px, cn = ci, COIN_MAP.get(ci, ci)
            else: print(f"❌ '{args.coin}' not found. Available: {list(PREFIX_TO_SYMBOL.keys())}"); sys.exit(1)
        optimize_coin(all_data, px, cn, n_trials=args.trials, n_jobs=args.jobs,
                      plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta,
                      plateau_warmup=args.plateau_warmup, study_suffix=effective_study_suffix,
                      resume_study=args.resume_study, holdout_days=args.holdout_days,
                      min_internal_oos_trades=args.min_internal_oos_trades,
                      min_total_trades=args.min_total_trades)