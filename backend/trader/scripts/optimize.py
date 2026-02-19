#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (v11.1: Fast CV).

v11.1: CRITICAL PERFORMANCE FIX
  v11.0 called run_backtest() 3x per trial (~30 min/trial = 50h for 100 trials).
  v11.1 uses fast_evaluate_fold() which trains ONE model per fold and simulates
  trading directly (~2s/fold). run_backtest() only used for final holdout.
  Expected: 100 trials in ~30-60 minutes instead of 50 hours.

Usage:
    python optimize.py --coin BTC --trials 100 --jobs 4
    python optimize.py --all --trials 100 --jobs 16
    python optimize.py --show
"""
import argparse, json, warnings, sys, os, logging, sqlite3, functools, traceback, time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from scripts.train_model import (
    Config, load_data, run_backtest, MLSystem,
    get_contract_spec, calculate_coinbase_fee,
)
from core.coin_profiles import (
    CoinProfile, COIN_PROFILES, get_coin_profile,
    BTC_EXTRA_FEATURES, SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_TRIALS = False

# ═══════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def _to_json_safe(obj):
    if isinstance(obj, dict): return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.generic): return obj.item()
    return obj

def init_db_wal(db_name="optuna_trading.db"):
    try:
        conn = sqlite3.connect(db_name); conn.execute("PRAGMA journal_mode=WAL;"); conn.execute("PRAGMA busy_timeout = 30000;"); conn.close()
    except Exception as e: print(f"⚠️ WAL mode failed: {e}")

def get_extra_features(coin_name):
    return {'BTC': BTC_EXTRA_FEATURES, 'SOL': SOL_EXTRA_FEATURES, 'DOGE': DOGE_EXTRA_FEATURES}.get(coin_name, [])

def _as_number(value, default=None):
    if value is None: return default
    if isinstance(value, (int, float)): return float(value)
    try: return float(value)
    except: return default

def _finite_metric(value, default=0.0):
    n = _as_number(value, default=default)
    return default if (n is None or not np.isfinite(n)) else float(n)

def _fmt_pct(v, d=1, fb="?"): n = _as_number(v); return f"{n:.{d}%}" if n is not None else fb
def _fmt_float(v, d=3, fb="?"): n = _as_number(v); return f"{n:.{d}f}" if n is not None else fb
def _set_reject_reason(trial, reason): trial.set_user_attr('reject_reason', reason)

def compute_deflated_sharpe(observed_sharpe, n_trades, n_trials=200, skewness=0.0, kurtosis=3.0):
    from scipy import stats
    if n_trades < 10 or n_trials < 2:
        return {'dsr': 0.0, 'p_value': 1.0, 'expected_max_sr': 0.0, 'significant_10pct': False, 'valid': True}
    em = 0.5772156649
    max_z = ((1 - em) * stats.norm.ppf(1 - 1.0/n_trials) + em * stats.norm.ppf(1 - 1.0/(n_trials * np.e)))
    exp_max_sr = max_z * np.sqrt(1.0 / n_trades)
    sr_std = np.sqrt((1 + 0.5*observed_sharpe**2 - skewness*observed_sharpe + ((kurtosis-3)/4.0)*observed_sharpe**2) / max(n_trades,1))
    if sr_std <= 0: sr_std = 0.001
    dsr_z = (observed_sharpe - exp_max_sr) / sr_std
    p = 1 - stats.norm.cdf(dsr_z)
    return {'dsr': round(float(dsr_z),4), 'p_value': round(float(p),4), 'expected_max_sr': round(float(exp_max_sr),4), 'significant_10pct': p < 0.10, 'valid': True}

# ═══════════════════════════════════════════════════════════════════════════
# DATA SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

def resolve_target_symbol(all_data, coin_prefix, coin_name):
    for sym in all_data:
        parts = sym.upper().split('-')
        if parts[0] in (coin_prefix.upper(), coin_name.upper()): return sym
    for sym in all_data:
        if coin_name.upper() in sym.upper() or coin_prefix.upper() in sym.upper(): return sym
    return None

def split_data_temporal(all_data, holdout_days=180):
    all_ends = [d['ohlcv'].index.max() for d in all_data.values() if len(d['ohlcv']) > 0]
    if not all_ends or holdout_days <= 0: return all_data, {}
    global_end = max(all_ends)
    holdout_start = global_end - pd.Timedelta(days=holdout_days)
    optim_data, holdout_data = {}, {}
    for sym, d in all_data.items():
        feat, ohlcv = d['features'], d['ohlcv']
        of, oo = feat[feat.index < holdout_start], ohlcv[ohlcv.index < holdout_start]
        if len(of) > 500: optim_data[sym] = {'features': of, 'ohlcv': oo}
        if len(feat) > 500: holdout_data[sym] = {'features': feat.copy(), 'ohlcv': ohlcv.copy()}
    return optim_data, holdout_data

def create_cv_splits(data, target_sym, n_folds=3, min_train_days=120):
    """Anchored walk-forward CV splits. Returns [(train_end, test_start, test_end), ...]"""
    ohlcv = data[target_sym]['ohlcv']
    start, end = ohlcv.index.min(), ohlcv.index.max()
    total_days = (end - start).days
    min_test_days = 60
    if total_days < min_train_days + min_test_days:
        boundary = start + pd.Timedelta(days=int(total_days * 0.7))
        return [(boundary, boundary, end)]
    n_folds = min(n_folds, (total_days - min_train_days) // min_test_days)
    if n_folds < 1: n_folds = 1
    test_zone_start = start + pd.Timedelta(days=min_train_days)
    fold_days = (end - test_zone_start).days // n_folds
    splits = []
    for i in range(n_folds):
        ts = test_zone_start + pd.Timedelta(days=i * fold_days)
        te = ts + pd.Timedelta(days=fold_days) if i < n_folds - 1 else end
        splits.append((ts, ts, te))
    return splits

# ═══════════════════════════════════════════════════════════════════════════
# v11.1: FAST LIGHTWEIGHT EVALUATOR (~2s per fold instead of ~10min)
# ═══════════════════════════════════════════════════════════════════════════

def fast_evaluate_fold(features, ohlcv, train_end, test_start, test_end, profile, config):
    """Train ONE model on data before train_end, simulate trading on test period."""
    system = MLSystem(config)
    cols = system.get_feature_columns(features.columns, profile.feature_columns)
    if not cols or len(cols) < 4: return None

    embargo_hours = max(config.train_embargo_hours, profile.label_forward_hours, 1)
    actual_train_end = train_end - pd.Timedelta(hours=embargo_hours)
    # Cap training window to match production's sliding window (default 120 days)
    train_window_start = actual_train_end - pd.Timedelta(days=config.train_lookback_days)
    train_feat = features[(features.index >= train_window_start) & (features.index < actual_train_end)]
    train_ohlcv = ohlcv[(ohlcv.index >= train_window_start) & (ohlcv.index < train_end + pd.Timedelta(hours=profile.label_forward_hours))]
    if len(train_feat) < config.min_train_samples: return None

    y = system.create_labels(train_ohlcv, train_feat, profile=profile)
    valid_idx = y.dropna().index
    X_all = train_feat.loc[valid_idx, cols]
    y_all = y.loc[valid_idx]
    if len(X_all) < config.min_train_samples: return None

    split_idx = int(len(X_all) * (1 - config.val_fraction))
    result = system.train(X_all.iloc[:split_idx], y_all.iloc[:split_idx],
                          X_all.iloc[split_idx:], y_all.iloc[split_idx:], profile=profile)
    if not result: return None
    model, scaler, iso, auc = result

    # --- SIMULATE TRADING ---
    test_feat = features[(features.index >= test_start) & (features.index <= test_end)]
    test_ohlcv = ohlcv[(ohlcv.index >= test_start) & (ohlcv.index <= test_end)]
    if len(test_feat) < 50: return None

    completed_trades = []
    active_pos = None
    last_exit = None
    equity = 100_000.0
    peak_equity = equity
    fee_rt = config.fee_pct_per_side * 2

    for ts in test_feat.index:
        if ts not in test_ohlcv.index: continue
        price = test_ohlcv.loc[ts, 'close']
        sma_200 = test_ohlcv.loc[ts, 'sma_200'] if 'sma_200' in test_ohlcv.columns else np.nan

        # CHECK EXITS
        if active_pos is not None:
            d = active_pos['dir']
            exit_price = exit_reason = None
            if d == 1:
                if price >= active_pos['tp']: exit_price, exit_reason = price, 'take_profit'
                elif price <= active_pos['sl']: exit_price, exit_reason = price, 'stop_loss'
            else:
                if price <= active_pos['tp']: exit_price, exit_reason = price, 'take_profit'
                elif price >= active_pos['sl']: exit_price, exit_reason = price, 'stop_loss'
            hours_held = (ts - active_pos['time']).total_seconds() / 3600
            if hours_held >= profile.max_hold_hours and not exit_price:
                exit_price, exit_reason = price, 'max_hold'
            if exit_price:
                raw = (exit_price - active_pos['entry']) / active_pos['entry'] * d
                net = raw - fee_rt
                notional = equity * profile.position_size * config.leverage
                pnl_d = net * notional
                completed_trades.append({'net_pnl': net, 'pnl_dollars': pnl_d, 'exit_reason': exit_reason})
                equity += pnl_d
                peak_equity = max(peak_equity, equity)
                last_exit = ts
                active_pos = None

        # CHECK ENTRIES
        if active_pos is None and equity > config.min_equity:
            if last_exit and (ts - last_exit).total_seconds() < profile.cooldown_hours * 3600: continue
            row = test_feat.loc[ts]
            ts_loc = test_ohlcv.index.get_loc(ts)
            if ts_loc < 24: continue

            ret_24h = price / test_ohlcv['close'].iloc[max(0, ts_loc-24)] - 1
            ret_72h = (price / test_ohlcv['close'].iloc[max(0, ts_loc-72)] - 1) if ts_loc >= 72 else 0
            sma_50 = test_ohlcv['close'].iloc[max(0,ts_loc-50):ts_loc].mean()
            ms = (1 if ret_24h > 0 else -1) + (1 if ret_72h > 0 else -1) + (1 if price > sma_50 else -1)
            if ms >= 2: direction = 1
            elif ms <= -2: direction = -1
            else: continue

            if not pd.isna(sma_200):
                if direction == 1 and price < sma_200: continue
                if direction == -1 and price > sma_200: continue

            vol_24h = test_ohlcv['close'].pct_change().rolling(24).std().get(ts, None)
            if vol_24h is None or pd.isna(vol_24h): continue
            if vol_24h < profile.min_vol_24h or vol_24h > profile.max_vol_24h: continue
            if abs(ret_72h) < profile.min_momentum_magnitude: continue

            x_in = np.nan_to_num(np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0)
            prob = float(iso.predict([model.predict_proba(scaler.transform(x_in))[0, 1]])[0])
            # Match production: require prob >= threshold + min_signal_edge
            if prob < profile.signal_threshold + config.min_signal_edge: continue

            # Match production: funding rate filter
            f_z = row.get('funding_rate_zscore', 0)
            if pd.isna(f_z): f_z = 0
            if direction == 1 and f_z > 2.5: continue
            if direction == -1 and f_z < -2.5: continue

            vol = vol_24h if vol_24h > 0 else 0.02
            active_pos = {
                'time': ts, 'entry': price, 'dir': direction, 'vol': vol,
                'tp': price * (1 + profile.vol_mult_tp * vol * direction),
                'sl': price * (1 - profile.vol_mult_sl * vol * direction),
            }

    # METRICS
    n = len(completed_trades)
    if n < 3:
        return {'n_trades': n, 'sharpe': -99.0, 'win_rate': 0, 'profit_factor': 0,
                'max_drawdown': 1.0, 'ann_return': -1.0, 'total_return': equity/100000-1, 'trades_per_year': 0}

    pnls = [t['net_pnl'] for t in completed_trades]
    pnl_d = [t['pnl_dollars'] for t in completed_trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    wr = len(wins) / len(pnls)
    avg, std = np.mean(pnls), np.std(pnls)
    sr = avg / std if std > 0 else 0.0
    gw = sum(wins) if wins else 0
    gl = abs(sum(losses)) if losses else 0.001
    pf = gw / gl if gl > 0 else 0.0

    eq_curve = 100000 + np.cumsum(pnl_d)
    peak = np.maximum.accumulate(eq_curve)
    dd = float(np.max((peak - eq_curve) / np.maximum(peak, 1.0)))
    total_ret = equity / 100000 - 1
    test_days = max((test_end - test_start).days, 1)
    ann_f = 365.0 / test_days
    ann_ret = (1 + total_ret) ** ann_f - 1 if total_ret > -1 else -1.0
    tpy = n * ann_f
    tpd = n / test_days
    ann_sr = sr * np.sqrt(tpd * 252) if tpd > 0 else 0.0

    return {'n_trades': n, 'sharpe': round(ann_sr, 4), 'win_rate': round(wr, 4),
            'profit_factor': round(min(pf, 5.0), 4), 'max_drawdown': round(dd, 4),
            'ann_return': round(ann_ret, 4), 'total_return': round(total_ret, 4),
            'trades_per_year': round(tpy, 1), 'avg_pnl': round(avg, 6)}

# ═══════════════════════════════════════════════════════════════════════════
# TRIAL PROFILE (9 tunable params)
# ═══════════════════════════════════════════════════════════════════════════

FIXED_ML = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 20}
FIXED_RISK = {'position_size': 0.12, 'vol_sizing_target': 0.025, 'cooldown_hours': 24.0, 'min_val_auc': 0.53}

def create_trial_profile(trial, coin_name):
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name],
        extra_features=get_extra_features(coin_name),
        signal_threshold=trial.suggest_float('signal_threshold', 0.65, 0.85, step=0.02),
        label_forward_hours=trial.suggest_int('label_forward_hours', 12, 48, step=12),
        label_vol_target=trial.suggest_float('label_vol_target', 1.2, 2.4, step=0.3),
        min_momentum_magnitude=trial.suggest_float('min_momentum_magnitude', 0.02, 0.10, step=0.02),
        vol_mult_tp=trial.suggest_float('vol_mult_tp', 3.0, 7.0, step=1.0),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', 2.0, 4.0, step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', 36, 96, step=12),
        min_vol_24h=trial.suggest_float('min_vol_24h', 0.005, 0.015, step=0.005),
        max_vol_24h=trial.suggest_float('max_vol_24h', 0.04, 0.08, step=0.02),
        cooldown_hours=FIXED_RISK['cooldown_hours'], position_size=FIXED_RISK['position_size'],
        vol_sizing_target=FIXED_RISK['vol_sizing_target'], min_val_auc=FIXED_RISK['min_val_auc'],
        n_estimators=FIXED_ML['n_estimators'], max_depth=FIXED_ML['max_depth'],
        learning_rate=FIXED_ML['learning_rate'], min_child_samples=FIXED_ML['min_child_samples'],
    )

def profile_from_params(params, coin_name):
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name],
        extra_features=get_extra_features(coin_name),
        signal_threshold=params.get('signal_threshold', 0.75),
        min_val_auc=params.get('min_val_auc', FIXED_RISK['min_val_auc']),
        label_forward_hours=params.get('label_forward_hours', 24),
        label_vol_target=params.get('label_vol_target', 1.8),
        min_momentum_magnitude=params.get('min_momentum_magnitude', 0.06),
        vol_mult_tp=params.get('vol_mult_tp', 5.0), vol_mult_sl=params.get('vol_mult_sl', 3.0),
        max_hold_hours=params.get('max_hold_hours', 72),
        cooldown_hours=params.get('cooldown_hours', FIXED_RISK['cooldown_hours']),
        min_vol_24h=params.get('min_vol_24h', 0.008), max_vol_24h=params.get('max_vol_24h', 0.06),
        position_size=params.get('position_size', FIXED_RISK['position_size']),
        vol_sizing_target=params.get('vol_sizing_target', FIXED_RISK['vol_sizing_target']),
        n_estimators=params.get('n_estimators', FIXED_ML['n_estimators']),
        max_depth=params.get('max_depth', FIXED_ML['max_depth']),
        learning_rate=params.get('learning_rate', FIXED_ML['learning_rate']),
        min_child_samples=params.get('min_child_samples', FIXED_ML['min_child_samples']),
    )

# ═══════════════════════════════════════════════════════════════════════════
# OBJECTIVE + STOPPER + SELECTION
# ═══════════════════════════════════════════════════════════════════════════

def objective(trial, optim_data, coin_prefix, coin_name, cv_splits, target_sym):
    profile = create_trial_profile(trial, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10, train_embargo_hours=24)
    features = optim_data[target_sym]['features']
    ohlcv_data = optim_data[target_sym]['ohlcv']

    fold_results, total_trades = [], 0
    for train_boundary, test_start, test_end in cv_splits:
        r = fast_evaluate_fold(features, ohlcv_data, train_boundary, test_start, test_end, profile, config)
        if r is not None:
            fold_results.append(r); total_trades += r['n_trades']

    if len(fold_results) < max(1, len(cv_splits) // 2):
        _set_reject_reason(trial, f'too_few_folds:{len(fold_results)}/{len(cv_splits)}'); return -99.0
    if total_trades < 20:
        _set_reject_reason(trial, f'too_few_trades:{total_trades}'); return -99.0

    sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in fold_results]
    mean_sr, min_sr = np.mean(sharpes), np.min(sharpes)
    std_sr = np.std(sharpes) if len(sharpes) > 1 else 0.0
    mean_wr = np.mean([r['win_rate'] for r in fold_results])
    mean_dd = np.mean([r['max_drawdown'] for r in fold_results])
    mean_pf = np.mean([r['profit_factor'] for r in fold_results])
    mean_ret = np.mean([r['total_return'] for r in fold_results])

    score = mean_sr
    if std_sr < 0.3 and len(sharpes) >= 2: score += 0.15
    elif std_sr > 0.8: score -= 0.25
    if min_sr > 0: score += 0.10
    elif min_sr < -0.5: score -= 0.30
    if min(max(0, mean_pf), 5.0) > 1.2: score += min(0.2, (mean_pf - 1.0) * 0.15)
    if mean_dd > 0.25: score -= (mean_dd - 0.25) * 2.0
    avg_tc = np.mean([r['n_trades'] for r in fold_results])
    if avg_tc < 5: score -= 0.5
    elif avg_tc < 8: score -= 0.2
    if mean_wr > 0.75 and total_trades < 40: score -= 0.3

    trial.set_user_attr('n_trades', total_trades)
    trial.set_user_attr('n_folds', len(fold_results))
    trial.set_user_attr('mean_sharpe', round(mean_sr, 3))
    trial.set_user_attr('min_sharpe', round(min_sr, 3))
    trial.set_user_attr('std_sharpe', round(std_sr, 3))
    trial.set_user_attr('mean_return', round(mean_ret, 4))
    trial.set_user_attr('win_rate', round(mean_wr, 3))
    trial.set_user_attr('profit_factor', round(min(mean_pf, 5), 3))
    trial.set_user_attr('max_drawdown', round(mean_dd, 4))
    trial.set_user_attr('sharpe', round(mean_sr, 3))
    trial.set_user_attr('oos_sharpe', round(mean_sr, 3))
    trial.set_user_attr('mean_oos_sharpe', round(mean_sr, 3))
    trial.set_user_attr('min_oos_sharpe', round(min_sr, 3))
    trial.set_user_attr('std_oos_sharpe', round(std_sr, 3))
    trial.set_user_attr('oos_return', round(mean_ret, 4))
    trial.set_user_attr('oos_trades', total_trades)
    tpy = np.mean([r['trades_per_year'] for r in fold_results])
    trial.set_user_attr('trades_per_month', round(tpy / 12.0, 1))
    ann_ret = np.mean([r['ann_return'] for r in fold_results])
    trial.set_user_attr('ann_return', round(ann_ret, 4))
    trial.set_user_attr('calmar', round(min(ann_ret / mean_dd if mean_dd > 0.01 else 0, 10.0), 3))

    if DEBUG_TRIALS:
        print(f"  T{trial.number}: score={score:.3f} SR={mean_sr:.3f}±{std_sr:.3f} min={min_sr:.3f} trades={total_trades} folds={len(fold_results)}")
    return score

class PlateauStopper:
    def __init__(self, patience=60, min_delta=0.02, warmup_trials=30):
        self.patience, self.min_delta, self.warmup_trials = max(1,patience), max(0,min_delta), max(0,warmup_trials)
        self.best_value = self.best_trial_number = None
    def __call__(self, study, trial):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        if len(completed) < self.warmup_trials: return
        if self.best_value is None: self.best_value = study.best_value; self.best_trial_number = study.best_trial.number; return
        if study.best_value > self.best_value + self.min_delta: self.best_value = study.best_value; self.best_trial_number = study.best_trial.number; return
        if sum(1 for t in completed if t.number > (self.best_trial_number or 0)) >= self.patience:
            print(f"\n🛑 Plateau: {self.patience} trials w/o improvement (best={self.best_value:.4f})"); study.stop()

def _select_best_trial(study, min_trades=20):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed: return study.best_trial
    robust, acceptable = [], []
    for t in completed:
        nt = int(t.user_attrs.get('n_trades', 0) or 0)
        if nt < min_trades: continue
        acceptable.append(t)
        sr = _as_number(t.user_attrs.get('mean_sharpe', t.user_attrs.get('sharpe')), 0) or 0
        if sr <= -90: sr = 0
        dd = _as_number(t.user_attrs.get('max_drawdown'), 1) or 1
        min_s = _as_number(t.user_attrs.get('min_sharpe'), None)
        std_s = _as_number(t.user_attrs.get('std_sharpe'), 1)
        if sr < -0.3 or dd > 0.35: continue
        if min_s is not None and min_s < -0.5: continue
        if std_s is not None and std_s > 1.0: continue
        robust.append(t)
    if robust:
        return max(robust, key=lambda t: _as_number(t.user_attrs.get('mean_sharpe', t.user_attrs.get('sharpe')), 0) or 0)
    return max(acceptable, key=lambda t: t.value) if acceptable else study.best_trial

# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT (full run_backtest — called ONCE at the end)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_holdout(holdout_data, params, coin_name, coin_prefix, holdout_days):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym: return None
    profile = profile_from_params(params, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10, train_embargo_hours=24, oos_eval_days=holdout_days)
    try: result = run_backtest({target_sym: holdout_data[target_sym]}, config, profile_overrides={coin_name: profile})
    except Exception as e: print(f"  ❌ Holdout error: {e}"); return None
    if not result: return None
    oos_sr = _finite_metric(result.get('oos_sharpe', 0))
    return {'holdout_sharpe': oos_sr if oos_sr > -90 else 0, 'holdout_return': _finite_metric(result.get('oos_return', 0)),
            'holdout_trades': int(result.get('oos_trades', 0) or 0), 'full_sharpe': _finite_metric(result.get('sharpe_annual', 0)),
            'full_pf': _finite_metric(result.get('profit_factor', 0)), 'full_dd': _finite_metric(result.get('max_drawdown', 1), 1),
            'full_trades': int(result.get('n_trades', 0) or 0)}

def assess_result_quality(rd):
    issues, warns = [], []
    m, h, dsr = rd.get('optim_metrics', {}), rd.get('holdout_metrics', {}), rd.get('deflated_sharpe', {})
    nt = int(m.get('n_trades', 0) or 0); sr = _finite_metric(m.get('mean_sharpe', m.get('sharpe', 0)))
    dd = _finite_metric(m.get('max_drawdown', 1), 1); ho_sr = _finite_metric(h.get('holdout_sharpe', 0))
    if nt < 20: issues.append(f'low_trades:{nt}')
    if sr < 0: issues.append(f'neg_sharpe:{sr:.3f}')
    if dd > 0.30: issues.append(f'high_dd:{dd:.1%}')
    if int(h.get('holdout_trades', 0) or 0) > 0 and ho_sr < -0.2: issues.append(f'bad_holdout:{ho_sr:.3f}')
    if not issues and len(warns) <= 1: rating = 'GOOD'
    elif not issues: rating = 'ACCEPTABLE'
    elif len(issues) <= 1: rating = 'MARGINAL'
    else: rating = 'POOR'
    return {'rating': rating, 'issues': issues, 'warnings': warns}

# ═══════════════════════════════════════════════════════════════════════════
# PATHS / PERSISTENCE / MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _db_path(): return SCRIPT_DIR / "optuna_trading.db"
def _sqlite_url(path): return f"sqlite:///{path.resolve()}"
def _candidate_results_dirs(): return [SCRIPT_DIR / "optimization_results", Path.cwd() / "optimization_results"]
def _persist_result_json(coin_name, data):
    for d in _candidate_results_dirs():
        try: d.mkdir(parents=True, exist_ok=True); p = d / f"{coin_name}_optimization.json"; open(p,'w').write(json.dumps(_to_json_safe(data), indent=2)); return p
        except: continue
    return None

def optimize_coin(all_data, coin_prefix, coin_name, n_trials=100, n_jobs=1,
                  plateau_patience=60, plateau_min_delta=0.02, plateau_warmup=30,
                  study_suffix="", resume_study=False, holdout_days=180,
                  min_internal_oos_trades=0, min_total_trades=0, n_cv_folds=3):
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym: print(f"❌ {coin_name}: no data after holdout split"); return None

    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start, optim_end = ohlcv.index.min(), ohlcv.index.max()
    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    holdout_end = holdout_data[holdout_sym]['ohlcv'].index.max() if holdout_sym else optim_end
    cv_splits = create_cv_splits(optim_data, target_sym, n_folds=n_cv_folds)

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} — v11.1 FAST CV")
    print(f"   Optim: {optim_start.date()} → {optim_end.date()} | Holdout: last {holdout_days}d (→{holdout_end.date()})")
    print(f"   CV folds: {len(cv_splits)} | Params: 9 tunable | Trials: {n_trials} | Jobs: {n_jobs}")
    for i, (tb, ts, te) in enumerate(cv_splits):
        print(f"     Fold {i}: train→{tb.date()} | test {ts.date()}→{te.date()}")
    print(f"   Est: ~{n_trials * len(cv_splits) * 3 / 60 / max(n_jobs,1):.0f} min")
    print(f"{'='*60}")

    study_name = f"optimize_{coin_name}{'_' + study_suffix if study_suffix else ''}"
    sampler = TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3))
    study = None
    for attempt in range(3):
        try:
            study = optuna.create_study(direction='maximize', sampler=sampler, study_name=study_name,
                                        storage=_sqlite_url(_db_path()), load_if_exists=resume_study); break
        except Exception as e:
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in str(e):
                study = optuna.load_study(study_name=study_name, storage=_sqlite_url(_db_path()), sampler=sampler); break
            if "database is locked" in str(e) and attempt < 2: time.sleep(0.3*(attempt+1)); continue
            raise
    if not study: print("❌ Could not create study"); return None

    obj = functools.partial(objective, optim_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, cv_splits=cv_splits, target_sym=target_sym)
    try: study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=sys.stderr.isatty(),
                        callbacks=[PlateauStopper(plateau_patience, plateau_min_delta, plateau_warmup)])
    except KeyboardInterrupt: print("\n🛑 Stopped.")
    except Exception as e: print(f"\n❌ {e}"); traceback.print_exc(); return None
    if not study.trials: print("No trials."); return None

    best = _select_best_trial(study, min_trades=min_total_trades or 20)
    if best.number != study.best_trial.number:
        print(f"\n🛡️ Selected #{best.number} over raw best #{study.best_trial.number}")

    print(f"\n✅ BEST {coin_name}: Score={_fmt_float(best.value)} | SR={_fmt_float(best.user_attrs.get('mean_sharpe'))} "
          f"min={_fmt_float(best.user_attrs.get('min_sharpe'))} | Trades={best.user_attrs.get('n_trades')} "
          f"| WR={_fmt_pct(best.user_attrs.get('win_rate'))} | DD={_fmt_pct(best.user_attrs.get('max_drawdown'),2)}")

    nc = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None])
    dsr = compute_deflated_sharpe(_finite_metric(best.user_attrs.get('mean_sharpe', 0)), int(best.user_attrs.get('n_trades', 0) or 0), nc)
    print(f"  📐 DSR: {dsr['dsr']:.3f} p={dsr['p_value']:.3f}")

    holdout_result = None
    if holdout_data and holdout_sym:
        print(f"\n🔬 HOLDOUT ({holdout_days}d, full walk-forward)...")
        holdout_result = evaluate_holdout(holdout_data, best.params, coin_name, coin_prefix, holdout_days)
        if holdout_result:
            h = holdout_result
            print(f"  SR={_fmt_float(h['holdout_sharpe'])} Ret={_fmt_pct(h['holdout_return'],2)} Trades={h['holdout_trades']} "
                  f"| Full: SR={_fmt_float(h['full_sharpe'])} PF={_fmt_float(h['full_pf'])} DD={_fmt_pct(h['full_dd'],2)}")

    result_data = {'coin': coin_name, 'prefix': coin_prefix, 'optim_score': best.value,
        'optim_metrics': dict(best.user_attrs), 'holdout_metrics': holdout_result or {},
        'params': best.params, 'n_trials': len(study.trials), 'n_cv_folds': len(cv_splits),
        'holdout_days': holdout_days, 'deflated_sharpe': dsr, 'version': 'v11.1', 'timestamp': datetime.now().isoformat()}
    result_data['quality'] = assess_result_quality(result_data)
    print(f"  🧪 Quality: {result_data['quality']['rating']}")
    p = _persist_result_json(coin_name, result_data)
    if p: print(f"  💾 {p}")

    print(f"\n  📝 CoinProfile(name='{coin_name}',")
    for k, v in sorted(best.params.items()):
        print(f"    {k}={f'{v:.4f}'.rstrip('0').rstrip('.') if isinstance(v, float) else v},")
    print(f"  )")
    return result_data

def show_results():
    results = []
    for d in _candidate_results_dirs():
        results.extend(d.glob("*_optimization.json")) if d.exists() else None
    if not results: print("No results."); return
    for p in sorted(results):
        r = json.load(open(p)); m = r.get('optim_metrics', {}); h = r.get('holdout_metrics', {})
        print(f"\n{r['coin']} — {r['n_trials']}t — {r.get('version','?')} | SR={_fmt_float(m.get('mean_sharpe', m.get('sharpe')))} "
              f"Trades={m.get('n_trades','?')} | Holdout: SR={_fmt_float(h.get('holdout_sharpe'))} Ret={_fmt_pct(h.get('holdout_return'),2)}")

def apply_runtime_preset(args):
    presets = {
        'robust180': {'plateau_patience': 60, 'plateau_warmup': 30, 'plateau_min_delta': 0.02, 'holdout_days': 180, 'min_internal_oos_trades': 8, 'min_total_trades': 20, 'n_cv_folds': 3},
        'robust120': {'plateau_patience': 50, 'plateau_warmup': 25, 'plateau_min_delta': 0.02, 'holdout_days': 120, 'min_internal_oos_trades': 6, 'min_total_trades': 15, 'n_cv_folds': 3},
        'quick':     {'plateau_patience': 30, 'plateau_warmup': 15, 'plateau_min_delta': 0.03, 'holdout_days': 90, 'min_internal_oos_trades': 5, 'min_total_trades': 10, 'n_cv_folds': 2},
    }
    name = getattr(args, 'preset', 'none')
    if name in (None, '', 'none'): return args
    cfg = presets.get(name)
    if cfg:
        for k, v in cfg.items(): setattr(args, k, v)
        print(f"🧭 Preset '{name}': " + ", ".join(f"{k}={v}" for k, v in cfg.items()))
    return args

COIN_MAP = {'BIP':'BTC','BTC':'BTC','ETP':'ETH','ETH':'ETH','XPP':'XRP','XRP':'XRP','SLP':'SOL','SOL':'SOL','DOP':'DOGE','DOGE':'DOGE'}
PREFIX_FOR_COIN = {'BTC':'BIP','ETH':'ETP','XRP':'XPP','SOL':'SLP','DOGE':'DOP'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v11.1 Fast CV Optimization")
    parser.add_argument("--coin", type=str); parser.add_argument("--all", action="store_true")
    parser.add_argument("--show", action="store_true"); parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1); parser.add_argument("--plateau-patience", type=int, default=60)
    parser.add_argument("--plateau-min-delta", type=float, default=0.02); parser.add_argument("--plateau-warmup", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="robust180", choices=["none","robust120","robust180","quick"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0); parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=3); parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--debug-trials", action="store_true")
    args = parser.parse_args(); args = apply_runtime_preset(args)
    if args.debug_trials: DEBUG_TRIALS = True
    if args.show: show_results(); sys.exit(0)
    init_db_wal(str(_db_path()))
    all_data = load_data()
    if not all_data: print("❌ No data."); sys.exit(1)
    coins = list(dict.fromkeys(COIN_MAP.values())) if args.all else [COIN_MAP.get(args.coin.upper(), args.coin.upper())] if args.coin else []
    if not coins: parser.print_help(); sys.exit(1)
    for cn in coins:
        optimize_coin(all_data, PREFIX_FOR_COIN.get(cn, cn), cn, n_trials=args.trials, n_jobs=args.jobs,
            plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup, study_suffix=args.study_suffix, resume_study=args.resume,
            holdout_days=args.holdout_days, min_internal_oos_trades=args.min_internal_oos_trades,
            min_total_trades=args.min_total_trades, n_cv_folds=args.n_cv_folds)