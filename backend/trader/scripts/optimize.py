#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (v11.1: Fast CV).

v11.1: CRITICAL PERFORMANCE FIX
  v11.0 called run_backtest() 3x per trial (~30 min/trial = 50h for 100 trials).
  v11.3 uses fast_evaluate_fold() which trains ONE model per fold and simulates
  trading directly (~2s/fold). run_backtest() only used for final holdout.
  Expected: 100 trials in ~30-60 minutes instead of 50 hours.

Usage:
    python optimize.py --coin BTC --trials 100 --jobs 4
    python optimize.py --all --trials 100 --jobs 16
    python optimize.py --show
"""
import argparse, json, warnings, sys, os, logging, sqlite3, functools, traceback, time, math
from datetime import datetime, timedelta
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fcntl

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
    BTC_EXTRA_FEATURES, ETH_EXTRA_FEATURES, XRP_EXTRA_FEATURES,
    SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
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
    return {
        'BTC': BTC_EXTRA_FEATURES,
        'ETH': ETH_EXTRA_FEATURES,
        'XRP': XRP_EXTRA_FEATURES,
        'SOL': SOL_EXTRA_FEATURES,
        'DOGE': DOGE_EXTRA_FEATURES,
    }.get(coin_name, [])

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


def _reject_score(observed: float | None = None, threshold: float | None = None, *, base: float = -6.0, scale: float = 8.0) -> float:
    """Return a strong but non-degenerate penalty for rejected trials.

    We intentionally avoid a flat sentinel like -99 so Optuna can still rank
    rejected trials by *how far* they are from meeting hard constraints.
    """
    if observed is None or threshold is None:
        return base - scale
    try:
        obs = float(observed)
        th = float(threshold)
    except (TypeError, ValueError):
        return base - scale
    if not np.isfinite(obs) or not np.isfinite(th):
        return base - scale
    gap = max(0.0, th - obs)
    norm = max(abs(th), 1e-6)
    penalty = scale * min(3.0, gap / norm)
    return base - penalty

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


def compute_probabilistic_sharpe(sharpes: List[float], benchmark_sr: float = 0.0) -> dict:
    """Estimate P(true Sharpe > benchmark) from walk-forward fold sharpes.

    The test is intentionally conservative: it treats each fold Sharpe as one
    independent out-of-sample estimate and computes a one-sided z-score for the
    mean Sharpe exceeding `benchmark_sr`.
    """
    valid = [float(s) for s in sharpes if s is not None and np.isfinite(s)]
    if len(valid) < 2:
        return {'valid': False, 'psr': 0.0, 'z_score': 0.0, 'n_folds': len(valid)}

    mean_sr = float(np.mean(valid))
    std_sr = float(np.std(valid, ddof=1))
    if std_sr <= 1e-12:
        psr = 1.0 if mean_sr > benchmark_sr else 0.0
        return {'valid': True, 'psr': psr, 'z_score': 999.0 if psr == 1.0 else -999.0, 'n_folds': len(valid)}

    z = (mean_sr - benchmark_sr) / (std_sr / math.sqrt(len(valid)))
    psr = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return {'valid': True, 'psr': float(psr), 'z_score': float(z), 'n_folds': len(valid)}

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

def create_cv_splits(data, target_sym, n_folds=3, min_train_days=120, purge_days=2):
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
    purge_delta = pd.Timedelta(days=max(0, purge_days))
    for i in range(n_folds):
        ts = test_zone_start + pd.Timedelta(days=i * fold_days)
        te = ts + pd.Timedelta(days=fold_days) if i < n_folds - 1 else end
        train_end = ts - purge_delta
        if train_end <= start + pd.Timedelta(days=min_train_days):
            train_end = ts
        splits.append((train_end, ts, te))
    return splits

# ═══════════════════════════════════════════════════════════════════════════
# v11.1: FAST LIGHTWEIGHT EVALUATOR (~2s per fold instead of ~10min)
# ═══════════════════════════════════════════════════════════════════════════

def fast_evaluate_fold(features, ohlcv, train_end, test_start, test_end, profile, config, symbol, fee_multiplier=1.0, pruned_only=True):
    """Train ONE model on data before train_end, simulate trading on test period."""
    system = MLSystem(config)
    feature_candidates = profile.resolve_feature_columns(
        use_pruned_features=bool(pruned_only),
        strict_pruned=bool(pruned_only),
    )
    cols = system.get_feature_columns(features.columns, feature_candidates)
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
    fee_multiplier = max(0.0, float(fee_multiplier))
    stressed_config = config
    if abs(fee_multiplier - 1.0) > 1e-9:
        stressed_config = Config(**{**config.__dict__,
                                    'fee_pct_per_side': config.fee_pct_per_side * fee_multiplier,
                                    'min_fee_per_contract': config.min_fee_per_contract * fee_multiplier})

    contract_spec = get_contract_spec(symbol)
    units_per_contract = float(contract_spec.get('units', 1.0) or 1.0)

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
                notional = equity * profile.position_size * config.leverage
                notional_per_contract = max(units_per_contract * active_pos['entry'], 1e-9)
                n_contracts = max(1, int(notional / notional_per_contract))
                entry_fee = calculate_coinbase_fee(n_contracts, active_pos['entry'], symbol, stressed_config)
                exit_fee = calculate_coinbase_fee(n_contracts, exit_price, symbol, stressed_config)
                fee_rt = (entry_fee + exit_fee) / max(notional, 1e-9)
                net = raw - fee_rt
                pnl_d = net * notional
                completed_trades.append({'raw_pnl': raw, 'net_pnl': net, 'pnl_dollars': pnl_d, 'exit_reason': exit_reason})
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
    raw_pnls = [t['raw_pnl'] for t in completed_trades]
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

    avg_raw = float(np.mean(raw_pnls)) if raw_pnls else 0.0
    fee_edge_ratio = abs((avg_raw - avg) / avg_raw) if abs(avg_raw) > 1e-9 else 1.0

    return {'n_trades': n, 'sharpe': round(ann_sr, 4), 'win_rate': round(wr, 4),
            'profit_factor': round(min(pf, 5.0), 4), 'max_drawdown': round(dd, 4),
            'ann_return': round(ann_ret, 4), 'total_return': round(total_ret, 4),
            'trades_per_year': round(tpy, 1), 'avg_pnl': round(avg, 6),
            'avg_raw_pnl': round(avg_raw, 6), 'fee_edge_ratio': round(float(fee_edge_ratio), 4)}

# ═══════════════════════════════════════════════════════════════════════════
# TRIAL PROFILE (9 tunable params)
# ═══════════════════════════════════════════════════════════════════════════

FIXED_ML = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 20}
FIXED_RISK = {
    'position_size': 0.12,
    'vol_sizing_target': 0.025,
    'min_val_auc': 0.53,
    'cooldown_hours': 24.0,
}

COIN_OPTIMIZATION_PRIORS = {
    # target_trades_per_year ~= at least weekly cadence while preserving quality
    'BTC': {'target_trades_per_year': 56.0, 'cooldown_min': 12.0, 'cooldown_max': 48.0},
    'ETH': {'target_trades_per_year': 64.0, 'cooldown_min': 8.0, 'cooldown_max': 36.0},
    'SOL': {'target_trades_per_year': 72.0, 'cooldown_min': 6.0, 'cooldown_max': 30.0},
    'XRP': {'target_trades_per_year': 60.0, 'cooldown_min': 8.0, 'cooldown_max': 36.0},
    'DOGE': {'target_trades_per_year': 52.0, 'cooldown_min': 10.0, 'cooldown_max': 42.0},
}

COIN_OBJECTIVE_GUARDS = {
    'BTC': {'min_total_trades': 16, 'min_avg_trades_per_fold': 4.0, 'min_expectancy': 0.0002},
    'ETH': {'min_total_trades': 20, 'min_avg_trades_per_fold': 5.0, 'min_expectancy': 0.00015},
    'SOL': {'min_total_trades': 20, 'min_avg_trades_per_fold': 5.0, 'min_expectancy': 0.0002},
    'XRP': {'min_total_trades': 18, 'min_avg_trades_per_fold': 4.0, 'min_expectancy': 0.00015},
    'DOGE': {'min_total_trades': 16, 'min_avg_trades_per_fold': 3.5, 'min_expectancy': 0.0002},
}

def create_trial_profile(trial, coin_name):
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    priors = COIN_OPTIMIZATION_PRIORS.get(coin_name, {})

    def clamp(v, low, high):
        return max(low, min(high, v))

    base_threshold = bp.signal_threshold if bp else 0.75
    base_fwd = bp.label_forward_hours if bp else 24
    base_label_vol = bp.label_vol_target if bp else 1.8
    base_mom = bp.min_momentum_magnitude if bp else 0.06
    base_tp = bp.vol_mult_tp if bp else 5.0
    base_sl = bp.vol_mult_sl if bp else 3.0
    base_hold = bp.max_hold_hours if bp else 72
    base_min_vol = bp.min_vol_24h if bp else 0.008
    base_max_vol = bp.max_vol_24h if bp else 0.06
    base_cooldown = bp.cooldown_hours if bp else 24.0
    cooldown_min = float(priors.get('cooldown_min', max(4.0, base_cooldown - 12.0)))
    cooldown_max = float(priors.get('cooldown_max', min(72.0, base_cooldown + 18.0)))

    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name],
        extra_features=get_extra_features(coin_name),
        signal_threshold=trial.suggest_float('signal_threshold', clamp(base_threshold - 0.10, 0.62, 0.88), clamp(base_threshold + 0.08, 0.68, 0.90), step=0.01),
        label_forward_hours=trial.suggest_int('label_forward_hours', int(clamp(base_fwd - 12, 12, 48)), int(clamp(base_fwd + 12, 12, 48)), step=12),
        label_vol_target=trial.suggest_float('label_vol_target', clamp(base_label_vol - 0.6, 1.0, 2.4), clamp(base_label_vol + 0.6, 1.2, 2.6), step=0.2),
        min_momentum_magnitude=trial.suggest_float('min_momentum_magnitude', clamp(base_mom - 0.04, 0.01, 0.12), clamp(base_mom + 0.04, 0.03, 0.14), step=0.01),
        vol_mult_tp=trial.suggest_float('vol_mult_tp', clamp(base_tp - 2.0, 2.0, 8.0), clamp(base_tp + 2.0, 3.0, 9.0), step=0.5),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', clamp(base_sl - 1.0, 1.5, 5.0), clamp(base_sl + 1.0, 2.0, 5.5), step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', int(clamp(base_hold - 24, 24, 120)), int(clamp(base_hold + 24, 36, 132)), step=12),
        min_vol_24h=trial.suggest_float('min_vol_24h', clamp(base_min_vol - 0.004, 0.003, 0.02), clamp(base_min_vol + 0.004, 0.006, 0.024), step=0.001),
        max_vol_24h=trial.suggest_float('max_vol_24h', clamp(base_max_vol - 0.02, 0.03, 0.10), clamp(base_max_vol + 0.02, 0.05, 0.12), step=0.005),
        cooldown_hours=trial.suggest_float('cooldown_hours', cooldown_min, cooldown_max, step=2.0),
        position_size=FIXED_RISK['position_size'],
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

def objective(
    trial,
    optim_data,
    coin_prefix,
    coin_name,
    cv_splits,
    target_sym,
    min_internal_oos_trades=0,
    target_trades_per_week=1.0,
    enable_fee_stress=True,
    fee_stress_multiplier=2.0,
    fee_blend_normal_weight=0.6,
    fee_blend_stressed_weight=0.4,
    pruned_only=True,
):
    min_fold_sharpe_hard = -0.1
    min_fold_win_rate = 0.35
    min_fold_win_rate_trades = max(5, int(min_internal_oos_trades) if min_internal_oos_trades else 8)

    profile = create_trial_profile(trial, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10,
                    train_embargo_hours=24, enforce_pruned_features=bool(pruned_only))
    features = optim_data[target_sym]['features']
    ohlcv_data = optim_data[target_sym]['ohlcv']

    fold_results, total_trades = [], 0
    stressed_fold_results = []
    blended_fold_sharpes = []
    w_normal = float(fee_blend_normal_weight)
    w_stress = float(fee_blend_stressed_weight)
    if w_normal <= 0 and w_stress <= 0:
        w_normal, w_stress = 1.0, 0.0
    w_total = w_normal + w_stress
    w_normal, w_stress = w_normal / w_total, w_stress / w_total

    for train_boundary, test_start, test_end in cv_splits:
        r = fast_evaluate_fold(features, ohlcv_data, train_boundary, test_start, test_end, profile, config,
                               target_sym, fee_multiplier=1.0, pruned_only=pruned_only)
        if r is not None:
            fold_results.append(r)
            total_trades += r['n_trades']

            if enable_fee_stress:
                stressed_r = fast_evaluate_fold(
                    features, ohlcv_data, train_boundary, test_start, test_end,
                    profile, config, target_sym, fee_multiplier=fee_stress_multiplier,
                    pruned_only=pruned_only,
                )
                if stressed_r is None:
                    _set_reject_reason(trial, 'missing_stressed_fold')
                    return _reject_score()
            else:
                stressed_r = dict(r)

            stressed_fold_results.append(stressed_r)
            blended_fold_sharpes.append((w_normal * (r['sharpe'] if r['sharpe'] > -90 else 0.0)) + (w_stress * (stressed_r['sharpe'] if stressed_r['sharpe'] > -90 else 0.0)))

    if len(fold_results) < max(1, len(cv_splits) // 2):
        _set_reject_reason(trial, f'too_few_folds:{len(fold_results)}/{len(cv_splits)}'); return _reject_score(len(fold_results), max(1, len(cv_splits) // 2))
    if min_internal_oos_trades > 0 and any(int(r.get('n_trades', 0) or 0) < min_internal_oos_trades for r in fold_results):
        worst_internal_trades = min(int(r.get('n_trades', 0) or 0) for r in fold_results)
        _set_reject_reason(trial, f'too_few_internal_oos_trades:{min_internal_oos_trades}')
        return _reject_score(worst_internal_trades, min_internal_oos_trades)
    if total_trades < 20:
        _set_reject_reason(trial, f'too_few_trades:{total_trades}'); return _reject_score(total_trades, 20)

    sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in fold_results]
    stressed_sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in stressed_fold_results]
    psr = compute_probabilistic_sharpe(sharpes, benchmark_sr=0.0)
    mean_sr, min_sr = np.mean(sharpes), np.min(sharpes)
    mean_stressed_sr = np.mean(stressed_sharpes) if stressed_sharpes else mean_sr
    mean_blended_sr = np.mean(blended_fold_sharpes) if blended_fold_sharpes else mean_sr
    std_sr = np.std(sharpes) if len(sharpes) > 1 else 0.0
    mean_wr = np.mean([r['win_rate'] for r in fold_results])
    mean_dd = np.mean([r['max_drawdown'] for r in fold_results])
    mean_pf = np.mean([r['profit_factor'] for r in fold_results])
    mean_ret = np.mean([r['total_return'] for r in fold_results])
    mean_fee_edge = np.mean([r.get('fee_edge_ratio', 1.0) for r in fold_results])
    mean_expectancy = np.mean([r.get('avg_pnl', 0.0) for r in fold_results])
    mean_raw_expectancy = np.mean([r.get('avg_raw_pnl', 0.0) for r in fold_results])
    stressed_expectancy = np.mean([r.get('avg_pnl', 0.0) for r in stressed_fold_results]) if stressed_fold_results else mean_expectancy
    stressed_fee_edge_ratio = np.mean([r.get('fee_edge_ratio', 1.0) for r in stressed_fold_results]) if stressed_fold_results else mean_fee_edge
    exp_std = np.std([r.get('avg_pnl', 0.0) for r in fold_results]) if len(fold_results) > 1 else 0.0
    tpy = np.mean([r.get('trades_per_year', 0.0) for r in fold_results])

    guards = COIN_OBJECTIVE_GUARDS.get(coin_name, {})
    guard_min_trades = int(guards.get('min_total_trades', 20))
    guard_min_avg_tc = float(guards.get('min_avg_trades_per_fold', 4.0))
    guard_min_exp = float(guards.get('min_expectancy', 0.0))
    avg_tc = np.mean([r['n_trades'] for r in fold_results])

    fold_metrics = [
        {
            'fold_idx': i,
            'n_trades': int(r.get('n_trades', 0) or 0),
            'sharpe': round(float(r.get('sharpe', 0.0) or 0.0), 6),
            'win_rate': round(float(r.get('win_rate', 0.0) or 0.0), 6),
            'profit_factor': round(float(r.get('profit_factor', 0.0) or 0.0), 6),
            'max_drawdown': round(float(r.get('max_drawdown', 0.0) or 0.0), 6),
            'total_return': round(float(r.get('total_return', 0.0) or 0.0), 6),
        }
        for i, r in enumerate(fold_results)
    ]

    if min_sr < min_fold_sharpe_hard:
        _set_reject_reason(trial, f'guard_min_fold_sharpe:{min_sr:.3f}<{min_fold_sharpe_hard:.3f}')
        return _reject_score(min_sr, min_fold_sharpe_hard)

    low_wr_folds = [
        f for f in fold_metrics
        if f['n_trades'] >= min_fold_win_rate_trades and f['win_rate'] < min_fold_win_rate
    ]
    if low_wr_folds:
        worst_fold = min(low_wr_folds, key=lambda f: f['win_rate'])
        _set_reject_reason(
            trial,
            (
                f"guard_min_fold_win_rate:fold{worst_fold['fold_idx']}="
                f"{worst_fold['win_rate']:.3f}<{min_fold_win_rate:.2f}"
                f"@trades>={min_fold_win_rate_trades}"
            ),
        )
        return _reject_score(worst_fold['win_rate'], min_fold_win_rate)

    if total_trades < guard_min_trades:
        _set_reject_reason(trial, f'guard_total_trades:{total_trades}<{guard_min_trades}')
        return _reject_score(total_trades, guard_min_trades)
    if avg_tc < guard_min_avg_tc:
        _set_reject_reason(trial, f'guard_avg_fold_trades:{avg_tc:.2f}<{guard_min_avg_tc:.2f}')
        return _reject_score(avg_tc, guard_min_avg_tc)
    if mean_expectancy < guard_min_exp:
        _set_reject_reason(trial, f'guard_expectancy:{mean_expectancy:.6f}<{guard_min_exp:.6f}')
        return _reject_score(mean_expectancy, guard_min_exp)
    if mean_raw_expectancy <= 0:
        _set_reject_reason(trial, f'raw_expectancy_nonpositive:{mean_raw_expectancy:.6f}')
        return _reject_score(mean_raw_expectancy, 1e-6)
    if enable_fee_stress and mean_stressed_sr < 0:
        _set_reject_reason(trial, f'negative_stressed_sharpe:{mean_stressed_sr:.6f}')
        return _reject_score(mean_stressed_sr, 0.0)
    if enable_fee_stress and stressed_expectancy <= 0:
        _set_reject_reason(trial, f'nonpositive_stressed_expectancy:{stressed_expectancy:.6f}')
        return _reject_score(stressed_expectancy, 1e-6)
    if psr.get('valid') and psr.get('psr', 0.0) < 0.55:
        _set_reject_reason(trial, f'low_psr:{psr.get("psr", 0.0):.3f}')
        return _reject_score(psr.get('psr', 0.0), 0.55)

    score = mean_blended_sr
    if std_sr < 0.3 and len(sharpes) >= 2: score += 0.15
    elif std_sr > 0.8: score -= 0.25
    if min_sr > 0: score += 0.10
    elif min_sr < -0.5: score -= 0.30
    if min(max(0, mean_pf), 5.0) > 1.2: score += min(0.2, (mean_pf - 1.0) * 0.15)
    if mean_dd > 0.25: score -= (mean_dd - 0.25) * 2.0
    if avg_tc < 5: score -= 0.5
    elif avg_tc < 8: score -= 0.2
    if mean_wr > 0.75 and total_trades < 40: score -= 0.3
    if mean_fee_edge > 0.35: score -= min(0.35, (mean_fee_edge - 0.35) * 0.8)
    if exp_std > 0.01: score -= min(0.30, (exp_std - 0.01) * 12)
    if psr.get('valid'):
        if psr['psr'] >= 0.80:
            score += 0.20
        elif psr['psr'] < 0.60:
            score -= 0.20

    target_tpy = max(1.0, float(target_trades_per_week) * 52.0)
    freq_ratio = (tpy / target_tpy) if target_tpy > 0 else 1.0
    if freq_ratio < 1.0:
        score -= min(0.60, (1.0 - freq_ratio) * 0.8)
    elif freq_ratio >= 1.15:
        score += min(0.20, (freq_ratio - 1.0) * 0.15)

    trial.set_user_attr('n_trades', total_trades)
    trial.set_user_attr('n_folds', len(fold_results))
    trial.set_user_attr('mean_sharpe', round(mean_sr, 3))
    trial.set_user_attr('min_sharpe', round(min_sr, 3))
    trial.set_user_attr('std_sharpe', round(std_sr, 3))
    trial.set_user_attr('blended_sharpe', round(mean_blended_sr, 3))
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
    trial.set_user_attr('psr', round(float(psr.get('psr', 0.0)), 4))
    trial.set_user_attr('psr_z', round(float(psr.get('z_score', 0.0)), 4))
    trial.set_user_attr('fee_edge_ratio', round(float(mean_fee_edge), 4))
    trial.set_user_attr('stressed_sharpe', round(float(mean_stressed_sr), 4))
    trial.set_user_attr('stressed_fee_edge_ratio', round(float(stressed_fee_edge_ratio), 4))
    trial.set_user_attr('stressed_expectancy', round(float(stressed_expectancy), 6))
    trial.set_user_attr('expectancy', round(float(mean_expectancy), 6))
    trial.set_user_attr('raw_expectancy', round(float(mean_raw_expectancy), 6))
    trial.set_user_attr('expectancy_std', round(float(exp_std), 6))
    trial.set_user_attr('trades_per_month', round(tpy / 12.0, 1))
    trial.set_user_attr('trades_per_year', round(tpy, 1))
    trial.set_user_attr('target_trades_per_week', round(float(target_trades_per_week), 2))
    trial.set_user_attr('frequency_ratio', round(float(freq_ratio), 3))
    trial.set_user_attr('fold_metrics', fold_metrics)
    trial.set_user_attr('stressed_fold_metrics', [
        {
            'fold_idx': i,
            'n_trades': int(r.get('n_trades', 0) or 0),
            'sharpe': round(float(r.get('sharpe', 0.0) or 0.0), 6),
            'win_rate': round(float(r.get('win_rate', 0.0) or 0.0), 6),
            'profit_factor': round(float(r.get('profit_factor', 0.0) or 0.0), 6),
            'max_drawdown': round(float(r.get('max_drawdown', 0.0) or 0.0), 6),
            'total_return': round(float(r.get('total_return', 0.0) or 0.0), 6),
        }
        for i, r in enumerate(stressed_fold_results)
    ])
    trial.set_user_attr('fold_constraints', {
        'min_fold_sharpe_hard': float(min_fold_sharpe_hard),
        'min_fold_win_rate': float(min_fold_win_rate),
        'min_fold_win_rate_trades': int(min_fold_win_rate_trades),
    })
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


def _candidate_trials_for_holdout(study, max_candidates=3, min_trades=20):
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    if not completed:
        return []

    accepted = []
    for t in completed:
        nt = int(t.user_attrs.get('n_trades', 0) or 0)
        if nt < min_trades:
            continue
        dd = _as_number(t.user_attrs.get('max_drawdown'), 1.0) or 1.0
        std_s = _as_number(t.user_attrs.get('std_sharpe'), 9.0) or 9.0
        min_s = _as_number(t.user_attrs.get('min_sharpe'), -9.0)
        if dd > 0.40 or std_s > 1.2 or (min_s is not None and min_s < -0.8):
            continue
        accepted.append(t)

    if not accepted:
        accepted = completed

    def _rank_key(t):
        psr = _as_number(t.user_attrs.get('psr'), 0.0) or 0.0
        mean_sr = _as_number(t.user_attrs.get('mean_sharpe', t.user_attrs.get('sharpe')), -9.0) or -9.0
        std_s = _as_number(t.user_attrs.get('std_sharpe'), 9.0) or 9.0
        dd = _as_number(t.user_attrs.get('max_drawdown'), 1.0) or 1.0
        return (float(t.value or -99), psr, mean_sr, -std_s, -dd)

    accepted_sorted = sorted(accepted, key=_rank_key, reverse=True)
    cap = max(1, int(max_candidates or 1))
    return accepted_sorted[:cap]


def _holdout_selection_score(holdout_metrics, cv_score=0.0):
    if not holdout_metrics:
        return -999.0
    ho_sr = _as_number(holdout_metrics.get('holdout_sharpe'), 0.0) or 0.0
    ho_ret = _as_number(holdout_metrics.get('holdout_return'), 0.0) or 0.0
    ho_trades = int(holdout_metrics.get('holdout_trades', 0) or 0)

    trade_term = min(1.0, ho_trades / 25.0)
    score = (0.60 * ho_sr) + (0.25 * ho_ret * 5.0) + (0.15 * trade_term)

    if ho_trades < 10:
        score -= 0.25
    if ho_sr <= 0:
        score -= 0.35
    if ho_ret <= 0:
        score -= 0.25

    return score + 0.10 * (_as_number(cv_score, 0.0) or 0.0)


def _passes_holdout_gate(holdout_metrics, min_trades=15, min_sharpe=0.0, min_return=0.0):
    if not holdout_metrics:
        return False
    ho_trades = int(holdout_metrics.get('holdout_trades', 0) or 0)
    ho_sr = _as_number(holdout_metrics.get('holdout_sharpe'), 0.0) or 0.0
    ho_ret = _as_number(holdout_metrics.get('holdout_return'), 0.0) or 0.0
    return ho_trades >= int(min_trades) and ho_sr >= float(min_sharpe) and ho_ret >= float(min_return)

# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT (full run_backtest — called ONCE at the end)
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_holdout(holdout_data, params, coin_name, coin_prefix, holdout_days, pruned_only=True):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym: return None
    profile = profile_from_params(params, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10,
                    train_embargo_hours=24, oos_eval_days=holdout_days,
                    enforce_pruned_features=bool(pruned_only))
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
    ho_ret = _finite_metric(h.get('holdout_return', 0), 0)
    ho_trades = int(h.get('holdout_trades', 0) or 0)
    if nt < 20: issues.append(f'low_trades:{nt}')
    if sr < 0: issues.append(f'neg_sharpe:{sr:.3f}')
    if dd > 0.30: issues.append(f'high_dd:{dd:.1%}')
    if ho_trades > 0:
        if ho_trades < 15: issues.append(f'holdout_too_few_trades:{ho_trades}')
        if ho_sr <= 0: issues.append(f'bad_holdout_sharpe:{ho_sr:.3f}')
        if ho_ret <= 0: issues.append(f'bad_holdout_return:{ho_ret:.3%}')
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


def _candidate_trial_ledger_paths() -> List[Path]:
    return [d / "trial_ledger.jsonl" for d in _candidate_results_dirs()]


def resolve_trial_ledger_path() -> Optional[Path]:
    for path in _candidate_trial_ledger_paths():
        if path.exists():
            return path
    for path in _candidate_trial_ledger_paths():
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        except OSError:
            continue
    return None


def append_trial_ledger_entry(
    coin_name: str,
    preset_name: str,
    run_id: str,
    completed_trials: int,
    timestamp: Optional[str] = None,
) -> Optional[Path]:
    ledger_path = resolve_trial_ledger_path()
    if ledger_path is None:
        return None

    ts = timestamp or datetime.now().isoformat()
    row = {
        'coin': coin_name,
        'preset': preset_name,
        'run_id': run_id,
        'timestamp': ts,
        'completed_trials': int(max(0, completed_trials)),
    }
    with open(ledger_path, 'a') as f:
        f.write(json.dumps(_to_json_safe(row), default=str) + "\n")
    return ledger_path


def aggregate_cumulative_trial_counts(ledger_path: Optional[Path] = None) -> Dict[str, object]:
    path = Path(ledger_path) if ledger_path else resolve_trial_ledger_path()
    by_coin: Dict[str, int] = {}
    global_total = 0
    last_timestamp = None

    if path is None or not path.exists():
        return {
            'ledger_path': str(path) if path else None,
            'ledger_timestamp': None,
            'coin_totals': by_coin,
            'global_total': global_total,
            'entries': 0,
        }

    entries = 0
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            coin = str(item.get('coin', '')).upper()
            trials = int(item.get('completed_trials', 0) or 0)
            ts = item.get('timestamp')
            if not coin or trials < 0:
                continue
            by_coin[coin] = by_coin.get(coin, 0) + trials
            global_total += trials
            entries += 1
            if ts:
                last_timestamp = str(ts)

    return {
        'ledger_path': str(path),
        'ledger_timestamp': last_timestamp,
        'coin_totals': by_coin,
        'global_total': global_total,
        'entries': entries,
    }

@contextmanager
def _study_storage_lock(db_path: Path):
    """Serialize Optuna SQLite storage initialization across processes."""
    lock_path = db_path.with_suffix(db_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def _persist_result_json(coin_name, data):
    for d in _candidate_results_dirs():
        try: d.mkdir(parents=True, exist_ok=True); p = d / f"{coin_name}_optimization.json"; open(p,'w').write(json.dumps(_to_json_safe(data), indent=2)); return p
        except: continue
    return None

def optimize_coin(all_data, coin_prefix, coin_name, n_trials=100, n_jobs=1,
                  plateau_patience=60, plateau_min_delta=0.02, plateau_warmup=30,
                  study_suffix="", resume_study=False, holdout_days=180,
                  min_internal_oos_trades=0, min_total_trades=0, n_cv_folds=3,
                  sampler_seed=42, holdout_candidates=3, require_holdout_pass=False,
                  holdout_min_trades=15, holdout_min_sharpe=0.0, holdout_min_return=0.0,
                  target_trades_per_week=1.0, preset_name="none", enable_fee_stress=True,
                  fee_stress_multiplier=2.0, fee_blend_normal_weight=0.6, fee_blend_stressed_weight=0.4,
                  pruned_only=True):
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym: print(f"❌ {coin_name}: no data after holdout split"); return None

    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start, optim_end = ohlcv.index.min(), ohlcv.index.max()
    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    holdout_end = holdout_data[holdout_sym]['ohlcv'].index.max() if holdout_sym else optim_end
    cv_splits = create_cv_splits(optim_data, target_sym, n_folds=n_cv_folds, purge_days=2)

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} — v11.3 FAST CV")
    print(f"   Optim: {optim_start.date()} → {optim_end.date()} | Holdout: last {holdout_days}d (→{holdout_end.date()})")
    print(f"   CV folds: {len(cv_splits)} | Params: 10 tunable | Trials: {n_trials} | Jobs: {n_jobs}")
    for i, (tb, ts, te) in enumerate(cv_splits):
        print(f"     Fold {i}: train→{tb.date()} | test {ts.date()}→{te.date()}")
    print(f"   Est: ~{n_trials * len(cv_splits) * 3 / 60 / max(n_jobs,1):.0f} min")
    print(f"{'='*60}")

    study_name = f"optimize_{coin_name}{'_' + study_suffix if study_suffix else ''}"
    sampler = TPESampler(seed=sampler_seed, n_startup_trials=min(10, n_trials // 3))
    study = None
    storage_url = _sqlite_url(_db_path())
    for attempt in range(10):
        try:
            with _study_storage_lock(_db_path()):
                study = optuna.create_study(
                    direction='maximize',
                    sampler=sampler,
                    study_name=study_name,
                    storage=storage_url,
                    load_if_exists=True,
                )
            break
        except Exception as e:
            err = str(e).lower()
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in err:
                with _study_storage_lock(_db_path()):
                    study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)
                break
            # SQLite schema/alembic races can happen when multiple processes initialize storage concurrently.
            transient_schema_race = (
                ("table" in err and "already exists" in err)
                or ("alembic_version" in err and "unique constraint failed" in err)
            )
            if transient_schema_race and attempt < 9:
                time.sleep(0.3 * (attempt + 1)); continue
            if "database is locked" in err and attempt < 9:
                time.sleep(0.4 * (attempt + 1)); continue
            raise
    if not study: print("❌ Could not create study"); return None

    obj = functools.partial(objective, optim_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, cv_splits=cv_splits, target_sym=target_sym,
                            min_internal_oos_trades=min_internal_oos_trades,
                            target_trades_per_week=target_trades_per_week,
                            enable_fee_stress=enable_fee_stress,
                            fee_stress_multiplier=fee_stress_multiplier,
                            fee_blend_normal_weight=fee_blend_normal_weight,
                            fee_blend_stressed_weight=fee_blend_stressed_weight,
                            pruned_only=pruned_only)
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
    run_id = f"{study_name}_{datetime.now().strftime('%Y%m%dT%H%M%S')}"
    ledger_path = append_trial_ledger_entry(
        coin_name=coin_name,
        preset_name=preset_name,
        run_id=run_id,
        completed_trials=nc,
    )
    cumulative_trials = aggregate_cumulative_trial_counts(ledger_path=ledger_path)
    print(f"  📐 DSR: {dsr['dsr']:.3f} p={dsr['p_value']:.3f}")

    holdout_result = None
    selection_meta = {}
    deployment_blocked = False
    blocked_reasons = []
    effective_holdout_min_trades = max(
        int(holdout_min_trades),
        int((max(0.1, float(target_trades_per_week)) * max(7, int(holdout_days)) / 7.0) * 0.60),
    )
    if holdout_data and holdout_sym:
        candidate_trials = _candidate_trials_for_holdout(
            study,
            max_candidates=max(1, int(holdout_candidates or 1)),
            min_trades=min_total_trades or 20,
        )
        if best not in candidate_trials:
            candidate_trials = [best] + candidate_trials

        print(f"\n🔬 HOLDOUT ({holdout_days}d, full walk-forward) — evaluating {len(candidate_trials)} candidate(s)...")
        ranked_candidates = []
        for cand in candidate_trials:
            holdout_metrics = evaluate_holdout(holdout_data, cand.params, coin_name, coin_prefix, holdout_days,
                                               pruned_only=pruned_only)
            if not holdout_metrics:
                continue
            sel_score = _holdout_selection_score(holdout_metrics, cv_score=cand.value)
            ranked_candidates.append((cand, holdout_metrics, sel_score))
            print(
                f"  Trial #{cand.number}: sel={sel_score:.3f} "
                f"SR={_fmt_float(holdout_metrics['holdout_sharpe'])} "
                f"Ret={_fmt_pct(holdout_metrics['holdout_return'],2)} Trades={holdout_metrics['holdout_trades']}"
            )

        if ranked_candidates:
            ranked_candidates.sort(key=lambda x: x[2], reverse=True)
            passing_candidates = [
                (c, m, sc) for c, m, sc in ranked_candidates
                if _passes_holdout_gate(
                    m,
                    min_trades=effective_holdout_min_trades,
                    min_sharpe=holdout_min_sharpe,
                    min_return=holdout_min_return,
                )
            ]
            selected_pool = passing_candidates if require_holdout_pass else ranked_candidates
            selected_trial, holdout_result, selected_score = selected_pool[0] if selected_pool else (None, None, None)
            if require_holdout_pass and not passing_candidates:
                deployment_blocked = True
                blocked_reasons.append(
                    f"holdout_gate_failed:trades>={effective_holdout_min_trades},sr>={holdout_min_sharpe},ret>={holdout_min_return}"
                )
                print("  🛑 Holdout gate failed for all candidates. Blocking deployment for this coin.")
                selected_trial, holdout_result, selected_score = ranked_candidates[0]
            if selected_trial.number != best.number:
                print(
                    f"  🧭 Holdout-guided selection: #{selected_trial.number} "
                    f"over #{best.number} (sel {selected_score:.3f})"
                )
                best = selected_trial
            selection_meta = {
                'mode': 'holdout_guided',
                'n_candidates': len(ranked_candidates),
                'n_passing_candidates': len(passing_candidates),
                'selected_trial': int(best.number),
                'selected_score': round(float(selected_score), 6),
                'require_holdout_pass': bool(require_holdout_pass),
                'holdout_gate': {
                    'min_trades': int(holdout_min_trades),
                    'effective_min_trades': int(effective_holdout_min_trades),
                    'min_sharpe': float(holdout_min_sharpe),
                    'min_return': float(holdout_min_return),
                },
                'deployment_blocked': bool(deployment_blocked),
                'candidates': [
                    {
                        'trial': int(c.number),
                        'selection_score': round(float(sc), 6),
                        'holdout_sharpe': round(float(m.get('holdout_sharpe', 0.0) or 0.0), 6),
                        'holdout_return': round(float(m.get('holdout_return', 0.0) or 0.0), 6),
                        'holdout_trades': int(m.get('holdout_trades', 0) or 0),
                        'passes_gate': _passes_holdout_gate(
                            m,
                            min_trades=effective_holdout_min_trades,
                            min_sharpe=holdout_min_sharpe,
                            min_return=holdout_min_return,
                        ),
                    }
                    for c, m, sc in ranked_candidates
                ],
            }
            h = holdout_result
            print(f"  ✅ Selected holdout: SR={_fmt_float(h['holdout_sharpe'])} Ret={_fmt_pct(h['holdout_return'],2)} "
                  f"Trades={h['holdout_trades']} | Full: SR={_fmt_float(h['full_sharpe'])} "
                  f"PF={_fmt_float(h['full_pf'])} DD={_fmt_pct(h['full_dd'],2)}")
        else:
            print("  ⚠️ Holdout evaluation returned no valid candidates.")

    result_data = {'coin': coin_name, 'prefix': coin_prefix, 'optim_score': best.value,
        'optim_metrics': dict(best.user_attrs), 'holdout_metrics': holdout_result or {},
        'params': best.params, 'n_trials': len(study.trials), 'n_cv_folds': len(cv_splits),
        'holdout_days': holdout_days, 'deflated_sharpe': dsr, 'version': 'v11.3', 'timestamp': datetime.now().isoformat(),
        'fee_stress': {
            'enabled': bool(enable_fee_stress),
            'multiplier': float(fee_stress_multiplier),
            'blend_normal_weight': float(fee_blend_normal_weight),
            'blend_stressed_weight': float(fee_blend_stressed_weight),
        },
        'selection_meta': selection_meta, 'deployment_blocked': deployment_blocked,
        'deployment_block_reasons': blocked_reasons,
        'pruned_only': bool(pruned_only),
        'run_id': run_id,
        'trial_ledger': {
            'ledger_path': cumulative_trials.get('ledger_path'),
            'ledger_timestamp': cumulative_trials.get('ledger_timestamp'),
            'coin_cumulative_trials': cumulative_trials.get('coin_totals', {}).get(coin_name, 0),
            'global_cumulative_trials': cumulative_trials.get('global_total', 0),
            'completed_trials_this_run': nc,
            'preset': preset_name,
        }}
    result_data['quality'] = assess_result_quality(result_data)
    print(f"  🧪 Quality: {result_data['quality']['rating']}")
    p = _persist_result_json(coin_name, result_data)
    if p: print(f"  💾 {p}")

    print(f"\n  📝 CoinProfile(name='{coin_name}',")
    for k, v in sorted(best.params.items()):
        print(f"    {k}={f'{v:.4f}'.rstrip('0').rstrip('.') if isinstance(v, float) else v},")
    print(f"  )")
    return result_data


def optimize_coin_multiseed(all_data, coin_prefix, coin_name, sampler_seeds=None, **kwargs):
    seeds = [int(s) for s in (sampler_seeds or [42])]
    if len(seeds) <= 1:
        return optimize_coin(all_data, coin_prefix, coin_name, sampler_seed=seeds[0], **kwargs)

    print(f"\n🌱 Multi-seed optimization for {coin_name}: seeds={seeds}")
    run_results = []
    base_suffix = kwargs.get('study_suffix', '')
    for seed in seeds:
        seed_kwargs = dict(kwargs)
        seed_kwargs['study_suffix'] = f"{base_suffix}_s{seed}" if base_suffix else f"s{seed}"
        r = optimize_coin(all_data, coin_prefix, coin_name, sampler_seed=seed, **seed_kwargs)
        if r:
            run_results.append(r)

    if not run_results:
        return None

    qualified = [
        r for r in run_results
        if not r.get('deployment_blocked', False)
        and _passes_holdout_gate(
            r.get('holdout_metrics', {}),
            min_trades=kwargs.get('holdout_min_trades', 15),
            min_sharpe=kwargs.get('holdout_min_sharpe', 0.0),
            min_return=kwargs.get('holdout_min_return', 0.0),
        )
    ]
    pool = qualified or run_results

    params_pool = [r.get('params', {}) for r in pool if r.get('params')]
    consensus_params = {}
    if params_pool:
        keys = set().union(*[p.keys() for p in params_pool])
        for k in keys:
            vals = [p[k] for p in params_pool if k in p]
            if not vals:
                continue
            if all(isinstance(v, (int, float)) for v in vals):
                med = float(np.median(vals))
                if all(isinstance(v, int) for v in vals):
                    consensus_params[k] = int(round(med))
                else:
                    consensus_params[k] = round(med, 6)
            else:
                consensus_params[k] = max(set(vals), key=vals.count)

    best_seed_result = max(pool, key=lambda r: _finite_metric(r.get('optim_score', -99), -99))
    if consensus_params:
        best_seed_result['params'] = consensus_params
        best_seed_result.setdefault('meta', {})['seed_consensus'] = {
            'seeds': seeds,
            'qualified_runs': len(qualified),
            'total_runs': len(run_results),
        }
        best_seed_result['quality'] = assess_result_quality(best_seed_result)
        p = _persist_result_json(coin_name, best_seed_result)
        if p:
            print(f"  💾 Consensus result saved to {p}")

    return best_seed_result

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
        'robust180': {'plateau_patience': 60, 'plateau_warmup': 30, 'plateau_min_delta': 0.02, 'holdout_days': 180, 'min_internal_oos_trades': 8, 'min_total_trades': 20, 'n_cv_folds': 5, 'holdout_candidates': 3, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0},
        'robust120': {'plateau_patience': 50, 'plateau_warmup': 25, 'plateau_min_delta': 0.02, 'holdout_days': 120, 'min_internal_oos_trades': 6, 'min_total_trades': 15, 'n_cv_folds': 5, 'holdout_candidates': 2, 'holdout_min_trades': 12, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0},
        'quick':     {'plateau_patience': 30, 'plateau_warmup': 15, 'plateau_min_delta': 0.03, 'holdout_days': 90, 'min_internal_oos_trades': 5, 'min_total_trades': 10, 'n_cv_folds': 2, 'holdout_candidates': 1, 'holdout_min_trades': 10, 'holdout_min_sharpe': 0.0, 'holdout_min_return': -0.01, 'require_holdout_pass': False, 'target_trades_per_week': 0.8},
        'paper_ready': {'plateau_patience': 80, 'plateau_warmup': 40, 'plateau_min_delta': 0.015, 'holdout_days': 240, 'min_internal_oos_trades': 10, 'min_total_trades': 28, 'n_cv_folds': 5, 'holdout_candidates': 4, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.05, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0},
    }
    name = getattr(args, 'preset', 'none')
    if name in (None, '', 'none'): return args
    cfg = presets.get(name)
    if cfg:
        arg_flags = {
            'plateau_patience': '--plateau-patience',
            'plateau_warmup': '--plateau-warmup',
            'plateau_min_delta': '--plateau-min-delta',
            'holdout_days': '--holdout-days',
            'min_internal_oos_trades': '--min-internal-oos-trades',
            'min_total_trades': '--min-total-trades',
            'n_cv_folds': '--n-cv-folds',
            'holdout_candidates': '--holdout-candidates',
            'holdout_min_trades': '--holdout-min-trades',
            'holdout_min_sharpe': '--holdout-min-sharpe',
            'holdout_min_return': '--holdout-min-return',
            'require_holdout_pass': '--require-holdout-pass',
            'target_trades_per_week': '--target-trades-per-week',
        }
        provided = set(sys.argv[1:])
        for k, v in cfg.items():
            flag = arg_flags.get(k)
            if flag and flag in provided:
                continue
            setattr(args, k, v)
        print(f"🧭 Preset '{name}': " + ", ".join(f"{k}={getattr(args, k)}" for k in cfg.keys()))
    return args

COIN_MAP = {'BIP':'BTC','BTC':'BTC','ETP':'ETH','ETH':'ETH','XPP':'XRP','XRP':'XRP','SLP':'SOL','SOL':'SOL','DOP':'DOGE','DOGE':'DOGE'}
PREFIX_FOR_COIN = {'BTC':'BIP','ETH':'ETP','XRP':'XPP','SOL':'SLP','DOGE':'DOP'}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="v11.3 Fast CV Optimization")
    parser.add_argument("--coin", type=str); parser.add_argument("--all", action="store_true")
    parser.add_argument("--show", action="store_true"); parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--jobs", type=int, default=1); parser.add_argument("--plateau-patience", type=int, default=60)
    parser.add_argument("--plateau-min-delta", type=float, default=0.02); parser.add_argument("--plateau-warmup", type=int, default=30)
    parser.add_argument("--holdout-days", type=int, default=180)
    parser.add_argument("--preset", type=str, default="paper_ready", choices=["none","robust120","robust180","quick", "paper_ready"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0); parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=3); parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--holdout-candidates", type=int, default=3,
                        help="Evaluate top-N CV candidates on holdout and pick the best")
    parser.add_argument("--require-holdout-pass", action="store_true",
                        help="Block deployment when no holdout candidate meets minimum gate")
    parser.add_argument("--holdout-min-trades", type=int, default=15)
    parser.add_argument("--holdout-min-sharpe", type=float, default=0.0)
    parser.add_argument("--holdout-min-return", type=float, default=0.0)
    parser.add_argument("--target-trades-per-week", type=float, default=1.0,
                        help="Trade-frequency objective target used during CV scoring")
    parser.add_argument("--disable-fee-stress", action="store_true",
                        help="Disable stressed-fee scoring in CV objective")
    parser.add_argument("--fee-stress-multiplier", type=float, default=2.0,
                        help="Multiplier for stressed fee schedule (applies to pct + min contract fee)")
    parser.add_argument("--fee-blend-normal-weight", type=float, default=0.6,
                        help="Blend weight for normal-fee fold Sharpe")
    parser.add_argument("--fee-blend-stressed-weight", type=float, default=0.4,
                        help="Blend weight for stressed-fee fold Sharpe")
    parser.add_argument("--pruned-only", action="store_true",
                        help="Require pruned feature artifacts during optimization and holdout")
    parser.add_argument("--allow-unpruned", action="store_false", dest="pruned_only",
                        help="Allow fallback to unpruned profile features if artifacts are missing")
    parser.add_argument("--sampler-seeds", type=str, default="")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--debug-trials", action="store_true")
    parser.set_defaults(pruned_only=True)
    args = parser.parse_args(); args = apply_runtime_preset(args)
    if args.debug_trials: DEBUG_TRIALS = True
    if args.show: show_results(); sys.exit(0)
    init_db_wal(str(_db_path()))
    all_data = load_data()
    if not all_data: print("❌ No data."); sys.exit(1)
    coins = list(dict.fromkeys(COIN_MAP.values())) if args.all else [COIN_MAP.get(args.coin.upper(), args.coin.upper())] if args.coin else []
    if not coins: parser.print_help(); sys.exit(1)
    seeds = [int(s.strip()) for s in args.sampler_seeds.split(',') if s.strip()] if args.sampler_seeds else [args.sampler_seed]
    for cn in coins:
        optimize_coin_multiseed(all_data, PREFIX_FOR_COIN.get(cn, cn), cn, sampler_seeds=seeds,
            n_trials=args.trials, n_jobs=args.jobs,
            plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup, study_suffix=args.study_suffix, resume_study=args.resume,
            holdout_days=args.holdout_days, min_internal_oos_trades=args.min_internal_oos_trades,
            min_total_trades=args.min_total_trades, n_cv_folds=args.n_cv_folds,
            holdout_candidates=args.holdout_candidates, require_holdout_pass=args.require_holdout_pass,
            holdout_min_trades=args.holdout_min_trades, holdout_min_sharpe=args.holdout_min_sharpe,
            holdout_min_return=args.holdout_min_return,
            target_trades_per_week=args.target_trades_per_week,
            enable_fee_stress=not args.disable_fee_stress,
            fee_stress_multiplier=args.fee_stress_multiplier,
            fee_blend_normal_weight=args.fee_blend_normal_weight,
            fee_blend_stressed_weight=args.fee_blend_stressed_weight,
            pruned_only=args.pruned_only,
            preset_name=args.preset)
