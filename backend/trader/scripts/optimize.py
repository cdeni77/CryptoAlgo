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
    get_contract_spec, calculate_pnl_exact,
)
from core.meta_labeling import calibrator_predict, primary_recall_threshold
from core.cv_splitters import CVFold, create_purged_embargo_splits, create_walk_forward_splits
from core.preprocessing_cv import fit_transform_fold
from core.coin_profiles import (
    CoinProfile, COIN_PROFILES, get_coin_profile,
    BTC_EXTRA_FEATURES, ETH_EXTRA_FEATURES, XRP_EXTRA_FEATURES,
    SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)
from core.metrics_significance import (
    compute_deflated_sharpe as compute_deflated_sharpe_metric,
    compute_psr_from_samples,
    evaluate_significance_gates,
)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

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


def _compute_frequency_bonus(freq_ratio: float) -> float:
    """Return a bounded soft bonus from trade-frequency ratio.

    This keeps the bonus informative for ranking while avoiding hard-veto behavior.
    """
    ratio = max(0.0, float(freq_ratio))
    floor_bonus = 0.40
    cap_bonus = 1.10
    slope = 2.0
    center = 1.0
    bonus = floor_bonus + (cap_bonus - floor_bonus) / (1.0 + math.exp(-slope * (ratio - center)))
    return float(min(cap_bonus, max(floor_bonus, bonus)))


def _set_partial_reject_attrs(trial, *, fold_results, stressed_fold_results=None, psr=None):
    stressed_fold_results = stressed_fold_results or []
    fold_trade_counts = [int(r.get('n_trades', 0) or 0) for r in (fold_results or [])]
    sharpes = [float(r.get('sharpe', 0.0) or 0.0) for r in (fold_results or [])]
    stressed_sharpes = [float(r.get('sharpe', 0.0) or 0.0) for r in stressed_fold_results]
    total_trades_partial = int(sum(fold_trade_counts))
    trial.set_user_attr('n_folds_evaluated', len(fold_trade_counts))
    trial.set_user_attr('total_trades_partial', total_trades_partial)
    trial.set_user_attr('fold_trade_counts', fold_trade_counts)
    if sharpes:
        trial.set_user_attr('mean_sr_partial', round(float(np.mean(sharpes)), 6))
        trial.set_user_attr('min_sr_partial', round(float(np.min(sharpes)), 6))
    if psr is not None:
        trial.set_user_attr('psr_partial', round(float(psr.get('psr', 0.0) or 0.0), 6))
    if stressed_sharpes:
        trial.set_user_attr('stressed_sr_partial', round(float(np.mean(stressed_sharpes)), 6))


def _reject_trial(
    trial,
    *,
    code,
    reason,
    stage,
    observed=None,
    threshold=None,
    base=-6.0,
    scale=8.0,
    fold_results=None,
    stressed_fold_results=None,
    psr=None,
    extra_attrs=None,
):
    _set_partial_reject_attrs(
        trial,
        fold_results=fold_results or [],
        stressed_fold_results=stressed_fold_results or [],
        psr=psr,
    )
    penalty = _reject_score(observed, threshold, base=base, scale=scale)
    trial.set_user_attr('reject_code', str(code))
    trial.set_user_attr('reject_reason', str(reason))
    trial.set_user_attr('reject_stage', str(stage))
    trial.set_user_attr('reject_observed', observed if observed is None else float(observed))
    trial.set_user_attr('reject_threshold', threshold if threshold is None else float(threshold))
    trial.set_user_attr('reject_penalty', float(penalty))
    if isinstance(extra_attrs, dict):
        for key, value in extra_attrs.items():
            trial.set_user_attr(str(key), value)
    return penalty


def estimate_holdout_trade_budget(holdout_days: int, target_trades_per_week: float, holdout_min_trades: int) -> tuple[float, int]:
    estimated_holdout_trades = max(0.0, float(target_trades_per_week)) * (max(0, int(holdout_days)) / 7.0)
    effective_holdout_floor = max(
        int(holdout_min_trades),
        int(estimated_holdout_trades * 0.60),
    )
    return estimated_holdout_trades, effective_holdout_floor


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
    """Backward-compatible wrapper around canonical DSR helper."""
    return compute_deflated_sharpe_metric(
        observed_sharpe=observed_sharpe,
        observations=n_trades,
        effective_test_count=n_trials,
        skewness=skewness,
        kurtosis=kurtosis,
    )


def compute_probabilistic_sharpe(sharpes: List[float], benchmark_sr: float = 0.0) -> dict:
    """Backward-compatible wrapper around canonical PSR helper."""
    return compute_psr_from_samples(sharpes, benchmark_sharpe=benchmark_sr)

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


def compute_purge_days(profile: CoinProfile) -> int:
    """Compute purge window from max holding horizon (in hours)."""
    max_hold_hours = max(1, int(getattr(profile, 'max_hold_hours', 24) or 24))
    return int(math.ceil(max_hold_hours / 24.0) + 1)

def create_cv_splits(
    data,
    target_sym,
    n_folds=3,
    min_train_days=120,
    cv_mode='walk_forward',
    purge_days=2,
    purge_bars=None,
    embargo_days=None,
    embargo_bars=None,
    embargo_frac=0.0,
):
    """Create CV folds with either walk-forward or purged+embargo behavior."""
    index = data[target_sym]['ohlcv'].index
    if cv_mode == 'purged_embargo':
        return create_purged_embargo_splits(
            index,
            n_folds=n_folds,
            min_train_days=min_train_days,
            purge_days=purge_days,
            purge_bars=purge_bars,
            embargo_days=embargo_days,
            embargo_bars=embargo_bars,
            embargo_frac=embargo_frac,
        )
    return create_walk_forward_splits(index, n_folds=n_folds, min_train_days=min_train_days, purge_days=purge_days)


# ═══════════════════════════════════════════════════════════════════════════
# v11.1: FAST LIGHTWEIGHT EVALUATOR (~2s per fold instead of ~10min)
# ═══════════════════════════════════════════════════════════════════════════

def fast_evaluate_fold(features, ohlcv, fold: CVFold, profile, config, symbol, fee_multiplier=1.0, pruned_only=True, diagnostics=None):
    """Train one model on train fold and simulate trading on the corresponding test fold."""
    def _set_diag(reason):
        if isinstance(diagnostics, dict):
            diagnostics['skip_reason'] = str(reason)

    system = MLSystem(config)
    feature_candidates = profile.resolve_feature_columns(
        use_pruned_features=bool(pruned_only),
        strict_pruned=bool(pruned_only),
    )
    cols = system.get_feature_columns(features.columns, feature_candidates)
    if not cols or len(cols) < 4:
        _set_diag('insufficient_features')
        return None

    train_feat = features.loc[fold.train_idx.intersection(features.index)]
    if len(train_feat) < config.min_train_samples:
        _set_diag('train_feat_too_small')
        return None

    y = system.create_labels(ohlcv, train_feat, profile=profile)
    valid_idx = y.dropna().index
    X_all = train_feat.loc[valid_idx, cols]
    y_all = y.loc[valid_idx]
    X_all, y_all = system.prepare_binary_training_set(X_all, y_all)
    if len(X_all) < config.min_train_samples:
        _set_diag('labeled_samples_too_small')
        return None

    split_idx = int(len(X_all) * (1 - config.val_fraction))
    if split_idx <= 0 or split_idx >= len(X_all):
        _set_diag('invalid_train_val_split')
        return None

    y_train = y_all.iloc[:split_idx]
    y_val = y_all.iloc[split_idx:]
    x_train_fold, x_val_fold, _ = fit_transform_fold(X_all.iloc[:split_idx], X_all.iloc[split_idx:], y_train)
    if y_train.nunique() < 2 or y_val.nunique() < 2:
        _set_diag('single_class_split')
        return None

    result = system.train(
        x_train_fold,
        y_train,
        x_val_fold,
        y_val,
        profile=profile,
        symbol=symbol,
    )
    if not result:
        _set_diag('model_training_failed')
        return {
            'n_trades': 0,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'ann_return': 0.0,
            'total_return': 0.0,
            'trades_per_year': 0.0,
            'avg_pnl': 0.0,
            'avg_raw_pnl': 0.0,
            'fee_edge_ratio': 0.0,
        }
    model, scaler, iso, auc, meta_artifacts, stage_metrics, member_meta = result

    # --- SIMULATE TRADING ---
    test_feat = features.loc[fold.test_idx.intersection(features.index)]
    test_ohlcv = ohlcv.loc[fold.test_idx.intersection(ohlcv.index)]
    if len(test_feat) < 50:
        _set_diag('test_window_too_small')
        return None

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
            row = test_feat.loc[ts]
            funding_hourly_bps = row.get('funding_rate_bps', 0.0)
            if pd.isna(funding_hourly_bps):
                funding_hourly_bps = 0.0
            active_pos['accum_funding'] += -(funding_hourly_bps / 10000.0) * active_pos['dir']
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
                notional = equity * profile.position_size * config.leverage
                notional_per_contract = max(units_per_contract * active_pos['entry'], 1e-9)
                n_contracts = max(1, int(notional / notional_per_contract))
                net, raw, _, _, _, _, pnl_d, _ = calculate_pnl_exact(
                    active_pos['entry'], exit_price, d,
                    active_pos['accum_funding'], n_contracts, symbol, stressed_config
                )
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
            if ret_24h * ret_72h < 0: continue

            x_in = np.nan_to_num(np.array([row.get(c, 0) for c in cols]).reshape(1, -1), nan=0.0)
            raw_prob = model.predict_proba(scaler.transform(x_in))[0, 1]
            prob = float(calibrator_predict(iso, np.array([raw_prob]))[0])
            # Match production: apply recall-oriented primary threshold.
            primary_cutoff = primary_recall_threshold(profile.signal_threshold, config.min_signal_edge)
            if prob < primary_cutoff: continue

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
                'accum_funding': 0.0,
            }

    # METRICS
    n = len(completed_trades)
    if n == 0:
        test_days = max((fold.test_end - fold.test_start).days, 1)
        return {
            'n_trades': 0,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'ann_return': 0.0,
            'total_return': 0.0,
            'trades_per_year': 0.0,
            'avg_pnl': 0.0,
            'avg_raw_pnl': 0.0,
            'fee_edge_ratio': 0.0,
        }
    if n < 3:
        pnls = [t['net_pnl'] for t in completed_trades]
        raw_pnls = [t['raw_pnl'] for t in completed_trades]
        wins = [p for p in pnls if p > 0]
        wr = len(wins) / max(len(pnls), 1)
        avg = float(np.mean(pnls))
        total_ret = equity / 100000 - 1
        test_days = max((fold.test_end - fold.test_start).days, 1)
        return {
            'n_trades': n,
            'sharpe': 0.0,
            'win_rate': round(wr, 4),
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'ann_return': 0.0,
            'total_return': round(total_ret, 4),
            'trades_per_year': round(n * 365.0 / test_days, 1),
            'avg_pnl': round(avg, 6),
            'avg_raw_pnl': round(float(np.mean(raw_pnls)), 6),
            'fee_edge_ratio': 0.0,
        }

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
    test_days = max((fold.test_end - fold.test_start).days, 1)
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
# TRIAL PROFILE (6 tunable params)
# ═══════════════════════════════════════════════════════════════════════════

FIXED_ML = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 20}
FIXED_RISK = {
    'position_size': 0.12,
    'vol_sizing_target': 0.025,
    'min_val_auc': 0.48,
    'min_vol_24h': 0.006,
    'max_vol_24h': 0.08,
    'min_momentum_magnitude': 0.04,
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
    'BTC': {'min_total_trades': 5, 'min_avg_trades_per_fold': 1.0, 'min_expectancy': 0.0},
    'ETH': {'min_total_trades': 5, 'min_avg_trades_per_fold': 1.0, 'min_expectancy': 0.0},
    'SOL': {'min_total_trades': 5, 'min_avg_trades_per_fold': 1.0, 'min_expectancy': 0.0},
    'XRP': {'min_total_trades': 5, 'min_avg_trades_per_fold': 1.0, 'min_expectancy': 0.0},
    'DOGE': {'min_total_trades': 5, 'min_avg_trades_per_fold': 1.0, 'min_expectancy': 0.0},
}

def create_trial_profile(trial, coin_name):
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    default_priors = COIN_OPTIMIZATION_PRIORS.get('ETH', {'target_trades_per_year': 60.0, 'cooldown_min': 8.0, 'cooldown_max': 36.0})
    priors = COIN_OPTIMIZATION_PRIORS.get(coin_name, default_priors)
    def clamp(v, low, high):
        return max(low, min(high, v))

    base_threshold = bp.signal_threshold if bp else 0.75
    base_fwd = bp.label_forward_hours if bp else 24
    base_label_vol = bp.label_vol_target if bp else 1.8
    base_tp = bp.vol_mult_tp if bp else 5.0
    base_sl = bp.vol_mult_sl if bp else 3.0
    base_hold = bp.max_hold_hours if bp else 72

    # Search dimensionality is intentionally constrained: we only add one
    # extra cadence parameter (min_vol_24h) while keeping momentum gating fixed.
    min_vol_floor = FIXED_RISK['min_vol_24h']

    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name],
        extra_features=get_extra_features(coin_name),
        signal_threshold=trial.suggest_float('signal_threshold', clamp(base_threshold - 0.15, 0.50, 0.75), clamp(base_threshold + 0.10, 0.62, 0.82), step=0.01),
        label_forward_hours=trial.suggest_int('label_forward_hours', int(clamp(base_fwd - 12, 12, 48)), int(clamp(base_fwd + 12, 12, 48)), step=12),
        label_vol_target=trial.suggest_float('label_vol_target', clamp(base_label_vol - 0.6, 1.0, 2.4), clamp(base_label_vol + 0.6, 1.2, 2.6), step=0.2),
        min_momentum_magnitude=FIXED_RISK['min_momentum_magnitude'],
        vol_mult_tp=trial.suggest_float('vol_mult_tp', clamp(base_tp - 2.0, 2.0, 8.0), clamp(base_tp + 2.0, 3.0, 9.0), step=0.5),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', clamp(base_sl - 1.0, 1.5, 5.0), clamp(base_sl + 1.0, 2.0, 5.5), step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', int(clamp(base_hold - 24, 24, 120)), int(clamp(base_hold + 24, 36, 132)), step=12),
        min_vol_24h=trial.suggest_float(
            'min_vol_24h',
            max(0.003, min_vol_floor - 0.002),
            min(0.012, min_vol_floor + 0.003),
            step=0.001,
        ),
        max_vol_24h=FIXED_RISK['max_vol_24h'],
        cooldown_hours=trial.suggest_float('cooldown_hours', priors['cooldown_min'], priors['cooldown_max']),
        position_size=FIXED_RISK['position_size'],
        vol_sizing_target=FIXED_RISK['vol_sizing_target'], min_val_auc=FIXED_RISK['min_val_auc'],
        n_estimators=FIXED_ML['n_estimators'], max_depth=FIXED_ML['max_depth'],
        learning_rate=FIXED_ML['learning_rate'], min_child_samples=FIXED_ML['min_child_samples'],
    )


def build_effective_params(params: Dict, coin_name: str) -> Dict:
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    base_cooldown = bp.cooldown_hours if bp and getattr(bp, 'cooldown_hours', None) is not None else 24.0
    return {
        'signal_threshold': params.get('signal_threshold', bp.signal_threshold if bp else 0.75),
        'label_forward_hours': params.get('label_forward_hours', bp.label_forward_hours if bp else 24),
        'label_vol_target': params.get('label_vol_target', bp.label_vol_target if bp else 1.8),
        'vol_mult_tp': params.get('vol_mult_tp', bp.vol_mult_tp if bp else 5.0),
        'vol_mult_sl': params.get('vol_mult_sl', bp.vol_mult_sl if bp else 3.0),
        'max_hold_hours': params.get('max_hold_hours', bp.max_hold_hours if bp else 72),
        # Tuned cadence + frozen low-impact gates
        'cooldown_hours': params.get('cooldown_hours', base_cooldown),
        'min_vol_24h': params.get('min_vol_24h', FIXED_RISK['min_vol_24h']),
        'max_vol_24h': FIXED_RISK['max_vol_24h'],
        'min_momentum_magnitude': FIXED_RISK['min_momentum_magnitude'],
        # Fixed risk/ML knobs
        'min_val_auc': FIXED_RISK['min_val_auc'],
        'position_size': FIXED_RISK['position_size'],
        'vol_sizing_target': FIXED_RISK['vol_sizing_target'],
        'n_estimators': FIXED_ML['n_estimators'],
        'max_depth': FIXED_ML['max_depth'],
        'learning_rate': FIXED_ML['learning_rate'],
        'min_child_samples': FIXED_ML['min_child_samples'],
    }

def profile_from_params(params, coin_name):
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    effective_params = build_effective_params(params, coin_name)
    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name],
        extra_features=get_extra_features(coin_name),
        signal_threshold=effective_params['signal_threshold'],
        min_val_auc=effective_params['min_val_auc'],
        label_forward_hours=effective_params['label_forward_hours'],
        label_vol_target=effective_params['label_vol_target'],
        min_momentum_magnitude=effective_params['min_momentum_magnitude'],
        vol_mult_tp=effective_params['vol_mult_tp'], vol_mult_sl=effective_params['vol_mult_sl'],
        max_hold_hours=effective_params['max_hold_hours'],
        cooldown_hours=effective_params['cooldown_hours'],
        min_vol_24h=effective_params['min_vol_24h'], max_vol_24h=effective_params['max_vol_24h'],
        position_size=effective_params['position_size'],
        vol_sizing_target=effective_params['vol_sizing_target'],
        n_estimators=effective_params['n_estimators'],
        max_depth=effective_params['max_depth'],
        learning_rate=effective_params['learning_rate'],
        min_child_samples=effective_params['min_child_samples'],
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
    target_trades_per_year=None,
    enable_fee_stress=True,
    fee_stress_multiplier=2.0,
    fee_blend_normal_weight=0.6,
    fee_blend_stressed_weight=0.4,
    pruned_only=True,
    min_total_trades_gate=0,
    min_fold_sharpe_hard=-0.1,
    min_fold_win_rate=0.30,
    min_psr=0.55,
    min_psr_cv=None,
    min_raw_expectancy=1e-6,
    min_stressed_expectancy=1e-6,
):
    min_fold_win_rate_trades = max(5, int(min_internal_oos_trades) if min_internal_oos_trades else 10)

    profile = create_trial_profile(trial, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10,
                    train_embargo_hours=24, enforce_pruned_features=bool(pruned_only),
                    min_val_auc=0.48, min_train_samples=100, signal_threshold=0.50)
    features = optim_data[target_sym]['features']
    ohlcv_data = optim_data[target_sym]['ohlcv']
    guards = COIN_OBJECTIVE_GUARDS.get(coin_name, {})
    guard_floor_trades = int(guards.get('min_total_trades', 5))
    required_total_trades_floor = max(4, int(min_total_trades_gate or 0), guard_floor_trades)
    early_trade_reject_tolerance = 0.10
    early_trade_projection_min_folds = 2

    fold_results, total_trades = [], 0
    stressed_fold_results = []
    blended_fold_sharpes = []
    fold_skip_reasons = {}
    w_normal = float(fee_blend_normal_weight)
    w_stress = float(fee_blend_stressed_weight)
    if w_normal <= 0 and w_stress <= 0:
        w_normal, w_stress = 1.0, 0.0
    w_total = w_normal + w_stress
    w_normal, w_stress = w_normal / w_total, w_stress / w_total

    for fold in cv_splits:
        fold_diag = {}
        r = fast_evaluate_fold(features, ohlcv_data, fold, profile, config,
                               target_sym, fee_multiplier=1.0, pruned_only=pruned_only, diagnostics=fold_diag)
        if r is not None:
            fold_results.append(r)
            total_trades += r['n_trades']

            if enable_fee_stress:
                stressed_r = fast_evaluate_fold(
                    features, ohlcv_data, fold,
                    profile, config, target_sym, fee_multiplier=fee_stress_multiplier,
                    pruned_only=pruned_only,
                )
                if stressed_r is None:
                    return _reject_trial(
                        trial,
                        code='MISSING_STRESSED_FOLD',
                        reason='missing_stressed_fold',
                        stage='fee_stress',
                        fold_results=fold_results,
                        stressed_fold_results=stressed_fold_results,
                    )
            else:
                stressed_r = dict(r)

            stressed_fold_results.append(stressed_r)
            blended_fold_sharpes.append((w_normal * (r['sharpe'] if r['sharpe'] > -90 else 0.0)) + (w_stress * (stressed_r['sharpe'] if stressed_r['sharpe'] > -90 else 0.0)))

            folds_observed = len(fold_results)
            if folds_observed >= early_trade_projection_min_folds:
                observed_avg_trades_per_fold = float(total_trades) / float(folds_observed)
                projected_total = observed_avg_trades_per_fold * float(len(cv_splits))
                projected_floor_with_tolerance = float(required_total_trades_floor) * (1.0 - early_trade_reject_tolerance)
                if projected_total < projected_floor_with_tolerance:
                    return _reject_trial(
                        trial,
                        code='EARLY_TRADE_STARVATION',
                        reason=(
                            f'early_trade_starvation:{projected_total:.2f}'
                            f'<{projected_floor_with_tolerance:.2f}'
                        ),
                        stage='fold_eval',
                        observed=projected_total,
                        threshold=required_total_trades_floor,
                        fold_results=fold_results,
                        stressed_fold_results=stressed_fold_results,
                        extra_attrs={
                            'observed_avg_trades_per_fold': round(observed_avg_trades_per_fold, 6),
                            'observed_trades_so_far': int(total_trades),
                            'observed_folds': int(folds_observed),
                            'projected_total_trades': round(projected_total, 6),
                            'projected_floor_with_tolerance': round(projected_floor_with_tolerance, 6),
                            'required_total_trades_floor': int(required_total_trades_floor),
                            'early_trade_reject_tolerance': float(early_trade_reject_tolerance),
                        },
                    )
        else:
            skip_reason = fold_diag.get('skip_reason', 'unknown_fold_skip')
            fold_skip_reasons[skip_reason] = int(fold_skip_reasons.get(skip_reason, 0)) + 1

    min_required_folds = max(1, len(cv_splits) // 3)
    if len(fold_results) < min_required_folds:
        return _reject_trial(
            trial,
            code='TOO_FEW_FOLDS',
            reason=f'too_few_folds:{len(fold_results)}/{len(cv_splits)}',
            stage='fold_eval',
            observed=len(fold_results),
            threshold=min_required_folds,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            extra_attrs={'fold_skip_reasons': fold_skip_reasons},
        )
    if min_internal_oos_trades > 0 and any(int(r.get('n_trades', 0) or 0) < min_internal_oos_trades for r in fold_results):
        worst_internal_trades = min(int(r.get('n_trades', 0) or 0) for r in fold_results)
        return _reject_trial(
            trial,
            code='TOO_FEW_INTERNAL_OOS_TRADES',
            reason=f'too_few_internal_oos_trades:{min_internal_oos_trades}',
            stage='fold_eval',
            observed=worst_internal_trades,
            threshold=min_internal_oos_trades,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
        )
    if total_trades < 4:
        return _reject_trial(
            trial,
            code='TOO_FEW_TRADES',
            reason=f'too_few_trades:{total_trades}',
            stage='fold_eval',
            observed=total_trades,
            threshold=4,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
        )

    sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in fold_results]
    stressed_sharpes = [r['sharpe'] if r['sharpe'] > -90 else 0.0 for r in stressed_fold_results]
    cv_psr_threshold = float(min_psr_cv) if min_psr_cv is not None else float(min_psr)
    psr = compute_psr_from_samples(
        sharpes,
        benchmark_sharpe=0.0,
        effective_observations=len(sharpes),
    )
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

    guard_min_trades = required_total_trades_floor
    guard_min_avg_tc = float(guards.get('min_avg_trades_per_fold', 1.0))
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
        return _reject_trial(
            trial,
            code='LOW_MIN_FOLD_SHARPE',
            reason=f'guard_min_fold_sharpe:{min_sr:.3f}<{min_fold_sharpe_hard:.3f}',
            stage='post_cv_gates',
            observed=min_sr,
            threshold=min_fold_sharpe_hard,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )

    low_wr_folds = [
        f for f in fold_metrics
        if f['n_trades'] >= min_fold_win_rate_trades and f['win_rate'] < min_fold_win_rate
    ]
    if low_wr_folds:
        worst_fold = min(low_wr_folds, key=lambda f: f['win_rate'])
        return _reject_trial(
            trial,
            code='LOW_FOLD_WIN_RATE',
            reason=(
                f"guard_min_fold_win_rate:fold{worst_fold['fold_idx']}="
                f"{worst_fold['win_rate']:.3f}<{min_fold_win_rate:.2f}"
                f"@trades>={min_fold_win_rate_trades}"
            ),
            stage='post_cv_gates',
            observed=worst_fold['win_rate'],
            threshold=min_fold_win_rate,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )

    if total_trades < guard_min_trades:
        return _reject_trial(
            trial,
            code='GUARD_TOTAL_TRADES',
            reason=f'guard_total_trades:{total_trades}<{guard_min_trades}',
            stage='post_cv_gates',
            observed=total_trades,
            threshold=guard_min_trades,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    if avg_tc < guard_min_avg_tc:
        return _reject_trial(
            trial,
            code='GUARD_AVG_FOLD_TRADES',
            reason=f'guard_avg_fold_trades:{avg_tc:.2f}<{guard_min_avg_tc:.2f}',
            stage='post_cv_gates',
            observed=avg_tc,
            threshold=guard_min_avg_tc,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    if mean_expectancy < guard_min_exp:
        return _reject_trial(
            trial,
            code='GUARD_EXPECTANCY',
            reason=f'guard_expectancy:{mean_expectancy:.6f}<{guard_min_exp:.6f}',
            stage='post_cv_gates',
            observed=mean_expectancy,
            threshold=guard_min_exp,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    if mean_raw_expectancy < float(min_raw_expectancy):
        return _reject_trial(
            trial,
            code='RAW_EXPECTANCY_NONPOSITIVE',
            reason=f'raw_expectancy_below_min:{mean_raw_expectancy:.6f}<{float(min_raw_expectancy):.6f}',
            stage='post_cv_gates',
            observed=mean_raw_expectancy,
            threshold=float(min_raw_expectancy),
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    if enable_fee_stress and mean_stressed_sr < 0:
        return _reject_trial(
            trial,
            code='NEG_STRESSED_SR',
            reason=f'negative_stressed_sharpe:{mean_stressed_sr:.6f}',
            stage='fee_stress',
            observed=mean_stressed_sr,
            threshold=0.0,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    if enable_fee_stress and stressed_expectancy < float(min_stressed_expectancy):
        return _reject_trial(
            trial,
            code='NONPOSITIVE_STRESSED_EXPECTANCY',
            reason=f'stressed_expectancy_below_min:{stressed_expectancy:.6f}<{float(min_stressed_expectancy):.6f}',
            stage='fee_stress',
            observed=stressed_expectancy,
            threshold=float(min_stressed_expectancy),
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )
    significance_gates = evaluate_significance_gates(
        psr_cv=psr,
        min_psr_cv=cv_psr_threshold,
    )
    if not significance_gates['psr_cv']['passed']:
        return _reject_trial(
            trial,
            code='LOW_PSR',
            reason=f'low_psr:{psr.get("psr", 0.0):.3f}<{cv_psr_threshold:.3f}',
            stage='psr',
            observed=psr.get('psr', 0.0),
            threshold=cv_psr_threshold,
            fold_results=fold_results,
            stressed_fold_results=stressed_fold_results,
            psr=psr,
        )

    robust_score = mean_blended_sr
    if std_sr < 0.3 and len(sharpes) >= 2: robust_score += 0.15
    elif std_sr > 0.8: robust_score -= 0.25
    if min_sr > 0: robust_score += 0.10
    elif min_sr < -0.5: robust_score -= 0.30
    if min(max(0, mean_pf), 5.0) > 1.2: robust_score += min(0.2, (mean_pf - 1.0) * 0.15)
    if mean_dd > 0.25: robust_score -= (mean_dd - 0.25) * 2.0
    if avg_tc < 5: robust_score -= 0.5
    elif avg_tc < 8: robust_score -= 0.2
    if mean_wr > 0.75 and total_trades < 40: robust_score -= 0.3
    if mean_fee_edge > 0.35: robust_score -= min(0.35, (mean_fee_edge - 0.35) * 0.8)
    if exp_std > 0.01: robust_score -= min(0.30, (exp_std - 0.01) * 12)
    if psr.get('valid'):
        if psr['psr'] >= 0.80:
            robust_score += 0.20
        elif psr['psr'] < 0.60:
            robust_score -= 0.20

    target_tpy = max(
        1.0,
        float(target_trades_per_year)
        if target_trades_per_year is not None
        else float(target_trades_per_week) * 52.0,
    )
    freq_ratio = (tpy / target_tpy) if target_tpy > 0 else 1.0
    frequency_bonus = _compute_frequency_bonus(freq_ratio)
    frequency_rank_adjustment = (frequency_bonus - 1.0) * 0.30
    score = robust_score + frequency_rank_adjustment

    trial.set_user_attr('n_trades', total_trades)
    trial.set_user_attr('n_folds', len(fold_results))
    trial.set_user_attr('mean_sharpe', round(mean_sr, 3))
    trial.set_user_attr('min_sharpe', round(min_sr, 3))
    trial.set_user_attr('std_sharpe', round(std_sr, 3))
    trial.set_user_attr('blended_sharpe', round(mean_blended_sr, 3))
    trial.set_user_attr('raw_sharpe_score', round(float(mean_blended_sr), 6))
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
    trial.set_user_attr('psr_meta', {
        'observations': int(psr.get('observations', 0) or 0),
        'benchmark_sharpe': float(psr.get('benchmark_sharpe', 0.0) or 0.0),
        'sharpe_estimate': float(psr.get('sharpe_estimate', 0.0) or 0.0),
        'skewness': float(psr.get('skewness', 0.0) or 0.0),
        'kurtosis': float(psr.get('kurtosis', 3.0) or 3.0),
        'fallback_moments_used': bool(psr.get('fallback_moments_used', False)),
        'sample_count': int(psr.get('sample_count', 0) or 0),
    })
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
    trial.set_user_attr('target_trades_per_year', round(float(target_tpy), 2))
    trial.set_user_attr('frequency_ratio', round(float(freq_ratio), 3))
    trial.set_user_attr('frequency_bonus', round(float(frequency_bonus), 3))
    trial.set_user_attr('raw_robust_score', round(float(robust_score), 6))
    trial.set_user_attr('frequency_adjusted_score', round(float(score), 6))
    trial.set_user_attr('fold_metrics', fold_metrics)
    trial.set_user_attr('fold_skip_reasons', fold_skip_reasons)
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
        'min_psr': float(min_psr),
        'min_psr_cv': float(cv_psr_threshold),
        'min_raw_expectancy': float(min_raw_expectancy),
        'min_stressed_expectancy': float(min_stressed_expectancy),
    })
    tunable_params = dict(trial.params)
    tunable_params['cooldown_hours'] = tunable_params.get('cooldown_hours', build_effective_params(trial.params, coin_name)['cooldown_hours'])
    trial.set_user_attr('tunable_params', tunable_params)
    trial.set_user_attr('effective_params', build_effective_params(trial.params, coin_name))
    ann_ret = np.mean([r['ann_return'] for r in fold_results])
    trial.set_user_attr('ann_return', round(ann_ret, 4))
    trial.set_user_attr('calmar', round(min(ann_ret / mean_dd if mean_dd > 0.01 else 0, 10.0), 3))

    if DEBUG_TRIALS:
        print(
            f"  T{trial.number}: raw_sharpe={mean_blended_sr:.3f} robust={robust_score:.3f} freq_adj={score:.3f} "
            f"freq_bonus={frequency_bonus:.3f} freq_rank_adj={frequency_rank_adjustment:.3f} "
            f"SR={mean_sr:.3f}±{std_sr:.3f} "
            f"min={min_sr:.3f} trades={total_trades} folds={len(fold_results)}"
        )
    logger.debug(
        "Trial %s scoring: raw_sharpe=%.4f robust_score=%.4f frequency_bonus=%.4f frequency_rank_adjustment=%.4f frequency_adjusted_score=%.4f "
        "tpy=%.2f target_tpy=%.2f",
        trial.number,
        mean_blended_sr,
        robust_score,
        frequency_bonus,
        frequency_rank_adjustment,
        score,
        tpy,
        target_tpy,
    )
    return score

class PlateauStopper:
    def __init__(
        self,
        patience=60,
        min_delta=0.02,
        warmup_trials=30,
        min_completed_trials=0,
        flatline_window=50,
        flatline_reject_ratio=0.70,
    ):
        self.patience = max(1, patience)
        self.min_delta = max(0, min_delta)
        self.warmup_trials = max(0, warmup_trials)
        self.min_completed_trials = max(0, min_completed_trials)
        self.flatline_window = max(10, int(flatline_window))
        self.flatline_reject_ratio = min(1.0, max(0.0, float(flatline_reject_ratio)))
        self.best_value = self.best_trial_number = None
        self._flatline_announced = False

    def __call__(self, study, trial):
        completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
        min_trials_required = max(self.warmup_trials, self.min_completed_trials)
        if len(completed) < min_trials_required:
            return

        recent = completed[-self.flatline_window:]
        reject_counts = {}
        for t in recent:
            code = t.user_attrs.get('reject_code')
            if code:
                reject_counts[code] = int(reject_counts.get(code, 0)) + 1
        if reject_counts:
            top_code, top_count = max(reject_counts.items(), key=lambda kv: kv[1])
            top_ratio = float(top_count) / float(max(1, len(recent)))
            if top_ratio >= self.flatline_reject_ratio and not self._flatline_announced:
                print(
                    f"\n⚠️ Flatline detector: {top_ratio:.0%} recent rejects are {top_code} "
                    f"(window={len(recent)}, best={study.best_value:.4f})."
                )
                self._flatline_announced = True

        if self.best_value is None:
            self.best_value = study.best_value
            self.best_trial_number = study.best_trial.number
            return
        if study.best_value > self.best_value + self.min_delta:
            self.best_value = study.best_value
            self.best_trial_number = study.best_trial.number
            return
        if sum(1 for t in completed if t.number > (self.best_trial_number or 0)) >= self.patience:
            print(
                f"\n🛑 Plateau: {self.patience} trials w/o improvement "
                f"(best={self.best_value:.4f}, completed={len(completed)}, "
                f"min_completed={min_trials_required})"
            )
            study.stop()

def _select_best_trial(study, min_trades=6):
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
    slices = holdout_metrics.get('holdout_slices', {}) if isinstance(holdout_metrics, dict) else {}
    slice_items = []
    if isinstance(slices, dict):
        for name, metrics in slices.items():
            if isinstance(metrics, dict):
                slice_items.append((name, metrics))
    if not slice_items:
        slice_items = [('legacy', holdout_metrics)]

    sharpe_vals = [_as_number(m.get('holdout_sharpe'), 0.0) or 0.0 for _, m in slice_items]
    return_vals = [_as_number(m.get('holdout_return'), 0.0) or 0.0 for _, m in slice_items]
    trade_vals = [int(m.get('holdout_trades', 0) or 0) for _, m in slice_items]

    median_sr = float(np.median(sharpe_vals)) if sharpe_vals else 0.0
    median_ret = float(np.median(return_vals)) if return_vals else 0.0
    median_trade_term = min(1.0, (float(np.median(trade_vals)) if trade_vals else 0.0) / 25.0)
    dispersion_penalty = 0.35 * (float(np.std(sharpe_vals)) if len(sharpe_vals) > 1 else 0.0)

    trade_min_penalty = 0.0
    for _, metrics in slice_items:
        ho_sr = _as_number(metrics.get('holdout_sharpe'), 0.0) or 0.0
        ho_ret = _as_number(metrics.get('holdout_return'), 0.0) or 0.0
        ho_trades = int(metrics.get('holdout_trades', 0) or 0)
        if ho_trades < 10:
            trade_min_penalty += 0.20
        if ho_sr <= 0:
            trade_min_penalty += 0.20
        if ho_ret <= 0:
            trade_min_penalty += 0.15

    score = (0.60 * median_sr) + (0.25 * median_ret * 5.0) + (0.15 * median_trade_term)
    score -= (trade_min_penalty + dispersion_penalty)
    return score + 0.10 * (_as_number(cv_score, 0.0) or 0.0)


def _passes_holdout_gate(
    holdout_metrics,
    min_trades=15,
    min_sharpe=0.0,
    min_return=0.0,
    min_psr_holdout=None,
    min_dsr=None,
):
    if not holdout_metrics:
        return False
    ho_trades = int(holdout_metrics.get('holdout_trades', 0) or 0)
    ho_sr = _as_number(holdout_metrics.get('holdout_sharpe'), 0.0) or 0.0
    ho_ret = _as_number(holdout_metrics.get('holdout_return'), 0.0) or 0.0

    psr_holdout = holdout_metrics.get('psr_holdout') if isinstance(holdout_metrics, dict) else None
    dsr_holdout = holdout_metrics.get('dsr_holdout') if isinstance(holdout_metrics, dict) else None
    significance = evaluate_significance_gates(
        psr_holdout=psr_holdout if isinstance(psr_holdout, dict) else None,
        dsr=dsr_holdout if isinstance(dsr_holdout, dict) else None,
        min_psr_holdout=min_psr_holdout,
        min_dsr=min_dsr,
    )

    base_gate = ho_trades >= int(min_trades) and ho_sr >= float(min_sharpe) and ho_ret >= float(min_return)
    return bool(base_gate and significance.get('all_passed', True))

def _compute_holdout_significance(holdout_metrics, *, completed_trials: int):
    if not holdout_metrics:
        return holdout_metrics
    hm = dict(holdout_metrics)
    ho_sr = _as_number(hm.get('holdout_sharpe'), 0.0) or 0.0
    ho_trades = int(hm.get('holdout_trades', 0) or 0)
    psr_holdout = compute_psr_from_samples([ho_sr], benchmark_sharpe=0.0, effective_observations=max(ho_trades, 1))
    dsr_holdout = compute_deflated_sharpe_metric(
        observed_sharpe=ho_sr,
        observations=max(ho_trades, 1),
        effective_test_count=max(1, int(completed_trials or 1)),
    )
    hm['psr_holdout'] = psr_holdout
    hm['dsr_holdout'] = dsr_holdout
    return hm

# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT (full run_backtest — called ONCE at the end)
# ═══════════════════════════════════════════════════════════════════════════

def _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days, pruned_only=True):
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00, max_ensemble_std=0.10,
                    train_embargo_hours=24, oos_eval_days=int(eval_days),
                    enforce_pruned_features=bool(pruned_only))
    try:
        result = run_backtest({target_sym: holdout_data[target_sym]}, config, profile_overrides={coin_name: profile})
    except Exception as e:
        print(f"  ❌ Holdout error ({eval_days}d): {e}")
        return None
    if not result:
        _set_diag('model_training_failed')
        return None
    oos_sr = _finite_metric(result.get('oos_sharpe', 0))
    return {
        'holdout_sharpe': oos_sr if oos_sr > -90 else 0,
        'holdout_return': _finite_metric(result.get('oos_return', 0)),
        'holdout_trades': int(result.get('oos_trades', 0) or 0),
        'full_sharpe': _finite_metric(result.get('sharpe_annual', 0)),
        'full_pf': _finite_metric(result.get('profit_factor', 0)),
        'full_dd': _finite_metric(result.get('max_drawdown', 1), 1),
        'full_trades': int(result.get('n_trades', 0) or 0),
    }


def _derive_top_level_holdout(holdout_slices, holdout_mode):
    if not holdout_slices:
        return {}
    recent = holdout_slices.get('recent90')
    if holdout_mode == 'single90' and isinstance(recent, dict):
        selected = dict(recent)
        selected['selected_slice'] = 'recent90'
        return selected

    values = [m for m in holdout_slices.values() if isinstance(m, dict)]
    if not values:
        return {}
    selected = {
        'holdout_sharpe': float(np.median([_as_number(m.get('holdout_sharpe'), 0.0) or 0.0 for m in values])),
        'holdout_return': float(np.median([_as_number(m.get('holdout_return'), 0.0) or 0.0 for m in values])),
        'holdout_trades': int(min(int(m.get('holdout_trades', 0) or 0) for m in values)),
        'full_sharpe': float(np.median([_as_number(m.get('full_sharpe'), 0.0) or 0.0 for m in values])),
        'full_pf': float(np.median([_as_number(m.get('full_pf'), 0.0) or 0.0 for m in values])),
        'full_dd': float(np.median([_as_number(m.get('full_dd'), 1.0) or 1.0 for m in values])),
        'full_trades': int(min(int(m.get('full_trades', 0) or 0) for m in values)),
        'selected_slice': 'median_composite',
    }
    return selected


def evaluate_holdout(holdout_data, params, coin_name, coin_prefix, holdout_days, pruned_only=True, holdout_mode='single90'):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym: return None
    profile = profile_from_params(params, coin_name)
    holdout_slices = {}

    recent_metrics = _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days=90, pruned_only=pruned_only)
    if recent_metrics:
        holdout_slices['recent90'] = recent_metrics

    sym_ohlcv = holdout_data[target_sym]['ohlcv']
    end_ts = sym_ohlcv.index.max()
    prior_end = end_ts - pd.Timedelta(days=90)
    prior_dataset = {
        target_sym: {
            'features': holdout_data[target_sym]['features'][holdout_data[target_sym]['features'].index <= prior_end],
            'ohlcv': sym_ohlcv[sym_ohlcv.index <= prior_end],
        }
    }
    if len(prior_dataset[target_sym]['ohlcv']) > 500:
        prior_metrics = _run_holdout_window(prior_dataset, target_sym, profile, coin_name, eval_days=90, pruned_only=pruned_only)
        if prior_metrics:
            holdout_slices['prior90'] = prior_metrics

    full_span_days = (sym_ohlcv.index.max() - sym_ohlcv.index.min()).days
    if full_span_days >= 180:
        full_metrics = _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days=180, pruned_only=pruned_only)
        if full_metrics:
            holdout_slices['full180'] = full_metrics

    top_level = _derive_top_level_holdout(holdout_slices, holdout_mode=holdout_mode)
    if not top_level:
        return None
    top_level['holdout_mode'] = holdout_mode
    top_level['holdout_slices'] = holdout_slices
    return top_level


def _proxy_fidelity_thresholds(
    sharpe_delta_max=0.35,
    trade_count_delta_max=8,
    return_delta_max=0.03,
    max_drawdown_delta_max=0.04,
):
    return {
        'sharpe_delta_max': float(max(0.0, sharpe_delta_max)),
        'trade_count_delta_max': int(max(0, trade_count_delta_max)),
        'return_delta_max': float(max(0.0, return_delta_max)),
        'max_drawdown_delta_max': float(max(0.0, max_drawdown_delta_max)),
    }


def _degrade_confidence_tier(tier: str) -> str:
    rank = ['REJECT', 'RESEARCH_ONLY', 'PAPER_READY', 'PROMOTION_READY']
    if tier not in rank:
        return tier
    idx = rank.index(tier)
    return rank[max(0, idx - 1)]


def calibrate_proxy_fidelity(
    *,
    holdout_data,
    coin_prefix,
    coin_name,
    study,
    max_candidates=3,
    min_trades=8,
    pruned_only=True,
    eval_days=90,
    thresholds=None,
):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym:
        return {'enabled': False, 'reason': 'missing_target_symbol'}

    thresholds = thresholds or _proxy_fidelity_thresholds()
    candidates = _candidate_trials_for_holdout(
        study,
        max_candidates=max(1, int(max_candidates or 1)),
        min_trades=max(0, int(min_trades or 0)),
    )
    if not candidates:
        return {
            'enabled': True,
            'reason': 'no_candidates',
            'n_candidates_requested': int(max_candidates),
            'n_candidates_evaluated': 0,
            'thresholds': thresholds,
            'summary': {},
            'samples': [],
            'warning': False,
        }

    features = holdout_data[target_sym]['features']
    ohlcv = holdout_data[target_sym]['ohlcv']
    test_end = ohlcv.index.max()
    test_start = test_end - pd.Timedelta(days=max(1, int(eval_days)))
    train_idx = features.index[features.index < test_start]
    test_idx = features.index[(features.index >= test_start) & (features.index <= test_end)]
    eval_fold = CVFold(
        train_idx=train_idx,
        test_idx=test_idx,
        train_end=train_idx.max() if len(train_idx) else test_start,
        test_start=test_start,
        test_end=test_end,
        purge_bars=0,
        embargo_bars=0,
    )

    samples = []
    for cand in candidates:
        profile = profile_from_params(cand.params, coin_name)
        config = Config(
            max_positions=1,
            leverage=4,
            min_signal_edge=0.00,
            max_ensemble_std=0.10,
            train_embargo_hours=24,
            oos_eval_days=int(eval_days),
            enforce_pruned_features=bool(pruned_only),
        )
        fast_metrics = fast_evaluate_fold(
            features,
            ohlcv,
            eval_fold,
            profile=profile,
            config=config,
            symbol=target_sym,
            pruned_only=pruned_only,
        )
        if not fast_metrics:
            continue
        backtest_metrics = _run_holdout_window(
            holdout_data,
            target_sym,
            profile,
            coin_name,
            eval_days=int(eval_days),
            pruned_only=pruned_only,
        )
        if not backtest_metrics:
            continue

        deltas = {
            'sharpe': float((_as_number(fast_metrics.get('sharpe'), 0.0) or 0.0) - (_as_number(backtest_metrics.get('holdout_sharpe'), 0.0) or 0.0)),
            'trade_count': int(fast_metrics.get('n_trades', 0) or 0) - int(backtest_metrics.get('holdout_trades', 0) or 0),
            'return': float((_as_number(fast_metrics.get('total_return'), 0.0) or 0.0) - (_as_number(backtest_metrics.get('holdout_return'), 0.0) or 0.0)),
            'max_drawdown': float((_as_number(fast_metrics.get('max_drawdown'), 0.0) or 0.0) - (_as_number(backtest_metrics.get('full_dd'), 0.0) or 0.0)),
        }
        samples.append({
            'trial': int(cand.number),
            'fast_evaluate_fold': {
                'sharpe': _finite_metric(fast_metrics.get('sharpe', 0.0), 0.0),
                'trade_count': int(fast_metrics.get('n_trades', 0) or 0),
                'return': _finite_metric(fast_metrics.get('total_return', 0.0), 0.0),
                'max_drawdown': _finite_metric(fast_metrics.get('max_drawdown', 0.0), 0.0),
            },
            'run_backtest': {
                'sharpe': _finite_metric(backtest_metrics.get('holdout_sharpe', 0.0), 0.0),
                'trade_count': int(backtest_metrics.get('holdout_trades', 0) or 0),
                'return': _finite_metric(backtest_metrics.get('holdout_return', 0.0), 0.0),
                'max_drawdown': _finite_metric(backtest_metrics.get('full_dd', 0.0), 0.0),
            },
            'delta': deltas,
            'abs_delta': {k: abs(v) for k, v in deltas.items()},
        })

    summary = {}
    if samples:
        metric_map = {
            'sharpe': 'sharpe_delta_max',
            'trade_count': 'trade_count_delta_max',
            'return': 'return_delta_max',
            'max_drawdown': 'max_drawdown_delta_max',
        }
        warning = False
        for metric, threshold_key in metric_map.items():
            vals = np.asarray([float(s['delta'][metric]) for s in samples], dtype=float)
            abs_vals = np.abs(vals)
            threshold = float(thresholds[threshold_key])
            exceeded = int(np.sum(abs_vals > threshold))
            summary[metric] = {
                'mean_delta': round(float(np.mean(vals)), 6),
                'median_delta': round(float(np.median(vals)), 6),
                'mean_abs_delta': round(float(np.mean(abs_vals)), 6),
                'max_abs_delta': round(float(np.max(abs_vals)), 6),
                'threshold': threshold,
                'exceeded_count': exceeded,
                'exceeded_ratio': round(float(exceeded / len(samples)), 6),
            }
            if exceeded > 0:
                warning = True
    else:
        warning = False

    return {
        'enabled': True,
        'eval_days': int(eval_days),
        'n_candidates_requested': int(max_candidates),
        'n_candidates_evaluated': len(samples),
        'thresholds': thresholds,
        'summary': summary,
        'samples': samples,
        'warning': bool(warning),
    }

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


def _compute_seed_stability(
    run_results: List[dict],
    *,
    holdout_min_trades: int,
    holdout_min_sharpe: float,
    holdout_min_return: float,
    min_psr_holdout: float | None = None,
    min_dsr: float | None = None,
) -> dict:
    seeds_total = len(run_results)
    if seeds_total <= 0:
        return {
            'seeds_total': 0,
            'seeds_passed_holdout': 0,
            'pass_rate': 0.0,
            'parameter_dispersion': {},
            'oos_sharpe_dispersion': {'std': 0.0, 'iqr': 0.0, 'median': 0.0},
        }

    passed = []
    for r in run_results:
        if _passes_holdout_gate(
            r.get('holdout_metrics', {}),
            min_trades=holdout_min_trades,
            min_sharpe=holdout_min_sharpe,
            min_return=holdout_min_return,
            min_psr_holdout=min_psr_holdout,
            min_dsr=min_dsr,
        ):
            passed.append(r)

    params_by_key: Dict[str, List[float]] = {}
    for result in run_results:
        params = result.get('tunable_params') or result.get('params') or {}
        if not isinstance(params, dict):
            continue
        for key, value in params.items():
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                params_by_key.setdefault(str(key), []).append(float(value))

    parameter_dispersion = {}
    for key, values in params_by_key.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            continue
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = float(q3 - q1)
        std = float(np.std(arr))
        median = float(np.median(arr))
        mean = float(np.mean(arr))
        norm_base = max(abs(median), 1e-9)
        parameter_dispersion[key] = {
            'count': int(arr.size),
            'iqr': iqr,
            'std': std,
            'median': median,
            'mean': mean,
            'iqr_over_median_abs': float(iqr / norm_base),
            'std_over_median_abs': float(std / norm_base),
        }

    oos_vals = []
    for result in run_results:
        holdout_metrics = result.get('holdout_metrics', {}) if isinstance(result, dict) else {}
        oos = _as_number(holdout_metrics.get('holdout_sharpe'), None)
        if oos is None:
            oos = _as_number(result.get('optim_metrics', {}).get('mean_oos_sharpe'), None)
        if oos is not None and np.isfinite(oos):
            oos_vals.append(float(oos))
    if oos_vals:
        oos_arr = np.asarray(oos_vals, dtype=float)
        oos_dispersion = {
            'count': int(oos_arr.size),
            'std': float(np.std(oos_arr)),
            'iqr': float(np.percentile(oos_arr, 75) - np.percentile(oos_arr, 25)),
            'median': float(np.median(oos_arr)),
            'mean': float(np.mean(oos_arr)),
        }
    else:
        oos_dispersion = {'count': 0, 'std': 0.0, 'iqr': 0.0, 'median': 0.0, 'mean': 0.0}

    return {
        'seeds_total': int(seeds_total),
        'seeds_passed_holdout': int(len(passed)),
        'pass_rate': float(len(passed) / max(1, seeds_total)),
        'parameter_dispersion': parameter_dispersion,
        'oos_sharpe_dispersion': oos_dispersion,
    }


def _derive_confidence_tier(
    *,
    holdout_passed: bool,
    seed_stability: Optional[dict],
    min_seed_pass_rate: float,
    max_seed_param_dispersion: float,
    max_seed_oos_sharpe_dispersion: float,
) -> str:
    if not holdout_passed:
        return 'SCREENED'
    if not seed_stability or int(seed_stability.get('seeds_total', 0) or 0) < 2:
        return 'PAPER_QUALIFIED'

    pass_rate = float(seed_stability.get('pass_rate', 0.0) or 0.0)
    oos_std = float(seed_stability.get('oos_sharpe_dispersion', {}).get('std', 0.0) or 0.0)
    pdisp = seed_stability.get('parameter_dispersion', {}) if isinstance(seed_stability, dict) else {}
    max_param_disp = 0.0
    if isinstance(pdisp, dict):
        for row in pdisp.values():
            if not isinstance(row, dict):
                continue
            max_param_disp = max(max_param_disp, float(row.get('iqr_over_median_abs', 0.0) or 0.0))

    if (
        pass_rate >= float(min_seed_pass_rate)
        and max_param_disp <= float(max_seed_param_dispersion)
        and oos_std <= float(max_seed_oos_sharpe_dispersion)
    ):
        return 'PROMOTION_READY'
    return 'PAPER_QUALIFIED'

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
                  plateau_min_completed=0,
                  study_suffix="", resume_study=False, holdout_days=90,
                  min_internal_oos_trades=0, min_total_trades=0, n_cv_folds=5,
                  sampler_seed=42, holdout_candidates=3, require_holdout_pass=False,
                  holdout_min_trades=15, holdout_min_sharpe=0.0, holdout_min_return=0.0,
                  holdout_mode='single90',
                  target_trades_per_week=1.0, target_trades_per_year=None,
                  preset_name="none", enable_fee_stress=True,
                  fee_stress_multiplier=2.0, fee_blend_normal_weight=0.6, fee_blend_stressed_weight=0.4,
                  min_fold_sharpe_hard=-0.1, min_fold_win_rate=0.30, min_psr=0.55,
                  min_psr_cv=None, min_psr_holdout=None, min_dsr=None,
                  min_raw_expectancy=1e-6, min_stressed_expectancy=1e-6,
                  pruned_only=True, gate_mode="initial_paper_qualification",
                  seed_stability_min_pass_rate=0.67,
                  seed_stability_max_param_dispersion=0.60,
                  seed_stability_max_oos_sharpe_dispersion=0.35,
                  proxy_fidelity_candidates=3,
                  proxy_fidelity_eval_days=90,
                  proxy_fidelity_sharpe_delta_max=0.35,
                  proxy_fidelity_trade_count_delta_max=8,
                  proxy_fidelity_return_delta_max=0.03,
                  proxy_fidelity_max_drawdown_delta_max=0.04,
                  cv_mode='walk_forward', purge_days=None, purge_bars=None, embargo_days=None,
                  embargo_bars=None, embargo_frac=0.0):
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym: print(f"❌ {coin_name}: no data after holdout split"); return None

    ohlcv = optim_data[target_sym]['ohlcv']
    optim_start, optim_end = ohlcv.index.min(), ohlcv.index.max()
    holdout_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    holdout_end = holdout_data[holdout_sym]['ohlcv'].index.max() if holdout_sym else optim_end
    active_profile = get_coin_profile(coin_name)
    purge_days = int(purge_days if purge_days is not None else compute_purge_days(active_profile))
    cv_splits = create_cv_splits(
        optim_data,
        target_sym,
        n_folds=n_cv_folds,
        cv_mode=cv_mode,
        purge_days=purge_days,
        purge_bars=purge_bars,
        embargo_days=embargo_days,
        embargo_bars=embargo_bars,
        embargo_frac=embargo_frac,
    )

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} — v11.3 FAST CV")
    print(f"   Selected config: n_cv_folds={n_cv_folds}, holdout_days={holdout_days}, holdout_mode={holdout_mode}, cv_mode={cv_mode}")
    print(f"   Optim: {optim_start.date()} → {optim_end.date()} | Holdout: last {holdout_days}d (→{holdout_end.date()})")
    print(f"   CV folds: {len(cv_splits)} | Purge: {purge_days}d (max_hold={active_profile.max_hold_hours}h) | Params: 6 tunable | Trials: {n_trials} | Jobs: {n_jobs}")
    print(
        f"   Gates: fold_sr>={min_fold_sharpe_hard:.2f}, fold_wr>={min_fold_win_rate:.2f}, "
        f"psr>={min_psr:.2f}, raw_exp>={min_raw_expectancy:.5f}"
    )
    for i, fold in enumerate(cv_splits):
        print(
            f"     Fold {i}: train_n={len(fold.train_idx)} test_n={len(fold.test_idx)} "
            f"| train_end={fold.train_end.date()} | test {fold.test_start.date()}→{fold.test_end.date()} "
            f"| purge_bars={fold.purge_bars} embargo_bars={fold.embargo_bars}"
        )
        logger.debug(
            "Fold %s (%s): train_end=%s test=[%s,%s] train_n=%s test_n=%s purge_bars=%s embargo_bars=%s",
            i,
            coin_name,
            fold.train_end,
            fold.test_start,
            fold.test_end,
            len(fold.train_idx),
            len(fold.test_idx),
            fold.purge_bars,
            fold.embargo_bars,
        )
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
                    load_if_exists=bool(resume_study),
                )
            break
        except Exception as e:
            err = str(e).lower()
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in err:
                if resume_study:
                    with _study_storage_lock(_db_path()):
                        study = optuna.load_study(study_name=study_name, storage=storage_url, sampler=sampler)
                    break
                raise RuntimeError(
                    f"Study '{study_name}' already exists but --resume was not set. "
                    "Pass --resume to continue the existing study or change --study-suffix."
                )
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

    study.set_user_attr('sampler_seed', int(sampler_seed))
    study.set_user_attr('study_name', study_name)
    study.set_user_attr('run_id', study_suffix or '')
    study.set_user_attr('target_trades_per_week', float(target_trades_per_week))
    study.set_user_attr('target_trades_per_year', float(target_trades_per_year) if target_trades_per_year is not None else float(target_trades_per_week) * 52.0)
    study.set_user_attr('gate_mode', str(gate_mode))
    study.set_user_attr('fee_stress_enabled', bool(enable_fee_stress))

    obj = functools.partial(objective, optim_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, cv_splits=cv_splits, target_sym=target_sym,
                            min_internal_oos_trades=min_internal_oos_trades,
                            min_total_trades_gate=min_total_trades,
                            target_trades_per_week=target_trades_per_week,
                            target_trades_per_year=target_trades_per_year,
                            enable_fee_stress=enable_fee_stress,
                            fee_stress_multiplier=fee_stress_multiplier,
                            fee_blend_normal_weight=fee_blend_normal_weight,
                            fee_blend_stressed_weight=fee_blend_stressed_weight,
                            min_fold_sharpe_hard=min_fold_sharpe_hard,
                            min_fold_win_rate=min_fold_win_rate,
                            min_psr=min_psr,
                            min_psr_cv=min_psr_cv,
                            min_raw_expectancy=min_raw_expectancy,
                            min_stressed_expectancy=min_stressed_expectancy,
                            pruned_only=pruned_only)
    min_completed_trials = max(int(plateau_min_completed or 0), int(max(1, n_trials) * 0.40))
    try: study.optimize(obj, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=sys.stderr.isatty(),
                        callbacks=[PlateauStopper(
                            plateau_patience,
                            plateau_min_delta,
                            plateau_warmup,
                            min_completed_trials,
                        )])
    except KeyboardInterrupt: print("\n🛑 Stopped.")
    except Exception as e: print(f"\n❌ {e}"); traceback.print_exc(); return None
    if not study.trials: print("No trials."); return None

    best = _select_best_trial(study, min_trades=min_total_trades or 6)
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
    est_ho_trades, effective_holdout_min_trades = estimate_holdout_trade_budget(
        holdout_days=holdout_days,
        target_trades_per_week=max(
            0.1,
            float(target_trades_per_year) / 52.0
            if target_trades_per_year is not None
            else float(target_trades_per_week),
        ),
        holdout_min_trades=holdout_min_trades,
    )
    if est_ho_trades < effective_holdout_min_trades:
        print(
            f"⚠️  Trade-starved holdout risk: est_holdout_trades={est_ho_trades:.1f} "
            f"< gate_floor={effective_holdout_min_trades}"
        )
    if holdout_data and holdout_sym:
        candidate_trials = _candidate_trials_for_holdout(
            study,
            max_candidates=max(1, int(holdout_candidates or 1)),
            min_trades=min_total_trades or 6,
        )
        if best not in candidate_trials:
            candidate_trials = [best] + candidate_trials

        print(f"\n🔬 HOLDOUT ({holdout_days}d, full walk-forward) — evaluating {len(candidate_trials)} candidate(s)...")
        ranked_candidates = []
        for cand in candidate_trials:
            holdout_metrics = evaluate_holdout(holdout_data, cand.params, coin_name, coin_prefix, holdout_days,
                                               pruned_only=pruned_only, holdout_mode=holdout_mode)
            holdout_metrics = _compute_holdout_significance(holdout_metrics, completed_trials=nc)
            if not holdout_metrics:
                continue
            sel_score = _holdout_selection_score(holdout_metrics, cv_score=cand.value)
            ranked_candidates.append((cand, holdout_metrics, sel_score))
            print(
                f"  Trial #{cand.number}: sel={sel_score:.3f} "
                f"SR={_fmt_float(holdout_metrics['holdout_sharpe'])} "
                f"Ret={_fmt_pct(holdout_metrics['holdout_return'],2)} Trades={holdout_metrics['holdout_trades']}"
            )
            slice_parts = []
            for slice_name in ('recent90', 'prior90', 'full180'):
                sm = holdout_metrics.get('holdout_slices', {}).get(slice_name)
                if not sm:
                    continue
                slice_parts.append(
                    f"{slice_name}:SR={_fmt_float(sm.get('holdout_sharpe'))},"
                    f"Ret={_fmt_pct(sm.get('holdout_return'),2)},T={int(sm.get('holdout_trades', 0) or 0)}"
                )
            if slice_parts:
                print(f"     slices[{holdout_mode}] -> " + " | ".join(slice_parts))

        if ranked_candidates:
            ranked_candidates.sort(key=lambda x: x[2], reverse=True)
            passing_candidates = [
                (c, m, sc) for c, m, sc in ranked_candidates
                if _passes_holdout_gate(
                    m,
                    min_trades=effective_holdout_min_trades,
                    min_sharpe=holdout_min_sharpe,
                    min_return=holdout_min_return,
                    min_psr_holdout=min_psr_holdout,
                    min_dsr=min_dsr,
                )
            ]
            selected_pool = passing_candidates if require_holdout_pass else ranked_candidates
            selected_trial, holdout_result, selected_score = selected_pool[0] if selected_pool else (None, None, None)
            if require_holdout_pass and not passing_candidates:
                deployment_blocked = True
                blocked_reasons.append(
                    f"holdout_gate_failed:trades>={effective_holdout_min_trades},sr>={holdout_min_sharpe},ret>={holdout_min_return},psr>={min_psr_holdout},dsr>={min_dsr}"
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
                'holdout_mode': holdout_mode,
                'require_holdout_pass': bool(require_holdout_pass),
                'holdout_gate': {
                    'min_trades': int(holdout_min_trades),
                    'effective_min_trades': int(effective_holdout_min_trades),
                    'min_sharpe': float(holdout_min_sharpe),
                    'min_return': float(holdout_min_return),
                    'min_psr_holdout': None if min_psr_holdout is None else float(min_psr_holdout),
                    'min_dsr': None if min_dsr is None else float(min_dsr),
                },
                'deployment_blocked': bool(deployment_blocked),
                'candidates': [
                    {
                        'trial': int(c.number),
                        'selection_score': round(float(sc), 6),
                        'holdout_sharpe': round(float(m.get('holdout_sharpe', 0.0) or 0.0), 6),
                        'holdout_return': round(float(m.get('holdout_return', 0.0) or 0.0), 6),
                        'holdout_trades': int(m.get('holdout_trades', 0) or 0),
                        'holdout_slices': m.get('holdout_slices', {}),
                        'passes_gate': _passes_holdout_gate(
                            m,
                            min_trades=effective_holdout_min_trades,
                            min_sharpe=holdout_min_sharpe,
                            min_return=holdout_min_return,
                            min_psr_holdout=min_psr_holdout,
                            min_dsr=min_dsr,
                        ),
                    }
                    for c, m, sc in ranked_candidates
                ],
            }
            h = holdout_result
            print(f"  ✅ Selected holdout[{holdout_mode}]: SR={_fmt_float(h['holdout_sharpe'])} Ret={_fmt_pct(h['holdout_return'],2)} "
                  f"Trades={h['holdout_trades']} | Full: SR={_fmt_float(h['full_sharpe'])} "
                  f"PF={_fmt_float(h['full_pf'])} DD={_fmt_pct(h['full_dd'],2)}")
            if h.get('holdout_slices'):
                print("     slice outcomes: " + ", ".join(
                    f"{k}(SR={_fmt_float(v.get('holdout_sharpe'))},Ret={_fmt_pct(v.get('holdout_return'),2)},T={int(v.get('holdout_trades',0) or 0)})"
                    for k, v in h['holdout_slices'].items() if isinstance(v, dict)
                ))
        else:
            print("  ⚠️ Holdout evaluation returned no valid candidates.")

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    reject_reason_counts = {}
    reject_code_counts = {}
    accepted_trials = 0
    for trial_item in completed_trials:
        reject_reason = trial_item.user_attrs.get('reject_reason')
        reject_code = trial_item.user_attrs.get('reject_code')
        if reject_reason:
            reject_reason_counts[reject_reason] = int(reject_reason_counts.get(reject_reason, 0)) + 1
        if reject_code:
            reject_code_counts[reject_code] = int(reject_code_counts.get(reject_code, 0)) + 1
        if not reject_reason and not reject_code:
            accepted_trials += 1

    effective_best_params = build_effective_params(best.params, coin_name)
    tunable_best_params = dict(best.params)
    tunable_best_params['cooldown_hours'] = tunable_best_params.get('cooldown_hours', effective_best_params['cooldown_hours'])
    holdout_passed = _passes_holdout_gate(
        holdout_result or {},
        min_trades=effective_holdout_min_trades,
        min_sharpe=holdout_min_sharpe,
        min_return=holdout_min_return,
        min_psr_holdout=min_psr_holdout,
        min_dsr=min_dsr,
    ) if holdout_result else False
    cv_psr_metric = {
        'valid': bool(best.user_attrs.get('psr_meta')),
        'psr': _as_number(best.user_attrs.get('psr'), 0.0) or 0.0,
    }
    holdout_psr_metric = (holdout_result or {}).get('psr_holdout') if isinstance(holdout_result, dict) else None
    significance_gates = evaluate_significance_gates(
        psr_cv=cv_psr_metric,
        psr_holdout=holdout_psr_metric if isinstance(holdout_psr_metric, dict) else None,
        dsr=dsr,
        min_psr_cv=min_psr_cv if min_psr_cv is not None else min_psr,
        min_psr_holdout=min_psr_holdout,
        min_dsr=min_dsr,
    )
    seed_stability = {
        'seeds_total': 1,
        'seeds_passed_holdout': 1 if holdout_passed else 0,
        'pass_rate': 1.0 if holdout_passed else 0.0,
        'parameter_dispersion': {},
        'oos_sharpe_dispersion': {
            'count': 1 if holdout_result else 0,
            'std': 0.0,
            'iqr': 0.0,
            'median': _finite_metric((holdout_result or {}).get('holdout_sharpe', 0.0), 0.0),
            'mean': _finite_metric((holdout_result or {}).get('holdout_sharpe', 0.0), 0.0),
        },
    }
    research_confidence_tier = _derive_confidence_tier(
        holdout_passed=bool(holdout_passed),
        seed_stability=seed_stability,
        min_seed_pass_rate=seed_stability_min_pass_rate,
        max_seed_param_dispersion=seed_stability_max_param_dispersion,
        max_seed_oos_sharpe_dispersion=seed_stability_max_oos_sharpe_dispersion,
    )
    selection_meta = dict(selection_meta)
    proxy_fidelity = calibrate_proxy_fidelity(
        holdout_data=holdout_data,
        coin_prefix=coin_prefix,
        coin_name=coin_name,
        study=study,
        max_candidates=proxy_fidelity_candidates,
        min_trades=min_total_trades or 6,
        pruned_only=pruned_only,
        eval_days=proxy_fidelity_eval_days,
        thresholds=_proxy_fidelity_thresholds(
            sharpe_delta_max=proxy_fidelity_sharpe_delta_max,
            trade_count_delta_max=proxy_fidelity_trade_count_delta_max,
            return_delta_max=proxy_fidelity_return_delta_max,
            max_drawdown_delta_max=proxy_fidelity_max_drawdown_delta_max,
        ),
    )
    proxy_fidelity_warning = bool(proxy_fidelity.get('warning', False))
    selection_meta['proxy_fidelity_warning'] = proxy_fidelity_warning
    selection_meta['proxy_fidelity'] = {
        'n_candidates_evaluated': int(proxy_fidelity.get('n_candidates_evaluated', 0) or 0),
        'eval_days': int(proxy_fidelity.get('eval_days', proxy_fidelity_eval_days) or proxy_fidelity_eval_days),
    }

    selection_meta['research_confidence_tier'] = research_confidence_tier
    if proxy_fidelity_warning:
        prior_tier = research_confidence_tier
        research_confidence_tier = _degrade_confidence_tier(research_confidence_tier)
        selection_meta['research_confidence_tier'] = research_confidence_tier
        selection_meta['proxy_fidelity_tier_adjustment'] = {
            'applied': True,
            'previous_tier': prior_tier,
            'new_tier': research_confidence_tier,
        }

    selection_meta['seed_stability'] = seed_stability
    selection_meta['seed_stability_thresholds'] = {
        'min_pass_rate': float(seed_stability_min_pass_rate),
        'max_param_dispersion': float(seed_stability_max_param_dispersion),
        'max_oos_sharpe_dispersion': float(seed_stability_max_oos_sharpe_dispersion),
    }
    result_data = {'coin': coin_name, 'prefix': coin_prefix, 'optim_score': best.value,
        'optim_metrics': dict(best.user_attrs), 'holdout_metrics': holdout_result or {},
        'params': effective_best_params,
        'tunable_params': tunable_best_params,
        'n_trials': len(study.trials), 'n_cv_folds': len(cv_splits),
        'holdout_days': holdout_days, 'holdout_mode': holdout_mode,
        'cv_mode': cv_mode,
        'cv_controls': {
            'purge_days': int(purge_days),
            'purge_bars': None if purge_bars is None else int(purge_bars),
            'embargo_days': None if embargo_days is None else int(embargo_days),
            'embargo_bars': None if embargo_bars is None else int(embargo_bars),
            'embargo_frac': float(embargo_frac or 0.0),
        },
        'deflated_sharpe': dsr, 'psr_cv': cv_psr_metric, 'psr_holdout': holdout_psr_metric or {},
        'significance_gates': significance_gates, 'version': 'v11.3', 'timestamp': datetime.now().isoformat(),
        'fee_stress': {
            'enabled': bool(enable_fee_stress),
            'multiplier': float(fee_stress_multiplier),
            'blend_normal_weight': float(fee_blend_normal_weight),
            'blend_stressed_weight': float(fee_blend_stressed_weight),
        },
        'selection_meta': selection_meta, 'deployment_blocked': deployment_blocked,
        'deployment_block_reasons': blocked_reasons,
        'proxy_fidelity': proxy_fidelity,
        'proxy_fidelity_warning': proxy_fidelity_warning,
        'seed_stability': seed_stability,
        'research_confidence_tier': research_confidence_tier,
        'gate_mode': gate_mode,
        'gate_profile': {
            'mode': gate_mode,
            'screen_threshold': float(resolve_gate_mode(gate_mode).get('screen_threshold', 38.0)),
            'holdout_min_trades': int(holdout_min_trades),
            'holdout_min_sharpe': float(holdout_min_sharpe),
            'holdout_min_return': float(holdout_min_return),
            'min_psr_cv': float(min_psr_cv) if min_psr_cv is not None else float(min_psr),
            'min_psr_holdout': None if min_psr_holdout is None else float(min_psr_holdout),
            'min_dsr': None if min_dsr is None else float(min_dsr),
            'escalation_policy': {
                'window_days': '14-28',
                'action': 'Tighten thresholds and switch to production_promotion after stable paper evidence.',
            },
        },
        'pruned_only': bool(pruned_only),
        'run_id': run_id,
        'trial_ledger': {
            'ledger_path': cumulative_trials.get('ledger_path'),
            'ledger_timestamp': cumulative_trials.get('ledger_timestamp'),
            'coin_cumulative_trials': cumulative_trials.get('coin_totals', {}).get(coin_name, 0),
            'global_cumulative_trials': cumulative_trials.get('global_total', 0),
            'completed_trials_this_run': nc,
            'preset': preset_name,
        },
        'optimization_diagnostics': {
            'completed_trials': len(completed_trials),
            'failed_trials': len(failed_trials),
            'pruned_trials': len(pruned_trials),
            'accepted_trials': int(accepted_trials),
            'reject_reason_counts': reject_reason_counts,
            'reject_code_counts': reject_code_counts,
        }}
    result_data['quality'] = assess_result_quality(result_data)
    print(f"  🧪 Quality: {result_data['quality']['rating']}")
    print(
        "  🧭 Research tier: "
        f"{research_confidence_tier} "
        f"(seed pass {seed_stability.get('seeds_passed_holdout', 0)}/{seed_stability.get('seeds_total', 0)})"
    )
    p = _persist_result_json(coin_name, result_data)
    if p: print(f"  💾 {p}")

    print(f"\n  📝 CoinProfile(name='{coin_name}',")
    for k, v in sorted(effective_best_params.items()):
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

    holdout_min_trades = int(kwargs.get('holdout_min_trades', 15) or 15)
    holdout_min_sharpe = float(kwargs.get('holdout_min_sharpe', 0.0) or 0.0)
    holdout_min_return = float(kwargs.get('holdout_min_return', 0.0) or 0.0)
    min_psr_holdout = kwargs.get('min_psr_holdout', None)
    min_dsr = kwargs.get('min_dsr', None)
    min_seed_pass_rate = float(kwargs.get('seed_stability_min_pass_rate', 0.67) or 0.67)
    max_seed_param_dispersion = float(kwargs.get('seed_stability_max_param_dispersion', 0.60) or 0.60)
    max_seed_oos_sharpe_dispersion = float(kwargs.get('seed_stability_max_oos_sharpe_dispersion', 0.35) or 0.35)

    seed_stability = _compute_seed_stability(
        run_results,
        holdout_min_trades=holdout_min_trades,
        holdout_min_sharpe=holdout_min_sharpe,
        holdout_min_return=holdout_min_return,
        min_psr_holdout=min_psr_holdout,
        min_dsr=min_dsr,
    )

    qualified = [
        r for r in run_results
        if not r.get('deployment_blocked', False)
        and _passes_holdout_gate(
            r.get('holdout_metrics', {}),
            min_trades=holdout_min_trades,
            min_sharpe=holdout_min_sharpe,
            min_return=holdout_min_return,
            min_psr_holdout=min_psr_holdout,
            min_dsr=min_dsr,
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
    holdout_passed = _passes_holdout_gate(
        best_seed_result.get('holdout_metrics', {}),
        min_trades=holdout_min_trades,
        min_sharpe=holdout_min_sharpe,
        min_return=holdout_min_return,
    )
    research_confidence_tier = _derive_confidence_tier(
        holdout_passed=bool(holdout_passed),
        seed_stability=seed_stability,
        min_seed_pass_rate=min_seed_pass_rate,
        max_seed_param_dispersion=max_seed_param_dispersion,
        max_seed_oos_sharpe_dispersion=max_seed_oos_sharpe_dispersion,
    )
    if bool(best_seed_result.get('proxy_fidelity_warning', False)):
        research_confidence_tier = _degrade_confidence_tier(research_confidence_tier)
    best_seed_result['seed_stability'] = seed_stability
    best_seed_result['research_confidence_tier'] = research_confidence_tier
    best_seed_result.setdefault('selection_meta', {})['research_confidence_tier'] = research_confidence_tier
    best_seed_result['selection_meta']['seed_stability_thresholds'] = {
        'min_pass_rate': float(min_seed_pass_rate),
        'max_param_dispersion': float(max_seed_param_dispersion),
        'max_oos_sharpe_dispersion': float(max_seed_oos_sharpe_dispersion),
    }
    best_seed_result['selection_meta']['seed_stability'] = seed_stability
    print(
        "  🌱 Seed stability: "
        f"pass_rate={seed_stability.get('pass_rate', 0.0):.0%} "
        f"({seed_stability.get('seeds_passed_holdout', 0)}/{seed_stability.get('seeds_total', 0)}) | "
        f"oos_std={seed_stability.get('oos_sharpe_dispersion', {}).get('std', 0.0):.3f} | "
        f"tier={research_confidence_tier}"
    )

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


GATE_MODE_CONFIGS = {
    'initial_paper_qualification': {
        'description': 'Lenient gate to allow controlled paper qualification of borderline models.',
        'screen_threshold': 38.0,
        'holdout_min_trades': 8,
        'holdout_min_sharpe': -0.1,
        'holdout_min_return': -0.05,
    },
    'production_promotion': {
        'description': 'Stricter gate for production promotion after paper evidence is proven.',
        'screen_threshold': 60.0,
        'holdout_min_trades': 15,
        'holdout_min_sharpe': 0.0,
        'holdout_min_return': 0.0,
    },
}


def resolve_gate_mode(mode_name: str) -> Dict[str, object]:
    return dict(GATE_MODE_CONFIGS.get(mode_name, GATE_MODE_CONFIGS['initial_paper_qualification']))

def apply_runtime_preset(args):
    presets = {
        'robust180': {'plateau_patience': 120, 'plateau_warmup': 60, 'plateau_min_delta': 0.015, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_internal_oos_trades': 8, 'min_total_trades': 20, 'n_cv_folds': 5, 'holdout_candidates': 3, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.67, 'seed_stability_max_param_dispersion': 0.60, 'seed_stability_max_oos_sharpe_dispersion': 0.35, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
        'robust120': {'plateau_patience': 90, 'plateau_warmup': 45, 'plateau_min_delta': 0.015, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_internal_oos_trades': 6, 'min_total_trades': 15, 'n_cv_folds': 5, 'holdout_candidates': 2, 'holdout_min_trades': 12, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.60, 'seed_stability_max_param_dispersion': 0.70, 'seed_stability_max_oos_sharpe_dispersion': 0.40, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
        'quick':     {'plateau_patience': 45, 'plateau_warmup': 20, 'plateau_min_delta': 0.03, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'single90', 'min_internal_oos_trades': 0, 'min_total_trades': 8, 'n_cv_folds': 2, 'holdout_candidates': 1, 'holdout_min_trades': 8, 'holdout_min_sharpe': -0.1, 'holdout_min_return': -0.05, 'require_holdout_pass': False, 'target_trades_per_week': 0.8, 'disable_fee_stress': True, 'min_fold_sharpe_hard': -0.5, 'min_fold_win_rate': 0.30, 'min_psr': 0.05, 'min_psr_cv': 0.05, 'min_psr_holdout': None, 'min_dsr': None, 'min_raw_expectancy': -0.0010, 'min_stressed_expectancy': -0.0010, 'seed_stability_min_pass_rate': 0.50, 'seed_stability_max_param_dispersion': 1.00, 'seed_stability_max_oos_sharpe_dispersion': 0.80},
        'paper_ready': {'plateau_patience': 150, 'plateau_warmup': 80, 'plateau_min_delta': 0.012, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_internal_oos_trades': 10, 'min_total_trades': 28, 'n_cv_folds': 5, 'holdout_candidates': 4, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.05, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.75, 'seed_stability_max_param_dispersion': 0.50, 'seed_stability_max_oos_sharpe_dispersion': 0.30, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
    }
    name = getattr(args, 'preset', 'none')
    if name in (None, '', 'none'): return args
    cfg = presets.get(name)
    if cfg:
        arg_flags = {
            'plateau_patience': '--plateau-patience',
            'plateau_warmup': '--plateau-warmup',
            'plateau_min_delta': '--plateau-min-delta',
            'plateau_min_completed': '--plateau-min-completed',
            'holdout_days': '--holdout-days',
            'holdout_mode': '--holdout-mode',
            'min_internal_oos_trades': '--min-internal-oos-trades',
            'min_total_trades': '--min-total-trades',
            'n_cv_folds': '--n-cv-folds',
            'cv_mode': '--cv-mode',
            'purge_days': '--purge-days',
            'purge_bars': '--purge-bars',
            'embargo_days': '--embargo-days',
            'embargo_bars': '--embargo-bars',
            'embargo_frac': '--embargo-frac',
            'holdout_candidates': '--holdout-candidates',
            'holdout_min_trades': '--holdout-min-trades',
            'holdout_min_sharpe': '--holdout-min-sharpe',
            'holdout_min_return': '--holdout-min-return',
            'require_holdout_pass': '--require-holdout-pass',
            'target_trades_per_week': '--target-trades-per-week',
            'target_trades_per_year': '--target-trades-per-year',
            'disable_fee_stress': '--disable-fee-stress',
            'min_fold_sharpe_hard': '--min-fold-sharpe-hard',
            'min_fold_win_rate': '--min-fold-win-rate',
            'min_psr': '--min-psr',
            'min_psr_cv': '--min-psr-cv',
            'min_psr_holdout': '--min-psr-holdout',
            'min_dsr': '--min-dsr',
            'min_raw_expectancy': '--min-raw-expectancy',
            'min_stressed_expectancy': '--min-stressed-expectancy',
            'seed_stability_min_pass_rate': '--seed-stability-min-pass-rate',
            'seed_stability_max_param_dispersion': '--seed-stability-max-param-dispersion',
            'seed_stability_max_oos_sharpe_dispersion': '--seed-stability-max-oos-sharpe-dispersion',
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
    parser.add_argument("--jobs", type=int, default=1); parser.add_argument("--plateau-patience", type=int, default=120)
    parser.add_argument("--plateau-min-delta", type=float, default=0.015); parser.add_argument("--plateau-warmup", type=int, default=60)
    parser.add_argument("--plateau-min-completed", type=int, default=0,
                        help="Never plateau-stop before this many completed trials (0 = auto 40%% of n_trials)")
    parser.add_argument("--holdout-days", type=int, default=90)
    parser.add_argument("--holdout-mode", type=str, default="single90", choices=["single90", "multi_slice"],
                        help="Holdout aggregation mode for selection + backward-compatible top-level metrics")
    parser.add_argument("--preset", type=str, default="paper_ready", choices=["none","robust120","robust180","quick", "paper_ready"])
    parser.add_argument("--min-internal-oos-trades", type=int, default=0); parser.add_argument("--min-total-trades", type=int, default=0)
    parser.add_argument("--n-cv-folds", type=int, default=5); parser.add_argument("--study-suffix", type=str, default="")
    parser.add_argument("--cv-mode", type=str, default="walk_forward", choices=["walk_forward", "purged_embargo"],
                        help="Cross-validation split mode")
    parser.add_argument("--purge-days", type=int, default=None,
                        help="Purge window in days for CV (defaults to max-hold-derived value)")
    parser.add_argument("--purge-bars", type=int, default=None,
                        help="Purge window in bars (overrides --purge-days)")
    parser.add_argument("--embargo-days", type=int, default=None,
                        help="Embargo window in days for purged_embargo CV")
    parser.add_argument("--embargo-bars", type=int, default=None,
                        help="Embargo window in bars (overrides --embargo-days)")
    parser.add_argument("--embargo-frac", type=float, default=0.0,
                        help="Embargo as a fraction of each test fold length")
    parser.add_argument("--sampler-seed", type=int, default=42)
    parser.add_argument("--holdout-candidates", type=int, default=3,
                        help="Evaluate top-N CV candidates on holdout and pick the best")
    parser.add_argument("--require-holdout-pass", action="store_true",
                        help="Block deployment when no holdout candidate meets minimum gate")
    parser.add_argument("--holdout-min-trades", type=int, default=8)
    parser.add_argument("--holdout-min-sharpe", type=float, default=-0.1)
    parser.add_argument("--holdout-min-return", type=float, default=-0.05)
    parser.add_argument("--gate-mode", type=str, default="initial_paper_qualification",
                        choices=sorted(GATE_MODE_CONFIGS.keys()),
                        help="Gate profile: initial paper qualification (lenient) vs production promotion (strict)")
    parser.add_argument("--target-trades-per-week", type=float, default=1.0,
                        help="Trade-frequency objective target used during CV scoring (default: 1.0 ~= 52 trades/year)")
    parser.add_argument("--target-trades-per-year", type=float, default=None,
                        help="Optional annual target; overrides --target-trades-per-week when set")
    parser.add_argument("--disable-fee-stress", action="store_true",
                        help="Disable stressed-fee scoring in CV objective")
    parser.add_argument("--fee-stress-multiplier", type=float, default=2.0,
                        help="Multiplier for stressed fee schedule (applies to pct + min contract fee)")
    parser.add_argument("--fee-blend-normal-weight", type=float, default=0.6,
                        help="Blend weight for normal-fee fold Sharpe")
    parser.add_argument("--fee-blend-stressed-weight", type=float, default=0.4,
                        help="Blend weight for stressed-fee fold Sharpe")
    parser.add_argument("--min-fold-sharpe-hard", type=float, default=-0.1,
                        help="Hard reject if any fold Sharpe is below this threshold")
    parser.add_argument("--min-fold-win-rate", type=float, default=0.30,
                        help="Hard reject when sufficiently-active folds fall below this win-rate")
    parser.add_argument("--min-psr", type=float, default=0.55,
                        help="Minimum probabilistic Sharpe ratio gate")
    parser.add_argument("--min-psr-cv", type=float, default=None,
                        help="Optional CV PSR gate override (defaults to --min-psr)")
    parser.add_argument("--min-psr-holdout", type=float, default=None,
                        help="Optional holdout PSR gate (disabled when unset)")
    parser.add_argument("--min-dsr", type=float, default=None,
                        help="Optional holdout DSR gate (disabled when unset)")
    parser.add_argument("--min-raw-expectancy", type=float, default=1e-6,
                        help="Minimum pre-fee expectancy gate")
    parser.add_argument("--min-stressed-expectancy", type=float, default=1e-6,
                        help="Minimum stressed-fee expectancy gate")
    parser.add_argument("--seed-stability-min-pass-rate", type=float, default=0.67,
                        help="Minimum holdout pass-rate across sampler seeds for PROMOTION_READY")
    parser.add_argument("--seed-stability-max-param-dispersion", type=float, default=0.60,
                        help="Maximum normalized (IQR/|median|) per-parameter seed dispersion for PROMOTION_READY")
    parser.add_argument("--seed-stability-max-oos-sharpe-dispersion", type=float, default=0.35,
                        help="Maximum holdout Sharpe std across seeds for PROMOTION_READY")
    parser.add_argument("--proxy-fidelity-candidates", type=int, default=3,
                        help="How many accepted candidates to sample for fast-vs-backtest calibration")
    parser.add_argument("--proxy-fidelity-eval-days", type=int, default=90,
                        help="Calibration evaluation horizon (days) shared by fast proxy and run_backtest")
    parser.add_argument("--proxy-fidelity-sharpe-delta-max", type=float, default=0.35,
                        help="Warning threshold for |fast_sharpe - backtest_sharpe|")
    parser.add_argument("--proxy-fidelity-trade-count-delta-max", type=int, default=8,
                        help="Warning threshold for |fast_trades - backtest_trades|")
    parser.add_argument("--proxy-fidelity-return-delta-max", type=float, default=0.03,
                        help="Warning threshold for |fast_return - backtest_return|")
    parser.add_argument("--proxy-fidelity-max-drawdown-delta-max", type=float, default=0.04,
                        help="Warning threshold for |fast_max_drawdown - backtest_max_drawdown|")
    parser.add_argument("--pruned-only", action="store_true",
                        help="Require pruned feature artifacts during optimization and holdout")
    parser.add_argument("--allow-unpruned", action="store_false", dest="pruned_only",
                        help="Allow fallback to unpruned profile features if artifacts are missing")
    parser.add_argument("--sampler-seeds", type=str, default="")
    parser.add_argument("--resume", action="store_true"); parser.add_argument("--debug-trials", action="store_true")
    parser.set_defaults(pruned_only=True)
    args = parser.parse_args(); args = apply_runtime_preset(args)
    gate_mode = resolve_gate_mode(args.gate_mode)
    provided_flags = set(sys.argv[1:])
    if "--holdout-min-trades" not in provided_flags:
        args.holdout_min_trades = int(gate_mode["holdout_min_trades"])
    if "--holdout-min-sharpe" not in provided_flags:
        args.holdout_min_sharpe = float(gate_mode["holdout_min_sharpe"])
    if "--holdout-min-return" not in provided_flags:
        args.holdout_min_return = float(gate_mode["holdout_min_return"])
    print(
        f"🛡️ Gate mode: {args.gate_mode} | trades>={args.holdout_min_trades}, "
        f"SR>={args.holdout_min_sharpe}, Ret>={args.holdout_min_return}, "
        f"PSR_HO>={args.min_psr_holdout}, DSR>={args.min_dsr}"
    )
    print("   Escalation policy: tighten thresholds after 14-28 days of stable paper evidence.")
    if args.debug_trials:
        DEBUG_TRIALS = True
        logging.basicConfig(level=logging.DEBUG, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if args.show: show_results(); sys.exit(0)
    init_db_wal(str(_db_path()))
    all_data = load_data()
    if not all_data: print("❌ No data."); sys.exit(1)
    coins = list(dict.fromkeys(COIN_MAP.values())) if args.all else [COIN_MAP.get(args.coin.upper(), args.coin.upper())] if args.coin else []
    if not coins: parser.print_help(); sys.exit(1)
    seeds = [int(s.strip()) for s in args.sampler_seeds.split(',') if s.strip()] if args.sampler_seeds else [args.sampler_seed]
    for cn in coins:
        purge_days = args.purge_days if args.purge_days is not None else compute_purge_days(get_coin_profile(cn))
        optimize_coin_multiseed(all_data, PREFIX_FOR_COIN.get(cn, cn), cn, sampler_seeds=seeds,
            n_trials=args.trials, n_jobs=args.jobs,
            plateau_patience=args.plateau_patience, plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup, plateau_min_completed=args.plateau_min_completed,
            study_suffix=args.study_suffix, resume_study=args.resume,
            holdout_days=args.holdout_days, min_internal_oos_trades=args.min_internal_oos_trades,
            min_total_trades=args.min_total_trades, n_cv_folds=args.n_cv_folds,
            holdout_candidates=args.holdout_candidates, require_holdout_pass=args.require_holdout_pass,
            holdout_min_trades=args.holdout_min_trades, holdout_min_sharpe=args.holdout_min_sharpe,
            holdout_min_return=args.holdout_min_return,
            holdout_mode=args.holdout_mode,
            target_trades_per_week=args.target_trades_per_week,
            target_trades_per_year=args.target_trades_per_year,
            enable_fee_stress=not args.disable_fee_stress,
            fee_stress_multiplier=args.fee_stress_multiplier,
            fee_blend_normal_weight=args.fee_blend_normal_weight,
            fee_blend_stressed_weight=args.fee_blend_stressed_weight,
            min_fold_sharpe_hard=args.min_fold_sharpe_hard,
            min_fold_win_rate=args.min_fold_win_rate,
            min_psr=args.min_psr,
            min_psr_cv=args.min_psr_cv,
            min_psr_holdout=args.min_psr_holdout,
            min_dsr=args.min_dsr,
            min_raw_expectancy=args.min_raw_expectancy,
            min_stressed_expectancy=args.min_stressed_expectancy,
            seed_stability_min_pass_rate=args.seed_stability_min_pass_rate,
            seed_stability_max_param_dispersion=args.seed_stability_max_param_dispersion,
            seed_stability_max_oos_sharpe_dispersion=args.seed_stability_max_oos_sharpe_dispersion,
            proxy_fidelity_candidates=args.proxy_fidelity_candidates,
            proxy_fidelity_eval_days=args.proxy_fidelity_eval_days,
            proxy_fidelity_sharpe_delta_max=args.proxy_fidelity_sharpe_delta_max,
            proxy_fidelity_trade_count_delta_max=args.proxy_fidelity_trade_count_delta_max,
            proxy_fidelity_return_delta_max=args.proxy_fidelity_return_delta_max,
            proxy_fidelity_max_drawdown_delta_max=args.proxy_fidelity_max_drawdown_delta_max,
            cv_mode=args.cv_mode,
            purge_bars=args.purge_bars,
            purge_days=purge_days,
            embargo_days=args.embargo_days,
            embargo_bars=args.embargo_bars,
            embargo_frac=args.embargo_frac,
            pruned_only=args.pruned_only,
            preset_name=args.preset,
            gate_mode=args.gate_mode)
