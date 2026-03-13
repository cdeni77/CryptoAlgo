#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (v11.1: Fast CV).

v11.1: CRITICAL PERFORMANCE FIX
  v11.0 called run_backtest() 3x per trial (~30 min/trial = 50h for 100 trials).
  v11.4 evaluates folds through execution gates aligned with train_model.py logic
  and uses realized fold outcomes for objective scoring. run_backtest() only used for final holdout.
  Expected: 100 trials in ~30-60 minutes instead of 50 hours.

Usage:
    python optimize.py --coin BTC --trials 100 --jobs 4
    python optimize.py --all --trials 100 --jobs 16
    python optimize.py --show
"""
import argparse, json, warnings, sys, os, logging, sqlite3, functools, traceback, time, math, tempfile
from datetime import datetime, timezone, timedelta
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional
import fcntl

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import brier_score_loss, roc_auc_score
from optuna.samplers import TPESampler

from scripts.train_model import (
    Config, load_data, run_backtest, MLSystem,
    _build_strategy_context, _calibrator_predict, _resolve_filter_policy,
    calculate_n_contracts, get_strategy_family, primary_recall_threshold,
    resolve_label_horizon, resolve_param, resolve_categorical_param,
    build_ensemble_member_specs, GATE_REASON_CODES,
)
from core.meta_labeling import calibrator_predict
from core.cv_splitters import CVFold, create_purged_embargo_splits, create_walk_forward_splits
from core.preprocessing_cv import fit_transform_fold
from core.coin_profiles import (
    CoinProfile, COIN_PROFILES, get_coin_profile,
    BTC_EXTRA_FEATURES, ETH_EXTRA_FEATURES, XRP_EXTRA_FEATURES,
    SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)
from core.reason_codes import ReasonCode
from core.metrics_significance import (
    compute_deflated_sharpe as compute_deflated_sharpe_metric,
    compute_psr_from_samples,
    evaluate_significance_gates,
)
from core.costs import ExchangeCostAssumptions, load_exchange_cost_assumptions
from core.overfit_diagnostics import (
    build_score_matrix_from_trials,
    compose_robustness_diagnostics,
    compute_pbo_from_matrix,
    make_stress_costs_block,
)
from core.study_significance import compose_study_significance_diagnostic

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
DEBUG_TRIALS = False
class PlateauStopper:
    """Stop an Optuna study when completed trials stop improving.

    Uses absolute delta on the objective value and only considers completed
    trials. Warmup/min_completed guardrails prevent premature stopping.
    """

    def __init__(self, patience: int, min_delta: float = 0.0, warmup: int = 0, min_completed: int = 0):
        self.patience = max(1, int(patience or 1))
        self.min_delta = max(0.0, float(min_delta or 0.0))
        self.warmup = max(0, int(warmup or 0))
        self.min_completed = max(0, int(min_completed or 0))
        self._best_value: float | None = None
        self._best_completed_count = 0

    def _is_better(self, current: float, reference: float, direction: optuna.study.StudyDirection) -> bool:
        if direction == optuna.study.StudyDirection.MINIMIZE:
            return current <= (reference - self.min_delta)
        return current >= (reference + self.min_delta)

    def __call__(self, study: optuna.Study, frozen_trial: optuna.trial.FrozenTrial) -> None:
        if frozen_trial.state != optuna.trial.TrialState.COMPLETE or frozen_trial.value is None:
            return

        completed = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        completed_count = len(completed)

        if completed_count <= self.warmup:
            return
        if completed_count < self.min_completed:
            return

        current_best = float(study.best_value)
        if self._best_value is None or self._is_better(current_best, self._best_value, study.direction):
            self._best_value = current_best
            self._best_completed_count = completed_count
            return

        stagnant_completed = completed_count - self._best_completed_count
        if stagnant_completed >= self.patience:
            study.set_user_attr('plateau_stop_reason', 'no_improvement')
            study.set_user_attr('plateau_patience', self.patience)
            study.set_user_attr('plateau_min_delta', self.min_delta)
            study.set_user_attr('plateau_last_improvement_completed', self._best_completed_count)
            study.stop()


# Example JSONL row: {"event_type":"reject","coin":"BTC","trial_number":12,"reason_code":"TOO_FEW_TRADES"}
class EventLogger:
    def __init__(self, path: Path | None):
        self.path = path

    def emit(self, event: dict):
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(event)
        row.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
        with open(self.path, 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(_to_json_safe(row)) + "\n")


EVENT_LOGGER: EventLogger | None = None


def _emit_event(event: dict):
    if EVENT_LOGGER is not None:
        EVENT_LOGGER.emit(event)


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


def resolve_target_symbol(all_data, coin_prefix, coin_name):
    """Resolve the symbol key to optimize for a given coin.

    Preference order:
      1) symbol names starting with the Coinbase CDE prefix (e.g. BIP, ETP)
      2) symbol names starting with the canonical coin ticker (e.g. BTC, ETH)
    Within each group, prefer the symbol with the most recent OHLCV timestamp.
    """
    if not all_data:
        return None

    prefix = str(coin_prefix or "").upper()
    ticker = str(coin_name or "").upper()
    symbols = list(all_data.keys())

    epoch_utc = pd.Timestamp(0, tz='UTC')

    def _latest_ts(sym):
        try:
            ohlcv = all_data[sym].get('ohlcv')
            if ohlcv is None or ohlcv.empty:
                return epoch_utc
            idx = ohlcv.index
            return idx.max() if len(idx) else epoch_utc
        except Exception:
            return epoch_utc

    for token in (prefix, ticker):
        if not token:
            continue
        candidates = [sym for sym in symbols if str(sym).upper().startswith(token)]
        if candidates:
            return max(candidates, key=_latest_ts)

    return None

def _as_number(value, default=None):
    if value is None: return default
    if isinstance(value, (int, float)): return float(value)
    try: return float(value)
    except: return default


def _finite_metric(value, default=0.0):
    n = _as_number(value, default=default)
    return default if (n is None or not np.isfinite(n)) else float(n)


def _bounded_sharpe_term(sharpe_value: float, scale: float = 2.0) -> float:
    return float(np.tanh(_finite_metric(sharpe_value, 0.0) / max(float(scale), 1e-9)))

def _fmt_pct(v, d=1, fb="?"): n = _as_number(v); return f"{n:.{d}%}" if n is not None else fb
def _fmt_float(v, d=3, fb="?"): n = _as_number(v); return f"{n:.{d}f}" if n is not None else fb


def _build_cost_config(cost_assumptions: ExchangeCostAssumptions | None) -> tuple[Config, dict]:
    config = Config()
    if cost_assumptions is None:
        return config, {
            'version': 'legacy_default',
            'cost_config_id': 'legacy_default',
            'source_path': None,
            'execution_fee_mode': 'bps',
            'exchange_fee_mode': 'per_contract_usd',
            'funding_interval_hours': 1,
            'assumption_profile': 'legacy',
            'applied': {'funding': True, 'slippage': True, 'impact': False},
        }

    slippage = cost_assumptions.slippage
    impact = cost_assumptions.impact
    funding = cost_assumptions.funding
    config.fee_pct_per_side = cost_assumptions.effective_fee_pct_per_side()
    config.min_fee_per_contract = float(cost_assumptions.effective_min_fee_per_contract())
    config.slippage_bps = float(slippage.bps_per_side)
    config.apply_funding = bool(funding.enabled)
    config.apply_slippage = bool(slippage.enabled)
    config.apply_impact = bool(impact.enabled)
    config.impact_bps_per_contract = float(impact.bps_per_contract)
    config.impact_max_bps_per_side = float(impact.max_bps_per_side)
    config.cost_config_path = cost_assumptions.source_path
    config.cost_config_version = cost_assumptions.version
    return config, cost_assumptions.to_metadata()


def estimate_holdout_trade_budget(holdout_days: int, target_trades_per_week: float, holdout_min_trades: int) -> tuple[float, int]:
    estimated_holdout_trades = max(0.0, float(target_trades_per_week)) * (max(0, int(holdout_days)) / 7.0)
    effective_holdout_floor = max(
        int(holdout_min_trades),
        int(estimated_holdout_trades * 0.60),
    )
    return estimated_holdout_trades, effective_holdout_floor



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


FIXED_ML = {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'min_child_samples': 20}
FIXED_RISK = {
    'position_size': 0.12,
    'vol_sizing_target': 0.025,
    'min_val_auc': 0.48,
}

COIN_OPTIMIZATION_PRIORS = {
    # Search priors are intentionally a bit wider than deployment defaults. This broadens
    # discovery while leaving holdout/promotion gates unchanged.
    'BTC': {'target_trades_per_year': 25.0, 'cooldown_min': 6.0, 'cooldown_max': 28.0, 'min_momentum_magnitude': (0.005, 0.030), 'max_vol_24h': (0.045, 0.110), 'max_ensemble_std': (0.08, 0.20), 'min_directional_agreement': (0.49, 0.69), 'meta_probability_threshold': (0.42, 0.62), 'min_vol_24h': (0.0001, 0.0030), 'min_trade_frequency_ratio': 0.55},
    'ETH': {'target_trades_per_year': 27.0, 'cooldown_min': 5.0, 'cooldown_max': 24.0, 'min_momentum_magnitude': (0.003, 0.028), 'max_vol_24h': (0.050, 0.100), 'max_ensemble_std': (0.08, 0.20), 'min_directional_agreement': (0.46, 0.66), 'meta_probability_threshold': (0.41, 0.61), 'min_vol_24h': (0.0004, 0.005), 'min_trade_frequency_ratio': 0.55},
    'SOL': {'target_trades_per_year': 33.0, 'cooldown_min': 3.0, 'cooldown_max': 16.0, 'min_momentum_magnitude': (0.003, 0.032), 'max_vol_24h': (0.055, 0.130), 'max_ensemble_std': (0.08, 0.20), 'min_directional_agreement': (0.44, 0.64), 'meta_probability_threshold': (0.40, 0.60), 'min_vol_24h': (0.0002, 0.005), 'min_trade_frequency_ratio': 0.58},
    'XRP': {'target_trades_per_year': 28.0, 'cooldown_min': 4.0, 'cooldown_max': 20.0, 'min_momentum_magnitude': (0.002, 0.028), 'max_vol_24h': (0.050, 0.110), 'max_ensemble_std': (0.08, 0.20), 'min_directional_agreement': (0.48, 0.68), 'meta_probability_threshold': (0.42, 0.62), 'min_vol_24h': (0.0004, 0.005), 'min_trade_frequency_ratio': 0.58},
    'DOGE': {'target_trades_per_year': 30.0, 'cooldown_min': 2.0, 'cooldown_max': 12.0, 'min_momentum_magnitude': (0.003, 0.032), 'max_vol_24h': (0.060, 0.150), 'max_ensemble_std': (0.08, 0.20), 'min_directional_agreement': (0.44, 0.64), 'meta_probability_threshold': (0.40, 0.60), 'min_vol_24h': (0.0002, 0.0045), 'min_trade_frequency_ratio': 0.60},
}

STRATEGY_FAMILIES = ('momentum_trend', 'breakout', 'mean_reversion', 'vol_overlay')
TRADE_FREQ_BUCKETS = ('conservative', 'balanced', 'aggressive')

STRATEGY_FAMILY_PRIORS = {
    'momentum_trend': {'min_momentum_multiplier': 1.00, 'cooldown_multiplier': 1.00, 'min_trade_frequency_ratio_delta': 0.00},
    'breakout': {'min_momentum_multiplier': 0.90, 'cooldown_multiplier': 0.85, 'min_trade_frequency_ratio_delta': 0.03},
    'mean_reversion': {'min_momentum_multiplier': 0.80, 'cooldown_multiplier': 0.95, 'min_trade_frequency_ratio_delta': -0.02},
    'vol_overlay': {'min_momentum_multiplier': 0.95, 'cooldown_multiplier': 1.10, 'min_trade_frequency_ratio_delta': -0.01},
}

TRADE_FREQ_BUCKET_PRIORS = {
    'conservative': {'cooldown_scale': 1.15, 'min_trade_frequency_ratio_delta': -0.06},
    'balanced': {'cooldown_scale': 1.00, 'min_trade_frequency_ratio_delta': 0.00},
    'aggressive': {'cooldown_scale': 0.70, 'min_trade_frequency_ratio_delta': 0.08},
}


def _compute_cv_oos_days(cv_splits: list[CVFold]) -> float:
    if not cv_splits:
        return 0.0
    total_days = 0.0
    for fold in cv_splits:
        start = pd.Timestamp(fold.test_start)
        end = pd.Timestamp(fold.test_end)
        span_days = max((end - start).total_seconds() / 86400.0, 0.0)
        total_days += span_days
    return float(total_days)


def _classify_activity_regime(trade_density_ratio: float, realized_trades: int) -> str:
    if realized_trades <= 0:
        return 'dead'
    if trade_density_ratio < 0.30 or realized_trades <= 5:
        return 'starved'
    if trade_density_ratio < 0.80 or realized_trades <= 12:
        return 'thin'
    return 'active'


def _trade_starvation_penalty(realized_trades: int, target_trades_floor: float) -> float:
    floor = max(float(target_trades_floor), 1.0)
    trades = int(realized_trades)
    ratio = float(trades) / floor
    shortfall = max(0.0, 1.0 - ratio)
    penalty = -0.40 * (shortfall ** 1.2)
    if trades == 0:
        return penalty - 1.00
    if trades <= 3:
        return penalty - 0.75
    if trades <= 5:
        return penalty - 0.50
    if trades <= 12:
        return penalty - 0.22
    return penalty


def _resolve_strategy_and_bucket(params: Dict | None) -> tuple[str, str]:
    params = params or {}
    strategy_family = str(params.get('strategy_family') or 'momentum_trend')
    trade_freq_bucket = str(params.get('trade_freq_bucket') or 'balanced')
    if strategy_family not in STRATEGY_FAMILIES:
        strategy_family = 'momentum_trend'
    if trade_freq_bucket not in TRADE_FREQ_BUCKETS:
        trade_freq_bucket = 'balanced'
    return strategy_family, trade_freq_bucket


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
    strategy_family = trial.suggest_categorical('strategy_family', STRATEGY_FAMILIES)
    trade_freq_bucket = trial.suggest_categorical('trade_freq_bucket', TRADE_FREQ_BUCKETS)
    family_prior = STRATEGY_FAMILY_PRIORS.get(strategy_family, STRATEGY_FAMILY_PRIORS['momentum_trend'])
    bucket_prior = TRADE_FREQ_BUCKET_PRIORS.get(trade_freq_bucket, TRADE_FREQ_BUCKET_PRIORS['balanced'])

    mm_low, mm_high = priors.get('min_momentum_magnitude', (0.03, 0.07))
    min_momentum_magnitude = trial.suggest_float(
        'min_momentum_magnitude',
        max(0.005, mm_low * family_prior['min_momentum_multiplier']),
        max(0.010, mm_high * family_prior['min_momentum_multiplier']),
        step=0.001,
    )
    max_vol_24h = trial.suggest_float('max_vol_24h', *priors.get('max_vol_24h', (0.05, 0.09)), step=0.001)
    max_ensemble_std = trial.suggest_float('max_ensemble_std', *priors.get('max_ensemble_std', (0.08, 0.13)), step=0.001)
    min_directional_agreement = trial.suggest_float('min_directional_agreement', *priors.get('min_directional_agreement', (0.60, 0.78)), step=0.01)
    meta_probability_threshold = trial.suggest_float('meta_probability_threshold', *priors.get('meta_probability_threshold', (0.50, 0.65)), step=0.01)

    min_vol_bounds = priors.get('min_vol_24h')
    if min_vol_bounds:
        min_vol_24h = trial.suggest_float('min_vol_24h', min_vol_bounds[0], min_vol_bounds[1], step=0.001)
    else:
        min_vol_24h = bp.min_vol_24h if bp else 0.004

    signal_lo = clamp(base_threshold - 0.07, 0.54, 0.73)
    signal_hi = clamp(base_threshold + 0.08, 0.62, 0.82)

    return CoinProfile(
        name=coin_name, prefixes=bp.prefixes if bp else [coin_name], extra_features=get_extra_features(coin_name),
        signal_threshold=trial.suggest_float('signal_threshold', signal_lo, signal_hi, step=0.01),
        label_forward_hours=trial.suggest_int('label_forward_hours', int(clamp(base_fwd - 12, 12, 48)), int(clamp(base_fwd + 12, 12, 48)), step=12),
        label_vol_target=trial.suggest_float('label_vol_target', clamp(base_label_vol - 0.6, 1.0, 2.4), clamp(base_label_vol + 0.6, 1.2, 2.6), step=0.2),
        min_momentum_magnitude=min_momentum_magnitude,
        vol_mult_tp=trial.suggest_float('vol_mult_tp', clamp(base_tp - 1.5, 3.5, 8.0), clamp(base_tp + 2.5, 5.0, 10.0), step=0.5),
        vol_mult_sl=trial.suggest_float('vol_mult_sl', clamp(base_sl - 1.0, 2.0, 5.0), clamp(base_sl + 1.0, 2.5, 5.5), step=0.5),
        max_hold_hours=trial.suggest_int('max_hold_hours', int(clamp(base_hold - 24, 48, 120)), int(clamp(base_hold + 36, 72, 168)), step=12),
        min_vol_24h=min_vol_24h,
        max_vol_24h=max_vol_24h,
        max_ensemble_std=max_ensemble_std,
        min_directional_agreement=min_directional_agreement,
        meta_probability_threshold=meta_probability_threshold,
        cooldown_hours=trial.suggest_float(
            'cooldown_hours',
            max(2.0, priors['cooldown_min'] * family_prior['cooldown_multiplier'] * bucket_prior['cooldown_scale']),
            max(6.0, priors['cooldown_max'] * family_prior['cooldown_multiplier'] * bucket_prior['cooldown_scale']),
        ),
        position_size=FIXED_RISK['position_size'],
        vol_sizing_target=FIXED_RISK['vol_sizing_target'], min_val_auc=FIXED_RISK['min_val_auc'],
        n_estimators=FIXED_ML['n_estimators'], max_depth=FIXED_ML['max_depth'],
        learning_rate=FIXED_ML['learning_rate'], min_child_samples=FIXED_ML['min_child_samples'],
        strategy_family=strategy_family,
        trade_freq_bucket=trade_freq_bucket,
    )


def build_effective_params(params: Dict, coin_name: str) -> Dict:
    bp = COIN_PROFILES.get(coin_name, COIN_PROFILES.get('ETH'))
    default_priors = COIN_OPTIMIZATION_PRIORS.get('ETH', {})
    priors = COIN_OPTIMIZATION_PRIORS.get(coin_name, default_priors)
    strategy_family, trade_freq_bucket = _resolve_strategy_and_bucket(params)
    base_cooldown = bp.cooldown_hours if bp and getattr(bp, 'cooldown_hours', None) is not None else 24.0
    return {
        'signal_threshold': params.get('signal_threshold', bp.signal_threshold if bp else 0.75),
        'label_forward_hours': params.get('label_forward_hours', bp.label_forward_hours if bp else 24),
        'label_vol_target': params.get('label_vol_target', bp.label_vol_target if bp else 1.8),
        'vol_mult_tp': params.get('vol_mult_tp', bp.vol_mult_tp if bp else 5.0),
        'vol_mult_sl': params.get('vol_mult_sl', bp.vol_mult_sl if bp else 3.0),
        'max_hold_hours': params.get('max_hold_hours', bp.max_hold_hours if bp else 72),
        'cooldown_hours': params.get('cooldown_hours', base_cooldown),
        'min_vol_24h': params.get('min_vol_24h', bp.min_vol_24h if bp else 0.004),
        'max_vol_24h': params.get('max_vol_24h', bp.max_vol_24h if bp else 0.09),
        'min_momentum_magnitude': params.get('min_momentum_magnitude', bp.min_momentum_magnitude if bp else 0.02),
        'max_ensemble_std': params.get('max_ensemble_std', bp.max_ensemble_std if bp else np.mean(priors.get('max_ensemble_std', (0.08, 0.13)))),
        'min_directional_agreement': params.get('min_directional_agreement', bp.min_directional_agreement if bp else np.mean(priors.get('min_directional_agreement', (0.60, 0.78)))),
        'meta_probability_threshold': params.get('meta_probability_threshold', bp.meta_probability_threshold if bp else np.mean(priors.get('meta_probability_threshold', (0.50, 0.65)))),
        'strategy_family': strategy_family,
        'trade_freq_bucket': trade_freq_bucket,
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
        max_ensemble_std=effective_params['max_ensemble_std'],
        min_directional_agreement=effective_params['min_directional_agreement'],
        meta_probability_threshold=effective_params['meta_probability_threshold'],
        position_size=effective_params['position_size'],
        vol_sizing_target=effective_params['vol_sizing_target'],
        n_estimators=effective_params['n_estimators'],
        max_depth=effective_params['max_depth'],
        learning_rate=effective_params['learning_rate'],
        min_child_samples=effective_params['min_child_samples'],
        strategy_family=effective_params['strategy_family'],
        trade_freq_bucket=effective_params['trade_freq_bucket'],
    )


def _score_model_quality(probs: np.ndarray, y_test: pd.Series) -> dict:
    if len(probs) == 0 or len(y_test) == 0:
        return {'auc': 0.5, 'brier': 1.0, 'prob_std': 0.0, 'prob_range': 0.0, 'score': -1.0}
    if len(np.unique(y_test)) < 2:
        auc = 0.5
    else:
        auc = float(roc_auc_score(y_test, probs))
    brier = float(brier_score_loss(y_test, probs))
    prob_std = float(np.std(probs))
    prob_range = float(np.percentile(probs, 95) - np.percentile(probs, 5))
    spread_bonus = min(0.1, prob_range * 0.5)
    baseline_brier = 0.25
    score = float((auc - 0.5) * 2.0 - (brier - baseline_brier) + spread_bonus)
    return {'auc': auc, 'brier': brier, 'prob_std': prob_std, 'prob_range': prob_range, 'score': score}


def _score_signal_density(probs: np.ndarray, ohlcv_test: pd.DataFrame, profile: CoinProfile) -> dict:
    n_bars = max(len(probs), 1)
    above_threshold = int(np.sum(probs >= float(profile.signal_threshold)))

    close = ohlcv_test.get('close', pd.Series(index=ohlcv_test.index, dtype=float))
    ret_72h = close.pct_change(72)
    momentum_pass = int(np.sum(ret_72h.abs() >= float(profile.min_momentum_magnitude)))

    vol_24h = close.pct_change().rolling(24).std()
    vol_pass = int(np.sum((vol_24h >= float(profile.min_vol_24h)) & (vol_24h <= float(profile.max_vol_24h))))

    signal_rate = above_threshold / n_bars
    momentum_rate = momentum_pass / n_bars
    vol_rate = vol_pass / n_bars

    joint_rate = max(signal_rate * momentum_rate * vol_rate, 1e-6)
    hours_per_trade = max(float(profile.cooldown_hours), 1.0 / joint_rate)
    estimated_tpy = float(8760.0 / hours_per_trade)
    filter_health = (signal_rate + momentum_rate + vol_rate) / 3.0
    capped_tpy = min(estimated_tpy, 100.0)
    tpy_score = float(min(1.0, np.log1p(capped_tpy) / np.log1p(100.0)))
    score = 0.5 * filter_health + 0.5 * tpy_score

    return {
        'signal_rate': signal_rate,
        'momentum_rate': momentum_rate,
        'vol_rate': vol_rate,
        'estimated_tpy': estimated_tpy,
        'score': score,
    }


def _score_label_quality(y_train: pd.Series, y_test: pd.Series) -> dict:
    train_pos_rate = float((y_train == 1).mean()) if len(y_train) else 0.0
    test_pos_rate = float((y_test == 1).mean()) if len(y_test) else 0.0
    balance_score = 1.0 - abs(train_pos_rate - 0.30) * 2.0
    drift = abs(train_pos_rate - test_pos_rate)
    consistency_score = 1.0 - min(drift * 5.0, 1.0)
    return {
        'train_pos_rate': train_pos_rate,
        'test_pos_rate': test_pos_rate,
        'score': float(0.6 * max(0.0, balance_score) + 0.4 * consistency_score),
    }


def _normalize_triple_barrier_labels(labels: pd.Series) -> pd.Series:
    """Normalize triple-barrier labels {-1, 0, 1} into binary {0, 1}.

    Neutral labels (0) are removed and direction labels are remapped
    from {-1, 1} to {0, 1}. If the input does not contain short-class
    labels (-1), values are returned unchanged apart from NaN dropping.
    """
    clean = labels.dropna()
    if clean.empty:
        return clean
    if (clean == -1).any():
        return clean.loc[clean != 0].map({-1: 0, 1: 1}).dropna().astype(int)
    return clean.astype(int)


def _score_fold_consistency(fold_aucs: list[float]) -> dict:
    if len(fold_aucs) < 2:
        return {'score': 0.5}
    cv = float(np.std(fold_aucs) / max(abs(np.mean(fold_aucs)), 0.01))
    return {'score': float(1.0 / (1.0 + cv))}


def _normalize_gate_reason_code(reason: str) -> Optional[str]:
    token = str(reason or '').strip()
    if not token:
        return None
    if token in GATE_REASON_CODES:
        return token
    lowered = token.lower()
    if lowered in GATE_REASON_CODES:
        return lowered
    return None


def _summarize_gate_counters_by_reason(gate_counts: Dict[str, int], total_checks: int) -> Dict[str, object]:
    normalized_counts = {reason: 0 for reason in GATE_REASON_CODES}
    for reason, count in (gate_counts or {}).items():
        normalized = _normalize_gate_reason_code(reason)
        if normalized is None:
            continue
        normalized_counts[normalized] += int(count or 0)

    checks = max(int(total_checks or 0), 0)
    rates = {
        reason: float(count / max(checks, 1))
        for reason, count in normalized_counts.items()
    }
    return {
        'total_checks': checks,
        'gate_counts': normalized_counts,
        'gate_rates': rates,
    }


def _derive_main_blockers(gate_summary: Dict[str, object], *, top_n: int = 3) -> List[Dict[str, object]]:
    counts = (gate_summary or {}).get('gate_counts', {}) if isinstance(gate_summary, dict) else {}
    rates = (gate_summary or {}).get('gate_rates', {}) if isinstance(gate_summary, dict) else {}
    ranked = sorted(
        [
            (reason, int(counts.get(reason, 0) or 0), float(rates.get(reason, 0.0) or 0.0))
            for reason in GATE_REASON_CODES
        ],
        key=lambda row: (row[1], row[2]),
        reverse=True,
    )
    return [
        {'reason_code': reason, 'count': count, 'rate': rate}
        for reason, count, rate in ranked
        if count > 0
    ][:max(1, int(top_n or 1))]


def _derive_starvation_signature(gate_summary: Dict[str, object]) -> Dict[str, object]:
    rates = (gate_summary or {}).get('gate_rates', {}) if isinstance(gate_summary, dict) else {}
    momentum_starve = float(rates.get('momentum_magnitude', 0.0) or 0.0) + float(rates.get('momentum_dir_agreement', 0.0) or 0.0)
    vol_starve = float(rates.get('vol_regime_low', 0.0) or 0.0)
    confidence_starve = float(rates.get('primary_threshold', 0.0) or 0.0) + float(rates.get('ensemble_agreement', 0.0) or 0.0)
    total = momentum_starve + vol_starve + confidence_starve
    if total <= 0.0:
        return {'label': 'insufficient_signal', 'shares': {}}

    shares = {
        'momentum_starved': float(momentum_starve / total),
        'vol_starved': float(vol_starve / total),
        'confidence_starved': float(confidence_starve / total),
    }
    label = max(shares, key=shares.get)
    return {
        'label': label,
        'shares': shares,
    }


def _summarize_fold_gate_counters(gate_counts: Dict[str, int], total_checks: int) -> Dict[str, object]:
    return _summarize_gate_counters_by_reason(gate_counts, total_checks)


def _extract_gate_summary_from_artifact(artifact_dir: Path, symbol: str) -> Dict[str, object]:
    files = sorted(artifact_dir.glob('gate_counters_backtest_*.json'))
    if not files:
        return _summarize_gate_counters_by_reason({}, 0)

    payload = json.loads(files[-1].read_text(encoding='utf-8'))
    symbols = payload.get('symbols', []) if isinstance(payload, dict) else []
    selected = None
    for item in symbols:
        if not isinstance(item, dict):
            continue
        if str(item.get('symbol')) == str(symbol):
            selected = item
            break
    if selected is None and symbols:
        selected = next((item for item in symbols if isinstance(item, dict)), None)

    if not selected:
        return _summarize_gate_counters_by_reason({}, 0)

    total_checks = int(selected.get('total_candidate_checks', 0) or 0)
    gate_counts = selected.get('gate_counts', {}) if isinstance(selected.get('gate_counts', {}), dict) else {}
    return _summarize_gate_counters_by_reason(gate_counts, total_checks)


def _aggregate_slice_gate_summaries(holdout_slices: Dict[str, dict]) -> Dict[str, object]:
    aggregate_counts = {reason: 0 for reason in GATE_REASON_CODES}
    total_checks = 0
    for metrics in (holdout_slices or {}).values():
        if not isinstance(metrics, dict):
            continue
        gate_summary = metrics.get('gate_counters', {}) if isinstance(metrics.get('gate_counters', {}), dict) else {}
        total_checks += int(gate_summary.get('total_checks', 0) or 0)
        counts = gate_summary.get('gate_counts', {}) if isinstance(gate_summary.get('gate_counts', {}), dict) else {}
        for reason in GATE_REASON_CODES:
            aggregate_counts[reason] += int(counts.get(reason, 0) or 0)
    return _summarize_gate_counters_by_reason(aggregate_counts, total_checks)

def _finalize_holdout_payload(metrics: Dict[str, object]) -> Dict[str, object]:
    payload = dict(metrics or {})
    gate_summary = payload.get('gate_counters', {}) if isinstance(payload.get('gate_counters', {}), dict) else {}
    payload['gate_counters'] = _summarize_gate_counters_by_reason(
        gate_summary.get('gate_counts', {}) if isinstance(gate_summary.get('gate_counts', {}), dict) else {},
        int(gate_summary.get('total_checks', 0) or 0),
    )
    payload['starvation_signature'] = _derive_starvation_signature(payload['gate_counters'])

    if int(payload.get('holdout_trades', 0) or 0) == 0:
        payload['main_blockers'] = _derive_main_blockers(payload['gate_counters'])
    else:
        payload.pop('main_blockers', None)
    return payload


def _select_median_composite_slice(holdout_slices: Dict[str, dict]) -> tuple[str, Dict[str, object]] | tuple[None, None]:
    slice_items = [(name, metrics) for name, metrics in (holdout_slices or {}).items() if isinstance(metrics, dict)]
    if not slice_items:
        return None, None

    sr_median = float(np.median([_as_number(metrics.get('holdout_sharpe'), 0.0) or 0.0 for _, metrics in slice_items]))
    ret_median = float(np.median([_as_number(metrics.get('holdout_return'), 0.0) or 0.0 for _, metrics in slice_items]))
    trades_median = float(np.median([int(metrics.get('holdout_trades', 0) or 0) for _, metrics in slice_items]))

    def _dist(item):
        _name, metrics = item
        sr = _as_number(metrics.get('holdout_sharpe'), 0.0) or 0.0
        ret = _as_number(metrics.get('holdout_return'), 0.0) or 0.0
        trades = float(int(metrics.get('holdout_trades', 0) or 0))
        return (
            abs(sr - sr_median),
            abs(ret - ret_median),
            abs(trades - trades_median),
            -trades,
        )

    selected_name, selected_metrics = min(slice_items, key=_dist)
    return selected_name, dict(selected_metrics)



def evaluate_fold_with_execution_gates(features, ohlcv, fold: CVFold, profile: CoinProfile, config: Config, symbol: str, pruned_only: bool = True):
    system = MLSystem(config)
    resolved_family = resolve_categorical_param('strategy_family', profile, config, Config.strategy_family)
    resolved_bucket = resolve_categorical_param('trade_freq_bucket', profile, config, Config.trade_freq_bucket)
    config.strategy_family = resolved_family
    config.trade_freq_bucket = resolved_bucket
    strategy_family = get_strategy_family(resolved_family)

    feature_candidates = profile.resolve_feature_columns(
        use_pruned_features=bool(pruned_only),
        strict_pruned=bool(pruned_only),
    )
    base_cols = system.get_feature_columns(features.columns, feature_candidates)
    if not base_cols or len(base_cols) < 4:
        return None

    train_feat = features.loc[fold.train_idx.intersection(features.index)]
    test_feat = features.loc[fold.test_idx.intersection(features.index)]
    if len(train_feat) < config.min_train_samples or len(test_feat) < 24:
        return None

    member_models = []
    for member_spec in build_ensemble_member_specs():
        if train_feat.empty:
            continue
        cutoff_ts = train_feat.index.max() - timedelta(days=int(member_spec.train_window_days))
        member_train_feat = train_feat.loc[train_feat.index >= cutoff_ts]
        if len(member_train_feat) < config.min_train_samples:
            continue

        member_cols = system.build_member_features(symbol, base_cols, member_spec)
        if len(member_cols) < 4:
            continue

        y = system.create_labels(ohlcv, member_train_feat, profile=profile)
        valid_idx = y.dropna().index
        X_all = member_train_feat.loc[valid_idx, member_cols]
        y_all = y.loc[valid_idx]
        X_all, y_all = system.prepare_binary_training_set(X_all, y_all)
        if len(X_all) < config.min_train_samples or y_all.nunique() < 2:
            continue

        split_idx = int(len(X_all) * (1 - config.val_fraction))
        if split_idx <= 0 or split_idx >= len(X_all):
            continue

        y_train = y_all.iloc[:split_idx]
        y_val = y_all.iloc[split_idx:]
        x_train_fold, x_val_fold, _ = fit_transform_fold(X_all.iloc[:split_idx], X_all.iloc[split_idx:], y_train)
        if y_train.nunique() < 2 or y_val.nunique() < 2:
            continue

        result = system.train(
            x_train_fold,
            y_train,
            x_val_fold,
            y_val,
            profile=profile,
            symbol=symbol,
            member_spec=member_spec,
            member_features=member_cols,
        )
        if not result:
            continue

        model, scaler, iso, auc, meta_artifacts, *_ = result
        member_models.append({
            'model': model,
            'scaler': scaler,
            'calibrator': iso,
            'meta': meta_artifacts,
            'cols': member_cols,
            'auc': float(auc),
        })

    if len(member_models) < 2:
        return None

    y_train_label_quality_raw = system.create_labels(ohlcv, train_feat, profile=profile).dropna()
    y_train_label_quality = _normalize_triple_barrier_labels(y_train_label_quality_raw)
    y_test_raw = system.create_labels(ohlcv, test_feat, profile=profile).dropna()
    if len(y_test_raw) < 20 or y_test_raw.nunique() < 2:
        return None
    y_test_binary = _normalize_triple_barrier_labels(y_test_raw)

    ensemble_probs = []
    y_test = []
    for ts, yv in y_test_binary.items():
        row = test_feat.loc[ts]
        probs = []
        for m in member_models:
            x_in = np.nan_to_num(np.array([row.get(c, 0.0) for c in m['cols']], dtype=float).reshape(1, -1), nan=0.0)
            raw_prob = m['model'].predict_proba(m['scaler'].transform(x_in))[0, 1]
            probs.append(float(_calibrator_predict(m['calibrator'], np.array([raw_prob]))[0]))
        if probs:
            ensemble_probs.append(float(np.mean(probs)))
            y_test.append(int(yv))

    if len(ensemble_probs) < 20 or len(set(y_test)) < 2:
        return None

    y_test_series = pd.Series(y_test)
    model_quality = _score_model_quality(np.array(ensemble_probs, dtype=float), y_test_series)
    label_quality = _score_label_quality(y_train_label_quality, y_test_series)

    gate_counts: Dict[str, int] = {}
    def _gate(reason: str):
        gate_counts[reason] = gate_counts.get(reason, 0) + 1

    bucket_prior = TRADE_FREQ_BUCKET_PRIORS.get(resolved_bucket, TRADE_FREQ_BUCKET_PRIORS['balanced'])
    cooldown_hours = max(float(profile.cooldown_hours) * float(bucket_prior.get('cooldown_scale', 1.0)) * 0.65, 0.0)
    last_entry_ts = None
    label_horizon = max(1, int(resolve_label_horizon(profile, config)))
    effective_threshold = resolve_param('signal_threshold', profile, config, Config.signal_threshold, mode='direct')
    primary_cutoff = primary_recall_threshold(effective_threshold, config.min_signal_edge * 0.5)
    primary_cutoff = max(0.45, primary_cutoff - 0.02)
    effective_meta_threshold = resolve_param('meta_probability_threshold', profile, config, Config.meta_probability_threshold, mode='direct')
    effective_meta_threshold = max(0.35, float(effective_meta_threshold) - 0.03)
    effective_directional_agreement = resolve_param('min_directional_agreement', profile, config, Config.min_directional_agreement, mode='direct')
    effective_directional_agreement = max(0.40, float(effective_directional_agreement) - 0.04)
    effective_max_ensemble_std = resolve_param('max_ensemble_std', profile, config, Config.max_ensemble_std, mode='direct')
    effective_momentum = resolve_param('min_momentum_magnitude', profile, config, Config.min_momentum_magnitude, mode='direct')
    effective_momentum = max(0.0015, float(effective_momentum) * 0.75)

    trade_returns: List[float] = []
    agreement_values: List[float] = []
    std_values: List[float] = []
    test_idx = test_feat.index.intersection(ohlcv.index)

    for ts in test_idx:
        if last_entry_ts is not None and cooldown_hours > 0:
            if (ts - last_entry_ts).total_seconds() < cooldown_hours * 3600:
                _gate('cooldown')
                continue

        row = features.loc[ts]
        ohlcv_row = ohlcv.loc[ts]
        price = float(ohlcv_row.get('close', np.nan))
        sma_200 = ohlcv_row.get('sma_200', np.nan)
        if pd.isna(price) or pd.isna(sma_200):
            _gate('missing_sma200')
            continue

        vol_24h = ohlcv['close'].pct_change().rolling(24).std().get(ts, None)
        if vol_24h is None or pd.isna(vol_24h) or float(vol_24h) < profile.min_vol_24h:
            _gate('vol_regime_low')
            continue
        if float(vol_24h) > profile.max_vol_24h:
            _gate('vol_regime_high')
            continue

        ts_loc = ohlcv.index.get_loc(ts)
        if ts_loc < 72:
            _gate('momentum_magnitude')
            continue

        ret_24h = (price / ohlcv['close'].iloc[ts_loc - 24] - 1)
        ret_72h = (price / ohlcv['close'].iloc[ts_loc - 72] - 1)
        if abs(ret_72h) < effective_momentum:
            _gate('momentum_magnitude')
            continue

        sma_50 = ohlcv['close'].iloc[max(0, ts_loc - 50):ts_loc].mean()
        strategy_context = _build_strategy_context(row, price, sma_200, ret_24h, ret_72h, sma_50, vol_24h=float(vol_24h))
        strategy_decision = strategy_family.evaluate(
            strategy_context,
            min_momentum_magnitude=effective_momentum,
            score_threshold=max(0.65, float(config.momentum_score_threshold) * 0.75),
            strict_mode=False,
        )
        direction = int(strategy_decision.direction)
        if not strategy_decision.gate_contributions.get('momentum_dir_agreement', direction != 0):
            _gate('momentum_dir_agreement')
            continue

        trend_violated = (direction == 1 and price < sma_200) or (direction == -1 and price > sma_200)
        trend_effect = _resolve_filter_policy(config.trend_filter_mode, trend_violated, rank_penalty=0.30, size_multiplier=0.70)
        if trend_effect.reject:
            _gate('trend_filter')
            continue
        size_multiplier = float(trend_effect.size_multiplier)

        f_z = row.get('funding_rate_zscore', 0)
        if pd.isna(f_z):
            f_z = 0
        funding_violated = (direction == 1 and f_z > 2.5) or (direction == -1 and f_z < -2.5)
        funding_effect = _resolve_filter_policy(config.funding_filter_mode, funding_violated, rank_penalty=0.25, size_multiplier=0.75)
        if funding_effect.reject:
            _gate('funding_filter')
            continue
        size_multiplier *= float(funding_effect.size_multiplier)

        probs: List[float] = []
        directional_votes: List[int] = []
        meta_probs: List[float] = []
        for m in member_models:
            x_in = np.nan_to_num(np.array([row.get(c, 0.0) for c in m['cols']], dtype=float).reshape(1, -1), nan=0.0)
            raw_prob = m['model'].predict_proba(m['scaler'].transform(x_in))[0, 1]
            cal_prob = float(_calibrator_predict(m['calibrator'], np.array([raw_prob]))[0])
            probs.append(cal_prob)
            directional_votes.append(1 if (cal_prob >= 0.5 and direction == 1) or (cal_prob < 0.5 and direction == -1) else 0)
            meta_artifacts = m['meta']
            if cal_prob >= primary_cutoff and meta_artifacts.model is not None and meta_artifacts.scaler is not None:
                meta_raw = meta_artifacts.model.predict_proba(meta_artifacts.scaler.transform(x_in))[0, 1]
                if meta_artifacts.calibrator is not None:
                    meta_probs.append(float(calibrator_predict(meta_artifacts.calibrator, np.array([meta_raw]))[0]))
                else:
                    meta_probs.append(float(meta_raw))

        if not probs:
            continue
        agreement = float(np.mean(directional_votes)) if directional_votes else 0.0
        prob = float(np.mean(probs))
        prob_std = float(np.std(probs))
        agreement_values.append(agreement)
        std_values.append(prob_std)

        if agreement < effective_directional_agreement:
            _gate('ensemble_agreement')
            continue

        if agreement < 0.999:
            relaxed_disagreement_cap = min(0.98, max(float(config.disagreement_confidence_cap), primary_cutoff + 0.08))
            prob = min(prob, relaxed_disagreement_cap)
        if prob < primary_cutoff or prob_std > effective_max_ensemble_std:
            _gate('primary_threshold' if prob < primary_cutoff else 'ensemble_std')
            continue

        if not meta_probs:
            if prob < (primary_cutoff + 0.035):
                _gate('meta_threshold')
                continue
        elif float(np.mean(meta_probs)) < effective_meta_threshold:
            soft_meta_floor = effective_meta_threshold - 0.04
            if prob < max(primary_cutoff + 0.02, soft_meta_floor):
                _gate('meta_threshold')
                continue

        n_contracts = calculate_n_contracts(100_000, price, symbol, config, vol_24h=float(vol_24h), profile=profile)
        n_contracts = int(np.floor(n_contracts * size_multiplier))
        if n_contracts < 1:
            _gate('no_contract_size')
            continue

        exit_loc = ts_loc + label_horizon
        if exit_loc >= len(ohlcv):
            continue

        exit_price = float(ohlcv['close'].iloc[exit_loc])
        trade_returns.append(float(((exit_price / price) - 1.0) * direction))
        last_entry_ts = ts

    fold_trades = int(len(trade_returns))
    if fold_trades:
        arr = np.array(trade_returns, dtype=float)
        fold_return = float(np.prod(1.0 + arr) - 1.0)
        expectancy = float(np.mean(arr))
        raw_sharpe = float((np.mean(arr) / max(np.std(arr), 1e-9)) * np.sqrt(fold_trades))
        if fold_trades < 3:
            sharpe = 0.0
        else:
            sharpe = float(np.clip(raw_sharpe, -3.0, 3.0))
    else:
        fold_return = 0.0
        expectancy = 0.0
        raw_sharpe = 0.0
        sharpe = 0.0

    return {
        'model_quality': model_quality,
        'label_quality': label_quality,
        'fold_metrics': {
            'trades': fold_trades,
            'raw_sharpe': raw_sharpe,
            'sharpe': sharpe,
            'return': fold_return,
            'expectancy': expectancy,
            'ensemble_agreement_mean': float(np.mean(agreement_values)) if agreement_values else 0.0,
            'ensemble_std_mean': float(np.mean(std_values)) if std_values else 0.0,
        },
        'gate_counters': _summarize_fold_gate_counters(gate_counts, total_checks=len(test_idx)),
    }


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
    pruned_only=True,
    base_config=None,
    **_kwargs,
):
    profile = create_trial_profile(trial, coin_name)
    seed_config = base_config or Config()
    config = Config(**{
        **seed_config.__dict__,
        'max_positions': 1,
        'leverage': 4,
        'min_signal_edge': 0.0,
        'enforce_pruned_features': bool(pruned_only),
        'min_train_samples': 100,
        'signal_threshold': 0.50,
    })
    config.strategy_family = profile.strategy_family
    config.trade_freq_bucket = profile.trade_freq_bucket

    features = optim_data[target_sym]['features']
    ohlcv = optim_data[target_sym]['ohlcv']
    fold_scores = []

    for fold_idx, fold in enumerate(cv_splits):
        result = evaluate_fold_with_execution_gates(features, ohlcv, fold, profile, config, target_sym, pruned_only=pruned_only)
        if result is not None:
            fold_scores.append(result)
            partial_model_q = float(np.mean([f['model_quality']['score'] for f in fold_scores]))
            partial_label_q = float(np.mean([f['label_quality']['score'] for f in fold_scores]))
            partial_consistency = _score_fold_consistency([f['model_quality']['auc'] for f in fold_scores])
            partial_metrics = [f.get('fold_metrics', {}) for f in fold_scores]
            partial_sharpe = float(np.mean([_finite_metric(m.get('sharpe'), 0.0) for m in partial_metrics]))
            partial_sharpe_term = _bounded_sharpe_term(partial_sharpe)
            partial_return = float(np.mean([_finite_metric(m.get('return'), 0.0) for m in partial_metrics]))
            partial_expectancy = float(np.mean([_finite_metric(m.get('expectancy'), 0.0) for m in partial_metrics]))

            partial_combined = (
                0.25 * partial_model_q +
                0.20 * partial_label_q +
                0.15 * partial_consistency['score'] +
                0.25 * partial_sharpe_term +
                0.10 * partial_expectancy +
                0.05 * np.tanh(partial_return * 5.0)
            )
            trial.report(float(partial_combined), step=int(fold_idx))

            if trial.should_prune():
                trial.set_user_attr('reject_stage', 'cv_fold')
                trial.set_user_attr('reject_code', str(ReasonCode.PRUNED.value))
                trial.set_user_attr('reject_reason', f'optuna_pruner:fold_{fold_idx}')
                trial.set_user_attr('prune_source', 'optuna_pruner')
                trial.set_user_attr('prune_fold_idx', int(fold_idx))
                trial.set_user_attr('n_folds', int(len(fold_scores)))
                trial.set_user_attr('prune_intermediate_value', round(float(partial_combined), 6))
                raise optuna.TrialPruned(f'pruned_at_fold_{fold_idx}')

    min_required_folds = max(1, len(cv_splits) // 3)
    if len(fold_scores) < min_required_folds:
        trial.set_user_attr('reject_code', str(ReasonCode.TOO_FEW_FOLDS))
        trial.set_user_attr('reject_reason', f'too_few_folds:{len(fold_scores)}/{len(cv_splits)}')
        trial.set_user_attr('n_folds', int(len(fold_scores)))
        return -1.0

    model_q = float(np.mean([f['model_quality']['score'] for f in fold_scores]))
    label_q = float(np.mean([f['label_quality']['score'] for f in fold_scores]))
    mean_auc = float(np.mean([f['model_quality']['auc'] for f in fold_scores]))
    consistency = _score_fold_consistency([f['model_quality']['auc'] for f in fold_scores])

    fold_metrics = [f.get('fold_metrics', {}) for f in fold_scores]
    fold_gate_counters = [f.get('gate_counters', {}) for f in fold_scores]
    fold_metrics_with_gates = [
        {
            'trades': int((metric or {}).get('trades', 0) or 0),
            'raw_sharpe': _finite_metric((metric or {}).get('raw_sharpe', (metric or {}).get('sharpe')), 0.0),
            'sharpe': _finite_metric((metric or {}).get('sharpe'), 0.0),
            'return': _finite_metric((metric or {}).get('return'), 0.0),
            'expectancy': _finite_metric((metric or {}).get('expectancy'), 0.0),
            'gate_counters': gate if isinstance(gate, dict) else {},
        }
        for metric, gate in zip(fold_metrics, fold_gate_counters)
    ]
    fold_trade_counts = [int(m.get('trades', 0) or 0) for m in fold_metrics]
    fold_sharpes = [_finite_metric(m.get('sharpe'), 0.0) for m in fold_metrics]
    fold_raw_sharpes = [_finite_metric(m.get('raw_sharpe', m.get('sharpe')), 0.0) for m in fold_metrics]

    realized_sharpe = float(np.mean([_finite_metric(m.get('sharpe'), 0.0) for m in fold_metrics]))
    realized_sharpe_raw = float(np.mean(fold_raw_sharpes)) if fold_raw_sharpes else 0.0
    realized_sharpe_term = _bounded_sharpe_term(realized_sharpe)
    realized_return = float(np.mean([_finite_metric(m.get('return'), 0.0) for m in fold_metrics]))
    realized_expectancy = float(np.mean([_finite_metric(m.get('expectancy'), 0.0) for m in fold_metrics]))
    realized_trades = int(sum(fold_trade_counts))

    default_priors = COIN_OPTIMIZATION_PRIORS.get('ETH', {})
    priors = COIN_OPTIMIZATION_PRIORS.get(coin_name, default_priors)
    family_prior = STRATEGY_FAMILY_PRIORS.get(profile.strategy_family, STRATEGY_FAMILY_PRIORS['momentum_trend'])
    bucket_prior = TRADE_FREQ_BUCKET_PRIORS.get(profile.trade_freq_bucket, TRADE_FREQ_BUCKET_PRIORS['balanced'])
    base_trade_frequency_ratio = float(priors.get('min_trade_frequency_ratio', 0.40) or 0.40)
    strategy_delta = float(family_prior.get('min_trade_frequency_ratio_delta', 0.0))
    bucket_delta = float(bucket_prior.get('min_trade_frequency_ratio_delta', 0.0))
    min_trade_frequency_ratio = float(np.clip(base_trade_frequency_ratio + strategy_delta + bucket_delta, 0.20, 0.90))
    cv_oos_days = _compute_cv_oos_days(cv_splits)
    cv_oos_years = max(cv_oos_days / 365.25, 1e-6)
    target_trades_per_year = float(priors.get('target_trades_per_year', 25.0) or 25.0)
    target_trades_floor = max(1.0, target_trades_per_year * cv_oos_years * min_trade_frequency_ratio)
    trade_density = float(realized_trades) / max(cv_oos_years, 1e-6)
    trade_density_ratio = float(realized_trades) / target_trades_floor

    activity_bonus = 0.20 * min(1.5, trade_density_ratio)
    if realized_trades >= max(20, int(math.ceil(target_trades_floor * 0.90))):
        activity_bonus += 0.08

    combined = (
        0.18 * model_q +
        0.14 * label_q +
        0.12 * consistency['score'] +
        0.20 * realized_sharpe_term +
        0.08 * realized_expectancy +
        0.04 * np.tanh(realized_return * 5.0) +
        activity_bonus
    )

    trade_floor_penalty = _trade_starvation_penalty(realized_trades, target_trades_floor)
    combined += trade_floor_penalty

    activity_regime = _classify_activity_regime(trade_density_ratio, realized_trades)

    trial.set_user_attr('n_folds', int(len(fold_scores)))
    trial.set_user_attr('mean_auc', round(mean_auc, 4))
    trial.set_user_attr('n_trades', int(realized_trades))
    trial.set_user_attr('cv_trade_density', round(trade_density, 6))
    trial.set_user_attr('cv_oos_days', round(cv_oos_days, 4))
    trial.set_user_attr('cv_oos_years', round(cv_oos_years, 6))
    trial.set_user_attr('realized_trades', int(realized_trades))
    trial.set_user_attr('realized_trades_per_year', round(trade_density, 6))
    trial.set_user_attr('cv_trade_density_ratio', round(trade_density_ratio, 6))
    trial.set_user_attr('target_trades_floor', round(target_trades_floor, 6))
    trial.set_user_attr('min_trade_frequency_ratio', round(min_trade_frequency_ratio, 6))
    trial.set_user_attr('activity_regime', activity_regime)
    trial.set_user_attr('trade_density_ratio', round(trade_density_ratio, 6))
    trial.set_user_attr('mean_sharpe', round(realized_sharpe, 6))
    trial.set_user_attr('mean_sharpe_raw', round(realized_sharpe_raw, 6))
    trial.set_user_attr('realized_sharpe_term', round(realized_sharpe_term, 6))
    trial.set_user_attr('min_sharpe', round(float(min([_finite_metric(m.get('sharpe'), 0.0) for m in fold_metrics] or [0.0])), 6))
    trial.set_user_attr('std_sharpe', round(float(np.std([_finite_metric(m.get('sharpe'), 0.0) for m in fold_metrics])) if fold_metrics else 0.0, 6))
    trial.set_user_attr('win_rate', round(float(np.mean([1.0 if _finite_metric(m.get('expectancy'), 0.0) > 0 else 0.0 for m in fold_metrics])) if fold_metrics else 0.0, 4))
    trial.set_user_attr('max_drawdown', round(max(0.0, 1.0 - max(realized_return, -1.0)), 4))
    psr_meta = compute_psr_from_samples(
        fold_sharpes,
        benchmark_sharpe=0.0,
        effective_observations=len(fold_sharpes),
    )
    if not psr_meta.get('valid', False) and psr_meta.get('reason') is None:
        psr_meta['reason'] = 'insufficient_observations'

    dsr_cv_meta = compute_deflated_sharpe_metric(
        observed_sharpe=float(np.mean(fold_sharpes)) if fold_sharpes else 0.0,
        observations=len(fold_sharpes),
        effective_test_count=max(len(cv_splits), len(fold_sharpes), 1),
    )
    if not dsr_cv_meta.get('valid', False) and dsr_cv_meta.get('reason') is None:
        dsr_cv_meta['reason'] = 'insufficient_observations'

    trial.set_user_attr('fold_metrics', _to_json_safe(fold_metrics_with_gates))
    trial.set_user_attr('fold_gate_counters', _to_json_safe(fold_gate_counters))
    trial.set_user_attr('fold_trade_counts', _to_json_safe(fold_trade_counts))
    aggregate_cv_gate_counters = _aggregate_slice_gate_summaries({
        f'fold_{idx}': {'gate_counters': gate_summary}
        for idx, gate_summary in enumerate(fold_gate_counters)
    })
    trial.set_user_attr('aggregated_gate_counters', _to_json_safe(aggregate_cv_gate_counters))
    trial.set_user_attr('main_blockers', _to_json_safe(_derive_main_blockers(aggregate_cv_gate_counters)))
    trial.set_user_attr('starvation_signature', _to_json_safe(_derive_starvation_signature(aggregate_cv_gate_counters)))
    trial.set_user_attr('fold_blocker_summary', _to_json_safe([
        {
            'fold': int(idx),
            'main_blockers': _derive_main_blockers(gate_summary, top_n=2),
        }
        for idx, gate_summary in enumerate(fold_gate_counters)
    ]))
    trial.set_user_attr('psr_meta', _to_json_safe(psr_meta))
    trial.set_user_attr('psr', float(psr_meta['psr']) if psr_meta.get('valid', False) else None)
    trial.set_user_attr('dsr_cv_meta', _to_json_safe(dsr_cv_meta))
    trial.set_user_attr('dsr_cv', float(dsr_cv_meta['dsr']) if dsr_cv_meta.get('valid', False) else None)
    trial.set_user_attr('realized_return', round(realized_return, 6))
    trial.set_user_attr('realized_expectancy', round(realized_expectancy, 6))
    trial.set_user_attr('trade_floor_penalty', round(trade_floor_penalty, 6))

    return float(combined)


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
        cv_trades = int(t.user_attrs.get('n_trades', 0) or 0)
        cv_density_ratio = _as_number(t.user_attrs.get('cv_trade_density_ratio'), None)
        if cv_density_ratio is None:
            cv_density_ratio = float(cv_trades) / max(float(min_trades), 1.0)
        activity_regime = str(t.user_attrs.get('activity_regime') or _classify_activity_regime(float(cv_density_ratio), cv_trades))
        regime_rank = {'dead': 0, 'starved': 1, 'thin': 2, 'active': 3}.get(activity_regime, 0)
        psr = _as_number(t.user_attrs.get('psr'), 0.0) or 0.0
        mean_sr = _as_number(t.user_attrs.get('mean_sharpe', t.user_attrs.get('sharpe')), -9.0) or -9.0
        realized_return = _as_number(t.user_attrs.get('realized_return'), -9.0) or -9.0
        std_s = _as_number(t.user_attrs.get('std_sharpe'), 9.0) or 9.0
        dd = _as_number(t.user_attrs.get('max_drawdown'), 1.0) or 1.0
        activity_score = min(3.0, float(cv_density_ratio)) + (0.06 * cv_trades)
        return (regime_rank, activity_score, float(mean_sr >= 0.0), float(realized_return >= 0.0), psr, mean_sr, float(t.value or -99), -std_s, -dd)

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
    median_trades = float(np.median(trade_vals)) if trade_vals else 0.0
    median_trade_term = min(1.8, median_trades / 18.0)
    dispersion_penalty = 0.30 * (float(np.std(sharpe_vals)) if len(sharpe_vals) > 1 else 0.0)

    trade_min_penalty = 0.0
    for _, metrics in slice_items:
        ho_sr = _as_number(metrics.get('holdout_sharpe'), 0.0) or 0.0
        ho_ret = _as_number(metrics.get('holdout_return'), 0.0) or 0.0
        ho_trades = int(metrics.get('holdout_trades', 0) or 0)
        if ho_trades <= 0:
            trade_min_penalty += 1.00
        elif ho_trades <= 3:
            trade_min_penalty += 0.75
        elif ho_trades <= 6:
            trade_min_penalty += 0.50
        elif ho_trades < 12:
            trade_min_penalty += 0.25
        if ho_sr < 0:
            trade_min_penalty += 0.25
        if ho_ret < 0:
            trade_min_penalty += 0.20

    score = (0.40 * median_sr) + (0.18 * median_ret * 5.0) + (0.42 * median_trade_term)
    score -= (trade_min_penalty + dispersion_penalty)
    return score + 0.08 * (_as_number(cv_score, 0.0) or 0.0)


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
    hm = _finalize_holdout_payload(dict(holdout_metrics))
    slices = hm.get('holdout_slices', {}) if isinstance(hm.get('holdout_slices', {}), dict) else {}
    if slices:
        hm['holdout_slices'] = {
            name: _finalize_holdout_payload(metrics)
            for name, metrics in slices.items()
            if isinstance(metrics, dict)
        }

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


def _candidate_holdout_summary(
    candidate_trial,
    holdout_metrics: Dict[str, object],
    selection_score: float,
    *,
    min_trades: int,
    min_sharpe: float,
    min_return: float,
    min_psr_holdout=None,
    min_dsr=None,
) -> Dict[str, object]:
    finalized = _finalize_holdout_payload(holdout_metrics if isinstance(holdout_metrics, dict) else {})
    trial_attrs = candidate_trial.user_attrs if isinstance(getattr(candidate_trial, 'user_attrs', None), dict) else {}
    summary = {
        'trial': int(candidate_trial.number),
        'selection_score': round(float(selection_score), 6),
        'cv_n_trades': int(trial_attrs.get('n_trades', 0) or 0),
        'cv_trade_density': _finite_metric(trial_attrs.get('cv_trade_density'), 0.0),
        'cv_trade_density_ratio': _finite_metric(trial_attrs.get('cv_trade_density_ratio'), 0.0),
        'activity_regime': str(trial_attrs.get('activity_regime') or 'unknown'),
        'holdout_sharpe': round(float(finalized.get('holdout_sharpe', 0.0) or 0.0), 6),
        'holdout_return': round(float(finalized.get('holdout_return', 0.0) or 0.0), 6),
        'holdout_trades': int(finalized.get('holdout_trades', 0) or 0),
        'gate_counters': finalized.get('gate_counters', {}),
        'starvation_signature': finalized.get('starvation_signature', _derive_starvation_signature(finalized.get('gate_counters', {}))),
        'holdout_slices': finalized.get('holdout_slices', {}),
        'slice_gate_counters': {
            k: _finalize_holdout_payload(v).get('gate_counters', {})
            for k, v in finalized.get('holdout_slices', {}).items()
            if isinstance(v, dict)
        },
        'passes_gate': _passes_holdout_gate(
            finalized,
            min_trades=min_trades,
            min_sharpe=min_sharpe,
            min_return=min_return,
            min_psr_holdout=min_psr_holdout,
            min_dsr=min_dsr,
        ),
    }
    if int(summary['holdout_trades']) == 0:
        summary['main_blockers'] = finalized.get('main_blockers', _derive_main_blockers(summary['gate_counters']))
    return summary

# ═══════════════════════════════════════════════════════════════════════════
# HOLDOUT (full run_backtest — called ONCE at the end)
# ═══════════════════════════════════════════════════════════════════════════

def _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days, pruned_only=True, base_config=None):
    seed_config = base_config or Config()
    config = Config(**{**seed_config.__dict__, 'max_positions': 1, 'leverage': 4, 'min_signal_edge': 0.00,
                    'max_ensemble_std': 0.10, 'train_embargo_hours': 24, 'oos_eval_days': int(eval_days),
                    'enforce_pruned_features': bool(pruned_only)})
    try:
        with tempfile.TemporaryDirectory(prefix='holdout_gate_counters_') as gate_tmp:
            gate_dir = Path(gate_tmp)
            result = run_backtest(
                {target_sym: holdout_data[target_sym]},
                config,
                profile_overrides={coin_name: profile},
                gate_artifact_dir=gate_dir,
            )
            gate_summary = _extract_gate_summary_from_artifact(gate_dir, target_sym)
    except Exception as e:
        print(f"  ❌ Holdout error ({eval_days}d): {e}")
        return None
    if not result:
        return None

    oos_sr = _finite_metric(result.get('oos_sharpe', 0))
    holdout_payload = {
        'holdout_sharpe': oos_sr if oos_sr > -90 else 0,
        'holdout_return': _finite_metric(result.get('oos_return', 0)),
        'holdout_trades': int(result.get('oos_trades', 0) or 0),
        'full_sharpe': _finite_metric(result.get('sharpe_annual', 0)),
        'full_pf': _finite_metric(result.get('profit_factor', 0)),
        'full_dd': _finite_metric(result.get('max_drawdown', 1), 1),
        'full_trades': int(result.get('n_trades', 0) or 0),
        'gate_counters': gate_summary,
    }
    return _finalize_holdout_payload(holdout_payload)


def _derive_top_level_holdout(holdout_slices, holdout_mode):
    if not holdout_slices:
        return {}
    recent = holdout_slices.get('recent90')
    if holdout_mode == 'single90' and isinstance(recent, dict):
        selected = dict(recent)
        selected['selected_slice'] = 'recent90'
        return _finalize_holdout_payload(selected)

    selected_slice, selected_metrics = _select_median_composite_slice(holdout_slices)
    if not selected_metrics:
        return {}

    selected = dict(selected_metrics)
    selected['selected_slice'] = 'median_composite'
    selected['selected_slice_source'] = selected_slice
    return _finalize_holdout_payload(selected)


def evaluate_holdout(holdout_data, params, coin_name, coin_prefix, holdout_days, pruned_only=True, holdout_mode='single90', base_config=None):
    target_sym = resolve_target_symbol(holdout_data, coin_prefix, coin_name)
    if not target_sym: return None
    profile = profile_from_params(params, coin_name)
    holdout_slices = {}

    recent_metrics = _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days=90, pruned_only=pruned_only, base_config=base_config)
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
        prior_metrics = _run_holdout_window(prior_dataset, target_sym, profile, coin_name, eval_days=90, pruned_only=pruned_only, base_config=base_config)
        if prior_metrics:
            holdout_slices['prior90'] = prior_metrics

    full_span_days = (sym_ohlcv.index.max() - sym_ohlcv.index.min()).days
    if full_span_days >= 180:
        full_metrics = _run_holdout_window(holdout_data, target_sym, profile, coin_name, eval_days=180, pruned_only=pruned_only, base_config=base_config)
        if full_metrics:
            holdout_slices['full180'] = full_metrics

    top_level = _derive_top_level_holdout(holdout_slices, holdout_mode=holdout_mode)
    if not top_level:
        return None
    top_level['holdout_mode'] = holdout_mode
    top_level['holdout_slices'] = holdout_slices
    return top_level


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


def _is_paper_facing_run(preset_name: str, gate_mode: str) -> bool:
    preset = str(preset_name or '').lower()
    gate = str(gate_mode or '').lower()
    return preset in {'paper_ready', 'robust120', 'robust180'} or gate in {
        'initial_paper_qualification',
        'production_promotion',
    }

# ═══════════════════════════════════════════════════════════════════════════
# PATHS / PERSISTENCE / MAIN
# ═══════════════════════════════════════════════════════════════════════════

def _db_path(): return SCRIPT_DIR / "optuna_trading.db"
def _sqlite_url(path): return f"sqlite:///{path.resolve()}"
def _candidate_results_dirs(): return [SCRIPT_DIR / "optimization_results", Path.cwd() / "optimization_results"]


def resolve_event_log_path(coin_name: str, run_id: str = "") -> Optional[Path]:
    safe_run = (run_id or "default").replace("/", "_")
    filename = f"{coin_name}_reject_prune_{safe_run}.jsonl"
    for base in _candidate_results_dirs():
        path = base / filename
        if path.exists():
            return path
    for base in _candidate_results_dirs():
        try:
            base.mkdir(parents=True, exist_ok=True)
            return base / filename
        except OSError:
            continue
    return None


def _candidate_trial_ledger_paths() -> List[Path]:
    return [d / "trial_ledger.jsonl" for d in _candidate_results_dirs()]


def _apply_cost_stress_config(base_config: Config, *, fee_multiplier: float = 1.0, slippage_multiplier: float = 1.0) -> Config:
    return Config(**{
        **base_config.__dict__,
        'fee_pct_per_side': float(base_config.fee_pct_per_side) * float(max(0.0, fee_multiplier)),
        'min_fee_per_contract': float(base_config.min_fee_per_contract) * float(max(0.0, fee_multiplier)),
        'slippage_bps': float(base_config.slippage_bps) * float(max(0.0, slippage_multiplier)),
    })


def _funding_adverse_adjustment(metrics: Dict[str, object], funding_bps_per_trade: float) -> Dict[str, object]:
    adjusted = dict(metrics or {})
    trades = int(adjusted.get('holdout_trades', 0) or 0)
    penalty = (float(funding_bps_per_trade) / 10000.0) * max(0, trades)
    base_return = float(_as_number(adjusted.get('holdout_return'), 0.0) or 0.0)
    adjusted['holdout_return'] = float(base_return - penalty)
    base_sr = float(_as_number(adjusted.get('holdout_sharpe'), 0.0) or 0.0)
    adjusted['holdout_sharpe'] = float(base_sr - min(0.5, penalty * 5.0))
    adjusted['funding_penalty_applied'] = {
        'funding_bps_per_trade': float(funding_bps_per_trade),
        'estimated_penalty_return': float(penalty),
        'trades': int(trades),
    }
    return adjusted


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


def _persist_paper_candidate_json(coin_name: str, payload: Dict) -> Optional[Path]:
    target_dir = Path("data") / "paper_candidates"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / f"{coin_name.lower()}.json"
        target_path.write_text(json.dumps(_to_json_safe(payload), indent=2), encoding='utf-8')
        return target_path
    except Exception:
        return None

def optimize_coin(all_data, coin_prefix, coin_name, n_trials=100, n_jobs=1,
                  plateau_patience=60, plateau_min_delta=0.02, plateau_warmup=30,
                  plateau_min_completed=0,
                  study_suffix="", resume_study=False, holdout_days=90,
                  min_total_trades=0, n_cv_folds=5,
                  sampler_seed=42, holdout_candidates=3, require_holdout_pass=False,
                  holdout_min_trades=15, holdout_min_sharpe=0.0, holdout_min_return=0.0,
                  holdout_mode='single90',
                  target_trades_per_week=1.0, target_trades_per_year=None,
                  preset_name="none", min_psr=0.55,
                  min_psr_cv=None, min_psr_holdout=None, min_dsr=None,
                  pruned_only=True, gate_mode="initial_paper_qualification",
                  seed_stability_min_pass_rate=0.67,
                  seed_stability_max_param_dispersion=0.60,
                  seed_stability_max_oos_sharpe_dispersion=0.35,
                  cv_mode='walk_forward', purge_days=None, purge_bars=None, embargo_days=None,
                  embargo_bars=None, embargo_frac=0.0,
                  cost_config_path=None,
                  proxy_fidelity_candidates=0,
                  proxy_fidelity_eval_days=0):
    enable_pbo_diagnostic = False
    enable_study_significance = False
    study_significance_bootstrap_iterations = 500
    study_significance_seed = 42
    study_significance_score_source = 'fold_sharpe'
    enable_cost_stress_diagnostics = False
    cost_stress_finalists = 2
    cost_stress_fee_multiplier = 1.5
    cost_stress_slippage_multiplier = 2.0
    cost_stress_funding_bps_per_trade = 2.0
    optim_data, holdout_data = split_data_temporal(all_data, holdout_days=holdout_days)
    target_sym = resolve_target_symbol(optim_data, coin_prefix, coin_name)
    if not target_sym: print(f"❌ {coin_name}: no data after holdout split"); return None

    global EVENT_LOGGER
    event_log_path = resolve_event_log_path(coin_name, study_suffix or '')
    EVENT_LOGGER = EventLogger(event_log_path)
    cost_assumptions = load_exchange_cost_assumptions(cost_config_path) if cost_config_path else None
    base_cost_config, cost_model_metadata = _build_cost_config(cost_assumptions)

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
    print("   Objective weights: model_quality=0.25, label_quality=0.20, fold_consistency=0.15, realized_sharpe=0.25, expectancy=0.10, return_term=0.05")
    print("   High-conviction mode: raised thresholds + longer cooldowns + wider TP")
    phase_label = 'discovery' if str(preset_name).lower() in {'discovery', 'paper_discovery'} else 'qualification'
    print(
        f"   Phase: {phase_label} | target_tpw={float(target_trades_per_week):.2f} "
        f"| target_tpy={float(target_trades_per_year) if target_trades_per_year is not None else float(target_trades_per_week) * 52.0:.1f}"
    )
    print(f"   Cost model: {cost_model_metadata.get('version')} ({cost_model_metadata.get('source_path') or 'built-in legacy defaults'})")
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
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=max(8, min(30, n_trials // 5)),
        n_warmup_steps=1,
        interval_steps=1,
    )
    study = None
    storage_url = _sqlite_url(_db_path())
    storage = optuna.storages.RDBStorage(
        url=storage_url,
        engine_kwargs={
            "connect_args": {"timeout": 30},
            "pool_pre_ping": True,
        },
    )
    for attempt in range(10):
        try:
            with _study_storage_lock(_db_path()):
                study = optuna.create_study(
                    direction='maximize',
                    sampler=sampler,
                    pruner=pruner,
                    study_name=study_name,
                    storage=storage,
                    load_if_exists=bool(resume_study),
                )
            break
        except Exception as e:
            err = str(e).lower()
            if isinstance(e, optuna.exceptions.DuplicatedStudyError) or "already exists" in err:
                if resume_study:
                    with _study_storage_lock(_db_path()):
                        study = optuna.load_study(study_name=study_name, storage=storage, sampler=sampler, pruner=pruner)
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
    study.set_user_attr('coin_name', coin_name)
    study.set_user_attr('target_trades_per_week', float(target_trades_per_week))
    study.set_user_attr('target_trades_per_year', float(target_trades_per_year) if target_trades_per_year is not None else float(target_trades_per_week) * 52.0)
    study.set_user_attr('gate_mode', str(gate_mode))
    study.set_user_attr('cost_config_version', cost_model_metadata.get('version'))
    study.set_user_attr('cost_config_path', cost_model_metadata.get('source_path'))

    obj = functools.partial(objective, optim_data=optim_data, coin_prefix=coin_prefix,
                            coin_name=coin_name, cv_splits=cv_splits, target_sym=target_sym,
                            pruned_only=pruned_only,
                            base_config=base_cost_config)
    min_completed_trials = max(int(plateau_min_completed or 0), int(max(1, n_trials) * 0.40))

    def _prune_event_callback(study, frozen_trial):
        if frozen_trial.state != optuna.trial.TrialState.PRUNED:
            return
        _emit_event({
            'event_type': 'prune',
            'coin': coin_name,
            'study_name': study.study_name,
            'trial_number': int(frozen_trial.number),
            'stage': frozen_trial.user_attrs.get('reject_stage', 'unknown'),
            'reason_code': frozen_trial.user_attrs.get('reject_code', ReasonCode.PRUNED),
            'reason': frozen_trial.user_attrs.get('reject_reason', 'optuna_pruned'),
            'metrics': {
                'n_trades': frozen_trial.user_attrs.get('n_trades'),
                'n_folds': frozen_trial.user_attrs.get('n_folds'),
            },
        })

    callbacks = [
        PlateauStopper(
            plateau_patience,
            plateau_min_delta,
            plateau_warmup,
            min_completed_trials,
        ),
        _prune_event_callback,
    ]
    optimize_jobs = max(1, int(n_jobs or 1))
    if optimize_jobs > 1:
        print(f"⚠️  Optuna SQLite storage is lock-prone with n_jobs>1; forcing n_jobs=1 for {coin_name}.")
        optimize_jobs = 1

    try:
        for attempt in range(5):
            try:
                study.optimize(
                    obj,
                    n_trials=n_trials,
                    n_jobs=optimize_jobs,
                    show_progress_bar=sys.stderr.isatty(),
                    callbacks=callbacks,
                )
                break
            except optuna.exceptions.StorageInternalError as e:
                err = str(e).lower()
                if "database is locked" in err and attempt < 4:
                    wait_s = 0.5 * (attempt + 1)
                    print(f"⚠️  SQLite busy during optimize ({coin_name}), retrying in {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                raise
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
    dsr = compute_deflated_sharpe_metric(
        _finite_metric(best.user_attrs.get('mean_sharpe', 0)),
        int(best.user_attrs.get('n_trades', 0) or 0),
        nc,
    )
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
    paper_facing_run = _is_paper_facing_run(preset_name, gate_mode)
    effective_require_holdout_pass = bool(require_holdout_pass or paper_facing_run)
    exploratory_best_trial = int(best.number)
    exploratory_best_score = round(float(best.value), 6) if best.value is not None else None
    promoted_paper_candidate = None
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
            selected_trial = None
            selected_score = None
            exploratory_trial, exploratory_holdout, exploratory_score = ranked_candidates[0]
            paper_candidate = passing_candidates[0] if passing_candidates else None

            if not passing_candidates:
                deployment_blocked = True
                blocked_reasons.extend([
                    'NO_HOLDOUT_CANDIDATE_PASSED_GATE',
                    (
                        f"holdout_gate_failed:trades>={effective_holdout_min_trades},sr>={holdout_min_sharpe},"
                        f"ret>={holdout_min_return},psr>={min_psr_holdout},dsr>={min_dsr}"
                    ),
                ])
                if effective_require_holdout_pass:
                    blocked_reasons.append('REQUIRE_HOLDOUT_PASS_ACTIVE')
                print("  🛑 Holdout gate failed for all candidates. Deployment remains blocked.")

            if paper_candidate:
                selected_trial, holdout_result, selected_score = paper_candidate
                promoted_paper_candidate = {
                    'trial': int(selected_trial.number),
                    'selection_score': round(float(selected_score), 6),
                }
            else:
                selected_trial, holdout_result, selected_score = exploratory_trial, exploratory_holdout, exploratory_score

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
                'require_holdout_pass_input': bool(require_holdout_pass),
                'paper_facing_run': bool(paper_facing_run),
                'require_holdout_pass_effective': bool(effective_require_holdout_pass),
                'holdout_gate': {
                    'min_trades': int(holdout_min_trades),
                    'effective_min_trades': int(effective_holdout_min_trades),
                    'min_sharpe': float(holdout_min_sharpe),
                    'min_return': float(holdout_min_return),
                    'min_psr_holdout': None if min_psr_holdout is None else float(min_psr_holdout),
                    'min_dsr': None if min_dsr is None else float(min_dsr),
                },
                'deployment_blocked': bool(deployment_blocked),
                'best_exploratory_trial': {
                    'trial': int(exploratory_trial.number),
                    'selection_score': round(float(exploratory_score), 6),
                },
                'paper_eligible_promoted_candidate': promoted_paper_candidate,
                'candidates': [
                    _candidate_holdout_summary(
                        c,
                        m,
                        sc,
                        min_trades=effective_holdout_min_trades,
                        min_sharpe=holdout_min_sharpe,
                        min_return=holdout_min_return,
                        min_psr_holdout=min_psr_holdout,
                        min_dsr=min_dsr,
                    )
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

    trade_frequency_diagnostics = {
        'best_cv_n_trades': int(best.user_attrs.get('n_trades', 0) or 0),
        'best_cv_trade_density': _finite_metric(best.user_attrs.get('cv_trade_density'), 0.0),
        'best_cv_trade_density_ratio': _finite_metric(best.user_attrs.get('cv_trade_density_ratio'), 0.0),
        'best_activity_regime': str(best.user_attrs.get('activity_regime') or 'unknown'),
        'target_trades_floor': _finite_metric(best.user_attrs.get('target_trades_floor'), 0.0),
        'min_trade_frequency_ratio': _finite_metric(best.user_attrs.get('min_trade_frequency_ratio'), 0.0),
        'holdout_trades': int((holdout_result or {}).get('holdout_trades', 0) or 0),
        'starvation_signature': (holdout_result or {}).get('starvation_signature', {}),
        'top_blockers': (holdout_result or {}).get('main_blockers', []),
    }

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
    cv_psr_metric = best.user_attrs.get('psr_meta') if isinstance(best.user_attrs.get('psr_meta'), dict) else {
        'valid': False,
        'psr': None,
        'reason': 'insufficient_observations',
    }
    cv_dsr_metric = best.user_attrs.get('dsr_cv_meta') if isinstance(best.user_attrs.get('dsr_cv_meta'), dict) else {
        'valid': False,
        'dsr': None,
        'reason': 'insufficient_observations',
    }
    holdout_psr_metric = (holdout_result or {}).get('psr_holdout') if isinstance(holdout_result, dict) else None
    significance_gates = evaluate_significance_gates(
        psr_cv=cv_psr_metric,
        psr_holdout=holdout_psr_metric if isinstance(holdout_psr_metric, dict) else None,
        dsr=cv_dsr_metric,
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
    if deployment_blocked:
        research_confidence_tier = 'SCREENED'
    selection_meta = dict(selection_meta)
    selection_meta['research_confidence_tier'] = research_confidence_tier

    selection_meta['seed_stability'] = seed_stability
    selection_meta['seed_stability_thresholds'] = {
        'min_pass_rate': float(seed_stability_min_pass_rate),
        'max_param_dispersion': float(seed_stability_max_param_dispersion),
        'max_oos_sharpe_dispersion': float(seed_stability_max_oos_sharpe_dispersion),
    }

    gate_profile = {
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
    }

    completed_trials_for_diag = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    pbo_diagnostic = None
    if enable_pbo_diagnostic:
        score_matrix = build_score_matrix_from_trials(completed_trials_for_diag, score_key='sharpe')
        pbo_diagnostic = compute_pbo_from_matrix(score_matrix, score_used='fold_metrics.sharpe')

    study_significance = compose_study_significance_diagnostic(
        enabled=bool(enable_study_significance),
        trials=completed_trials_for_diag,
        bootstrap_iterations=int(study_significance_bootstrap_iterations),
        random_seed=int(study_significance_seed),
        score_source=str(study_significance_score_source),
    )

    stress_costs = None
    if enable_cost_stress_diagnostics and holdout_data and holdout_sym:
        finalists = _candidate_trials_for_holdout(
            study,
            max_candidates=max(1, int(cost_stress_finalists or 1)),
            min_trades=min_total_trades or 6,
        )
        finalist_rows = []
        for finalist in finalists:
            baseline_metrics = evaluate_holdout(
                holdout_data,
                finalist.params,
                coin_name,
                coin_prefix,
                holdout_days,
                pruned_only=pruned_only,
                holdout_mode=holdout_mode,
                base_config=base_cost_config,
            )
            if not baseline_metrics:
                continue

            scenario_metrics = {
                'fees_plus_50pct': evaluate_holdout(
                    holdout_data,
                    finalist.params,
                    coin_name,
                    coin_prefix,
                    holdout_days,
                    pruned_only=pruned_only,
                    holdout_mode=holdout_mode,
                    base_config=_apply_cost_stress_config(base_cost_config, fee_multiplier=cost_stress_fee_multiplier),
                ) or {},
                'slippage_x2': evaluate_holdout(
                    holdout_data,
                    finalist.params,
                    coin_name,
                    coin_prefix,
                    holdout_days,
                    pruned_only=pruned_only,
                    holdout_mode=holdout_mode,
                    base_config=_apply_cost_stress_config(base_cost_config, slippage_multiplier=cost_stress_slippage_multiplier),
                ) or {},
            }
            if bool(getattr(base_cost_config, 'apply_funding', True)):
                scenario_metrics['adverse_funding'] = _funding_adverse_adjustment(
                    baseline_metrics,
                    funding_bps_per_trade=cost_stress_funding_bps_per_trade,
                )

            stress_block = make_stress_costs_block(
                baseline_metrics=baseline_metrics,
                scenario_metrics=scenario_metrics,
            )
            finalist_rows.append({
                'trial_number': int(finalist.number),
                'score': float(finalist.value),
                **stress_block,
            })

        stress_costs = {
            'enabled': True,
            'finalists_requested': int(max(1, int(cost_stress_finalists or 1))),
            'finalists_evaluated': int(len(finalist_rows)),
            'scenarios': {
                'fees_plus_50pct': {'fee_multiplier': float(cost_stress_fee_multiplier)},
                'slippage_x2': {'slippage_multiplier': float(cost_stress_slippage_multiplier)},
                'adverse_funding': {'funding_bps_per_trade': float(cost_stress_funding_bps_per_trade)},
            },
            'finalists': finalist_rows,
        }

    robustness_diagnostics = compose_robustness_diagnostics(
        enabled=bool(enable_pbo_diagnostic or enable_cost_stress_diagnostics),
        pbo=pbo_diagnostic,
        stress_costs=stress_costs,
    )

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
        'deflated_sharpe': dsr, 'dsr_cv': cv_dsr_metric, 'psr_cv': cv_psr_metric, 'psr_holdout': holdout_psr_metric or {},
        'significance_gates': significance_gates, 'version': 'v11.3', 'timestamp': datetime.now().isoformat(),
        'cost_model': cost_model_metadata,
        'selection_meta': selection_meta, 'deployment_blocked': deployment_blocked,
        'deployment_block_reasons': blocked_reasons,
        'best_exploratory_trial': {
            'trial': exploratory_best_trial,
            'optim_score': exploratory_best_score,
        },
        'paper_eligible_promoted_candidate': promoted_paper_candidate,
        'robustness_diagnostics': robustness_diagnostics,
        'study_significance': study_significance,
        'seed_stability': seed_stability,
        'research_confidence_tier': research_confidence_tier,
        'gate_mode': gate_mode,
        'gate_profile': gate_profile,
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
            'reject_prune_log_path': str(event_log_path) if event_log_path else None,
            'study_significance_enabled': bool(enable_study_significance),
        },
        'trade_frequency_diagnostics': trade_frequency_diagnostics}
    result_data['quality'] = assess_result_quality(result_data)
    print(f"  🧪 Quality: {result_data['quality']['rating']}")
    print(
        "  🧭 Research tier: "
        f"{research_confidence_tier} "
        f"(seed pass {seed_stability.get('seeds_passed_holdout', 0)}/{seed_stability.get('seeds_total', 0)})"
    )
    p = _persist_result_json(coin_name, result_data)
    if p: print(f"  💾 {p}")
    if event_log_path:
        print(f"  🧾 Reject/prune log: {event_log_path}")

    print(f"\n  📝 CoinProfile(name='{coin_name}',")
    for k, v in sorted(effective_best_params.items()):
        print(f"    {k}={f'{v:.4f}'.rstrip('0').rstrip('.') if isinstance(v, float) else v},")
    print(f"  )")
    EVENT_LOGGER = None
    return result_data


def aggregate_multiseed_results(
    coin_name,
    seeds,
    run_results,
    *,
    holdout_min_trades=15,
    holdout_min_sharpe=0.0,
    holdout_min_return=0.0,
    min_psr_holdout=None,
    min_dsr=None,
    min_seed_pass_rate=0.67,
    max_seed_param_dispersion=0.60,
    max_seed_oos_sharpe_dispersion=0.35,
    failed_seeds=None,
    emit_artifacts=True,
):
    if not run_results:
        return None

    holdout_min_trades = int(holdout_min_trades or 15)
    holdout_min_sharpe = float(holdout_min_sharpe or 0.0)
    holdout_min_return = float(holdout_min_return or 0.0)
    min_seed_pass_rate = float(min_seed_pass_rate or 0.67)
    max_seed_param_dispersion = float(max_seed_param_dispersion or 0.60)
    max_seed_oos_sharpe_dispersion = float(max_seed_oos_sharpe_dispersion or 0.35)
    failed_seeds = [int(seed) for seed in (failed_seeds or [])]

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

    best_seed_result.setdefault('meta', {})['seed_consensus'] = {
        'seeds': seeds,
        'qualified_runs': len(qualified),
        'total_runs': len(run_results),
        'failed_seeds': failed_seeds,
    }
    best_seed_result['consensus_params'] = consensus_params
    best_seed_result['consensus_revalidated'] = False
    best_seed_result['evaluated_params_source'] = 'selected_seed'

    # Guardrail: top-level params must always correspond to top-level holdout_metrics.
    if best_seed_result.get('evaluated_params_source') != 'selected_seed':
        raise ValueError("Top-level params must come from selected seed unless holdout is revalidated.")

    seed_params = best_seed_result.get('params')
    if not isinstance(seed_params, dict) or not seed_params:
        raise ValueError("Selected-seed params missing; refusing to persist inconsistent result.")

    holdout_metrics = best_seed_result.get('holdout_metrics')
    if not isinstance(holdout_metrics, dict) or not holdout_metrics:
        raise ValueError("Selected-seed holdout metrics missing; refusing to persist inconsistent result.")

    best_seed_result['quality'] = assess_result_quality(best_seed_result)
    if emit_artifacts:
        p = _persist_result_json(coin_name, best_seed_result)
        if p:
            print(f"  💾 Consensus result saved to {p}")

    final_tier = str(best_seed_result.get('research_confidence_tier') or '')
    if emit_artifacts and (
        not bool(best_seed_result.get('deployment_blocked', False))
        and _passes_holdout_gate(
            best_seed_result.get('holdout_metrics', {}),
            min_trades=holdout_min_trades,
            min_sharpe=holdout_min_sharpe,
            min_return=holdout_min_return,
            min_psr_holdout=min_psr_holdout,
            min_dsr=min_dsr,
        )
        and final_tier in {'PAPER_QUALIFIED', 'PROMOTION_READY'}
    ):
        paper_candidate_payload = {
            'coin': coin_name,
            'evaluated_params': best_seed_result['params'],
            'holdout_metrics': best_seed_result['holdout_metrics'],
            'holdout_passed': True,
            'holdout_gate_result': 'pass',
            'run_id': best_seed_result.get('run_id'),
            'study_name': best_seed_result.get('meta', {}).get('study_name') or best_seed_result.get('study_name'),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'research_confidence_tier': final_tier,
            'gate_profile': best_seed_result.get('gate_profile', {}),
        }
        paper_candidate_path = _persist_paper_candidate_json(coin_name, paper_candidate_payload)
        if paper_candidate_path:
            print(f"  📄 Final paper candidate artifact: {paper_candidate_path}")

    return best_seed_result


def optimize_coin_multiseed(all_data, coin_prefix, coin_name, sampler_seeds=None, **kwargs):
    seeds = [int(s) for s in (sampler_seeds or [42])]
    if len(seeds) <= 1:
        return optimize_coin(all_data, coin_prefix, coin_name, sampler_seed=seeds[0], **kwargs)

    print(f"\n🌱 Multi-seed optimization for {coin_name}: seeds={seeds}")
    run_results = []
    base_suffix = kwargs.get('study_suffix', '')
    failed_seeds = []
    for seed in seeds:
        seed_kwargs = dict(kwargs)
        seed_kwargs['study_suffix'] = f"{base_suffix}_s{seed}" if base_suffix else f"s{seed}"
        r = optimize_coin(all_data, coin_prefix, coin_name, sampler_seed=seed, **seed_kwargs)
        if r:
            run_results.append(r)
        else:
            failed_seeds.append(int(seed))

    return aggregate_multiseed_results(
        coin_name,
        seeds,
        run_results,
        holdout_min_trades=kwargs.get('holdout_min_trades', 15),
        holdout_min_sharpe=kwargs.get('holdout_min_sharpe', 0.0),
        holdout_min_return=kwargs.get('holdout_min_return', 0.0),
        min_psr_holdout=kwargs.get('min_psr_holdout', None),
        min_dsr=kwargs.get('min_dsr', None),
        min_seed_pass_rate=kwargs.get('seed_stability_min_pass_rate', 0.67),
        max_seed_param_dispersion=kwargs.get('seed_stability_max_param_dispersion', 0.60),
        max_seed_oos_sharpe_dispersion=kwargs.get('seed_stability_max_oos_sharpe_dispersion', 0.35),
        failed_seeds=failed_seeds,
        emit_artifacts=True,
    )

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
        'robust180': {'plateau_patience': 120, 'plateau_warmup': 60, 'plateau_min_delta': 0.015, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_total_trades': 20, 'n_cv_folds': 5, 'holdout_candidates': 3, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.67, 'seed_stability_max_param_dispersion': 0.60, 'seed_stability_max_oos_sharpe_dispersion': 0.35, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
        'robust120': {'plateau_patience': 90, 'plateau_warmup': 45, 'plateau_min_delta': 0.015, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_total_trades': 15, 'n_cv_folds': 5, 'holdout_candidates': 2, 'holdout_min_trades': 12, 'holdout_min_sharpe': 0.0, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.60, 'seed_stability_max_param_dispersion': 0.70, 'seed_stability_max_oos_sharpe_dispersion': 0.40, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
        'quick':     {'plateau_patience': 45, 'plateau_warmup': 20, 'plateau_min_delta': 0.03, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'single90', 'min_total_trades': 8, 'n_cv_folds': 2, 'holdout_candidates': 1, 'holdout_min_trades': 8, 'holdout_min_sharpe': -0.1, 'holdout_min_return': -0.05, 'require_holdout_pass': False, 'target_trades_per_week': 0.8, 'min_psr': 0.05, 'min_psr_cv': 0.05, 'min_psr_holdout': None, 'min_dsr': None, 'seed_stability_min_pass_rate': 0.50, 'seed_stability_max_param_dispersion': 1.00, 'seed_stability_max_oos_sharpe_dispersion': 0.80},
        'discovery': {'plateau_patience': 70, 'plateau_warmup': 30, 'plateau_min_delta': 0.02, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'single90', 'min_total_trades': 10, 'n_cv_folds': 4, 'holdout_candidates': 2, 'holdout_min_trades': 10, 'holdout_min_sharpe': -0.05, 'holdout_min_return': -0.03, 'require_holdout_pass': False, 'target_trades_per_week': 0.6, 'seed_stability_min_pass_rate': 0.55, 'seed_stability_max_param_dispersion': 0.90, 'seed_stability_max_oos_sharpe_dispersion': 0.60, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
        'paper_ready': {'plateau_patience': 150, 'plateau_warmup': 80, 'plateau_min_delta': 0.012, 'plateau_min_completed': 0, 'holdout_days': 90, 'holdout_mode': 'multi_slice', 'min_total_trades': 28, 'n_cv_folds': 5, 'holdout_candidates': 4, 'holdout_min_trades': 15, 'holdout_min_sharpe': 0.05, 'holdout_min_return': 0.0, 'require_holdout_pass': True, 'target_trades_per_week': 1.0, 'seed_stability_min_pass_rate': 0.75, 'seed_stability_max_param_dispersion': 0.50, 'seed_stability_max_oos_sharpe_dispersion': 0.30, 'min_psr_cv': None, 'min_psr_holdout': None, 'min_dsr': None},
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
            'min_psr': '--min-psr',
            'min_psr_cv': '--min-psr-cv',
            'min_psr_holdout': '--min-psr-holdout',
            'min_dsr': '--min-dsr',
            'seed_stability_min_pass_rate': '--seed-stability-min-pass-rate',
            'seed_stability_max_param_dispersion': '--seed-stability-max-param-dispersion',
            'seed_stability_max_oos_sharpe_dispersion': '--seed-stability-max-oos-sharpe-dispersion',
            'cost_config_path': '--cost-config-path',
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
    parser.add_argument("--preset", type=str, default="paper_ready", choices=["none","robust120","robust180","quick", "paper_ready", "discovery"])
    parser.add_argument("--min-total-trades", type=int, default=0)
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
    parser.add_argument("--cost-config-path", type=str, default=None,
                        help="Optional path to versioned exchange cost assumptions JSON")
    parser.add_argument("--min-psr", type=float, default=0.55,
                        help="Minimum probabilistic Sharpe ratio gate")
    parser.add_argument("--min-psr-cv", type=float, default=None,
                        help="Optional CV PSR gate override (defaults to --min-psr)")
    parser.add_argument("--min-psr-holdout", type=float, default=None,
                        help="Optional holdout PSR gate (disabled when unset)")
    parser.add_argument("--min-dsr", type=float, default=None,
                        help="Optional holdout DSR gate (disabled when unset)")
    parser.add_argument("--seed-stability-min-pass-rate", type=float, default=0.67,
                        help="Minimum holdout pass-rate across sampler seeds for PROMOTION_READY")
    parser.add_argument("--seed-stability-max-param-dispersion", type=float, default=0.60,
                        help="Maximum normalized (IQR/|median|) per-parameter seed dispersion for PROMOTION_READY")
    parser.add_argument("--seed-stability-max-oos-sharpe-dispersion", type=float, default=0.35,
                        help="Maximum holdout Sharpe std across seeds for PROMOTION_READY")
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
        optimize_coin_multiseed(
            all_data,
            PREFIX_FOR_COIN.get(cn, cn),
            cn,
            sampler_seeds=seeds,
            n_trials=args.trials,
            n_jobs=args.jobs,
            plateau_patience=args.plateau_patience,
            plateau_min_delta=args.plateau_min_delta,
            plateau_warmup=args.plateau_warmup,
            plateau_min_completed=args.plateau_min_completed,
            study_suffix=args.study_suffix,
            resume_study=args.resume,
            holdout_days=args.holdout_days,
            min_total_trades=args.min_total_trades,
            n_cv_folds=args.n_cv_folds,
            holdout_candidates=args.holdout_candidates,
            require_holdout_pass=args.require_holdout_pass,
            holdout_min_trades=args.holdout_min_trades,
            holdout_min_sharpe=args.holdout_min_sharpe,
            holdout_min_return=args.holdout_min_return,
            holdout_mode=args.holdout_mode,
            target_trades_per_week=args.target_trades_per_week,
            target_trades_per_year=args.target_trades_per_year,
            min_psr=args.min_psr,
            min_psr_cv=args.min_psr_cv,
            min_psr_holdout=args.min_psr_holdout,
            min_dsr=args.min_dsr,
            seed_stability_min_pass_rate=args.seed_stability_min_pass_rate,
            seed_stability_max_param_dispersion=args.seed_stability_max_param_dispersion,
            seed_stability_max_oos_sharpe_dispersion=args.seed_stability_max_oos_sharpe_dispersion,
            cv_mode=args.cv_mode,
            purge_bars=args.purge_bars,
            purge_days=purge_days,
            embargo_days=args.embargo_days,
            embargo_bars=args.embargo_bars,
            embargo_frac=args.embargo_frac,
            pruned_only=args.pruned_only,
            preset_name=args.preset,
            gate_mode=args.gate_mode,
            cost_config_path=args.cost_config_path,
        )
