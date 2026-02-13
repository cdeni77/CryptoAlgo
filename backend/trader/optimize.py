#!/usr/bin/env python3
"""
optimize.py — Per-coin Optuna parameter optimization (Parallel Enabled).

Runs the existing backtest filtered to a single coin at a time,
optimizing threshold, exits, label params, and ML hyperparams.

Usage:
    python optimize.py --coin BIP --trials 50 --jobs 4
    python optimize.py --all --trials 200 --jobs 16
    python optimize.py --show                        # Show saved results
"""
import argparse
import json
import warnings
import sys
import io
import os
import logging
import sqlite3
import functools  # <--- Critical for multiprocessing
import traceback
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

# Force single-threaded linear algebra BEFORE importing numpy/pandas/sklearn
# This prevents 16 workers x 20 threads = 320 threads crashing the CPU.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import optuna
from optuna.samplers import TPESampler

# Import your existing logic
from train_model import Config, load_data, run_backtest
from coin_profiles import (
    CoinProfile, COIN_PROFILES, 
    BTC_EXTRA_FEATURES, SOL_EXTRA_FEATURES, DOGE_EXTRA_FEATURES,
)

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------
warnings.filterwarnings('ignore')
# Turn off Optuna/LightGBM logging to keep console clean during parallel runs
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

RESULTS_DIR = Path("./optimization_results")
RESULTS_DIR.mkdir(exist_ok=True)

# Map coin prefix to symbol (filled at runtime)
PREFIX_TO_SYMBOL: Dict[str, str] = {}
DEBUG_TRIALS = False

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------

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


def _fmt_pct(value, decimals: int = 1, fallback: str = "?") -> str:
    n = _as_number(value)
    return f"{n:.{decimals}%}" if n is not None else fallback


def _fmt_float(value, decimals: int = 3, fallback: str = "?") -> str:
    n = _as_number(value)
    return f"{n:.{decimals}f}" if n is not None else fallback


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

def objective(trial: optuna.Trial, all_data: Dict, coin_prefix: str, coin_name: str) -> float:
    """
    Optuna objective: run single-coin backtest, return composite score.
    Thread-safe implementation for parallel execution.
    """
    profile = create_trial_profile(trial, coin_name)

    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        _set_reject_reason(trial, f'missing_symbol:{coin_prefix}/{coin_name}')
        return -99.0

    # Filter data to only this coin (reduces pickling overhead to workers)
    single_data = {target_sym: all_data[target_sym]}

    # Config: 1 position max for single-coin test
    # IMPORTANT: We force n_jobs=1 for LightGBM here via code if possible,
    # but the environment variables at the top of the file are the real safeguard.
    config = Config(
        max_positions=1,
        leverage=4,
        min_signal_edge=0.00,
        max_ensemble_std=0.10,
        train_embargo_hours=24,
        oos_eval_days=60,
    )

    # Capture stdout in normal mode to reduce noise, but allow full logs when debugging.
    captured_output = io.StringIO()
    original_stdout = sys.stdout
    start_ts = time.time()

    try:
        if not DEBUG_TRIALS:
            sys.stdout = captured_output
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
    finally:
        sys.stdout = original_stdout  # Restore stdout immediately

    trial.set_user_attr('elapsed_sec', round(time.time() - start_ts, 3))
    if DEBUG_TRIALS and captured_output.getvalue().strip():
        tail = '\n'.join(captured_output.getvalue().strip().splitlines()[-8:])
        print(f"\n--- trial {trial.number} backtest tail ---\n{tail}\n--- end tail ---")

    if result is None:
        _set_reject_reason(trial, 'result_none')
        return -99.0

    # Scoring Logic
    n_trades = int(result.get('n_trades', 0) or 0)
    sharpe = float(result.get('sharpe_annual', 0.0) or 0.0)
    pf = float(result.get('profit_factor', 0.0) or 0.0)
    dd = float(result.get('max_drawdown', 1.0) or 1.0)
    wr = float(result.get('win_rate', 0.0) or 0.0)
    ann_ret = float(result.get('ann_return', -1.0) or -1.0)
    trades_per_year = float(result.get('trades_per_year', 0.0) or 0.0)
    oos_sharpe = float(result.get('oos_sharpe', 0.0) or 0.0)
    oos_return = float(result.get('oos_return', 0.0) or 0.0)
    oos_trades = int(result.get('oos_trades', 0) or 0)

    # Treat sentinel values as missing data instead of a hard catastrophic score.
    if sharpe <= -90:
        sharpe = 0.0
    if oos_sharpe <= -90:
        oos_sharpe = 0.0

    pf_bonus = max(0, (pf - 1.0)) * 0.5 if pf > 0 else 0.0
    dd_penalty = max(0.0, dd - 0.30) * 3.0

    # === Final: Crypto-selective (favor 3–12 trades/year per coin) ===
    trade_penalty = 0.0

    # Keep this as a soft penalty so Optuna can still rank and escape bad regions
    # instead of collapsing to identical -99.0 values.
    if n_trades < 15:
        _set_reject_reason(trial, f'too_few_trades:{n_trades}')
        trade_penalty += min(3.0, (15 - n_trades) * 0.2)

    if trades_per_year < 10:  # <~1/year total — noticeable for ultra-selective
        trade_penalty += 0.5 + max(0.0, (5 - trades_per_year) * 0.05)
    elif trades_per_year < 25:  # Light if a bit sparse
        trade_penalty += 0.25
    elif trades_per_year > 100:  # Overtrading penalty
        trade_penalty += 0.5

    # Bonus for ideal single-coin selectivity (high-conviction, low noise)
    if 30 <= trades_per_year <= 80: 
        trade_penalty -= 0.4

    # Force generalization: prioritize out-of-sample behavior over in-sample optics
    score = (0.6 * oos_sharpe) + (0.4 * sharpe) + pf_bonus - dd_penalty - trade_penalty

    if oos_trades < 5:
        score -= 0.5
    if oos_return < 0:
        score -= min(1.0, abs(oos_return) * 10)

    if ann_ret < -0.05:
        score = min(score, -1.0)

    # Store metrics in trial (useful for analysis later)
    trial.set_user_attr('n_trades', n_trades)
    trial.set_user_attr('win_rate', round(wr, 3))
    trial.set_user_attr('ann_return', round(ann_ret, 4))
    trial.set_user_attr('sharpe', round(sharpe, 3))
    trial.set_user_attr('profit_factor', round(pf, 3))
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


# -----------------------------------------------------------------------------
# MAIN OPTIMIZATION LOOP
# -----------------------------------------------------------------------------

def optimize_coin(all_data: Dict, coin_prefix: str, coin_name: str, 
                  n_trials: int = 50, n_jobs: int = 1,
                  plateau_patience: int = 80, plateau_min_delta: float = 0.02,
                  plateau_warmup: int = 40,
                  study_suffix: str = ""):
    """Run Optuna optimization for a single coin with parallel support."""
    
    # 1. Setup Persistent Storage (Required for parallel jobs)
    # This allows multiple processes to write results to the same DB
    storage_url = "sqlite:///optuna_trading.db"
    study_name = f"optimize_{coin_name}{'_' + study_suffix if study_suffix else ''}"

    print(f"\n{'='*60}")
    print(f"🚀 OPTIMIZING {coin_name} ({coin_prefix})")
    print(f"   Trials: {n_trials} | Cores: {n_jobs} | Storage: {storage_url}")
    print(f"   Plateau stop: patience={plateau_patience}, min_delta={plateau_min_delta}, warmup={plateau_warmup}")
    print(f"{'='*60}")

    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42, n_startup_trials=min(10, n_trials // 3)),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True 
    )

    # 2. Run Optimization
    # CRITICAL CHANGE: Use functools.partial instead of lambda.
    # Lambdas cannot be pickled, so joblib falls back to threading (single core).
    # partials ARE pickleable, so joblib can spawn real parallel processes.
    objective_func = functools.partial(
        objective, 
        all_data=all_data, 
        coin_prefix=coin_prefix, 
        coin_name=coin_name
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
            show_progress_bar=True,
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

    # 3. Report Results
    best = study.best_trial

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
    print(f"\n✅ BEST RESULT for {coin_name}:")
    print(f"  Score:        {_fmt_float(best.value, 3)}")
    print(f"  Trades:       {best.user_attrs.get('n_trades', '?')}")
    print(f"  Win Rate:     {_fmt_pct(best.user_attrs.get('win_rate'), 1)}")
    print(f"  Ann Return:   {_fmt_pct(best.user_attrs.get('ann_return'), 2)}")
    print(f"  Sharpe:       {_fmt_float(best.user_attrs.get('sharpe'), 3)}")
    print(f"  Profit Factor:{_fmt_float(best.user_attrs.get('profit_factor'), 3)}")
    print(f"  Max Drawdown: {_fmt_pct(best.user_attrs.get('max_drawdown'), 2)}")

    # 4. Save JSON
    result_data = {
        'coin': coin_name,
        'prefix': coin_prefix,
        'score': best.value,
        'metrics': dict(best.user_attrs),
        'params': best.params,
        'n_trials': len(study.trials),
        'timestamp': datetime.now().isoformat(),
    }
    result_path = RESULTS_DIR / f"{coin_name}_optimization.json"
    with open(result_path, 'w') as f:
        json.dump(result_data, f, indent=2)
    print(f"\n  💾 Saved to {result_path}")

    # 5. Generate Code Snippet (Very helpful for copy-pasting)
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
    results = sorted(RESULTS_DIR.glob("*_optimization.json"))
    if not results:
        print("No optimization results found.")
        return

    print(f"\n{'='*80}")
    print(f"📊 OPTIMIZATION RESULTS SUMMARY")
    print(f"{'='*80}")

    for rpath in results:
        with open(rpath) as f:
            r = json.load(f)
        m = r.get('metrics', {})
        print(f"\n{r['coin']} ({r.get('prefix','?')}) — {r['n_trials']} trials — {r['timestamp'][:16]}")
        print(
            f"  Score: {_fmt_float(r.get('score'), 3)} | "
            f"Sharpe: {_fmt_float(m.get('sharpe'), 3)} | "
            f"WR: {_fmt_pct(m.get('win_rate'), 1)} | "
            f"PF: {_fmt_float(m.get('profit_factor'), 3)} | "
            f"DD: {_fmt_pct(m.get('max_drawdown'), 1)} | "
            f"Trades: {m.get('n_trades', '?')}"
        )

# -----------------------------------------------------------------------------
# RUNTIME CONFIG
# -----------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(description="Per-coin Optuna parameter optimization")
    parser.add_argument("--coin", type=str, help="Coin prefix or name (e.g. BIP, BTC)")
    parser.add_argument("--all", action="store_true", help="Optimize all coins")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel jobs (-1 = all cores)")
    parser.add_argument("--show", action="store_true", help="Show saved results")
    parser.add_argument("--study-suffix", type=str, default="",
                        help="Optional suffix to isolate studies per launch (useful for parallel runs)")
    parser.add_argument("--plateau-patience", type=int, default=80,
                        help="Stop if best score does not improve for this many trials")
    parser.add_argument("--plateau-min-delta", type=float, default=0.02,
                        help="Minimum best-score improvement to reset plateau counter")
    parser.add_argument("--plateau-warmup", type=int, default=40,
                        help="Minimum completed trials before plateau checks start")
    parser.add_argument("--debug-trials", action="store_true",
                        help="Show per-trial backtest logs/exceptions for debugging")
    args = parser.parse_args()

    DEBUG_TRIALS = args.debug_trials

    # Initialize SQLite WAL mode BEFORE running anything else
    init_db_wal()

    if args.show:
        show_results()
        sys.exit(0)

    if not args.coin and not args.all:
        parser.print_help()
        sys.exit(1)

    # Load data once in the main process
    # (On Windows, this is pickled to workers. On Linux, it's efficient copy-on-write)
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
                    study_suffix=args.study_suffix,
                )
    else:
        # Single coin
        coin_input = args.coin.upper()
        coin_name = COIN_MAP.get(coin_input, coin_input)
        prefix = PREFIX_FOR_COIN.get(coin_name, coin_input)
        
        if prefix not in PREFIX_TO_SYMBOL:
             # Try direct match
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
            study_suffix=args.study_suffix,
        )
