#!/usr/bin/env python3
"""
ACCURATE Optuna Optimization — uses real run_backtest
======================================================
Slower than fast version (~10-12 min/trial) but results are trustworthy.
Loads data ONCE, then passes to run_backtest for each trial.

Usage:
  python optimize_v7_accurate.py --n-trials 30    # Quick overnight run
  python optimize_v7_accurate.py --n-trials 60    # Full run
"""
import argparse
import sys
import math
import warnings
import io
from contextlib import redirect_stdout
from datetime import datetime

warnings.filterwarnings('ignore')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    print("❌ pip install optuna --break-system-packages")
    sys.exit(1)

from train_model import Config, load_data, run_backtest


def create_config(trial: optuna.Trial) -> Config:
    """Only 5 params. Small search space = fewer trials needed."""
    return Config(
        signal_threshold=trial.suggest_float("threshold", 0.62, 0.78, step=0.02),
        min_momentum_magnitude=trial.suggest_float("momentum", 0.04, 0.09, step=0.01),
        vol_mult_tp=trial.suggest_float("TP", 3.0, 5.5, step=0.5),
        vol_mult_sl=trial.suggest_float("SL", 1.5, 2.75, step=0.25),
        max_hold_hours=trial.suggest_int("hold", 48, 144, step=24),
        # Fixed
        leverage=4, min_val_auc=0.54,
        excluded_symbols=['BIP', 'DOP'],
        trailing_active=False, breakeven_trigger=999.0, trailing_mult=999.0,
        max_positions=3, position_size=0.15, vol_sizing_target=0.025,
    )


def objective(trial, data):
    config = create_config(trial)
    
    # Suppress output
    f = io.StringIO()
    with redirect_stdout(f):
        result = run_backtest(data, config)
    
    if result is None or result.get('n_trades', 0) < 50:
        return -10.0
    
    sharpe = result.get('sharpe_annual', -99)
    pf = result.get('profit_factor', 0)
    dd = result.get('max_drawdown', 1.0)
    ret = result.get('total_return', -1)
    n = result['n_trades']
    
    if sharpe < -5 or ret < -0.9 or pf <= 0:
        return -10.0
    
    dd_penalty = max(0, dd - 0.45) * 3.0
    trade_penalty = max(0, (80 - n) / 80) * 0.3
    score = sharpe * math.sqrt(pf) - dd_penalty - trade_penalty
    
    trial.set_user_attr("trades", n)
    trial.set_user_attr("return", ret)
    trial.set_user_attr("dd", dd)
    trial.set_user_attr("pf", pf)
    trial.set_user_attr("sharpe", sharpe)
    trial.set_user_attr("ann_ret", result.get('ann_return', 0))
    trial.set_user_attr("wr", result.get('win_rate', 0))
    trial.set_user_attr("raw_pnl", result.get('avg_raw_pnl', 0))
    
    return score


def fmt(trial):
    p, a = trial.params, trial.user_attrs
    return (f"  Score:{trial.value:+.3f} Sharpe:{a.get('sharpe',0):.2f} "
            f"Ret:{a.get('return',0):.0%} PF:{a.get('pf',0):.2f} "
            f"DD:{a.get('dd',0):.0%} WR:{a.get('wr',0):.0%} "
            f"N:{a.get('trades',0)}\n"
            f"  thr={p['threshold']:.2f} mom={p['momentum']:.2f} "
            f"TP={p['TP']:.1f} SL={p['SL']:.2f} hold={p['hold']}h")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔬 ACCURATE Optuna — v7 Momentum (real backtest)")
    print("=" * 60)
    print(f"Trials: {args.n_trials} (~{args.n_trials * 10 / 60:.0f}h estimated)")
    print(f"Symbols: ETH + XRP + SOL")
    print()
    
    data = load_data()
    
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    best_score = -999
    start = datetime.now()
    
    def callback(study, trial):
        nonlocal best_score
        elapsed = (datetime.now() - start).total_seconds()
        n_done = trial.number + 1
        avg_time = elapsed / n_done
        remaining = avg_time * (args.n_trials - n_done) / 60
        
        status = f"[{n_done}/{args.n_trials}] {avg_time:.0f}s/trial ~{remaining:.0f}min left"
        
        if trial.value is not None and trial.value > -5:
            print(f"{status}")
            print(fmt(trial))
            
            if trial.value > best_score:
                best_score = trial.value
                print(f"  ⭐ NEW BEST\n")
            else:
                print()
    
    study.optimize(
        lambda t: objective(t, data),
        n_trials=args.n_trials,
        callbacks=[callback],
    )
    
    elapsed = (datetime.now() - start).total_seconds()
    
    # Results
    print("=" * 60)
    print(f"DONE in {elapsed/60:.0f} min")
    print("=" * 60)
    
    good = sorted([t for t in study.trials if t.value and t.value > -5],
                   key=lambda t: t.value, reverse=True)
    
    print(f"\n🥇 TOP 5:")
    for i, t in enumerate(good[:5], 1):
        print(f"#{i} (trial {t.number}):")
        print(fmt(t))
        print()
    
    if good:
        top5 = good[:min(5, len(good))]
        print("PARAM CONVERGENCE (top 5):")
        for p in ['threshold', 'momentum', 'TP', 'SL', 'hold']:
            v = [t.params[p] for t in top5]
            print(f"  {p:12s}: {min(v):.2f} - {max(v):.2f}")
        
        profitable = sum(1 for t in good[:10] if t.user_attrs.get('return', -1) > 0)
        print(f"\n✅ {profitable}/{min(10, len(good))} top configs profitable")
    
    # Best config
    b = study.best_trial
    bp = b.params
    print(f"\n{'='*60}")
    print(f"OPTIMAL: thr={bp['threshold']:.2f} mom={bp['momentum']:.2f} "
          f"TP={bp['TP']:.1f} SL={bp['SL']:.2f} hold={bp['hold']}h")
    print(f"{'='*60}")
    
    # Final run with full output
    print("\n🔄 FULL BACKTEST WITH BEST PARAMS:\n")
    best_config = Config(
        signal_threshold=bp['threshold'],
        min_momentum_magnitude=bp['momentum'],
        vol_mult_tp=bp['TP'],
        vol_mult_sl=bp['SL'],
        max_hold_hours=bp['hold'],
        leverage=4, min_val_auc=0.54,
        excluded_symbols=['BIP', 'DOP'],
        trailing_active=False, breakeven_trigger=999.0, trailing_mult=999.0,
        max_positions=3, position_size=0.15, vol_sizing_target=0.025,
    )
    run_backtest(data, best_config)


if __name__ == "__main__":
    main()