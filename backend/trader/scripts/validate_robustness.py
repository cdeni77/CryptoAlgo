#!/usr/bin/env python3
"""
validate_robustness.py — Post-optimization robustness validation suite (v11).

v11 CHANGES:
  - CV consistency check (new)
  - Tighter readiness thresholds
  - Better trade PnL reconstruction
  - Handles v11 optimization metrics

Usage:
    python validate_robustness.py --coin BTC
    python validate_robustness.py --all
    python validate_robustness.py --show
"""
import argparse
import json
import warnings
import sys
import os
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).resolve().parent



def aggregate_cumulative_trial_counts(ledger_path: Optional[Path] = None) -> Dict[str, object]:
    from scripts.optimize import aggregate_cumulative_trial_counts as _aggregate_trial_counts

    return _aggregate_trial_counts(ledger_path=ledger_path)


def resolve_dsr_trial_count(coin_name: str, scope: str = "coin", ledger_path: Optional[Path] = None) -> Dict[str, object]:
    counts = aggregate_cumulative_trial_counts(ledger_path=ledger_path)
    trial_scope = (scope or "coin").lower()
    coin_key = coin_name.upper()

    if trial_scope == "global":
        n_trials_used = int(counts.get('global_total', 0) or 0)
    else:
        n_trials_used = int(counts.get('coin_totals', {}).get(coin_key, 0) or 0)

    return {
        'n_trials_used': max(0, n_trials_used),
        'scope': trial_scope,
        'ledger_timestamp': counts.get('ledger_timestamp'),
        'ledger_path': counts.get('ledger_path'),
        'coin_cumulative_trials': int(counts.get('coin_totals', {}).get(coin_key, 0) or 0),
        'global_cumulative_trials': int(counts.get('global_total', 0) or 0),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. MONTE CARLO TRADE SIMULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def monte_carlo_shuffle(trade_pnls, n_sims=1000, initial_equity=100_000.0):
    if len(trade_pnls) < 10:
        return {'valid': False, 'reason': 'too_few_trades'}
    rng = np.random.default_rng(42)
    max_drawdowns = []
    for _ in range(n_sims):
        shuffled = rng.permutation(trade_pnls)
        equity_curve = initial_equity + np.cumsum(shuffled)
        peak = np.maximum.accumulate(equity_curve)
        dd = np.max((peak - equity_curve) / np.maximum(peak, 1.0))
        max_drawdowns.append(dd)
    max_drawdowns = np.array(max_drawdowns)
    total_returns = (initial_equity + np.sum(trade_pnls)) / initial_equity - 1
    return {
        'valid': True, 'n_sims': n_sims,
        'mc_dd_median': float(np.median(max_drawdowns)),
        'mc_dd_95th': float(np.percentile(max_drawdowns, 95)),
        'mc_dd_99th': float(np.percentile(max_drawdowns, 99)),
        'prob_ruin_25pct': float(np.mean(max_drawdowns > 0.25)),
        'prob_ruin_50pct': float(np.mean(max_drawdowns > 0.50)),
    }


def monte_carlo_resample(trade_pnls, n_sims=1000, initial_equity=100_000.0):
    if len(trade_pnls) < 10:
        return {'valid': False, 'reason': 'too_few_trades'}
    n_trades = len(trade_pnls)
    rng = np.random.default_rng(123)
    sharpe_ratios, max_drawdowns, total_returns = [], [], []
    for _ in range(n_sims):
        sampled = rng.choice(trade_pnls, size=n_trades, replace=True)
        equity_curve = initial_equity + np.cumsum(sampled)
        peak = np.maximum.accumulate(equity_curve)
        dd = np.max((peak - equity_curve) / np.maximum(peak, 1.0))
        avg, std = np.mean(sampled), np.std(sampled)
        sr = avg / std if std > 0 else 0.0
        max_drawdowns.append(dd)
        sharpe_ratios.append(sr)
        total_returns.append((equity_curve[-1] / initial_equity) - 1)
    sr_arr = np.array(sharpe_ratios)
    ret_arr = np.array(total_returns)
    dd_arr = np.array(max_drawdowns)
    return {
        'valid': True, 'n_sims': n_sims,
        'sharpe_5th': float(np.percentile(sr_arr, 5)),
        'sharpe_median': float(np.median(sr_arr)),
        'sharpe_95th': float(np.percentile(sr_arr, 95)),
        'return_5th': float(np.percentile(ret_arr, 5)),
        'return_median': float(np.median(ret_arr)),
        'dd_median': float(np.median(dd_arr)),
        'dd_95th': float(np.percentile(dd_arr, 95)),
        'prob_negative_sharpe': float(np.mean(sr_arr < 0)),
        'prob_loss': float(np.mean(ret_arr < 0)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 2. PARAMETER SENSITIVITY
# ═══════════════════════════════════════════════════════════════════════════

def check_parameter_sensitivity(all_data, best_params, coin_name, coin_prefix, perturbation_pct=0.10):
    from scripts.train_model import Config, run_backtest
    from scripts.optimize import profile_from_params, resolve_target_symbol

    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        return {'valid': False, 'reason': 'symbol_not_found'}

    # v11: Only 9 tunable params
    perturbable = {
        'signal_threshold', 'label_forward_hours', 'label_vol_target',
        'min_momentum_magnitude', 'vol_mult_tp', 'vol_mult_sl',
        'max_hold_hours', 'min_vol_24h', 'max_vol_24h',
    }

    baseline_profile = profile_from_params(best_params, coin_name)
    single_data = {target_sym: all_data[target_sym]}
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24)

    try:
        baseline_result = run_backtest(single_data, config,
                                       profile_overrides={coin_name: baseline_profile})
    except Exception:
        return {'valid': False, 'reason': 'baseline_backtest_failed'}
    if baseline_result is None:
        return {'valid': False, 'reason': 'baseline_none'}

    baseline_sharpe = float(baseline_result.get('sharpe_annual', 0) or 0)
    if baseline_sharpe <= -90: baseline_sharpe = 0.0
    baseline_pf = float(baseline_result.get('profit_factor', 0) or 0)

    sensitivity_results = {}
    sharpe_deltas = []

    for param_name in perturbable:
        if param_name not in best_params: continue
        original_val = best_params[param_name]
        if not isinstance(original_val, (int, float)) or original_val == 0: continue

        delta = abs(original_val * perturbation_pct)
        if isinstance(original_val, int):
            delta = max(1, int(delta))
        perturbed_vals = [original_val - delta, original_val + delta]

        param_sharpes = []
        for pval in perturbed_vals:
            pval = max(1 if isinstance(original_val, int) else 0.001,
                       int(pval) if isinstance(original_val, int) else float(pval))
            test_params = dict(best_params)
            test_params[param_name] = pval
            try:
                test_profile = profile_from_params(test_params, coin_name)
                result = run_backtest(single_data, config,
                                      profile_overrides={coin_name: test_profile})
                s = float(result.get('sharpe_annual', 0) or 0) if result else 0.0
                if s <= -90: s = 0.0
                param_sharpes.append(s)
            except Exception:
                param_sharpes.append(0.0)

        if param_sharpes:
            avg_n = np.mean(param_sharpes)
            drop = baseline_sharpe - avg_n
            sharpe_deltas.append(drop)
            sensitivity_results[param_name] = {
                'baseline_sharpe': round(baseline_sharpe, 3),
                'neighbor_sharpes': [round(s, 3) for s in param_sharpes],
                'avg_neighbor_sharpe': round(avg_n, 3),
                'sharpe_drop': round(drop, 3),
            }

    avg_drop = np.mean(sharpe_deltas) if sharpe_deltas else 0.0
    max_drop = np.max(sharpe_deltas) if sharpe_deltas else 0.0
    fragile = ((avg_drop > baseline_sharpe * 0.30) or (max_drop > baseline_sharpe * 0.50)) if sharpe_deltas else True

    return {
        'valid': True, 'baseline_sharpe': round(baseline_sharpe, 3),
        'baseline_pf': round(baseline_pf, 3),
        'n_params_tested': len(sensitivity_results),
        'avg_sharpe_drop': round(avg_drop, 3),
        'max_sharpe_drop': round(max_drop, 3),
        'fragile': fragile, 'per_param': sensitivity_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. REGIME SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def test_regime_splits(all_data, best_params, coin_name, coin_prefix):
    from scripts.train_model import Config, run_backtest
    from scripts.optimize import profile_from_params, resolve_target_symbol

    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        return {'valid': False, 'reason': 'symbol_not_found'}

    ohlcv = all_data[target_sym]['ohlcv']
    if len(ohlcv) < 500:
        return {'valid': False, 'reason': 'insufficient_data'}

    monthly_returns = ohlcv['close'].resample('30D').last().pct_change().dropna()
    regime_periods = {'bull': [], 'bear': [], 'sideways': []}
    for date, ret in monthly_returns.items():
        if ret > 0.05: regime_periods['bull'].append(date)
        elif ret < -0.05: regime_periods['bear'].append(date)
        else: regime_periods['sideways'].append(date)

    profile = profile_from_params(best_params, coin_name)
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24)

    regime_results = {}
    profitable_count = 0
    tested_count = 0

    for regime_name, dates in regime_periods.items():
        if len(dates) < 2: continue
        regime_start = min(dates) - pd.Timedelta(days=120)
        regime_end = max(dates) + pd.Timedelta(days=30)
        regime_data = {}
        for sym, d in all_data.items():
            feat = d['features'][(d['features'].index >= regime_start) & (d['features'].index <= regime_end)]
            ohlcv_r = d['ohlcv'][(d['ohlcv'].index >= regime_start) & (d['ohlcv'].index <= regime_end)]
            if len(feat) > 200:
                regime_data[sym] = {'features': feat, 'ohlcv': ohlcv_r}
        if not regime_data: continue
        try:
            result = run_backtest(regime_data, config, profile_overrides={coin_name: profile})
            if result:
                sr = float(result.get('sharpe_annual', 0) or 0)
                if sr <= -90: sr = 0.0
                ret = float(result.get('ann_return', 0) or 0)
                trades = int(result.get('n_trades', 0) or 0)
                tested_count += 1
                if sr > 0 or ret > 0: profitable_count += 1
                regime_results[regime_name] = {
                    'sharpe': round(sr, 3), 'return': round(ret, 4),
                    'trades': trades, 'profitable': sr > 0 or ret > 0,
                }
        except Exception:
            continue

    return {
        'valid': tested_count > 0, 'tested_regimes': tested_count,
        'profitable_regimes': profitable_count,
        'regime_dependent': profitable_count < max(1, tested_count - 1),
        'results': regime_results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. CV CONSISTENCY CHECK (v11 NEW)
# ═══════════════════════════════════════════════════════════════════════════

def check_cv_consistency(optim_metrics):
    mean_oos = optim_metrics.get('mean_oos_sharpe', optim_metrics.get('oos_sharpe'))
    min_oos = optim_metrics.get('min_oos_sharpe')
    std_oos = optim_metrics.get('std_oos_sharpe')
    n_folds = optim_metrics.get('n_folds', 1)

    if mean_oos is None:
        return {'valid': False, 'reason': 'no_cv_metrics'}

    checks = {
        'mean_oos_positive': float(mean_oos) > 0,
        'min_fold_acceptable': min_oos is None or float(min_oos) > -0.3,
        'low_variance': std_oos is None or float(std_oos) < 0.5,
        'multiple_folds': n_folds >= 2,
    }
    passed = sum(checks.values())

    return {
        'valid': True,
        'mean_oos_sharpe': float(mean_oos) if mean_oos is not None else None,
        'min_oos_sharpe': float(min_oos) if min_oos is not None else None,
        'std_oos_sharpe': float(std_oos) if std_oos is not None else None,
        'n_folds': n_folds, 'checks': checks,
        'checks_passed': passed, 'checks_total': len(checks),
        'consistent': passed >= len(checks) - 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 5. READINESS SCORE (v11)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ReadinessCheck:
    name: str
    passed: bool
    weight: float
    detail: str = ""


def compute_readiness_score(mc_shuffle, mc_resample, sensitivity, dsr,
                            regime, cv_consistency, holdout_metrics, optim_metrics):
    checks = []

    if mc_shuffle.get('valid'):
        checks.append(ReadinessCheck(
            'mc_dd_acceptable', mc_shuffle['mc_dd_95th'] < 0.30, 15.0,
            f"MC 95th DD={mc_shuffle['mc_dd_95th']:.1%}"))
        checks.append(ReadinessCheck(
            'mc_ruin_low', mc_shuffle['prob_ruin_25pct'] < 0.10, 10.0,
            f"P(DD>25%)={mc_shuffle['prob_ruin_25pct']:.1%}"))

    if mc_resample.get('valid'):
        checks.append(ReadinessCheck(
            'mc_sharpe_positive', mc_resample['sharpe_5th'] > 0, 15.0,
            f"5th pctl Sharpe={mc_resample['sharpe_5th']:.3f}"))

    if sensitivity.get('valid'):
        checks.append(ReadinessCheck(
            'param_stability', not sensitivity['fragile'], 15.0,
            f"Fragile={sensitivity['fragile']}, avg_drop={sensitivity['avg_sharpe_drop']:.3f}"))

    if dsr.get('valid'):
        sig = dsr.get('significant_at_10pct', dsr.get('significant_10pct', False))
        checks.append(ReadinessCheck(
            'dsr_significant', sig, 10.0,
            f"DSR={dsr.get('dsr', 0):.3f}, p={dsr.get('p_value', 1):.3f}"))

    if regime.get('valid'):
        checks.append(ReadinessCheck(
            'multi_regime', not regime['regime_dependent'], 10.0,
            f"Profitable {regime['profitable_regimes']}/{regime['tested_regimes']} regimes"))

    if cv_consistency.get('valid'):
        checks.append(ReadinessCheck(
            'cv_consistent', cv_consistency['consistent'], 15.0,
            f"CV {cv_consistency['checks_passed']}/{cv_consistency['checks_total']} passed"))

    psr = float(optim_metrics.get('psr', 0.0) or 0.0)
    checks.append(ReadinessCheck(
        'psr_confident', psr >= 0.60, 10.0,
        f"PSR={psr:.2f} (need >=0.60)"))

    ho_sharpe = holdout_metrics.get('holdout_sharpe', 0)
    ho_trades = holdout_metrics.get('holdout_trades', 0)
    ho_return = holdout_metrics.get('holdout_return', 0)

    if ho_trades > 0:
        checks.append(ReadinessCheck(
            'holdout_positive', ho_trades >= 15 and ho_sharpe > 0 and ho_return > 0, 10.0,
            f"Holdout trades={ho_trades}, SR={ho_sharpe:.3f}, ret={ho_return:.2%}"))

    n_trades = int(optim_metrics.get('n_trades', 0) or 0)
    checks.append(ReadinessCheck(
        'sufficient_trades', n_trades >= 30, 5.0,
        f"Trades={n_trades} (need >=30)"))

    optim_sr = optim_metrics.get('mean_oos_sharpe',
                optim_metrics.get('sharpe', optim_metrics.get('oos_sharpe', 0)))
    if optim_sr and float(optim_sr) > 0 and ho_sharpe and float(ho_sharpe) > 0:
        decay = 1.0 - (float(ho_sharpe) / float(optim_sr))
        checks.append(ReadinessCheck(
            'sharpe_decay', decay < 0.50, 10.0,
            f"Decay={decay:.0%}"))

    total_weight = sum(c.weight for c in checks)
    if total_weight == 0:
        return {'score': 0, 'rating': 'UNKNOWN', 'details': [], 'n_checks': 0,
                'checks_passed': 0, 'checks_failed': 0}

    weighted_score = sum(c.weight for c in checks if c.passed) / total_weight * 100

    if weighted_score >= 80: rating = 'READY'
    elif weighted_score >= 60: rating = 'CAUTIOUS'
    elif weighted_score >= 40: rating = 'WEAK'
    else: rating = 'REJECT'

    return {
        'score': round(weighted_score, 1), 'rating': rating,
        'n_checks': len(checks),
        'checks_passed': sum(1 for c in checks if c.passed),
        'checks_failed': sum(1 for c in checks if not c.passed),
        'details': [{'name': c.name, 'passed': c.passed, 'weight': c.weight, 'detail': c.detail}
                    for c in checks],
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_validation(
    coin_name,
    optimization_result,
    all_data,
    mc_shuffle_sims=1000,
    mc_resample_sims=1000,
    dsr_trial_scope="coin",
):
    from scripts.train_model import Config, run_backtest
    from scripts.optimize import (
        profile_from_params, resolve_target_symbol,
        COIN_MAP, PREFIX_FOR_COIN, compute_deflated_sharpe,
    )

    params = optimization_result.get('params', {})
    coin_prefix = optimization_result.get('prefix', PREFIX_FOR_COIN.get(coin_name, coin_name))
    holdout_metrics = optimization_result.get('holdout_metrics', {})
    optim_metrics = optimization_result.get('optim_metrics', {})

    print(f"\n{'='*70}")
    print(f"🔬 ROBUSTNESS VALIDATION — {coin_name} (v11)")
    print(f"{'='*70}")

    target_sym = resolve_target_symbol(all_data, coin_prefix, coin_name)
    if not target_sym:
        print(f"  ❌ Cannot resolve symbol for {coin_name}")
        return {'valid': False, 'reason': 'symbol_not_found', 'coin': coin_name}

    # Run baseline backtest
    print(f"  📊 Running full backtest...")
    profile = profile_from_params(params, coin_name)
    single_data = {target_sym: all_data[target_sym]}
    config = Config(max_positions=1, leverage=4, min_signal_edge=0.00,
                    max_ensemble_std=0.10, train_embargo_hours=24)

    try:
        result = run_backtest(single_data, config, profile_overrides={coin_name: profile})
    except Exception as e:
        print(f"  ❌ Backtest failed: {e}")
        return {'valid': False, 'reason': f'backtest_error: {e}', 'coin': coin_name}

    if result is None:
        return {'valid': False, 'reason': 'backtest_none', 'coin': coin_name}

    n_trades = int(result.get('n_trades', 0))
    sharpe = float(result.get('sharpe_annual', 0) or 0)
    if sharpe <= -90: sharpe = 0.0

    # Use real trade-level PnLs from backtest when available
    trade_pnls = np.array(result.get('trade_pnls', []) or [], dtype=float)

    # 1. MC Shuffle
    print(f"  🎲 Monte Carlo shuffle...")
    mc_shuffle = (
        monte_carlo_shuffle(trade_pnls, n_sims=mc_shuffle_sims)
        if len(trade_pnls) >= 10 else {'valid': False}
    )
    if mc_shuffle.get('valid'):
        print(f"     DD 95th: {mc_shuffle['mc_dd_95th']:.1%} | P(ruin): {mc_shuffle['prob_ruin_25pct']:.1%}")

    # 2. MC Resample
    print(f"  🎲 Monte Carlo resample...")
    mc_resample = (
        monte_carlo_resample(trade_pnls, n_sims=mc_resample_sims)
        if len(trade_pnls) >= 10 else {'valid': False}
    )
    if mc_resample.get('valid'):
        print(f"     Sharpe 5th: {mc_resample['sharpe_5th']:.3f} | P(loss): {mc_resample['prob_loss']:.1%}")

    # 3. Param Sensitivity
    print(f"  🔧 Parameter sensitivity...")
    sensitivity = check_parameter_sensitivity(all_data, params, coin_name, coin_prefix)
    if sensitivity.get('valid'):
        print(f"     Fragile: {sensitivity['fragile']} | Avg drop: {sensitivity['avg_sharpe_drop']:.3f}")

    # 4. DSR
    print(f"  📐 Deflated Sharpe Ratio...")
    dsr_meta = resolve_dsr_trial_count(coin_name, scope=dsr_trial_scope)
    n_trials_for_dsr = int(dsr_meta.get('n_trials_used', 0) or 0)
    if n_trials_for_dsr < 2:
        n_trials_for_dsr = int(optimization_result.get('n_trials', 100) or 100)
        dsr_meta['n_trials_used'] = n_trials_for_dsr
        dsr_meta['fallback_to_single_run'] = True
    else:
        dsr_meta['fallback_to_single_run'] = False

    oos_sr = float(optim_metrics.get('mean_oos_sharpe', optim_metrics.get('oos_sharpe', sharpe)) or 0)
    dsr = compute_deflated_sharpe(oos_sr, n_trades, n_trials_for_dsr)
    if dsr.get('valid'):
        print(
            f"     DSR: {dsr['dsr']:.3f} (p={dsr['p_value']:.3f}) "
            f"| trials={n_trials_for_dsr} ({dsr_meta['scope']})"
        )

    # 5. Regime Split
    print(f"  🌤️ Regime split test...")
    regime = test_regime_splits(all_data, params, coin_name, coin_prefix)
    if regime.get('valid'):
        print(f"     Profitable: {regime['profitable_regimes']}/{regime['tested_regimes']} regimes")

    # 6. CV Consistency (v11)
    print(f"  📊 CV consistency check...")
    cv_consistency = check_cv_consistency(optim_metrics)
    if cv_consistency.get('valid'):
        print(f"     Consistent: {cv_consistency['consistent']} "
              f"({cv_consistency['checks_passed']}/{cv_consistency['checks_total']})")

    # 7. Readiness Score
    print(f"\n  🏁 Computing readiness score...")
    readiness = compute_readiness_score(
        mc_shuffle, mc_resample, sensitivity, dsr,
        regime, cv_consistency, holdout_metrics, optim_metrics,
    )

    emoji = {'READY': '✅', 'CAUTIOUS': '⚠️', 'WEAK': '🟡', 'REJECT': '❌'}.get(readiness['rating'], '?')
    print(f"\n  {emoji} {coin_name}: {readiness['rating']} — Score: {readiness['score']:.0f}/100")
    print(f"     Checks: {readiness['checks_passed']}/{readiness['n_checks']} passed")
    for detail in readiness['details']:
        status = '✅' if detail['passed'] else '❌'
        print(f"     {status} {detail['name']}: {detail['detail']}")

    # Save
    validation_data = {
        'coin': coin_name, 'version': 'v11',
        'timestamp': datetime.now().isoformat(),
        'mc_shuffle': mc_shuffle, 'mc_resample': mc_resample,
        'parameter_sensitivity': sensitivity,
        'deflated_sharpe': dsr, 'dsr_metadata': dsr_meta, 'regime_split': regime,
        'cv_consistency': cv_consistency,
        'readiness': readiness,
    }

    for d in [SCRIPT_DIR / "optimization_results", Path.cwd() / "optimization_results"]:
        try:
            d.mkdir(parents=True, exist_ok=True)
            p = d / f"{coin_name}_validation.json"
            with open(p, 'w') as f:
                json.dump(validation_data, f, indent=2, default=str)
            print(f"  💾 Saved to {p}")
            break
        except (PermissionError, OSError):
            continue

    return validation_data


def show_validation_results():
    candidates = [
        SCRIPT_DIR / "optimization_results",
        Path.cwd() / "optimization_results",
    ]
    results = []
    for d in candidates:
        if d.exists():
            results.extend(d.glob("*_validation.json"))

    if not results:
        print("No validation results found.")
        return

    print(f"\n{'='*80}")
    print(f"🔬 VALIDATION RESULTS SUMMARY (v11)")
    print(f"{'='*80}")

    for rpath in sorted(results):
        with open(rpath) as f:
            r = json.load(f)

        readiness = r.get('readiness', {})
        rating = readiness.get('rating', '?')
        score = readiness.get('score', 0)
        emoji = {'READY': '✅', 'CAUTIOUS': '⚠️', 'WEAK': '🟡', 'REJECT': '❌'}.get(rating, '?')

        dsr = r.get('deflated_sharpe', {})
        dsr_meta = r.get('dsr_metadata', {})
        mc = r.get('mc_shuffle', {})
        sens = r.get('parameter_sensitivity', {})
        cv = r.get('cv_consistency', {})

        print(f"\n{emoji} {r.get('coin', '?')} — Score: {score:.0f}/100 — {rating}")
        print(f"   Checks: {readiness.get('checks_passed', 0)}/{readiness.get('n_checks', 0)} passed")

        if dsr.get('valid'):
            print(
                f"   DSR: {dsr.get('dsr', 0):.3f} (p={dsr.get('p_value', 1):.3f})"
                f" | trials={dsr_meta.get('n_trials_used', '?')} ({dsr_meta.get('scope', 'coin')})"
            )
        if mc.get('valid'):
            print(f"   MC DD 95th: {mc.get('mc_dd_95th', 0):.1%} | P(ruin): {mc.get('prob_ruin_25pct', 0):.1%}")
        if sens.get('valid'):
            print(f"   Param Fragile: {sens.get('fragile', '?')} | Avg drop: {sens.get('avg_sharpe_drop', 0):.3f}")
        if cv.get('valid'):
            print(f"   CV Consistent: {cv.get('consistent', '?')} | "
                  f"Mean OOS SR: {cv.get('mean_oos_sharpe', '?')}")


COIN_MAP = {
    'BIP': 'BTC', 'BTC': 'BTC', 'ETP': 'ETH', 'ETH': 'ETH',
    'XPP': 'XRP', 'XRP': 'XRP', 'SLP': 'SOL', 'SOL': 'SOL',
    'DOP': 'DOGE', 'DOGE': 'DOGE',
}
ALL_COINS = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-optimization robustness validation (v11)")
    parser.add_argument("--coin", type=str)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--mc-shuffle-sims", type=int, default=1000,
                        help="Monte Carlo shuffle simulation count")
    parser.add_argument("--mc-resample-sims", type=int, default=1000,
                        help="Monte Carlo resample simulation count")
    parser.add_argument("--fast", action="store_true",
                        help="Use faster validation defaults for same-day screening")
    parser.add_argument("--dsr-trial-scope", type=str, default="coin", choices=["coin", "global"],
                        help="Use cumulative trial count by coin or globally when computing DSR")
    args = parser.parse_args()

    if args.fast:
        args.mc_shuffle_sims = min(args.mc_shuffle_sims, 300)
        args.mc_resample_sims = min(args.mc_resample_sims, 300)

    if args.show:
        show_validation_results()
        sys.exit(0)

    if args.all:
        coins_to_validate = ALL_COINS
    elif args.coin:
        coins_to_validate = [COIN_MAP.get(args.coin.upper(), args.coin.upper())]
    else:
        parser.print_help()
        sys.exit(1)

    print("📂 Loading data...")
    sys.path.insert(0, str(SCRIPT_DIR))
    from scripts.train_model import load_data
    all_data = load_data()
    if not all_data:
        print("❌ No data loaded.")
        sys.exit(1)

    for coin in coins_to_validate:
        opt_file = None
        for d in [SCRIPT_DIR / "optimization_results", Path.cwd() / "optimization_results"]:
            candidate = d / f"{coin}_optimization.json"
            if candidate.exists():
                opt_file = candidate
                break

        if not opt_file:
            print(f"\n⚠️ No optimization result for {coin}, skipping.")
            continue

        with open(opt_file) as f:
            opt_result = json.load(f)

        run_validation(
            coin,
            opt_result,
            all_data,
            mc_shuffle_sims=max(50, args.mc_shuffle_sims),
            mc_resample_sims=max(50, args.mc_resample_sims),
            dsr_trial_scope=args.dsr_trial_scope,
        )
