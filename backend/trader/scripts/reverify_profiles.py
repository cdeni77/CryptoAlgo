#!/usr/bin/env python3
"""
reverify_profiles.py — Re-verify each coin's existing COIN_PROFILES config
against today's data using weekly-retrain + pruned features (live-accurate).

Coins with verified-winning configs baked into coin_profiles.py may have been
overlooked by gap_search (which only tests a subset of strategies). This script
uses each coin's own profile strategy/params and runs a full 5yr walk-forward
backtest to confirm whether the config STILL passes in today's regime.

Usage:
    python -m scripts.reverify_profiles --coins BTC,LINK,LTC,SOL,ADA,SHIB,PEPE,NEAR,SUI,BCH,XLM,XRP --jobs 4
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

RESULTS_PATH = Path('optimization_results/reverify_profiles.json')

# Pass gate — want real edge, not just break-even
PASS_SR = 0.20
PASS_TPY = 12

_WORKER_DATA: Optional[Dict] = None


def _init_worker() -> None:
    global _WORKER_DATA
    from scripts.train_model import load_data
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _WORKER_DATA = load_data()


def _verify_one(coin: str) -> Dict[str, Any]:
    global _WORKER_DATA
    import core.coin_profiles as cp
    from scripts.train_model import Config, run_backtest

    if _WORKER_DATA is None:
        _init_worker()

    profile = cp.COIN_PROFILES[coin]
    filtered = {sym: d for sym, d in _WORKER_DATA.items()
                if any(sym.startswith(p) for p in profile.prefixes)}
    config = Config(leverage=4, enforce_pruned_features=True)

    t0 = time.time()
    result: Dict[str, Any] = {
        'coin': coin,
        'strategy_family': profile.strategy_family,
        'cooldown_hours': profile.cooldown_hours,
        'min_momentum_magnitude': profile.min_momentum_magnitude,
        'vol_mult_tp': profile.vol_mult_tp,
        'vol_mult_sl': profile.vol_mult_sl,
        'sharpe_annual': -99.0,
        'win_rate': 0.0,
        'profit_factor': 0.0,
        'trades_per_year': 0.0,
        'final_equity': 100_000.0,
        'elapsed_s': 0.0,
        'error': None,
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bt = run_backtest(filtered, config)
        if bt:
            result.update({
                'sharpe_annual':   bt.get('sharpe_annual', -99.0),
                'win_rate':        bt.get('win_rate', 0.0),
                'profit_factor':   bt.get('profit_factor', 0.0),
                'trades_per_year': bt.get('trades_per_year', 0.0),
                'final_equity':    bt.get('final_equity', 100_000.0),
            })
    except Exception as exc:
        result['error'] = str(exc)
    result['elapsed_s'] = round(time.time() - t0, 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=str, required=True,
                        help='Comma-separated coin list e.g. BTC,LINK,LTC')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(',') if c.strip()]
    print('=' * 80, flush=True)
    print(f'  RE-VERIFY PROFILES — {len(coins)} coins, jobs={args.jobs}', flush=True)
    print(f'  Pass gate: SR>={PASS_SR} AND tpy>={PASS_TPY}', flush=True)
    print('=' * 80, flush=True)

    t_start = time.time()
    results: List[Dict[str, Any]] = []

    with Pool(processes=args.jobs, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_verify_one, coins):
            sr = r.get('sharpe_annual', -99)
            tpy = r.get('trades_per_year', 0)
            wr = r.get('win_rate', 0)
            passed = sr >= PASS_SR and tpy >= PASS_TPY and not r.get('error')
            mark = '★' if passed else ('~' if sr >= 0 else '✗')
            print(f'  {mark} {r["coin"]:<5} {r["strategy_family"]:<16} '
                  f'cd={r["cooldown_hours"]:>4}h mm={r["min_momentum_magnitude"]:.3f} '
                  f'tp={r["vol_mult_tp"]:.1f}/sl={r["vol_mult_sl"]:.1f}  '
                  f'SR={sr:+.3f} WR={wr:.1%} tpy={tpy:.0f}  [{r["elapsed_s"]:.0f}s]'
                  + (f'  ERR:{r["error"][:80]}' if r.get('error') else ''),
                  flush=True)
            results.append(r)

    elapsed = time.time() - t_start
    print(f'\nDone — {len(results)} coins in {elapsed/60:.1f}m', flush=True)

    results.sort(key=lambda r: r['sharpe_annual'], reverse=True)
    passed_coins = [r for r in results
                    if r['sharpe_annual'] >= PASS_SR
                    and r['trades_per_year'] >= PASS_TPY
                    and not r.get('error')]
    print(f'\nPASSED ({len(passed_coins)}): '
          + ', '.join(r['coin'] for r in passed_coins), flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps({
        'pass_gate': {'sr': PASS_SR, 'tpy': PASS_TPY},
        'elapsed_min': round(elapsed / 60, 1),
        'results': results,
    }, indent=2, default=str))
    print(f'Saved → {RESULTS_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
