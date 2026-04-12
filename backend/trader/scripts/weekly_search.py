#!/usr/bin/env python3
"""
weekly_search.py — Live-accurate parameter search (weekly retrain + pruned features).

Why this exists:
  comprehensive_search.py and gap_search.py use a fast 30d-retrain SCREEN,
  then verify top candidates with weekly retrain. In the April 2026 market
  regime the screen passes reliably but verify almost always fails (screen
  SR - verify SR > 0.5 routinely). This script eliminates that gap by running
  the weekly retrain on EVERY combo — no screen, no surprises.

Tradeoff: ~3-5x slower per combo than screen. Keep grid small.

Usage (from backend/trader/):
    python -m scripts.weekly_search --jobs 18
"""
from __future__ import annotations

import argparse
import contextlib
import io
import itertools
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

RESULTS_DIR = Path('optimization_results')
SUMMARY_PATH = RESULTS_DIR / 'weekly_search_summary.json'

# Winner gate (lower than comprehensive_search — we're using live-accurate settings)
WINNER_SR = 0.20
WINNER_TPY = 12
NEARMISS_SR = 0.10
NEARMISS_TPY = 10

EXIT_PAIRS: List[Tuple[float, float]] = [
    (4.5, 3.0),   # 1.50 ratio — proven on AVAX/DOGE
    (5.0, 3.5),   # 1.43 ratio — proven on ETH/DOT
]

# ── Coin × strategy grid — targets shadow coins with near-breakeven reverify SR ──
# Reverify SR/tpy (weekly retrain) observed today:
#   LTC  +0.213/19  (PASSED — already deploying)
#   BCH  -0.090/12  (near-breakeven, try different families)
#   LINK -0.158/14  (near-breakeven, try different families)
#   XRP  -0.256/27  (high trade count but losing — need tighter filter)
#   ADA  -0.343/27  (same pattern)
#   SOL  -0.359/25  (same pattern)
COIN_CONFIG: Dict[str, Dict[str, Any]] = {
    # ── Currently-live coins (FULL tier) — re-search from scratch ──
    'AVAX': {
        # Current: breakout/8h/0.015 tp=4.5/sl=3.0 — verify fresh with today's regime
        'strategies': ['breakout', 'btc_lead', 'mean_reversion', 'momentum_trend'],
        'cooldown_hours':         [8, 18, 36],
        'min_momentum_magnitude': [0.015, 0.025, 0.040],
    },
    'DOGE': {
        # Current: btc_lead/6h/0.010 tp=5.0/sl=3.5
        'strategies': ['btc_lead', 'momentum_trend', 'breakout', 'mean_reversion'],
        'cooldown_hours':         [6, 12, 24],
        'min_momentum_magnitude': [0.010, 0.020, 0.035],
    },
    'DOT': {
        # Current: mean_reversion/48h/0.065 tp=5.0/sl=3.5
        'strategies': ['mean_reversion', 'btc_lead', 'breakout', 'trend_pullback'],
        'cooldown_hours':         [24, 48, 72],
        'min_momentum_magnitude': [0.040, 0.055, 0.075],
    },
    'ETH': {
        # Current: mean_reversion/24h/0.025 tp=5.0/sl=3.5
        'strategies': ['mean_reversion', 'breakout', 'btc_lead', 'momentum_trend'],
        'cooldown_hours':         [12, 24, 48],
        'min_momentum_magnitude': [0.018, 0.028, 0.042],
    },
    # ── Shadow / inactive coins ──
    'LINK': {
        'strategies': ['momentum_trend', 'mean_reversion', 'btc_lead', 'trend_pullback'],
        'cooldown_hours':         [18, 36, 48],
        'min_momentum_magnitude': [0.020, 0.035, 0.050],
    },
    'BCH': {
        'strategies': ['breakout', 'mean_reversion', 'btc_lead', 'trend_pullback'],
        'cooldown_hours':         [24, 48, 72],
        'min_momentum_magnitude': [0.020, 0.040, 0.065],
    },
    'ADA': {
        'strategies': ['mean_reversion', 'btc_lead', 'trend_pullback', 'breakout'],
        'cooldown_hours':         [48, 72, 96],
        'min_momentum_magnitude': [0.030, 0.055, 0.080],
    },
    'SOL': {
        'strategies': ['mean_reversion', 'trend_pullback', 'btc_lead', 'breakout'],
        'cooldown_hours':         [36, 60, 96],
        'min_momentum_magnitude': [0.035, 0.050, 0.075],
    },
    'XRP': {
        'strategies': ['breakout', 'mean_reversion', 'trend_pullback', 'btc_lead'],
        'cooldown_hours':         [24, 48, 72],
        'min_momentum_magnitude': [0.020, 0.035, 0.055],
    },
    'BTC': {
        'strategies': ['mean_reversion', 'trend_pullback', 'breakout', 'btc_lead'],
        'cooldown_hours':         [48, 72, 120],
        'min_momentum_magnitude': [0.020, 0.035, 0.050],
    },
    # ── Remaining coins with full feature data ──
    'LTC': {
        # Current: momentum_trend/36h/0.012 — reverify passed (SR=+0.213). Probe nearby.
        'strategies': ['momentum_trend', 'mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [18, 36, 60],
        'min_momentum_magnitude': [0.010, 0.020, 0.035],
    },
    'NEAR': {
        # Current: momentum_trend/24h/0.030 — reverify fail (-0.408, overtrading at 32/yr)
        'strategies': ['mean_reversion', 'trend_pullback', 'btc_lead', 'breakout'],
        'cooldown_hours':         [36, 60, 96],
        'min_momentum_magnitude': [0.030, 0.050, 0.080],
    },
    'SUI': {
        'strategies': ['mean_reversion', 'trend_pullback', 'btc_lead', 'momentum_trend'],
        'cooldown_hours':         [36, 60, 96],
        'min_momentum_magnitude': [0.030, 0.050, 0.075],
    },
    'XLM': {
        # Current: breakout/18h/0.018 — reverify fail (-0.453)
        'strategies': ['breakout', 'btc_lead', 'momentum_trend', 'mean_reversion'],
        'cooldown_hours':         [12, 24, 48],
        'min_momentum_magnitude': [0.015, 0.025, 0.040],
    },
    'SHIB': {
        # Current: mean_reversion/24h/0.050 — reverify fail (-1.478, 27% WR)
        'strategies': ['btc_lead', 'breakout', 'momentum_trend', 'mean_reversion'],
        'cooldown_hours':         [48, 72, 120],
        'min_momentum_magnitude': [0.060, 0.100, 0.160],
    },
    'PEPE': {
        # Current: momentum_trend/18h/0.065 — reverify errored (limited data?)
        'strategies': ['momentum_trend', 'breakout', 'btc_lead', 'mean_reversion'],
        'cooldown_hours':         [18, 36, 60],
        'min_momentum_magnitude': [0.050, 0.080, 0.130],
    },
}


# ── Worker data (loaded ONCE per worker) ─────────────────────────────────
_WORKER_DATA: Optional[Dict] = None


def _init_worker() -> None:
    global _WORKER_DATA
    from scripts.train_model import load_data
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        _WORKER_DATA = load_data()


def _run_one(args: Tuple) -> Dict[str, Any]:
    coin, strategy, cooldown, min_mom, vol_mult_tp, vol_mult_sl = args

    global _WORKER_DATA
    import core.coin_profiles as cp
    from scripts.train_model import Config, run_backtest

    if _WORKER_DATA is None:
        _init_worker()

    tmp_models = tempfile.mkdtemp(prefix=f'ws_{coin.lower()}_')
    cp.MODELS_DIR = Path(tmp_models)

    test_profile = replace(
        cp.COIN_PROFILES[coin],
        strategy_family=strategy,
        cooldown_hours=float(cooldown),
        min_momentum_magnitude=float(min_mom),
        vol_mult_tp=float(vol_mult_tp),
        vol_mult_sl=float(vol_mult_sl),
        n_estimators=100,  # Match verify/live
    )

    filtered_data = {
        sym: d for sym, d in _WORKER_DATA.items()
        if any(sym.startswith(p) for p in test_profile.prefixes)
    }

    # KEY: weekly retrain (7d default) + pruned features — live-accurate
    config = Config(leverage=4, enforce_pruned_features=True)

    result: Dict[str, Any] = {
        'coin': coin, 'strategy_family': strategy,
        'cooldown_hours': cooldown, 'min_momentum_magnitude': min_mom,
        'vol_mult_tp': vol_mult_tp, 'vol_mult_sl': vol_mult_sl,
        'sharpe_annual': -99.0, 'win_rate': 0.0,
        'profit_factor': 0.0, 'trades_per_year': 0.0,
        'final_equity': 100_000.0, 'error': None,
    }

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            bt = run_backtest(filtered_data, config, profile_overrides={coin: test_profile})
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
    finally:
        shutil.rmtree(tmp_models, ignore_errors=True)
    return result


def _build_worker_args(coins: List[str]) -> List[Tuple]:
    worker_args: List[Tuple] = []
    for coin in coins:
        cfg = COIN_CONFIG.get(coin)
        if cfg is None:
            print(f'  [skip] no config for {coin}', flush=True)
            continue
        combos = itertools.product(
            cfg['strategies'],
            cfg['cooldown_hours'],
            cfg['min_momentum_magnitude'],
            EXIT_PAIRS,
        )
        for strat, cd, mm, (tp, sl) in combos:
            worker_args.append((coin, strat, cd, mm, tp, sl))
    return worker_args


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=str,
                        default='AVAX,DOGE,DOT,ETH,LTC,LINK,BCH,ADA,SOL,XRP,BTC,NEAR,SUI,XLM,SHIB,PEPE')
    parser.add_argument('--jobs', type=int, default=18)
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(',') if c.strip()]
    worker_args = _build_worker_args(coins)
    n_total = len(worker_args)

    print('=' * 80, flush=True)
    print(f'  WEEKLY SEARCH — live-accurate (7d retrain, pruned features, n_est=100)', flush=True)
    print(f'  Coins: {", ".join(coins)}  |  {n_total} combos  |  {args.jobs} workers', flush=True)
    print(f'  Winner gate: SR>={WINNER_SR} AND tpy>={WINNER_TPY}', flush=True)
    print(f'  Near-miss:   SR>={NEARMISS_SR} AND tpy>={NEARMISS_TPY}', flush=True)
    print(f'  Est: ~{n_total * 270 / args.jobs / 60:.0f}m @ 270s/combo', flush=True)
    print('=' * 80, flush=True)

    t_start = time.time()
    results: List[Dict[str, Any]] = []
    n_wins = 0
    n_near = 0

    with Pool(processes=args.jobs, initializer=_init_worker) as pool:
        for r in pool.imap_unordered(_run_one, worker_args):
            completed = len(results) + 1
            sr = r.get('sharpe_annual', -99)
            tpy = r.get('trades_per_year', 0)
            err = r.get('error')

            is_win = sr >= WINNER_SR and tpy >= WINNER_TPY and not err
            is_near = sr >= NEARMISS_SR and tpy >= NEARMISS_TPY and not is_win and not err

            if is_win:
                n_wins += 1
                print(f'  ★ [{completed:4d}/{n_total}] {r["coin"]:<5} {r["strategy_family"]:<16} '
                      f'cd={r["cooldown_hours"]:>3}h mm={r["min_momentum_magnitude"]:.3f} '
                      f'tp={r["vol_mult_tp"]:.1f}/sl={r["vol_mult_sl"]:.1f}  '
                      f'SR={sr:+.3f} WR={r["win_rate"]:.1%} tpy={tpy:.0f}  [{n_wins} wins]',
                      flush=True)
            elif is_near:
                n_near += 1
                print(f'  ~ [{completed:4d}/{n_total}] {r["coin"]:<5} {r["strategy_family"]:<16} '
                      f'cd={r["cooldown_hours"]:>3}h mm={r["min_momentum_magnitude"]:.3f} '
                      f'tp={r["vol_mult_tp"]:.1f}/sl={r["vol_mult_sl"]:.1f}  '
                      f'SR={sr:+.3f} WR={r["win_rate"]:.1%} tpy={tpy:.0f}  [near-miss #{n_near}]',
                      flush=True)
            elif completed % 20 == 0 or err:
                elapsed = time.time() - t_start
                rate = completed / elapsed if elapsed > 0 else 1
                eta = (n_total - completed) / rate / 60
                print(f'  [{completed:4d}/{n_total}] {n_wins}★ {n_near}~  ETA {eta:.0f}m',
                      flush=True)

            results.append(r)

    elapsed = time.time() - t_start
    print(f'\nDone — {n_total} combos in {elapsed/60:.1f}m. Winners={n_wins}, near-miss={n_near}',
          flush=True)

    # ── Summary — top per coin ──
    per_coin: Dict[str, List[Dict]] = {}
    for r in results:
        per_coin.setdefault(r['coin'], []).append(r)

    print('\n' + '=' * 90)
    print('  TOP RESULTS (weekly retrain — live-accurate)')
    print('=' * 90)
    for coin in sorted(per_coin):
        valid = [r for r in per_coin[coin] if not r.get('error')]
        valid.sort(key=lambda r: r['sharpe_annual'], reverse=True)
        print(f'\n  {coin}:')
        for r in valid[:5]:
            mark = '★' if (r['sharpe_annual'] >= WINNER_SR and r['trades_per_year'] >= WINNER_TPY) \
                   else ('~' if r['sharpe_annual'] >= NEARMISS_SR and r['trades_per_year'] >= NEARMISS_TPY
                         else ' ')
            print(f'    {mark} {r["strategy_family"]:<16} cd={r["cooldown_hours"]:>3}h '
                  f'mm={r["min_momentum_magnitude"]:.3f} '
                  f'tp={r["vol_mult_tp"]:.1f}/sl={r["vol_mult_sl"]:.1f}  '
                  f'SR={r["sharpe_annual"]:+.3f} WR={r["win_rate"]:.1%} tpy={r["trades_per_year"]:.0f}')

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps({
        'ts': datetime.now(timezone.utc).isoformat(),
        'gate': {'winner_sr': WINNER_SR, 'winner_tpy': WINNER_TPY,
                 'nearmiss_sr': NEARMISS_SR, 'nearmiss_tpy': NEARMISS_TPY},
        'elapsed_min': round(elapsed / 60, 1),
        'n_combos': n_total,
        'n_winners': n_wins,
        'n_nearmiss': n_near,
        'results': results,
    }, indent=2, default=str))
    print(f'\nSaved → {SUMMARY_PATH}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
