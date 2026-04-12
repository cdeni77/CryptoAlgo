#!/usr/bin/env python3
"""
comprehensive_search.py — Targeted strategy search with new signal types.

NEW vs overnight_exit_search.py:
  - Two new strategy classes: btc_lead, autocorr_regime
  - Exit variation is IN the search (no separate phase) → no exit selection bias
  - Coin-specific strategy lists (not all 11 strategies for every coin)
  - Near-miss tracking (tpy >= 15) to surface almost-passing configs
  - Winner gate: SR >= 0.30 AND tpy >= 20 (hard gate); near-miss: SR >= 0.20 AND tpy >= 15

Usage (from backend/trader/):
    python -m scripts.comprehensive_search --jobs 18
    python -m scripts.comprehensive_search --jobs 18 --coins ETH,LINK
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

RESULTS_DIR  = Path('optimization_results')
SUMMARY_PATH = RESULTS_DIR / 'comprehensive_search_summary.json'
REPORT_PATH  = Path('/tmp/comprehensive_search_results.txt')
LOG_PATH     = Path('/tmp/comprehensive_search.log')

SEARCH_RETRAIN_DAYS = 30   # monthly retrains — keeps screen fast; verify uses 7d (live-accurate)
SEARCH_N_EST        = 50   # note: n_estimators in ensemble uses member spec ranges, not this

WINNER_SHARPE  = 0.30
WINNER_TPY     = 20
NEARMISS_SHARPE = 0.15
NEARMISS_TPY    = 15

# Exit pairs to test (tp > sl enforced — best-performing pairs from 5yr backtest evidence)
EXIT_PAIRS: List[Tuple[float, float]] = [
    (4.5, 3.0),   # ratio 1.50 — AVAX/DOGE proven winner
    (5.0, 3.5),   # ratio 1.43 — ETH/DOT proven winner
]

# ── Per-coin strategy lists and parameter grids ───────────────────────────
# btc_lead: BTC leads alts 4-24h; uses btc_rel_return_4h/24h (real data on all alts)
# autocorr_regime: adapts momentum vs mean-reversion based on ret_autocorr_lag1
#                  (only AVAX and SOL have autocorr features — returns 0 for others)
#
# Grid: 3 strategies × 3 cooldowns × 3 mm × 2 exits = 54 combos/coin
# Budget: 16 coins × 54 = 864 combos ≈ 10-24h @ 18 workers with 5yr feature data
#
# Regions informed by 5yr walk-forward backtest results (2026-04-04):
#   PROVEN (+PnL): ETH(breakout/36h/0.025), AVAX(breakout/12h/0.020), DOGE(btc_lead/4h/0.012), DOT(mean_rev/48h/0.065→10/yr)
#   FAILING:       SOL(38%WR,-$10K), LINK(46%WR,-$5K), SHIB(31%WR,-$34K), ADA(40%WR,-$3K)
#   NEW TERRITORY: BTC, XRP, LTC (shadow), NEAR/SUI/BCH/XLM (repeatedly failed)
COIN_CONFIG: Dict[str, Dict[str, Any]] = {
    # ── Proven 5yr winners: search near known-good region ──────────────────
    'ETH': {
        # 5yr: breakout/36h/0.025 → +$8K, 21/yr, 49% WR ✅
        'strategies': ['breakout', 'btc_lead', 'mean_reversion'],
        'cooldown_hours':         [24, 36, 48],
        'min_momentum_magnitude': [0.018, 0.025, 0.038],
    },
    'AVAX': {
        # 5yr: breakout/12h/0.020 → +$10K, 26/yr, 50% WR ✅
        'strategies': ['breakout', 'autocorr_regime', 'btc_lead'],
        'cooldown_hours':         [8, 12, 18],
        'min_momentum_magnitude': [0.015, 0.022, 0.035],
    },
    'DOGE': {
        # 5yr: btc_lead/4h/0.012 → +$6K, 25/yr, 48% WR ✅
        'strategies': ['btc_lead', 'momentum_trend', 'breakout'],
        'cooldown_hours':         [4, 6, 9],
        'min_momentum_magnitude': [0.010, 0.015, 0.025],
    },
    # ── DOT: 5yr winner but only 10/yr — need shorter cooldown for volume ──
    'DOT': {
        # 5yr: mean_reversion/48h/0.065 → +$26K, 10/yr (too low), 56% WR
        # Try shorter cd (18-36h) to push tpy above 20 while keeping WR high
        'strategies': ['mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [18, 24, 36],
        'min_momentum_magnitude': [0.035, 0.050, 0.065],
    },
    # ── Near-miss v3 targeted regions (2026-04-05 follow-up) ─────────────────
    'SOL': {
        # v2: autocorr_regime/48h/0.055 → SR=-0.135 verify. Try shorter cd + slightly lower mm
        # Also: v2 best screen was 50.7% WR at 15/yr — need shorter cd to push tpy above gate
        'strategies': ['autocorr_regime', 'mean_reversion', 'btc_lead'],
        'cooldown_hours':         [32, 36, 42],
        'min_momentum_magnitude': [0.045, 0.055, 0.070],
    },
    'LINK': {
        # v2: all combos failed verify (best screen SR=+0.18 but verify dropped). Try tighter region
        # Shorter cd (18-30h) + lower mm (0.025-0.050) — previous 36-60h range was too sparse
        'strategies': ['btc_lead', 'breakout', 'mean_reversion'],
        'cooldown_hours':         [18, 24, 30],
        'min_momentum_magnitude': [0.025, 0.035, 0.050],
    },
    'SHIB': {
        # 5yr: mean_reversion/24h/0.050 → -$34K, 12/yr, 31% WR ❌ (biggest loser)
        # Very long cd + very high mm — only trade massive confirmed moves
        'strategies': ['btc_lead', 'breakout', 'momentum_trend'],
        'cooldown_hours':         [72, 96, 120],
        'min_momentum_magnitude': [0.100, 0.140, 0.180],
    },
    'ADA': {
        # 5yr: breakout/36h/0.018 → -$3K, 21/yr, 40% WR (marginal) ❌
        # Higher mm + longer cd to improve WR toward 50%
        'strategies': ['mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [36, 48, 60],
        'min_momentum_magnitude': [0.035, 0.055, 0.080],
    },
    'PEPE': {
        # Only 9.2MB feature data (new listing) — limited 5yr coverage
        # Conservative: higher mm + longer cd to avoid noise on sparse data
        'strategies': ['momentum_trend', 'btc_lead', 'breakout'],
        'cooldown_hours':         [12, 24, 36],
        'min_momentum_magnitude': [0.050, 0.080, 0.120],
    },
    # ── Near-miss v3 targeted: shadow coins with tighter parameter regions ──
    'BTC': {
        # v2: mean_reversion/36h/0.020 screen SR=+0.22 but verify failed (WR dropped)
        # Try shorter cd (18-36h) + lower mm (0.015-0.040) — tighter momentum filter
        'strategies': ['mean_reversion', 'breakout', 'trend_pullback'],
        'cooldown_hours':         [18, 24, 36],
        'min_momentum_magnitude': [0.015, 0.025, 0.040],
    },
    'XRP': {
        # v2: btc_lead/9h/0.020 best screen but failed verify. Try shorter cd + lower mm
        'strategies': ['btc_lead', 'breakout', 'mean_reversion'],
        'cooldown_hours':         [6, 9, 12],
        'min_momentum_magnitude': [0.015, 0.025, 0.040],
    },
    'LTC': {
        # v2: mean_reversion/24h/0.032 → 52.7% WR, 16/yr screen; verify dropped to 13.8/yr (just below 15 gate)
        # Try shorter cd (12-24h) to push tpy above 15 at verify time
        'strategies': ['mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [12, 18, 24],
        'min_momentum_magnitude': [0.022, 0.032, 0.045],
    },
    # ── Repeatedly failing coins: last-chance focused regions ──────────────
    'NEAR': {
        # Rounds 1+2 failed; try high-selectivity: long cd + high mm only
        'strategies': ['mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [48, 72, 96],
        'min_momentum_magnitude': [0.055, 0.080, 0.110],
    },
    'SUI': {
        # autocorr_regime available (SOL feature set); limited data
        'strategies': ['btc_lead', 'autocorr_regime', 'breakout'],
        'cooldown_hours':         [48, 72, 96],
        'min_momentum_magnitude': [0.070, 0.100, 0.140],
    },
    'BCH': {
        # BTC-lite; multiple rounds failed with cd≤96h, try different strategies
        'strategies': ['mean_reversion', 'btc_lead', 'breakout'],
        'cooldown_hours':         [36, 60, 96],
        'min_momentum_magnitude': [0.030, 0.055, 0.080],
    },
    'XLM': {
        # Must use short cd (4-12h) to reach 20/yr — low volatility coin
        'strategies': ['breakout', 'btc_lead', 'momentum_trend'],
        'cooldown_hours':         [5, 8, 12],
        'min_momentum_magnitude': [0.010, 0.015, 0.022],
    },
}


# ── Worker data (loaded ONCE per worker at pool startup) ──────────────────
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

    tmp_models = tempfile.mkdtemp(prefix=f'cs_{coin.lower()}_')
    cp.MODELS_DIR = Path(tmp_models)

    test_profile = replace(
        cp.COIN_PROFILES[coin],
        strategy_family=strategy,
        cooldown_hours=float(cooldown),
        min_momentum_magnitude=float(min_mom),
        vol_mult_tp=float(vol_mult_tp),
        vol_mult_sl=float(vol_mult_sl),
        n_estimators=SEARCH_N_EST,
    )

    filtered_data = {
        sym: d for sym, d in _WORKER_DATA.items()
        if any(sym.startswith(p) for p in test_profile.prefixes)
    }

    config = Config(
        leverage=4,
        enforce_pruned_features=True,   # matches live system — critical fix for TPY/SR accuracy
        retrain_frequency_days=SEARCH_RETRAIN_DAYS,
    )

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


def _run_batch(pool: Pool, worker_args: List[Tuple],
               label: str, n_total: int, t_start: float) -> List[Dict]:
    results: List[Dict] = []
    n_winners = 0
    n_nearmiss = 0
    t0 = time.time()

    for result in pool.imap_unordered(_run_one, worker_args):
        completed = len(results) + 1
        sr  = result.get('sharpe_annual', -99)
        tpy = result.get('trades_per_year', 0)
        err = result.get('error')

        is_win  = sr >= WINNER_SHARPE  and tpy >= WINNER_TPY  and not err
        is_near = sr >= NEARMISS_SHARPE and tpy >= NEARMISS_TPY and not is_win and not err

        if is_win:
            n_winners += 1
            elapsed = time.time() - t_start
            print(f"  ★ [{label} {completed:4d}/{n_total}] {result['coin']:<5} "
                  f"{result['strategy_family']:<16} cd={result['cooldown_hours']:>3}h "
                  f"mm={result['min_momentum_magnitude']:.3f} "
                  f"tp={result['vol_mult_tp']:.1f}/sl={result['vol_mult_sl']:.1f}  "
                  f"SR={sr:+.3f} WR={result['win_rate']:.1%} tpy={tpy:.0f}  "
                  f"[{n_winners} wins, {elapsed/3600:.1f}h elapsed]",
                  flush=True)
        elif is_near:
            n_nearmiss += 1
            print(f"  ~ [{label} {completed:4d}/{n_total}] {result['coin']:<5} "
                  f"{result['strategy_family']:<16} cd={result['cooldown_hours']:>3}h "
                  f"mm={result['min_momentum_magnitude']:.3f} "
                  f"tp={result['vol_mult_tp']:.1f}/sl={result['vol_mult_sl']:.1f}  "
                  f"SR={sr:+.3f} WR={result['win_rate']:.1%} tpy={tpy:.0f}  "
                  f"[near-miss #{n_nearmiss}]",
                  flush=True)
        elif completed % 100 == 0 or err:
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 1
            eta  = (n_total - completed) / rate / 60
            print(f"  [{label} {completed:4d}/{n_total}] {n_winners}★ {n_nearmiss}~ "
                  f"ETA {eta:.0f}m",
                  flush=True)

        results.append(result)

    elapsed = time.time() - t0
    print(f"  '{label}' done: {len(results)} combos in {elapsed/60:.1f}m, "
          f"{n_winners} winners, {n_nearmiss} near-misses", flush=True)
    return results


def _verify_candidate(coin: str, candidate: Dict) -> Optional[Dict]:
    import core.coin_profiles as cp
    from scripts.train_model import Config, run_backtest, load_data

    test_profile = replace(
        cp.COIN_PROFILES[coin],
        strategy_family=candidate['strategy_family'],
        cooldown_hours=float(candidate['cooldown_hours']),
        min_momentum_magnitude=float(candidate['min_momentum_magnitude']),
        vol_mult_tp=float(candidate['vol_mult_tp']),
        vol_mult_sl=float(candidate['vol_mult_sl']),
        n_estimators=100,
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        all_data = load_data()
    filtered = {sym: d for sym, d in all_data.items()
                if any(sym.startswith(p) for p in test_profile.prefixes)}
    config = Config(leverage=4, enforce_pruned_features=True)
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            r = run_backtest(filtered, config, profile_overrides={coin: test_profile})
        return r
    except Exception as e:
        print(f"    verify error: {e}")
        return None


def _top_n(results: List[Dict], coin: str, n: int = 5, min_tpy: float = 10.0) -> List[Dict]:
    valid = [r for r in results
             if r['coin'] == coin and not r.get('error')
             and r.get('trades_per_year', 0) >= min_tpy]
    return sorted(valid, key=lambda r: r['sharpe_annual'], reverse=True)[:n]


def _print_summary(label: str, per_coin_results: Dict[str, List[Dict]],
                   verified: Dict[str, Optional[Dict]]) -> None:
    lines: List[str] = []
    lines.append('=' * 90)
    lines.append(f'  {label}')
    lines.append('=' * 90)
    lines.append(f"  {'Coin':<5} {'Status':<26} {'Strategy':<16} {'cd':>4} {'mm':>7} "
                 f"{'tp':>4} {'sl':>4}  {'Screen SR':>9} {'Verify SR':>9} {'WR':>6} {'tpy':>6}")
    lines.append('  ' + '-' * 87)

    passed, failed, nearmiss_list = [], [], []
    for coin, all_results in per_coin_results.items():
        top = _top_n(all_results, coin, n=1, min_tpy=NEARMISS_TPY)
        if not top:
            top = _top_n(all_results, coin, n=1, min_tpy=0)
        if not top:
            lines.append(f"  {coin:<5}  no valid results")
            continue

        best = top[0]
        scr_sr  = best['sharpe_annual']
        scr_tpy = best['trades_per_year']
        v = verified.get(coin)
        ver_sr  = v['sharpe_annual']  if v else float('nan')
        ver_tpy = v['trades_per_year'] if v else 0.0

        passes = v and ver_sr >= WINNER_SHARPE and ver_tpy >= WINNER_TPY
        near   = (scr_sr >= NEARMISS_SHARPE and scr_tpy >= NEARMISS_TPY) and not passes

        if passes:
            status = f'✅ PASS SR={ver_sr:+.3f}'
            passed.append(coin)
        elif near:
            status = f'~ NEAR  SR={scr_sr:+.3f}'
            nearmiss_list.append(coin)
        else:
            status = f'❌ FAIL SR={ver_sr:+.3f}' if v else f'✗  no edge SR={scr_sr:+.3f}'
            failed.append(coin)

        lines.append(
            f"  {coin:<5} {status:<26} {best['strategy_family']:<16} "
            f"{best['cooldown_hours']:>3}h {best['min_momentum_magnitude']:>7.3f} "
            f"{best['vol_mult_tp']:>4.1f} {best['vol_mult_sl']:>4.1f}  "
            f"{scr_sr:>9.3f} {ver_sr:>9.3f} {best['win_rate']:>5.1%} {scr_tpy:>6.1f}"
        )

    lines.append('  ' + '-' * 87)
    lines.append(f"\n  ✅ PASSED ({len(passed)}): {', '.join(passed) or 'none'}")
    lines.append(f"  ~  NEAR-MISS ({len(nearmiss_list)}): {', '.join(nearmiss_list) or 'none'}")
    lines.append(f"  ❌ Failed: {len(failed)} coins")

    if passed:
        lines.append('\n  WINNING PARAMS — deploy these to coin_profiles.py:')
        lines.append('  ' + '-' * 60)
        for coin in passed:
            top = _top_n(per_coin_results[coin], coin, n=1)[0]
            v   = verified[coin]
            lines.append(f"  # ── {coin} ──")
            lines.append(f"  strategy_family        = '{top['strategy_family']}'")
            lines.append(f"  cooldown_hours         = {top['cooldown_hours']}")
            lines.append(f"  min_momentum_magnitude = {top['min_momentum_magnitude']}")
            lines.append(f"  vol_mult_tp            = {top['vol_mult_tp']}")
            lines.append(f"  vol_mult_sl            = {top['vol_mult_sl']}")
            if v:
                lines.append(f"  # Verified: SR={v['sharpe_annual']:+.3f}  "
                              f"WR={v['win_rate']:.1%}  {v['trades_per_year']:.1f}/yr  "
                              f"PnL=${v['final_equity']-100_000:+,.0f}")
            lines.append('')

    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobs',    type=int, default=18)
    parser.add_argument('--coins',   type=str, default=','.join(COIN_CONFIG.keys()))
    parser.add_argument('--no-verify', action='store_true')
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(',') if c.strip()]
    coins = [c for c in coins if c in COIN_CONFIG]

    RESULTS_DIR.mkdir(exist_ok=True)

    # Build combo list
    all_combos: List[Tuple] = []
    for coin in coins:
        cfg = COIN_CONFIG[coin]
        for s, cd, mm, (tp, sl) in itertools.product(
            cfg['strategies'],
            cfg['cooldown_hours'],
            cfg['min_momentum_magnitude'],
            EXIT_PAIRS,
        ):
            all_combos.append((coin, s, cd, mm, tp, sl))

    n_total = len(all_combos)
    est_h   = n_total / args.jobs * 250 / 3600

    ts_start = datetime.now(timezone.utc)
    print(f"\n{'='*80}")
    print(f"  COMPREHENSIVE STRATEGY SEARCH")
    print(f"  Started: {ts_start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Coins: {', '.join(coins)}")
    print(f"  Total combos: {n_total}  (exits baked in — no separate phase 2)")
    print(f"  Jobs: {args.jobs}  Est: {est_h:.1f}h")
    print(f"  New strategies: btc_lead, autocorr_regime")
    print(f"  Winner gate: SR≥{WINNER_SHARPE} AND tpy≥{WINNER_TPY}")
    print(f"  Near-miss:   SR≥{NEARMISS_SHARPE} AND tpy≥{NEARMISS_TPY}")
    print(f"{'='*80}\n")

    from scripts.train_model import load_data
    check = load_data()
    if not check:
        print("❌ No data."); sys.exit(1)
    print(f"✅ Data: {len(check)} symbols.  Creating pool...\n")
    del check

    t_global = time.time()
    with Pool(processes=args.jobs, initializer=_init_worker) as pool:
        print(f"\n{'='*80}")
        print(f"  SEARCH — {n_total} combos across {len(coins)} coins")
        print(f"{'='*80}")

        all_results = _run_batch(pool, all_combos, 'SEARCH', n_total, t_global)

    # Organize per-coin
    per_coin: Dict[str, List[Dict]] = {c: [] for c in coins}
    for r in all_results:
        if r['coin'] in per_coin:
            per_coin[r['coin']].append(r)

    # Print per-coin top-10
    for coin in coins:
        top10 = _top_n(per_coin[coin], coin, n=10, min_tpy=NEARMISS_TPY)
        if not top10:
            top10 = _top_n(per_coin[coin], coin, n=10, min_tpy=0)
        print(f"\n{'='*90}")
        print(f"  TOP RESULTS — {coin}")
        print(f"{'='*90}")
        print(f"  {'Strategy':<16} {'cd':>4} {'mm':>7} {'tp':>4} {'sl':>4}  "
              f"{'SR':>7} {'WR':>6} {'PF':>5} {'tpy':>6} {'PnL':>10}")
        for r in top10:
            pnl = r['final_equity'] - 100_000
            star = '★' if r['sharpe_annual'] >= WINNER_SHARPE and r['trades_per_year'] >= WINNER_TPY else \
                   '~' if r['sharpe_annual'] >= NEARMISS_SHARPE and r['trades_per_year'] >= NEARMISS_TPY else ''
            print(f"  {r['strategy_family']:<16} {r['cooldown_hours']:>3}h "
                  f"{r['min_momentum_magnitude']:>7.3f} {r['vol_mult_tp']:>4.1f} "
                  f"{r['vol_mult_sl']:>4.1f}  "
                  f"{r['sharpe_annual']:>7.3f} {r['win_rate']:>5.1%} "
                  f"{r['profit_factor']:>5.2f} {r['trades_per_year']:>6.1f} "
                  f"${pnl:>+9,.0f}  {star}")

    # Save raw results
    ts = ts_start.strftime('%Y%m%dT%H%M%SZ')
    raw_path = RESULTS_DIR / f'comprehensive_search_{ts}.json'
    raw_path.write_text(json.dumps({
        'run_at': ts,
        'per_coin_top10': {
            c: _top_n(per_coin[c], c, n=10, min_tpy=0)
            for c in coins
        },
    }, indent=2, default=str))
    print(f"\n  💾 Raw results → {raw_path}")

    # Verification
    verified: Dict[str, Optional[Dict]] = {c: None for c in coins}
    if not args.no_verify:
        print(f"\n\n{'='*80}")
        print(f"  VERIFICATION — weekly retrain + pruned features")
        print(f"{'='*80}")

        to_verify: List[Tuple[str, Dict]] = []
        for coin in coins:
            # Verify top-3 winners; also verify top near-miss if no winner
            winners = [r for r in per_coin[coin]
                       if r['sharpe_annual'] >= WINNER_SHARPE
                       and r['trades_per_year'] >= WINNER_TPY
                       and not r.get('error')]
            winners = sorted(winners, key=lambda r: r['sharpe_annual'], reverse=True)[:3]

            if not winners:
                nm = [r for r in per_coin[coin]
                      if r['sharpe_annual'] >= NEARMISS_SHARPE
                      and r['trades_per_year'] >= NEARMISS_TPY
                      and not r.get('error')]
                nm = sorted(nm, key=lambda r: r['sharpe_annual'], reverse=True)[:1]
                winners = nm

            for w in winners:
                to_verify.append((coin, w))

        for i, (coin, candidate) in enumerate(to_verify, 1):
            print(f"\n  [{i}/{len(to_verify)}] Verifying {coin} "
                  f"{candidate['strategy_family']}/cd={candidate['cooldown_hours']}h/"
                  f"mm={candidate['min_momentum_magnitude']:.3f}/"
                  f"tp={candidate['vol_mult_tp']}/sl={candidate['vol_mult_sl']}  "
                  f"(screen SR={candidate['sharpe_annual']:+.3f}, "
                  f"tpy={candidate['trades_per_year']:.1f})...",
                  flush=True)
            t0 = time.time()
            v = _verify_candidate(coin, candidate)
            elapsed = time.time() - t0
            if v:
                passes = v['sharpe_annual'] >= WINNER_SHARPE and v['trades_per_year'] >= WINNER_TPY
                mark   = '✅ PASS' if passes else '❌ FAIL'
                print(f"    {mark}  SR={v['sharpe_annual']:+.3f}  WR={v['win_rate']:.1%}  "
                      f"tpy={v['trades_per_year']:.1f}  PnL=${v['final_equity']-100_000:+,.0f}  "
                      f"[{elapsed:.0f}s]",
                      flush=True)
                if passes or verified[coin] is None:
                    verified[coin] = v
                    verified[coin]['_params'] = candidate
            else:
                print(f"    ❌ verify error  [{elapsed:.0f}s]", flush=True)

    # Final report
    total_elapsed = (time.time() - t_global) / 3600
    report_header = (
        f"\n{'='*90}\n"
        f"  COMPREHENSIVE SEARCH — FINAL REPORT\n"
        f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  {total_elapsed:.1f}h total\n"
        f"  Coins: {', '.join(coins)}  |  {n_total} combos\n"
        f"{'='*90}\n"
    )
    summary = _print_summary('RESULTS SUMMARY', per_coin, verified)
    report  = report_header + summary

    print(report)
    REPORT_PATH.write_text(report)
    print(f"\n  Report → {REPORT_PATH}")

    # JSON summary
    passed_coins = [c for c in coins
                    if verified.get(c) and
                    verified[c]['sharpe_annual'] >= WINNER_SHARPE and
                    verified[c]['trades_per_year'] >= WINNER_TPY]
    summary_json = {
        'run_at':       ts,
        'total_combos': n_total,
        'elapsed_hours': round(total_elapsed, 2),
        'passed':       passed_coins,
        'per_coin': {
            c: {
                'best_screen': _top_n(per_coin[c], c, n=1, min_tpy=0)[0] if per_coin[c] else None,
                'verified':    verified.get(c),
            }
            for c in coins
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary_json, indent=2, default=str))
    print(f"  JSON  → {SUMMARY_PATH}")


if __name__ == '__main__':
    main()
