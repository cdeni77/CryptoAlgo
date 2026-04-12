#!/usr/bin/env python3
"""
gap_search.py — Fills strategy and parameter gaps for coins that failed comprehensive_search.

Targets: ETH, BTC, XRP, LINK, LTC

What was already tested in comprehensive_search (NOT repeated here):
  ETH:  breakout, btc_lead, vol_overlay, squeeze_breakout, momentum_trend
        cooldowns 30-60h, min_moms 0.025-0.050
  BTC:  mean_reversion, trend_pullback, breakout, squeeze_breakout, funding_carry
        cooldowns 36-96h, min_moms 0.018-0.060
  LINK: btc_lead, momentum_trend, breakout, vol_overlay, trend_pullback
        cooldowns 24-72h, min_moms 0.030-0.090
  XRP:  btc_lead, breakout_expansion, breakout, mean_reversion
        cooldowns 9-36h, min_moms 0.018-0.075
  LTC:  btc_lead, mean_reversion, breakout, momentum_trend
        cooldowns 12-36h, min_moms 0.015-0.055

What this search covers (the gaps):
  ETH:  trend_pullback + oi_divergence (both untested) + breakout at lower mm (0.020-0.028)
  BTC:  breakout_expansion + oi_divergence + vol_overlay + momentum_trend (all untested)
        + ultra-long cooldowns (96-168h) + ultra-low min_moms (0.006-0.018)
  LINK: mean_reversion + breakout_expansion (both untested) + lower mm region (0.020-0.040)
  XRP:  momentum_trend + trend_pullback (both untested) + lower mm region (0.010-0.028)
  LTC:  trend_pullback + breakout_expansion + vol_overlay (all untested)

Usage (from backend/trader/):
    python -m scripts.gap_search --jobs 18
    python -m scripts.gap_search --jobs 18 --coins XRP,LINK,LTC
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
SUMMARY_PATH = RESULTS_DIR / 'gap_search_summary.json'
REPORT_PATH  = Path('/tmp/gap_search_results.txt')
LOG_PATH     = Path('/tmp/gap_search.log')

SEARCH_RETRAIN_DAYS = 30   # monthly — consistent with comprehensive_search
SEARCH_N_EST        = 50

WINNER_SHARPE   = 0.30
WINNER_TPY      = 20
NEARMISS_SHARPE = 0.15
NEARMISS_TPY    = 15

# Same exit pairs as comprehensive_search — all tp > sl enforced
EXIT_PAIRS: List[Tuple[float, float]] = [
    (4.0, 3.0),   # ratio 1.33 — moderate (AVAX/DOGE winner)
    (4.5, 3.0),   # ratio 1.50
    (5.0, 3.5),   # ratio 1.43 — wide (SOL/ADA winner)
]

# ── Per-coin GAP configs ──────────────────────────────────────────────────────
# Each coin lists ONLY strategies and parameter regions NOT covered in comprehensive_search.
# ETH near-miss: breakout/48h/0.030 @ 19.3/yr — go lower on mm + test trend_pullback/oi_divergence
# BTC: zero winners at 1h — last-resort ultra-conservative sweep + 4 untested families
# LINK near-miss: vol_overlay/36h/0.048 @ 19.1/yr SR=0.183 — mean_reversion + breakout_expansion
# XRP near-miss: mean_reversion/12h/0.050 @ 23.1/yr SR=0.209 — momentum_trend + trend_pullback
# LTC: breakout/24h/0.025 verified SR=-0.524 — trend_pullback + breakout_expansion + vol_overlay
COIN_CONFIG: Dict[str, Dict[str, Any]] = {
    'ETH': {
        # Untested families: trend_pullback, oi_divergence
        # Under-explored region: breakout at mm=0.020-0.028 (near-miss was 0.030, just 0.7/yr short)
        # ETH has eth_pullback_depth_72h — built for trend_pullback
        'strategies': ['trend_pullback', 'oi_divergence', 'breakout'],
        'cooldown_hours':         [36, 42, 48],
        'min_momentum_magnitude': [0.020, 0.024, 0.028, 0.032],
    },
    'BTC': {
        # Untested families: breakout_expansion, oi_divergence, vol_overlay, momentum_trend
        # btc_lead/autocorr_regime N/A for BTC itself
        # Ultra-conservative params: very long cooldowns + very small min_moms
        # (BTC 1h is efficient — only trades on genuine macro signals)
        'strategies': ['breakout_expansion', 'oi_divergence', 'vol_overlay', 'momentum_trend'],
        'cooldown_hours':         [72, 96, 120, 168],
        'min_momentum_magnitude': [0.006, 0.010, 0.015, 0.020],
    },
    'LINK': {
        # Untested families: mean_reversion, breakout_expansion
        # Under-explored region: lower mm (0.020-0.040) — prev min was 0.030
        # LINK tracks BTC/ETH; when they pump, LINK overshoots → mean_reversion edge
        'strategies': ['mean_reversion', 'breakout_expansion', 'momentum_trend'],
        'cooldown_hours':         [18, 24, 36, 48],
        'min_momentum_magnitude': [0.020, 0.028, 0.036, 0.048],
    },
    'XRP': {
        # Untested families: momentum_trend, trend_pullback
        # Under-explored region: lower mm (0.010-0.025) — prev min was 0.018
        # XRP has xrp_compression_ratio + xrp_breakout_distance — fits trend_pullback well
        'strategies': ['momentum_trend', 'trend_pullback', 'squeeze_breakout'],
        'cooldown_hours':         [9, 12, 18, 24],
        'min_momentum_magnitude': [0.010, 0.015, 0.022, 0.030],
    },
    'LTC': {
        # Untested families: trend_pullback, breakout_expansion, vol_overlay
        # LTC = BTC lite — impulse + pullback pattern is its natural rhythm
        # LTC has BTC's full mean-reversion feature set + BTC-relative features
        'strategies': ['trend_pullback', 'breakout_expansion', 'vol_overlay'],
        'cooldown_hours':         [18, 24, 30, 36],
        'min_momentum_magnitude': [0.015, 0.020, 0.028, 0.038],
    },
}


# ── Worker (loaded once per worker at pool startup) ───────────────────────────
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

    tmp_models = tempfile.mkdtemp(prefix=f'gs_{coin.lower()}_')
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
        enforce_pruned_features=False,
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
    n_winners  = 0
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
                  f"{result['strategy_family']:<18} cd={result['cooldown_hours']:>3}h "
                  f"mm={result['min_momentum_magnitude']:.3f} "
                  f"tp={result['vol_mult_tp']:.1f}/sl={result['vol_mult_sl']:.1f}  "
                  f"SR={sr:+.3f} WR={result['win_rate']:.1%} tpy={tpy:.0f}  "
                  f"[{n_winners} wins, {elapsed/3600:.1f}h elapsed]",
                  flush=True)
        elif is_near:
            n_nearmiss += 1
            print(f"  ~ [{label} {completed:4d}/{n_total}] {result['coin']:<5} "
                  f"{result['strategy_family']:<18} cd={result['cooldown_hours']:>3}h "
                  f"mm={result['min_momentum_magnitude']:.3f} "
                  f"tp={result['vol_mult_tp']:.1f}/sl={result['vol_mult_sl']:.1f}  "
                  f"SR={sr:+.3f} WR={result['win_rate']:.1%} tpy={tpy:.0f}  "
                  f"[near-miss #{n_nearmiss}]",
                  flush=True)
        elif completed % 50 == 0 or err:
            elapsed = time.time() - t0
            rate = completed / elapsed if elapsed > 0 else 1
            eta  = (n_total - completed) / rate / 60
            flag = f"  ERR: {result.get('error','?')[:60]}" if err else ''
            print(f"  [{label} {completed:4d}/{n_total}] {n_winners}★ {n_nearmiss}~  "
                  f"ETA {eta:.0f}m{flag}",
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
                   verified: Dict[str, Optional[Dict]]) -> str:
    lines: List[str] = []
    lines.append('=' * 90)
    lines.append(f'  {label}')
    lines.append('=' * 90)
    lines.append(f"  {'Coin':<5} {'Status':<26} {'Strategy':<18} {'cd':>4} {'mm':>7} "
                 f"{'tp':>4} {'sl':>4}  {'Screen SR':>9} {'Verify SR':>9} {'WR':>6} {'tpy':>6}")
    lines.append('  ' + '-' * 90)

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
        ver_sr  = v['sharpe_annual']   if v else float('nan')
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
            f"  {coin:<5} {status:<26} {best['strategy_family']:<18} "
            f"{best['cooldown_hours']:>3}h {best['min_momentum_magnitude']:>7.3f} "
            f"{best['vol_mult_tp']:>4.1f} {best['vol_mult_sl']:>4.1f}  "
            f"{scr_sr:>9.3f} {ver_sr:>9.3f} {best['win_rate']:>5.1%} {scr_tpy:>6.1f}"
        )

    lines.append('  ' + '-' * 90)
    lines.append(f"\n  ✅ PASSED ({len(passed)}): {', '.join(passed) or 'none'}")
    lines.append(f"  ~  NEAR-MISS ({len(nearmiss_list)}): {', '.join(nearmiss_list) or 'none'}")
    lines.append(f"  ❌ Failed: {len(failed)} coins — {', '.join(failed) or 'none'}")

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
    parser = argparse.ArgumentParser(description='Gap strategy search for failing coins')
    parser.add_argument('--jobs',      type=int, default=18)
    parser.add_argument('--coins',     type=str, default=','.join(COIN_CONFIG.keys()))
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
    print(f"  GAP STRATEGY SEARCH")
    print(f"  Started: {ts_start.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Coins: {', '.join(coins)}")
    print(f"  Total combos: {n_total}  (untested families + unexplored param regions)")
    print(f"  Jobs: {args.jobs}  Est: {est_h:.1f}h")
    print(f"  Winner gate: SR≥{WINNER_SHARPE} AND tpy≥{WINNER_TPY}")
    print(f"  Near-miss:   SR≥{NEARMISS_SHARPE} AND tpy≥{NEARMISS_TPY}")
    print()
    for coin in coins:
        cfg = COIN_CONFIG[coin]
        n = len(cfg['strategies']) * len(cfg['cooldown_hours']) * len(cfg['min_momentum_magnitude']) * len(EXIT_PAIRS)
        print(f"  {coin:<5}: {cfg['strategies']}  ({n} combos)")
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
        print(f"  {'Strategy':<18} {'cd':>4} {'mm':>7} {'tp':>4} {'sl':>4}  "
              f"{'SR':>7} {'WR':>6} {'PF':>5} {'tpy':>6} {'PnL':>10}")
        for r in top10:
            pnl  = r['final_equity'] - 100_000
            mark = ('★' if r['sharpe_annual'] >= WINNER_SHARPE and r['trades_per_year'] >= WINNER_TPY
                    else '~' if r['sharpe_annual'] >= NEARMISS_SHARPE and r['trades_per_year'] >= NEARMISS_TPY
                    else '')
            print(f"  {r['strategy_family']:<18} {r['cooldown_hours']:>3}h "
                  f"{r['min_momentum_magnitude']:>7.3f} {r['vol_mult_tp']:>4.1f} "
                  f"{r['vol_mult_sl']:>4.1f}  "
                  f"{r['sharpe_annual']:>7.3f} {r['win_rate']:>5.1%} "
                  f"{r['profit_factor']:>5.2f} {r['trades_per_year']:>6.1f} "
                  f"${pnl:>+9,.0f}  {mark}")

    # Save raw results
    ts = ts_start.strftime('%Y%m%dT%H%M%SZ')
    raw_path = RESULTS_DIR / f'gap_search_{ts}.json'
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
            v  = _verify_candidate(coin, candidate)
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
        f"  GAP SEARCH — FINAL REPORT\n"
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
                    if verified.get(c)
                    and verified[c]['sharpe_annual'] >= WINNER_SHARPE
                    and verified[c]['trades_per_year'] >= WINNER_TPY]
    summary_json = {
        'run_at':        ts,
        'total_combos':  n_total,
        'elapsed_hours': round(total_elapsed, 2),
        'passed':        passed_coins,
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
