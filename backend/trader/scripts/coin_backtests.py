#!/usr/bin/env python3
"""
coin_backtests.py — Full 5-year individual backtest for each deployed coin.

Runs a verified backtest (weekly retrain, pruned features, n_estimators=100)
for each coin and prints a detailed per-coin report.

Usage (from backend/trader/):
    python -m scripts.coin_backtests
    python -m scripts.coin_backtests --coins SOL,AVAX,DOGE,ADA
"""
from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPORT_PATH = Path('/tmp/coin_backtests_report.txt')


def _run_backtest_for_coin(coin: str, all_data: dict) -> dict | None:
    import core.coin_profiles as cp
    from scripts.train_model import Config, run_backtest

    profile = cp.COIN_PROFILES.get(coin)
    if not profile:
        print(f"  ⚠️  No profile for {coin}")
        return None

    filtered = {
        sym: d for sym, d in all_data.items()
        if any(sym.startswith(p) for p in profile.prefixes)
    }
    if not filtered:
        print(f"  ⚠️  No data for {coin} (prefixes: {profile.prefixes})")
        return None

    config = Config(
        leverage=4,
        enforce_pruned_features=True,  # use pruned features like live trading does
        retrain_frequency_days=7,       # weekly retrain — matches verified backtest
    )

    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            result = run_backtest(filtered, config)
        return result
    except Exception as exc:
        print(f"  ❌ {coin} error: {exc}")
        return None


def fmt_pnl(v: float) -> str:
    return f"${v:+,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--coins', type=str, default='SOL,AVAX,DOGE,ADA',
                        help='Comma-separated list of coins to backtest')
    args = parser.parse_args()

    coins = [c.strip().upper() for c in args.coins.split(',') if c.strip()]

    print(f"\n{'='*80}")
    print(f"  5-YEAR INDIVIDUAL COIN BACKTESTS")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"  Coins: {', '.join(coins)}")
    print(f"  Config: weekly retrain | pruned features | leverage=4 | $100k")
    print(f"{'='*80}\n")

    print("⏳ Loading data...")
    from scripts.train_model import load_data
    all_data = load_data()
    if not all_data:
        print("❌ No data available.")
        sys.exit(1)
    print(f"✅ Loaded {len(all_data)} symbols\n")

    results: dict[str, dict | None] = {}
    for coin in coins:
        import core.coin_profiles as cp
        profile = cp.COIN_PROFILES.get(coin)
        strat   = profile.strategy_family if profile else '?'
        cd      = profile.cooldown_hours  if profile else '?'
        mm      = profile.min_momentum_magnitude if profile else '?'
        tp      = profile.vol_mult_tp if profile else '?'
        sl      = profile.vol_mult_sl if profile else '?'
        print(f"  [{coins.index(coin)+1}/{len(coins)}] Backtesting {coin}  "
              f"({strat}/cd={cd}h/mm={mm}/tp={tp}/sl={sl})...",
              flush=True)
        t0 = time.time()
        r  = _run_backtest_for_coin(coin, all_data)
        elapsed = time.time() - t0
        if r:
            sr  = r.get('sharpe_annual', float('nan'))
            wr  = r.get('win_rate', 0.0)
            tpy = r.get('trades_per_year', 0.0)
            pnl = r.get('final_equity', 100_000.0) - 100_000.0
            pf  = r.get('profit_factor', 0.0)
            dd  = r.get('max_drawdown', float('nan'))
            ok  = '✅' if sr >= 0.30 and tpy >= 20 else '⚠️ '
            print(f"    {ok}  SR={sr:+.3f}  WR={wr:.1%}  {tpy:.1f}/yr  "
                  f"PF={pf:.3f}  DD={dd:.1%}  PnL={fmt_pnl(pnl)}  [{elapsed:.0f}s]")
        else:
            print(f"    ❌ failed  [{elapsed:.0f}s]")
        results[coin] = r

    # ── Full report ──────────────────────────────────────────────────────────
    lines: list[str] = []
    lines.append('=' * 90)
    lines.append(f"  5-YEAR INDIVIDUAL COIN BACKTESTS — FINAL REPORT")
    lines.append(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append('=' * 90)
    lines.append(
        f"  {'Coin':<6} {'Strategy':<18} {'cd':>4} {'mm':>7} {'tp':>4} {'sl':>4}  "
        f"{'SR':>7} {'WR':>6} {'PF':>5} {'tpy':>6} {'DD':>7} {'PnL':>11}  Status"
    )
    lines.append('  ' + '-' * 86)

    passed, failed = [], []
    for coin in coins:
        r = results.get(coin)
        import core.coin_profiles as cp
        profile = cp.COIN_PROFILES.get(coin)
        strat = profile.strategy_family if profile else '?'
        cd    = profile.cooldown_hours  if profile else '?'
        mm    = profile.min_momentum_magnitude if profile else '?'
        tp    = profile.vol_mult_tp if profile else '?'
        sl    = profile.vol_mult_sl if profile else '?'

        if r:
            sr  = r.get('sharpe_annual', float('nan'))
            wr  = r.get('win_rate', 0.0)
            tpy = r.get('trades_per_year', 0.0)
            pnl = r.get('final_equity', 100_000.0) - 100_000.0
            pf  = r.get('profit_factor', 0.0)
            dd  = r.get('max_drawdown', 0.0)
            ok  = sr >= 0.30 and tpy >= 20
            status = '✅ PASS' if ok else '❌ FAIL'
            if ok:
                passed.append(coin)
            else:
                failed.append(coin)
            lines.append(
                f"  {coin:<6} {strat:<18} {cd:>3}h {mm:>7.3f} {tp:>4.1f} {sl:>4.1f}  "
                f"{sr:>7.3f} {wr:>5.1%} {pf:>5.3f} {tpy:>6.1f} {dd:>6.1%} "
                f"{fmt_pnl(pnl):>11}  {status}"
            )
        else:
            failed.append(coin)
            lines.append(
                f"  {coin:<6} {strat:<18} {cd:>3}h {mm:>7.3f} {tp:>4.1f} {sl:>4.1f}  "
                f"{'ERROR':>7}  ❌ ERROR"
            )

    lines.append('  ' + '-' * 86)
    lines.append(f"\n  ✅ Passed ({len(passed)}): {', '.join(passed) or 'none'}")
    lines.append(f"  ❌ Failed ({len(failed)}): {', '.join(failed) or 'none'}")

    # Combined portfolio stats
    valid = [results[c] for c in coins if results.get(c)]
    if valid:
        total_pnl   = sum(r.get('final_equity', 100_000) - 100_000 for r in valid)
        avg_sr      = sum(r.get('sharpe_annual', 0) for r in valid) / len(valid)
        avg_wr      = sum(r.get('win_rate', 0) for r in valid) / len(valid)
        total_trades = sum(r.get('trades_per_year', 0) for r in valid)
        lines.append(f"\n  Portfolio summary ({len(valid)} coins):")
        lines.append(f"    Combined PnL:   {fmt_pnl(total_pnl)}")
        lines.append(f"    Avg Sharpe:     {avg_sr:+.3f}")
        lines.append(f"    Avg Win Rate:   {avg_wr:.1%}")
        lines.append(f"    Combined tpy:   {total_trades:.1f}/yr")

    lines.append('')
    report = '\n'.join(lines)
    print(f"\n{report}")
    REPORT_PATH.write_text(report, encoding='utf-8')
    print(f"  Report → {REPORT_PATH}\n")


if __name__ == '__main__':
    main()
