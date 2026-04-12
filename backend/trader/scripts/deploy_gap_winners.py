#!/usr/bin/env python3
"""
deploy_gap_winners.py — Reads gap_search_summary.json, deploys verified winners
to coin_profiles.py and docker-compose.yml, then restarts the trader container.

Usage:
    python -m scripts.deploy_gap_winners
    python -m scripts.deploy_gap_winners --dry-run   # print changes only
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

SUMMARY_PATH      = Path('optimization_results/gap_search_summary.json')
PROFILES_PATH     = Path('core/coin_profiles.py')
COMPOSE_PATH      = Path('../../docker-compose.yml')   # relative to backend/trader/

WINNER_SHARPE = 0.30
WINNER_TPY    = 20.0

# Current deployed coins (from prior comprehensive_search)
ALREADY_DEPLOYED = {'SOL', 'AVAX', 'DOGE', 'ADA'}


def _read_profiles() -> str:
    return PROFILES_PATH.read_text(encoding='utf-8')


def _find_coin_block(content: str, coin: str) -> tuple[int, int]:
    """Return (start_line, end_line) indices of coin's profile block (exclusive)."""
    lines = content.splitlines()
    start = None
    depth = 0
    for i, line in enumerate(lines):
        if start is None:
            if re.search(rf"^\s+'{coin}':\s*CoinProfile\(", line):
                start = i
                depth = 0
        if start is not None:
            depth += line.count('(') - line.count(')')
            if depth <= 0 and i > start:
                return start, i + 1
    raise ValueError(f"Could not find profile block for {coin}")


def _update_field_in_block(lines: list[str], start: int, end: int,
                            field: str, value) -> bool:
    """Update a field within a block of lines. Returns True if found+replaced."""
    pattern = re.compile(rf'^(\s+{re.escape(field)}\s*=\s*)(.+?)(,?\s*(?:#.*)?)$')
    for i in range(start, end):
        m = pattern.match(lines[i])
        if m:
            indent = m.group(1)
            comment = f'  # gap_search winner'
            if isinstance(value, str):
                lines[i] = f"{indent}'{value}',{comment}"
            else:
                lines[i] = f"{indent}{value},{comment}"
            return True
    return False


def _insert_field_before_closing(lines: list[str], end: int,
                                  field: str, value) -> None:
    """Insert a new field line just before the closing '),' of the block."""
    # Find the last non-empty line before end that ends with a comma
    insert_at = end - 1
    while insert_at > 0 and not lines[insert_at].strip().startswith(')'):
        insert_at -= 1
    # Detect indentation from nearby lines
    indent = '        '
    for i in range(end - 2, end - 6, -1):
        if i >= 0 and '=' in lines[i]:
            indent = re.match(r'^(\s+)', lines[i]).group(1)
            break
    comment = '  # gap_search winner'
    if isinstance(value, str):
        new_line = f"{indent}{field}='{value}',{comment}"
    else:
        new_line = f"{indent}{field}={value},{comment}"
    lines.insert(insert_at, new_line)


def update_coin_profile(coin: str, params: dict, verified: dict, dry_run: bool) -> None:
    """Apply gap_search winner params to coin_profiles.py."""
    content = _read_profiles()
    lines   = content.splitlines()

    start, end = _find_coin_block(content, coin)

    wr  = verified.get('win_rate', 0.0)
    pf  = verified.get('profit_factor', 1.0)
    kelly_wr    = round(wr, 4)
    kelly_ratio = round(pf * (1 - wr) / max(wr, 1e-6), 4)

    updates = {
        'strategy_family':        params['strategy_family'],
        'cooldown_hours':         float(params['cooldown_hours']),
        'min_momentum_magnitude': float(params['min_momentum_magnitude']),
        'vol_mult_tp':            float(params['vol_mult_tp']),
        'vol_mult_sl':            float(params['vol_mult_sl']),
        'kelly_win_rate':         kelly_wr,
        'kelly_payoff_ratio':     kelly_ratio,
    }

    print(f"\n  {coin} → deploying:")
    for k, v in updates.items():
        print(f"    {k} = {v!r}")
    print(f"    # Verified: SR={verified['sharpe_annual']:+.3f}  "
          f"WR={wr:.1%}  {verified['trades_per_year']:.1f}/yr  "
          f"PnL=${verified['final_equity']-100_000:+,.0f}")

    if dry_run:
        print("    [dry-run — no file written]")
        return

    for field, value in updates.items():
        found = _update_field_in_block(lines, start, end, field, value)
        if not found:
            # Field not present yet (e.g. kelly fields on new coins) — insert it
            _insert_field_before_closing(lines, end, field, value)
            # Recompute block boundaries after insertion
            content2 = '\n'.join(lines) + '\n'
            start, end = _find_coin_block(content2, coin)
            lines = content2.splitlines()

    # Update header comment for this coin
    for i in range(max(0, start - 3), start):
        if '# ──' in lines[i] and coin in lines[i]:
            lines[i] = (f"    # ── {coin}: gap_search {__import__('datetime').date.today()}: "
                        f"{params['strategy_family']}/{int(params['cooldown_hours'])}h/"
                        f"{params['min_momentum_magnitude']:.3f}, "
                        f"tp={params['vol_mult_tp']}/sl={params['vol_mult_sl']}")
            break

    PROFILES_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"    ✅ coin_profiles.py updated")


def update_docker_compose(new_coins: list[str], dry_run: bool) -> None:
    """Add new winners to active-coins and tier-map in docker-compose.yml."""
    compose_path = COMPOSE_PATH.resolve()
    if not compose_path.exists():
        print(f"  ⚠️  docker-compose.yml not found at {compose_path} — skipping")
        return

    content = compose_path.read_text(encoding='utf-8')

    # Extract current active-coins
    ac_match = re.search(r'--active-coins\s+([\w,]+)', content)
    tm_match = re.search(r"--tier-map\s+'(\{[^']+\})'", content)

    if not ac_match or not tm_match:
        print("  ⚠️  Could not parse active-coins/tier-map — skipping docker update")
        return

    current_active = set(ac_match.group(1).split(','))
    all_active = sorted(current_active | set(new_coins))
    tier_map: dict = json.loads(tm_match.group(1))

    for coin in new_coins:
        tier_map[coin] = 'FULL'

    new_ac = ','.join(all_active)
    new_tm = json.dumps(tier_map, separators=(',', ':'))

    new_content = re.sub(
        r'--active-coins\s+[\w,]+',
        f'--active-coins {new_ac}',
        content,
    )
    new_content = re.sub(
        r"--tier-map\s+'[^']+'",
        f"--tier-map '{new_tm}'",
        new_content,
    )

    print(f"\n  docker-compose.yml → active-coins: {new_ac}")
    if dry_run:
        print("    [dry-run — no file written]")
        return

    compose_path.write_text(new_content, encoding='utf-8')
    print(f"    ✅ docker-compose.yml updated")


def restart_trader(dry_run: bool) -> None:
    compose_dir = COMPOSE_PATH.resolve().parent
    if dry_run:
        print("\n  [dry-run] would run: docker compose restart trader")
        return
    print("\n  Restarting trader container...")
    result = subprocess.run(
        ['docker', 'compose', 'restart', 'trader'],
        cwd=compose_dir, capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("  ✅ trader restarted")
    else:
        print(f"  ⚠️  restart failed: {result.stderr.strip()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Print changes without writing files')
    parser.add_argument('--summary', type=Path, default=SUMMARY_PATH,
                        help='Path to gap_search_summary.json')
    args = parser.parse_args()

    if not args.summary.exists():
        print(f"❌ Summary not found: {args.summary}")
        sys.exit(1)

    summary = json.loads(args.summary.read_text())

    passed = summary.get('passed', [])
    per_coin = summary.get('per_coin', {})

    print(f"\n{'='*70}")
    print(f"  DEPLOY GAP WINNERS")
    print(f"  Gap search ran {summary.get('total_combos','?')} combos in "
          f"{summary.get('elapsed_hours','?')}h")
    print(f"  Passed: {passed or 'none'}")
    print(f"{'='*70}")

    if not passed:
        print("\n  No new winners to deploy.")
        # Still print near-misses for reference
        print("\n  Near-misses (screen only, not verified):")
        for coin, data in per_coin.items():
            best = data.get('best_screen')
            if best and best.get('sharpe_annual', -99) >= 0.15 and best.get('trades_per_year', 0) >= 15:
                print(f"    {coin}: {best['strategy_family']}/cd={best['cooldown_hours']}h/"
                      f"mm={best['min_momentum_magnitude']:.3f}  "
                      f"SR={best['sharpe_annual']:+.3f} tpy={best['trades_per_year']:.0f}")
        return

    new_winners = [c for c in passed if c not in ALREADY_DEPLOYED]
    if not new_winners:
        print(f"\n  All winners ({passed}) are already deployed — no changes needed.")
        return

    # Update coin_profiles.py for each new winner
    for coin in new_winners:
        data = per_coin.get(coin, {})
        verified = data.get('verified')
        if not verified:
            print(f"  ⚠️  {coin}: no verified data in summary — skipping")
            continue
        params = verified.get('_params', {})
        if not params:
            print(f"  ⚠️  {coin}: no _params in verified data — skipping")
            continue
        update_coin_profile(coin, params, verified, args.dry_run)

    # Update docker-compose.yml
    update_docker_compose(new_winners, args.dry_run)

    # Restart trader
    restart_trader(args.dry_run)

    print(f"\n{'='*70}")
    print(f"  ✅ Deployed {len(new_winners)} new winner(s): {', '.join(new_winners)}")
    print(f"  Total active coins: "
          f"{sorted(ALREADY_DEPLOYED | set(new_winners))}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
