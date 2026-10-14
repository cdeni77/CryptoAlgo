"""How many basis points does our Coinbase displacement differ from Kalshi's index?

A window's strike is the previous window's settlement value, so consecutive
`expiration_value`s give the venue's own displacement. Prints the signed
difference per symbol. Run: `python -m research.validate._validate_proxy_bias`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import datetime as dt                                          # noqa: E402

DATA = Path(os.getenv('COLLECT_DATA')
            or Path(__file__).resolve().parents[2] / 'data' / 'collection')
SETTLEMENTS = DATA / 'kalshi_settlements.jsonl'
WINDOW = dt.timedelta(minutes=15)


def bias_bp(ours: float, theirs: float) -> Optional[float]:
    """Signed basis points of ours against theirs. Signed, to expose a bias."""
    if not theirs:
        return None
    return (ours - theirs) / theirs * 1e4


def displacement_bp(*, settle, strike) -> Optional[float]:
    """The move from strike, in basis points."""
    if not strike or settle is None:
        return None
    return (settle - strike) / strike * 1e4


def venue_moves(rows: list) -> list:
    """The venue's own (strike, settle) per window, from consecutive markets.

    Chains only across an exact fifteen-minute step within one symbol. A gap is
    skipped, never bridged.
    """
    by_symbol: dict = {}
    for r in rows:
        value = r.get('expiration_value')
        try:
            opened = dt.datetime.fromisoformat(str(r['window_open']))
        except (KeyError, TypeError, ValueError):
            continue
        by_symbol.setdefault(r.get('symbol'), {})[opened] = value

    out = []
    for symbol, series in by_symbol.items():
        for opened in sorted(series):
            settle = series.get(opened)
            strike = series.get(opened - WINDOW)
            # A null on either end breaks the chain.
            if settle is None or strike is None:
                continue
            out.append({'symbol': symbol, 'window_open': opened,
                        'strike': float(strike), 'settle': float(settle)})
    out.sort(key=lambda m: (m['symbol'], m['window_open']))
    return out


def _load_settlements() -> list:
    rows = []
    if not SETTLEMENTS.exists():
        return rows
    with open(SETTLEMENTS) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def _our_moves(symbols) -> dict:
    """Our own (strike, settle) per window, from `core/windows.py`.

    Reuses the real builder, so this tests the target the model trains on.
    """
    from core.datastore import ResearchStore
    from core.windows import build_windows

    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    bars = store.read('minute_bars')
    if bars is None or not len(bars):
        return {}
    out = {}
    for symbol in sorted(symbols):
        part = bars[bars['symbol'] == symbol] if 'symbol' in bars.columns else bars
        if not len(part):
            continue
        frame, _report = build_windows(part, symbol)
        # one row per (window, offset); the target is per window
        frame = frame.drop_duplicates('window_open')
        for row in frame.itertuples():
            strike, settle = row.strike, row.settle_price
            if strike != strike or settle != settle:      # NaN
                continue
            out[(symbol, row.window_open.to_pydatetime())] = (
                float(strike), float(settle))
    return out


def main() -> int:
    rows = _load_settlements()
    moves = venue_moves(rows)
    print(f'{len(rows):,} Kalshi settlements, '
          f'{len(moves):,} chain into a (strike, settle) pair')
    if not moves:
        print('nothing to compare — is kalshi_settlements.jsonl populated?')
        return 0

    months = sorted({m['window_open'].strftime('%Y-%m') for m in moves})
    print(f'months covered: {", ".join(months)}')
    print('\nNOTE: expiration_value exists only from 2026-06 (Kalshi retention).')
    print('The binary agreement in _validate_label reaches 2025-12; this does not.')

    ours = _our_moves({m['symbol'] for m in moves})
    if not ours:
        print('\nno local windows to compare against — run scripts.sync_store first')
        return 0

    import statistics
    per = {}
    flips = 0
    paired = 0
    for m in moves:
        key = (m['symbol'], m['window_open'])
        if key not in ours:
            continue
        our_strike, our_settle = ours[key]
        theirs = displacement_bp(settle=m['settle'], strike=m['strike'])
        mine = displacement_bp(settle=our_settle, strike=our_strike)
        if theirs is None or mine is None:
            continue
        paired += 1
        per.setdefault(m['symbol'], []).append(mine - theirs)
        if (mine >= 0) != (theirs >= 0):
            flips += 1

    print(f'\n{paired:,} windows compared')
    print(f'{"symbol":10} {"n":>7} {"median bp":>10} {"mean bp":>9} {"p05":>8} {"p95":>8}')
    for symbol, diffs in sorted(per.items()):
        d = sorted(diffs)
        if not d:
            continue
        print(f'{symbol:10} {len(d):>7,} {statistics.median(d):>10.2f} '
              f'{statistics.fmean(d):>9.2f} {d[len(d)//20]:>8.2f} '
              f'{d[-max(len(d)//20, 1)]:>8.2f}')
    if paired:
        print(f'\nlabel flips (our sign != venue sign): {flips:,} '
              f'= {flips/paired*100:.2f}%')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
