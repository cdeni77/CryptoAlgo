"""C2 — is the venue's label aligned with ours, or just different?

Our `outcome` comes from Coinbase bars: settlement is the mean of the minute
before the window closes, and up wins on `>=` the strike. The venue settles on
CF Benchmarks BRTI. They disagree near ties, which is expected and harmless.

**What this is really testing is ALIGNMENT, not noise.** Disagreement
concentrated in sub-2bp moves is two oracles splitting a coin flip.
Disagreement that is flat across move size is a window-alignment bug -- the
same shape as the Polymarket slug trap, where every number looked valid and
the mapping was off by one window. Pass/fail is pre-registered in
`research/MIGRATION_CHECKS.md` C2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core import datastore
from core.config import DEFAULT_CONFIG
from core.windows import build_window_panel

BUCKETS = [(0, 1), (1, 2), (2, 5), (5, 10), (10, 25), (25, 1e9)]


def main() -> int:
    store = datastore.ResearchStore()
    bars = {}
    for symbol in DEFAULT_CONFIG.symbols:
        b = store.read('minute_bars', symbols=[symbol], start='2025-12-01')
        if len(b):
            bars[symbol] = b.set_index(pd.to_datetime(b['event_time'], utc=True))
    windows, _ = build_window_panel(bars, DEFAULT_CONFIG, offsets=(3,))
    windows['window_open'] = pd.to_datetime(windows['window_open'], utc=True)
    print(f'our windows: {len(windows):,}')

    s = store.read('venue_settlements')
    s = s[s['venue'] == 'kalshi'] if 'venue' in s.columns else s
    s = s.dropna(subset=['symbol', 'window_open', 'settled_up'])
    if 'available_time' in s.columns:
        s = s.sort_values('available_time')
    s = s.drop_duplicates(['symbol', 'window_open'], keep='last')
    s['window_open'] = pd.to_datetime(s['window_open'], utc=True)
    s = s[['symbol', 'window_open', 'settled_up']].rename(
        columns={'settled_up': 'venue_outcome'})
    print(f'venue settlements: {len(s):,}')

    before = len(windows)
    m = windows.merge(s, on=['symbol', 'window_open'], how='inner')
    assert len(m) <= before
    m = m.dropna(subset=['outcome', 'venue_outcome', 'settle_price', 'strike'])
    print(f'windows with BOTH labels: {len(m):,}\n')

    ours = pd.to_numeric(m['outcome'], errors='coerce').astype(float)
    theirs = pd.to_numeric(m['venue_outcome'], errors='coerce').astype(float)
    disagree = (ours != theirs)
    move_bps = (np.abs(pd.to_numeric(m['settle_price'], errors='coerce')
                       / pd.to_numeric(m['strike'], errors='coerce') - 1.0)
                * 10_000.0)

    print(f'OVERALL disagreement: {disagree.mean():.2%} '
          f'({int(disagree.sum()):,} of {len(m):,})\n')
    print(f"{'|move| from strike':>22s} {'windows':>9s} {'disagree':>9s} {'rate':>8s}")
    rates = []
    for lo, hi in BUCKETS:
        sel = (move_bps >= lo) & (move_bps < hi)
        n = int(sel.sum())
        if not n:
            continue
        r = float(disagree[sel].mean())
        rates.append((lo, r))
        label = f'{lo}-{hi}bp' if hi < 1e9 else f'{lo}bp+'
        print(f'{label:>22s} {n:>9,} {int(disagree[sel].sum()):>9,} {r:>7.1%}')

    print()
    monotone = all(rates[i][1] >= rates[i + 1][1] - 1e-9
                   for i in range(len(rates) - 1))
    top = rates[-1][1] if rates else float('nan')
    print(f'monotonically falling with move size: {monotone}')
    print(f'disagreement above 25bp: {top:.2%}  (near zero is the pass)')
    print()
    if disagree.mean() <= 0.05 and monotone and top < 0.01:
        print('PASS — near-ties only. Two oracles splitting a coin flip.')
    else:
        print('FAIL — disagreement is not concentrated in near-ties.')
        print('  That is a WINDOW-ALIGNMENT bug, not a better label.')
        print('  Do not train on it; find the offset.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
