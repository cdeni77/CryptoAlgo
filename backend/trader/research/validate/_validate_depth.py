"""Do live-recorded and Predexon-backfilled Kalshi books agree on the same minute?

Compares `venue_depth` rows by source and judges the bid gap against quote
staleness: a gap that grows with staleness is timing, a flat gap is a different
book. Run: `python research/validate/_validate_depth.py` from backend/trader.
"""

from __future__ import annotations

# Put backend/trader on sys.path, so `core` imports when run as a script.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os

import numpy as np
import pandas as pd

from core.datastore import ResearchStore

pd.set_option('display.width', 200)
# Both quotes must be within this many seconds of the minute mark, or the
# comparison measures the time between them, not the books.
MAX_AGE = float(os.getenv('MAX_AGE', '15'))


# Staleness bins, in seconds, for the trend the verdict is based on.
AGE_BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9))
# About one tick. The freshest bin must land within it.
ONE_TICK = 0.0125
MIN_PAIRS = 40


def agreement_verdict(bins):
    """(ok, why) from [(upper_age_s, n, median_abs_diff)] ordered by freshness.

    Judges the shape, not the level. The same book sampled at different
    instants gives a gap that grows with staleness; a different book gives a
    flat gap. The freshest bin must also be within one tick.
    """
    usable = [b for b in bins if b[1] >= 15]
    if sum(b[1] for b in usable) < MIN_PAIRS or len(usable) < 3:
        return False, 'not enough overlapping pairs to judge'
    fresh, stale = usable[0], usable[-1]
    if fresh[2] > ONE_TICK:
        return False, (f'the freshest quotes still differ by '
                       f'{fresh[2]*100:.2f}c, above one tick')
    if stale[2] <= fresh[2] * 1.5:
        return False, ('the gap is flat in staleness, so it is not timing — '
                       'the backfill may describe a different book')
    return True, (f'the gap grows {fresh[2]*100:.2f}c -> {stale[2]*100:.2f}c '
                  f'with staleness, which is timing: the same book sampled at '
                  f'different instants')


def main() -> int:
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    depth = store.read('venue_depth')
    depth = depth[depth['venue'] == 'kalshi']
    key = ['symbol', 'window_open', 'offset_minutes']
    live = depth[depth['source'] == 'live'].drop_duplicates(key, keep='last')
    back = depth[depth['source'] == 'backfill'].drop_duplicates(key, keep='last')
    print(f'live {len(live):,} rows, backfill {len(back):,} rows')

    both = live.merge(back, on=key, suffixes=('_live', '_back'))
    print(f'overlapping (symbol, window, minute): {len(both):,}')

    # Keep only pairs where both quotes are near the mark. The live row is
    # stamped at the poll, up to a minute late; the backfill row is the last
    # book change at or before the mark.
    age_live = pd.to_numeric(both['quote_age_seconds_live'], errors='coerce')
    age_back = pd.to_numeric(both['quote_age_seconds_back'], errors='coerce')
    print(f'  quote age  live median {age_live.median():.1f}s p90 '
          f'{age_live.quantile(0.9):.1f}s   backfill median '
          f'{age_back.median():.1f}s p90 {age_back.quantile(0.9):.1f}s')
    close = (age_live.abs() <= MAX_AGE) & (age_back.abs() <= MAX_AGE)
    both = both[close.fillna(False)]
    print(f'  {len(both):,} pairs where BOTH quotes are within {MAX_AGE:.0f}s '
          f'of the mark\n')
    if len(both) < 20:
        print('Not enough overlap yet — the backfill stops a day before the')
        print('recorder started. Re-run once the collection reaches 2026-08-25.')
        return 0

    for field in ('yes_bid', 'yes_ask'):
        a = both[f'{field}_live'].astype(float)
        b = both[f'{field}_back'].astype(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        diff = (a[ok] - b[ok]).abs()
        print(f'{field:9s} n={int(ok.sum()):>6}  exact {100*(diff < 1e-9).mean():5.1f}%  '
              f'within 1c {100*(diff <= 0.0101).mean():5.1f}%  '
              f'median |diff| {diff.median():.4f}  p95 {diff.quantile(0.95):.4f}')

    for field in ('depth_bid_total', 'depth_ask_total', 'levels_bid'):
        a = both[f'{field}_live'].astype(float)
        b = both[f'{field}_back'].astype(float)
        ok = np.isfinite(a) & np.isfinite(b) & (a + b > 0)
        if ok.sum() < 10:
            continue
        ratio = (b[ok] + 1) / (a[ok] + 1)
        print(f'{field:16s} n={int(ok.sum()):>6}  median backfill/live {ratio.median():.3f}  '
              f'p10 {ratio.quantile(0.10):.3f}  p90 {ratio.quantile(0.90):.3f}')

    # The verdict is the shape across staleness. See `agreement_verdict`.
    print('\nprice gap against how stale the WORSE of the two quotes is:')
    age_worst = pd.concat([age_live.abs(), age_back.abs()], axis=1).max(axis=1)
    a = both['yes_bid_live'].astype(float)
    b = both['yes_bid_back'].astype(float)
    gap = (a - b).abs()
    rows = []
    for lo, hi in AGE_BINS:
        sel = (age_worst >= lo) & (age_worst < hi) & np.isfinite(gap)
        n = int(sel.sum())
        if n:
            rows.append((hi, n, float(gap[sel].median())))
            label = f'{lo:.0f}-{hi:.0f}s' if hi < 1e9 else f'{lo:.0f}s+'
            print(f'  {label:>10} n={n:>6,}  median |bid diff| '
                  f'{gap[sel].median()*100:>5.2f}c')
    ok, why = agreement_verdict(rows)
    print(f'\nVERDICT: {"PASS" if ok else "FAIL"} — {why}')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
