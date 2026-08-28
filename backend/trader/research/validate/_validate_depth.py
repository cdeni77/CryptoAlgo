"""Where live recording and Predexon backfill describe the same minute, do they agree?

This check was impossible until both landed in one table. It is the only
independent evidence that the 70-day backfill — which every book-derived feature
will be trained on — describes the same object our recorder sees.

A disagreement in `yes_bid`/`yes_ask` is a pricing error and would invalidate the
economics. A disagreement in depth is a coverage difference: Predexon returns
book CHANGES, and the state at minute m is the last change at or before it, so
a quiet minute is carried forward while our recorder polls the live ladder.
Those are different measurements of the same thing and small gaps are expected;
large ones are not.
"""

from __future__ import annotations

# This file moved into research/validate/ during a repo cleanup; `core`/`scripts`
# are packages rooted at backend/trader/, which Python does not add to
# sys.path automatically for a script run from a subdirectory (only the
# script's OWN directory is added). Without this, every `from core...`
# import below raises ModuleNotFoundError the moment the file is not sitting
# directly in backend/trader/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import os

import numpy as np
import pandas as pd

from core.datastore import ResearchStore

pd.set_option('display.width', 200)
# Both observations must sit this close to the minute mark, or the
# comparison measures the time between them rather than the books.
MAX_AGE = float(os.getenv('MAX_AGE', '15'))


# Staleness bins, in seconds, for the trend the verdict is based on.
AGE_BINS = ((0, 2), (2, 5), (5, 10), (10, 20), (20, 40), (40, 1e9))
# One tick. The freshest bin has to land here or the shape does not matter.
ONE_TICK = 0.0125
MIN_PAIRS = 40


def agreement_verdict(bins):
    """(ok, why) from [(upper_age_s, n, median_abs_diff)] ordered by freshness.

    The question is the SHAPE, not the level. Two books sampled tens of seconds
    apart differ by cents on a market that moves ~8.4pp a minute, so a fixed
    threshold on the level mostly measures the sampling lag. A backfill that
    describes a DIFFERENT book disagrees regardless of how fresh either quote
    is — flat in this variable — while the same book sampled at different
    instants produces a monotone ramp.

    Necessary and not sufficient: the freshest bin must still land near one
    tick, or a steep trend would excuse a book that is simply wrong.
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

    # **Compare like with like, or measure the clock instead of the book.** The
    # live row is stamped at the poll instant, up to a minute after the mark it
    # is filed under; the backfill row is the last book CHANGE at or before that
    # mark. Unfiltered, the two can be sixty seconds apart — and this market
    # moves a measured ~8.4pp per minute, so a four-cent median difference is
    # what the clock alone produces. The first run of this reported exactly that
    # and it says nothing about whether the backfill is the same book.
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

    # The verdict is the SHAPE across staleness, not a threshold on the level.
    # See `agreement_verdict`: a fixed threshold here failed the backfill at
    # 1.8c when the two sides were a median 34s apart.
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
