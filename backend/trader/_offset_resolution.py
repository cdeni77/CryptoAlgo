"""Where in the window does the edge actually live, at one-minute resolution?

The system scores four offsets — 3, 6, 9, 12 — and trades one of them. +12m was
chosen as the best of those four. That is a choice among four candidates, and if
the real peak sits at +7m or +13m nothing in the pipeline can see it.

**A finer grid is not more information.** Every offset in a window shares one
settlement, cross-validation splits on the window, and standard errors come from
fold dispersion. Going from four offsets to fourteen multiplies rows by 3.5 and
adds no new labels at all. What it buys is resolution on a curve we have only
ever sampled at four points — and, live, more chances to clear the gate and get
filled without breaking one-entry-per-window.

**What it costs is selection.** Picking the best of fourteen is far more
overfittable than picking the best of four. So this runs on the five-year
Coinbase history, which cannot be fitted to the 70-day Kalshi trading period,
and the Kalshi quotes are kept back to validate whatever this finds. Reported
for every offset, winners and losers, with the fold count beside the mean —
because with fourteen candidates a nominal t is nearly meaningless and agreement
across folds is the honest guard.

Pre-registered before the run: the metric is out-of-sample log-loss skill of the
model over the baseline, per offset. An offset is INTERESTING only if its mean
skill beats the +12m incumbent AND at least 5 of 6 folds agree. Both, not either.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from core.backtest import walk_forward
from core.baseline import log_loss
from core.config import DEFAULT_CONFIG
from core.dataset import Dataset, load_minute_bars
from core.datastore import ResearchStore

pd.set_option('display.width', 200)
GRID = tuple(range(1, 15))
INCUMBENT = 12


def main() -> int:
    config = DEFAULT_CONFIG.with_overrides(
        decision_offsets=GRID, entry_offsets=(INCUMBENT,))
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    bars = load_minute_bars(config, store=store)
    dataset = Dataset.build(bars, config)
    print(f'{len(dataset.window_index):,} windows, '
          f'{dataset.window_index.min().date()} .. '
          f'{dataset.window_index.max().date()}', flush=True)
    print(f'offsets {GRID}, incumbent +{INCUMBENT}m\n', flush=True)

    result = walk_forward(dataset, config, trade=False)
    scored = result.scored
    scored = scored[np.isfinite(scored['baseline_probability'])
                    & np.isfinite(scored['model_probability'])
                    & scored['outcome'].isin([0, 1, True, False])].copy()
    scored['outcome'] = scored['outcome'].astype(float)

    rows = []
    for offset, chunk in scored.groupby('offset'):
        per_fold = []
        for _fold, part in chunk.groupby('fold'):
            y = part['outcome'].to_numpy(float)
            if len(y) < 50 or len(np.unique(y)) < 2:
                continue
            per_fold.append(
                log_loss(y, part['baseline_probability'].to_numpy(float))
                - log_loss(y, part['model_probability'].to_numpy(float)))
        arr = np.array(per_fold)
        if arr.size == 0:
            continue
        t = (arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
             if arr.size > 2 and arr.std(ddof=1) > 0 else float('nan'))
        rows.append({'offset': int(offset), 'n': len(chunk),
                     'mean_skill': arr.mean(), 't': t,
                     'folds+': f'{int((arr > 0).sum())}/{len(arr)}',
                     'folds_pos': int((arr > 0).sum()), 'folds': len(arr)})

    table = pd.DataFrame(rows).sort_values('offset')
    base = table.loc[table['offset'] == INCUMBENT, 'mean_skill']
    incumbent = float(base.iloc[0]) if len(base) else float('nan')
    table['vs_+12m'] = table['mean_skill'] - incumbent
    table['INTERESTING'] = np.where(
        (table['mean_skill'] > incumbent) & (table['folds_pos'] >= 5), 'YES', '')

    print('=' * 100)
    print('LOG-LOSS SKILL OVER THE BASELINE, BY OFFSET — every offset, in order')
    print('=' * 100)
    print(table[['offset', 'n', 'mean_skill', 't', 'folds+', 'vs_+12m',
                 'INTERESTING']].to_string(
        index=False, float_format=lambda v: f'{v:+.6f}'))
    hits = table[table['INTERESTING'] == 'YES']
    print(f'\n{len(hits)} of {len(table) - 1} alternatives beat +{INCUMBENT}m '
          f'on both criteria (chance alone gives ~{0.05 * (len(table) - 1):.1f} '
          f'at a nominal 5%).')
    if len(hits):
        best = hits.sort_values('mean_skill').iloc[-1]
        print(f"best alternative: +{int(best['offset'])}m at "
              f"{best['mean_skill']:+.6f} ({best['folds+']} folds)")
    print('\nThis is the Coinbase history. Nothing here is a configuration '
          'change until it survives on the held-back Kalshi quotes.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
