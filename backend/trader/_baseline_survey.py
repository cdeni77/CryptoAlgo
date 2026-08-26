"""A survey of the thing that IS the strategy: the barrier baseline.

**Why here and not in the economics.** Every money test in this project is
confined to the 70 days of Kalshi quotes, and that is where overfitting lives —
seven arms have already been rejected on that sample. But at +12m the model adds
nothing measurable over `F(x/sigma)` (paired t = -0.06), so the baseline *is* the
strategy, and baseline quality is measurable on **1.8M out-of-sample rows across
five years** without needing a single quote. That is a hundred times the sample
and it cannot be fitted to the trading period.

**Pre-registered before any arm ran:**

* The metric is out-of-sample baseline log loss at +12m, walk-forward, plus
  calibration error. Not money.
* The control is the shipped configuration. An arm is INTERESTING only if it
  beats the control on log loss by more than `MATERIAL` AND does so in at least
  5 of 6 folds. Both, not either.
* `MATERIAL` = 0.0005, half the model's own measured skill over the baseline
  (0.00100). An improvement to the null smaller than the correction sitting on
  top of it is not worth carrying.
* Every arm is reported, including the ones that lose. With ~14 arms a nominal
  0.05 needs roughly t > 3.0 to survive Bonferroni; the fold count is the
  primary guard and the t is reported beside it.
* Arms are parameters of sigma and the barrier — each has a mechanism. This is
  not a grid over things that merely differ.

The predecessor of this project ran a 27-cell survey whose best cell was its own
control. That is the failure this structure exists to make visible rather than
avoidable.
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

pd.set_option('display.width', 240)
MATERIAL = 0.0005
OFFSET = 12


def ece(y: np.ndarray, p: np.ndarray, bins: int = 20) -> float:
    frame = pd.DataFrame({'y': y, 'p': p})
    try:
        cut = pd.qcut(frame['p'], bins, duplicates='drop')
    except ValueError:
        return float('nan')
    grouped = frame.groupby(cut, observed=True)
    total = sum(abs(g['y'].mean() - g['p'].mean()) * len(g) for _, g in grouped)
    return float(total / len(frame))


def evaluate(dataset, config, label):
    result = walk_forward(dataset, config, trade=False)
    scored = result.scored
    scored = scored[(scored['offset'] == OFFSET)
                    & np.isfinite(scored['baseline_probability'])
                    & scored['outcome'].isin([0, 1, True, False])].copy()
    scored['outcome'] = scored['outcome'].astype(float)
    y = scored['outcome'].to_numpy(float)
    p = scored['baseline_probability'].to_numpy(float)
    per_fold = {}
    for fold, chunk in scored.groupby('fold'):
        yy = chunk['outcome'].to_numpy(float)
        per_fold[int(fold)] = log_loss(
            yy, chunk['baseline_probability'].to_numpy(float))
    return {'label': label, 'n': len(scored), 'baseline_ll': log_loss(y, p),
            'ECE': ece(y, p), 'per_fold': per_fold}


def main() -> int:
    base = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    bars = load_minute_bars(base, store=store)
    dataset = Dataset.build(bars, base)
    print(f'{len(dataset.window_index):,} windows, '
          f'{dataset.window_index.min().date()} .. '
          f'{dataset.window_index.max().date()}')
    print(f'metric: out-of-sample baseline log loss at +{OFFSET}m, '
          f'material = {MATERIAL}\n')

    arms = [
        ('CONTROL (shipped)', {}),
        # Tail. The scale/tail pair is not separately identified from binary
        # outcomes, so a fixed nu is a real alternative rather than a subset.
        ('normal tail', {'baseline_distribution': 'normal'}),
        ('student_t nu=3', {'baseline_nu': 3.0}),
        ('student_t nu=8', {'baseline_nu': 8.0}),
        ('student_t nu=15', {'baseline_nu': 15.0}),
        # The sigma floor. A dead-quiet minute otherwise divides by ~0 and the
        # baseline returns 0 or 1 with total confidence.
        ('sigma floor 0.2bp', {'min_sigma_bps_per_minute': 0.2}),
        ('sigma floor 1.0bp', {'min_sigma_bps_per_minute': 1.0}),
        ('sigma floor 2.0bp', {'min_sigma_bps_per_minute': 2.0}),
        # HAR lookbacks. Short carries state, long carries level.
        ('vol short only (15,60)', {'vol_lookbacks_minutes': (15, 60)}),
        ('vol long only (240,1440)', {'vol_lookbacks_minutes': (240, 1440)}),
        ('vol +5m (5,15,60,240,1440)',
         {'vol_lookbacks_minutes': (5, 15, 60, 240, 1440)}),
        ('vol no daily (15,60,240)', {'vol_lookbacks_minutes': (15, 60, 240)}),
        # Intraday seasonality on sigma.
        ('seasonality off', {'seasonality_smooth_minutes': 0}),
        ('seasonality smooth 11', {'seasonality_smooth_minutes': 11}),
        ('seasonality smooth 61', {'seasonality_smooth_minutes': 61}),
    ]

    results = []
    for label, overrides in arms:
        config = base.with_overrides(**overrides) if overrides else base
        try:
            results.append(evaluate(dataset, config, label))
            print(f'  ran {label}', flush=True)
        except Exception as exc:                      # noqa: BLE001
            print(f'  FAILED {label}: {str(exc)[:80]}', flush=True)

    control = next(r for r in results if r['label'].startswith('CONTROL'))
    rows = []
    for r in results:
        diffs = [control['per_fold'][f] - r['per_fold'][f]
                 for f in sorted(control['per_fold'])
                 if f in r['per_fold']]
        arr = np.array(diffs) if diffs else np.array([np.nan])
        t = (arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
             if len(arr) > 2 and arr.std(ddof=1) > 0 else np.nan)
        better = control['baseline_ll'] - r['baseline_ll']
        rows.append({
            'arm': r['label'], 'n': r['n'], 'baseline_ll': r['baseline_ll'],
            'vs_control': better, 'ECE': r['ECE'],
            'folds_better': int((arr > 0).sum()), 'folds': len(arr), 't': t,
            'INTERESTING': 'YES' if (better > MATERIAL
                                     and int((arr > 0).sum()) >= 5) else '',
        })
    table = pd.DataFrame(rows).sort_values('vs_control', ascending=False)
    print('\n' + '=' * 120)
    print(f'BASELINE SURVEY at +{OFFSET}m — every arm, sorted. '
          f'INTERESTING needs > {MATERIAL} AND >= 5/6 folds.')
    print('=' * 120)
    print(table.to_string(index=False, float_format=lambda v: f'{v:+.5f}'))
    hits = table[table['INTERESTING'] == 'YES']
    print(f'\n{len(hits)} of {len(table) - 1} arms interesting '
          f'(chance alone would give ~{0.05 * (len(table) - 1):.1f} at a nominal 5%).')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
