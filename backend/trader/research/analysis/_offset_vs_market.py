"""Does any offset beat +12m against the MARKET, not against the baseline?

This is the question the whole offset argument has been stuck on, and it is
answerable for the first time.

`_offset_resolution.py` (in this same directory) measures model-minus-baseline at every minute and finds
skill declining with offset, +12m near the bottom. That reproduces a gradient
already known from the four-offset run and is **not** the trading question. The
case for +12m rests on two things measured against the market: the baseline's
calibration error falls monotonically to its minimum there, and the market is
most wrong there — `base - mkt` was +0.00419 at +12m against +0.00050 at +9m on
the 70-day quote archive, which only carried offsets 2, 3, 4, 6, 9, 12 and 14.

What blocked the finer question was quotes. `venue_quotes` has seven irregular
offsets; the model's own grid has four. Now `venue_depth` carries a reconstructed
market price at **every minute** of every backfilled window, so `base - mkt` is
computable at +4m exactly as at +12m.

Metric: log loss of the market's mid against log loss of the baseline, per
offset, positive meaning the baseline beats the price. Inference is clustered on
the UTC day — windows within a day share regime and the three symbols are ~0.7
correlated inside a window, so an unclustered t is fiction.

**The BASELINE only, deliberately.** The promoted artifact was fitted on offsets
3, 6, 9 and 12, and `ForecastModel.verify` refuses to score under a different
grid — correctly, since the gates were evaluated on that one. Asking LightGBM
for a prediction at +4m would be extrapolation dressed as a measurement. The
baseline has no such problem: seasonality, the volatility models and the
scale/tail pair are all offset-agnostic, so `F(x/sigma)` is exactly as valid at
+4m as at +12m. And the baseline is where the edge was measured to live anyway —
at +12m it supplies 94% of it.

**This does not select an offset by itself.** It is one leg of three: this,
calibration (`baseline ECE`), and whether the edge survives fees and the spread.
An offset that wins here and is badly calibrated is not tradeable.
"""

from __future__ import annotations

# This file moved into research/analysis/ during a repo cleanup; `core`/`scripts`
# are packages rooted at backend/trader/, which Python does not add to
# sys.path automatically for a script run from a subdirectory (only the
# script's OWN directory is added). Without this, every `from core...`
# import below raises ModuleNotFoundError the moment the file is not sitting
# directly in backend/trader/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import math
import os

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from core.dataset import (Dataset, FoldFit, apply_fold, apply_seasonality,
                          load_minute_bars)
from core.promotion import load_live

pd.set_option('display.width', 210)
GRID = tuple(range(1, 15))
EPS = 1e-6
# A quote older than this is carry-forward, not an observation.
MAX_QUOTE_AGE = float(os.getenv('MAX_QUOTE_AGE', '20'))


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> int:
    from core.datastore import ResearchStore

    config = DEFAULT_CONFIG.with_overrides(decision_offsets=GRID,
                                           entry_offsets=None)
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    depth = store.read('venue_depth')
    depth = depth[(depth['venue'] == 'kalshi')].copy()
    depth['window_open'] = pd.to_datetime(depth['window_open'], utc=True)
    depth['yes_bid'] = pd.to_numeric(depth['yes_bid'], errors='coerce')
    depth['yes_ask'] = pd.to_numeric(depth['yes_ask'], errors='coerce')
    depth = depth[np.isfinite(depth['yes_bid']) & np.isfinite(depth['yes_ask'])]
    # A crossed or absurd quote is not a price; drop rather than clip, because a
    # clipped one still enters the log loss as if it were an observation.
    depth = depth[(depth['yes_ask'] >= depth['yes_bid'])
                  & (depth['yes_bid'] > 0) & (depth['yes_ask'] < 1)]
    # **Refuse a stale quote outright.** Predexon serves book CHANGES, so a
    # quiet minute carries the previous state forward, and comparing a fresh
    # baseline against a price that is a minute old is not a measurement of
    # anything. The first run of this had no such filter and reported the
    # baseline beating the market by 0.246 nats at +14m (t = 8.4) — entirely
    # carry-forward, because the collector then fetched only to +13.5m.
    age = pd.to_numeric(depth['quote_age_seconds'], errors='coerce')
    fresh = age.notna() & (age <= MAX_QUOTE_AGE)
    dropped = int((~fresh).sum())
    depth = depth[fresh]
    print(f'dropped {dropped:,} rows whose quote was older than '
          f'{MAX_QUOTE_AGE}s (or unaged)')
    depth['market_probability'] = (depth['yes_bid'] + depth['yes_ask']) / 2.0
    depth = depth.drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                  keep='last')
    print(f'{len(depth):,} quoted (symbol, window, minute) rows, '
          f'{depth["window_open"].nunique():,} windows, '
          f'{depth["window_open"].min().date()} .. {depth["window_open"].max().date()}')

    # Load the artifact under the configuration it was FITTED with, so
    # `verify` passes, then reuse only its scoring bundle — which is
    # offset-agnostic — against a dataset built on the finer grid.
    model = load_live(config=DEFAULT_CONFIG)
    lo = (depth['window_open'].min() - pd.Timedelta(days=3)).tz_convert(None)
    hi = (depth['window_open'].max() + pd.Timedelta(hours=1)).tz_convert(None)
    bars = load_minute_bars(config, store=store, start=lo, end=hi)
    dataset = Dataset.build(bars, config)
    bundle = model.scoring
    states = {s: apply_seasonality(dataset.states[s], bundle.seasonality[s])
              for s in dataset.states if s in bundle.seasonality}
    fit = FoldFit(seasonality=bundle.seasonality, vol_models=bundle.vol_models,
                  baseline=bundle.baseline, train_windows=model.n_train_windows,
                  states=states)
    scored = apply_fold(dataset, fit, dataset.window_index, config,
                        groups=model.groups or None)

    joined = scored.merge(
        depth[['symbol', 'window_open', 'offset_minutes', 'market_probability',
               'source']],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    joined = joined[joined['outcome'].isin([0, 1, True, False])].copy()
    joined['outcome'] = joined['outcome'].astype(float)
    joined['day'] = joined['window_open'].dt.floor('D')
    print(f'joined: {len(joined):,} rows over {joined["day"].nunique()} days\n')
    if len(joined) < 500:
        print('too few joined rows — let the backfill run further')
        return 1

    rows = []
    for offset, chunk in joined.groupby('offset'):
        per_day = []
        for _day, part in chunk.groupby('day'):
            y = part['outcome'].to_numpy(float)
            if len(y) < 20 or len(np.unique(y)) < 2:
                continue
            mkt = log_loss(y, part['market_probability'].to_numpy(float))
            base = log_loss(y, part['baseline_probability'].to_numpy(float))
            per_day.append((mkt - base, mkt))
        if len(per_day) < 5:
            continue
        arr = np.array(per_day)
        def stat(col):
            v = arr[:, col]
            t = (v.mean() / (v.std(ddof=1) / math.sqrt(len(v)))
                 if v.std(ddof=1) > 0 else float('nan'))
            return v.mean(), t
        base_mean, base_t = stat(0)
        rows.append({'offset': int(offset), 'n': len(chunk),
                     'days': len(arr),
                     'market_ll': arr[:, 1].mean(),
                     'base_minus_mkt': base_mean, 't_base': base_t,
                     'days_base_pos': int((arr[:, 0] > 0).sum())})

    table = pd.DataFrame(rows).sort_values('offset')
    print('=' * 110)
    print('BASELINE vs THE MARKET, BY OFFSET  (positive = F(x/sigma) beats the price)')
    print('=' * 110)
    print(table.to_string(index=False, float_format=lambda v: f'{v:+.5f}'))
    print(f'\n(quotes older than {MAX_QUOTE_AGE:.0f}s excluded as '
          f'carry-forward rather than observation)')

    best = table.sort_values('base_minus_mkt').iloc[-1]
    incumbent = table[table['offset'] == 12]
    print(f"\nbest offset on base - mkt: +{int(best['offset'])}m at "
          f"{best['base_minus_mkt']:+.5f} (t {best['t_base']:+.2f}, "
          f"{best['days_base_pos']}/{best['days']} days positive)")
    if len(incumbent):
        row = incumbent.iloc[0]
        print(f"incumbent +12m:            {row['base_minus_mkt']:+.5f} "
              f"(t {row['t_base']:+.2f}, {row['days_base_pos']}/{row['days']} days)")
    print('\nOne leg of three. An offset that wins here still has to be well')
    print('calibrated and still has to clear fees and the spread.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
