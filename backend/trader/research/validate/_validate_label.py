"""Does our Coinbase-derived label agree with how the market actually settled?

**This tests the target, which nothing else in the repository does.** Every
window, every feature, every gate and every backtest in this project rests on
`core/windows.py` deciding UP or DOWN from Coinbase one-minute bars: the strike
is the mean over [t0-1min, t0), the settlement value the mean over [t1-1min, t1),
and `>=` decides. The market settles on CF Benchmarks BRTI. `CLAUDE.md` calls
that basis an unmeasured risk, and it stayed unmeasured because nothing here
held the venue's answer.

`venue_settlements` now does, for both venues. So:

  * **Kalshi vs us** — the direct test. A disagreement rate of x% means x% of
    every training label in five years is wrong in the same way, and no model
    fixes a mislabelled target. It also bounds the backtest: a window we called
    UP and the venue settled DOWN is a trade the backtest paid us for and the
    venue did not.
  * **Polymarket vs us** — the same test against a venue settling on Binance.
  * **Kalshi vs Polymarket** — two independent settlement sources on the same
    fifteen minutes. Where those two disagree, neither is a clean benchmark and
    the window is genuinely ambiguous rather than mislabelled.

Disagreement should concentrate where the two ends of the window are close
together — a near-tie on one index is a coin flip on another — so the rate is
reported against |settlement - strike| in basis points. If it is flat in that
variable, something structural is wrong rather than something marginal.
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

from core.config import DEFAULT_CONFIG
from core.dataset import Dataset, load_minute_bars
from core.datastore import ResearchStore

pd.set_option('display.width', 200)


def main() -> int:
    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    settled = store.read('venue_settlements')
    if not len(settled):
        print('venue_settlements is empty — run scripts.collect_settlements')
        return 1

    bars = load_minute_bars(config, store=store)
    dataset = Dataset.build(bars, config)
    # `windows` carries one row per (symbol, window, offset); the label and the
    # two ends of the comparison are identical across offsets, so collapse to
    # one row per window before joining or every disagreement is counted four
    # times and the rate is unchanged but the n is a fiction.
    ours = dataset.windows[['symbol', 'window_open', 'strike',
                            'settle_price', 'outcome']].drop_duplicates(
        ['symbol', 'window_open'], keep='first').copy()
    ours = ours[ours['outcome'].isin([0, 1, True, False])]
    ours['outcome'] = ours['outcome'].astype(bool)
    ours['window_open'] = pd.to_datetime(ours['window_open'], utc=True)
    print(f'our labels: {len(ours):,} windows  '
          f'{ours["window_open"].min().date()} .. {ours["window_open"].max().date()}')

    settled['window_open'] = pd.to_datetime(settled['window_open'], utc=True)
    for venue, part in settled.groupby('venue'):
        part = part.drop_duplicates(['symbol', 'window_open'], keep='last')
        joined = ours.merge(part[['symbol', 'window_open', 'settled_up']],
                            on=['symbol', 'window_open'], how='inner')
        if not len(joined):
            print(f'\n{venue}: no overlap with our windows')
            continue
        agree = joined['outcome'] == joined['settled_up'].astype(bool)
        move_bp = (1e4 * (joined['settle_price'] - joined['strike'])
                   / joined['strike']).abs()
        print(f'\n{"=" * 78}\n{venue.upper()} vs our Coinbase label — '
              f'{len(joined):,} shared windows\n{"=" * 78}')
        print(f'  agree {100 * agree.mean():.3f}%   '
              f'disagree {int((~agree).sum()):,}')
        print(f'  our base rate {joined["outcome"].mean():.4f}   '
              f'venue base rate {joined["settled_up"].astype(bool).mean():.4f}')
        bands = pd.cut(move_bp, [0, 1, 2, 5, 10, 25, 1e9],
                       labels=['<1bp', '1-2bp', '2-5bp', '5-10bp',
                               '10-25bp', '>25bp'])
        table = pd.DataFrame({'move': bands, 'disagree': ~agree})
        by_band = table.groupby('move', observed=True)['disagree'].agg(
            ['size', 'mean'])
        by_band.columns = ['windows', 'disagree_rate']
        print(by_band.to_string(float_format=lambda v: f'{v:.4f}'))
        print('  Concentration in the narrow bands is a near-tie effect and is')
        print('  benign. A flat rate across bands would mean something structural.')

    both = settled.pivot_table(index=['symbol', 'window_open'], columns='venue',
                               values='settled_up', aggfunc='last')
    if {'kalshi', 'polymarket'}.issubset(both.columns):
        pair = both.dropna(subset=['kalshi', 'polymarket'])
        if len(pair):
            same = pair['kalshi'].astype(bool) == pair['polymarket'].astype(bool)
            print(f'\n{"=" * 78}\nKALSHI vs POLYMARKET — {len(pair):,} shared '
                  f'windows, two independent settlement sources\n{"=" * 78}')
            print(f'  agree {100 * same.mean():.3f}%   '
                  f'disagree {int((~same).sum()):,}')
            print('  Where these two disagree, no label is clean and the window')
            print('  is ambiguous rather than ours being wrong.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
