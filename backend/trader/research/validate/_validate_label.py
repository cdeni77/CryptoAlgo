"""Does our Coinbase-derived label agree with how each venue settled?

Compares `core/windows.py` labels with `venue_settlements` (Kalshi, Polymarket),
and the two venues with each other, by |settlement - strike| band in bp.
Run: `python research/validate/_validate_label.py` from backend/trader.
"""

from __future__ import annotations

# Put backend/trader on sys.path, so `core` imports when run as a script.
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
    # `windows` has one row per (symbol, window, offset). Collapse to one row
    # per window, or each window is counted once per offset.
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
