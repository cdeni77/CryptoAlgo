"""What the quote backfill actually got, before anyone looks at a result.

Reports the things `DECISION_RULE.md` Appendix A commits to checking: rows per
symbol and offset, the two-sided share, the exclusion count against its 5%
ceiling, spread by week — SOL opened at 232 contracts of volume against BTC's
78,128, so its convergence is worth watching rather than asserting — and the
count that would actually reach `windows_evaluated`.

That last one is the correction that matters most. `windows_evaluated` is not the
number of symbol-windows held. `purged_walk_forward` cuts the timeline into
`n_folds + 2` edges, so 6 folds is 7 blocks, and block 0 is never a test block —
it is the seed training set. `total_windows` sums test folds only, so evaluated
is 6/7 of held. Reading held as evaluated overstates the sample by 17% and moves
the date the gate can pass by twelve days.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from core.datastore import ResearchStore

FOLD_BLOCKS, TEST_BLOCKS = 7, 6      # linspace(0, N, n_folds+2); block 0 is seed
GATE = 20_000
EXCLUSION_CEILING = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=None, help='defaults to $RESEARCH_STORE')
    args = parser.parse_args()

    store = ResearchStore(args.store or os.getenv('RESEARCH_STORE'))
    frame = store.read('venue_quotes')
    if frame.empty:
        print('no venue_quotes rows; run `python -m scripts.backfill_quotes` first')
        return 1
    frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
    frame['usable'] = frame['usable'].astype(bool)
    pd.set_option('display.width', 200)

    print(f'{len(frame):,} rows, {frame["window_open"].min()} .. {frame["window_open"].max()}')
    print(f'{frame["window_open"].dt.floor("D").nunique()} distinct UTC days\n')

    print('rows per symbol and offset')
    table = frame.pivot_table(index='symbol', columns='offset_minutes',
                              values='usable', aggfunc=['size', 'mean'])
    print(table.to_string(float_format=lambda v: f'{v:.4f}'), '\n')

    print('exclusions (DECISION_RULE.md Appendix A fixes this list; ceiling 5%)')
    bad = frame.loc[~frame['usable']]
    if bad.empty:
        print('  none\n')
    else:
        counts = bad['exclude_reason'].value_counts()
        for reason, n in counts.items():
            print(f'  {reason:<22} {n:>7,}  ({100*n/len(frame):.3f}% of rows)')
    share = 1.0 - frame['usable'].mean()
    verdict = 'OK' if share <= EXCLUSION_CEILING else 'OVER CEILING — TEST VOID'
    print(f'  total excluded         {100*share:.3f}%   {verdict}\n')

    print('median spread by week and symbol, in cents (SOL convergence)')
    ok = frame.loc[frame['usable']].copy()
    ok['week'] = ok['window_open'].dt.to_period('W').dt.start_time.dt.date
    weekly = ok.pivot_table(index='week', columns='symbol', values='spread',
                            aggfunc='median') * 100
    volume = ok.pivot_table(index='week', columns='symbol', values='volume',
                            aggfunc='median')
    print(weekly.to_string(float_format=lambda v: f'{v:.2f}'), '\n')
    print('median volume by week and symbol')
    print(volume.to_string(float_format=lambda v: f'{v:,.0f}'), '\n')

    print('what actually reaches windows_evaluated')
    held = ok.drop_duplicates(['symbol', 'window_open']).shape[0]
    evaluated = held * TEST_BLOCKS // FOLD_BLOCKS
    print(f'  usable symbol-windows held      {held:,}')
    print(f'  x {TEST_BLOCKS}/{FOLD_BLOCKS} (block 0 is seed, never tested)   '
          f'{evaluated:,}')
    print(f'  gate                            {GATE:,}'
          f'   {"PASSES" if evaluated >= GATE else f"short by {GATE-evaluated:,}"}')
    if evaluated < GATE:
        need_held = -(-GATE * FOLD_BLOCKS // TEST_BLOCKS)
        per_symbol = (need_held - held) / max(ok['symbol'].nunique(), 1)
        days = ok['window_open'].dt.floor('D').nunique()
        rate = held / max(ok['symbol'].nunique(), 1) / max(days, 1)
        print(f'  held needed                     {need_held:,}'
              f'  ({per_symbol:,.0f} more per symbol)')
        if rate > 0:
            print(f'  at {rate:.1f}/symbol/day          {per_symbol/rate:.1f} more days')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
