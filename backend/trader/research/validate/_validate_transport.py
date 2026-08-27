"""Do the two samplers describe the same book?

The migration does not flip on an argument. During Phase 1 `venue_ladder` holds
a REST row AND a WS row for the same minute — both survive a read because
`transport` is part of the event key — and this prints where they disagree.

**A disagreement is not automatically the stream's fault.** The REST call and
the cache sample are taken at slightly different instants and this book moves
hundreds of times a second, so some divergence is the market, not the transport.
What matters is the shape: small, symmetric, concentrated in prices far from the
touch, and shrinking as `book_age_ms` falls. Structural disagreement — a
consistently higher top of book, a level count that scales — is a fold error.

The offline version of this check already passed on a captured window: 11 of 11
comparisons agreed exactly on the best bid on both sides. This is the same
question asked continuously, in production, where the stream can also go stale
or silently die.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from core.datastore import ResearchStore


def _top(raw) -> float:
    levels = json.loads(raw or '[]')
    return max((p for p, _ in levels), default=float('nan'))


def _prices(raw) -> set:
    return {p for p, _ in json.loads(raw or '[]')}


def compare(store: ResearchStore | None = None) -> pd.DataFrame:
    store = store or ResearchStore(os.getenv('RESEARCH_STORE'))
    rows = store.read('venue_ladder')
    if rows.empty or 'transport' not in rows.columns:
        return pd.DataFrame()
    key = ['symbol', 'market_ticker', 'event_time']
    rest = rows[rows['transport'] == 'rest'].set_index(key)
    ws = rows[rows['transport'] == 'ws'].set_index(key)
    if rest.empty or ws.empty:
        return pd.DataFrame()
    both = rest.join(ws, how='inner', lsuffix='_rest', rsuffix='_ws')
    if both.empty:
        return both
    for side in ('yes', 'no'):
        both[f'top_{side}_rest'] = both[f'{side}_levels_rest'].map(_top)
        both[f'top_{side}_ws'] = both[f'{side}_levels_ws'].map(_top)
        # NaN means "this side of the book was empty", and both empty is
        # AGREEMENT. `nan == nan` is False, so comparing directly scored every
        # empty-vs-empty minute as a disagreement and reported 33% agreement on
        # a side where the two transports never actually differed.
        rest_top, ws_top = both[f'top_{side}_rest'], both[f'top_{side}_ws']
        both[f'top_{side}_same'] = (
            (rest_top == ws_top) | (rest_top.isna() & ws_top.isna()))
        both[f'{side}_both_empty'] = rest_top.isna() & ws_top.isna()
        both[f'drift_{side}'] = [
            len(_prices(a) ^ _prices(b))
            for a, b in zip(both[f'{side}_levels_rest'], both[f'{side}_levels_ws'])]
    both['size_ratio'] = (both['yes_total_ws']
                          / both['yes_total_rest'].replace(0, np.nan))
    return both


def main() -> int:
    both = compare()
    if both.empty:
        print('no paired minutes yet — let both samplers run for a while')
        return 1
    n = len(both)
    print(f'paired minutes: {n}')
    for side in ('yes', 'no'):
        agree = both[f'top_{side}_same'].mean()
        empty = both[f'{side}_both_empty'].sum()
        print(f'  best {side.upper()} bid identical: {agree:>7.2%}   '
              f'median ladder drift: {both[f"drift_{side}"].median():.0f} prices'
              f'   (both empty on {empty})')
    print(f'  median size ratio ws/rest: {both["size_ratio"].median():.4f}')
    print(f'  median book age at sample: {both["book_age_ms_ws"].median():.0f} ms')

    print('\nby cache staleness:')
    buckets = pd.cut(both['book_age_ms_ws'], [-1, 500, 2000, 10000, 1e12],
                     labels=['<0.5s', '0.5-2s', '2-10s', '>10s'])
    table = both.groupby(buckets, observed=True).agg(
        minutes=('top_yes_same', 'size'),
        yes_top_agree=('top_yes_same', 'mean'),
        no_top_agree=('top_no_same', 'mean'),
        drift=('drift_yes', 'median'))
    print(table.to_string())

    worst = both.nlargest(5, 'drift_yes')
    if worst['drift_yes'].max() > 2:
        print('\nlargest YES ladder disagreements:')
        print(worst[['market_ticker_rest' if 'market_ticker_rest' in worst
                     else 'minute_into_window_rest',
                     'drift_yes', 'book_age_ms_ws',
                     'top_yes_rest', 'top_yes_ws']].to_string())
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
