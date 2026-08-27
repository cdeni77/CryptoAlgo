"""Do the two samplers describe the same book?

The migration does not flip on an argument. During Phase 1 `venue_ladder` holds
a REST row AND a WS row for the same minute — both survive a read because
`transport` is part of the event key — and this prints where they disagree.

**REST is the LAGGIER reference, so 100% agreement is not the target and never
will be reached.** Measured: a stream frame reaches us ~34ms after the venue
stamps it (p50 33.6, p5-p95 31-43), while a REST orderbook round trip is ~73ms
p50 — so its snapshot describes the market ~40-60ms before we parse it. Folding
a captured window to `REST_arrival + offset` puts zero ladder drift across
-100ms..0ms and RISING drift at +25ms and beyond, which says our fold is already
ahead of the REST response by the time it lands.

So a disagreement here is the two samples describing different instants, and the
stream is the one describing the later — which is the point of it. What this
check is actually for is the SHAPE of the disagreement: it must stay small,
symmetric, and confined to prices near the touch. The signature of a real fold
error is different and unmistakable — total resting volume drifting away from
1.0000, or a consistent one-sided bias. Measured over 1,362 production minutes:
median volume ratio 1.0000, 7.3% of minutes low against 8.1% high, and 86.9%
with byte-identical price sets.

The definitive test of the fold is not this one. It is
`tests/test_stream_kalshi.py`, which replays real frames against REST snapshots
captured at the same instants, where the question has no timing ambiguity.

What this check IS for in production, which the replay cannot do: catching a
stream that has gone stale or silently died.
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
    # **How far apart the two samples actually were.** This is the number that
    # makes the comparison honest. The recorder fetches REST and then reads the
    # cache, so the stream row is up to a round trip FRESHER — and measured
    # against a captured window, that alone takes top-of-book agreement from
    # 100% at 0-100ms to 91.7% at 150ms and 66.7% at 500ms. Disagreement at
    # large skew is the book moving, not the fold being wrong; only disagreement
    # at small skew would be evidence against the stream.
    both['skew_ms'] = (
        (both['available_time_ws'] - both['available_time_rest'])
        .dt.total_seconds().abs() * 1000.0)
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

    print(f'  median skew between the two samples: '
          f'{both["skew_ms"].median():.0f} ms')

    print('\nby sampling skew — the two rows are NOT taken at the same instant:')
    skew = pd.cut(both['skew_ms'], [-1, 25, 100, 250, 1e9],
                  labels=['<25ms', '25-100ms', '100-250ms', '>250ms'])
    print(both.groupby(skew, observed=True).agg(
        minutes=('top_yes_same', 'size'),
        yes_top_agree=('top_yes_same', 'mean'),
        no_top_agree=('top_no_same', 'mean')).to_string())

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
