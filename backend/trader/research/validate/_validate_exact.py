"""Price the backfilled tick series at the exact instant the recorder polled.

`_validate_depth.py` compares the two sources at a shared minute mark, and the
best it can do is bound the gap: the live recorder polls once a minute, so its
row can sit thirty seconds from the mark, and at a measured ~8.4pp per minute
that alone explains a two-cent median difference. It cannot separate "different
book" from "different moment".

This can. `data/book_full.jsonl` holds every book CHANGE, so the backfill's view
at any instant is recoverable — including the exact microsecond the live
recorder wrote its row. Matched that way there is no clock difference left, and
any remaining disagreement is the two sources describing the same book
differently.

Reported separately because they answer different questions: price agreement
says whether the economics can be trusted; total resting size says whether the
depth can; level COUNT is the one that moved not at all under the time filter in
the looser test, at ~58% of live, which is the shape of a structural difference
rather than a timing one.
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

import bisect
import json
import os

import numpy as np
import pandas as pd

from core.datastore import ResearchStore

pd.set_option('display.width', 200)
FIELDS = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')
IX = {name: i for i, name in enumerate(FIELDS)}


def main() -> int:
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    live = store.read('venue_ladder')
    live['window_open'] = pd.to_datetime(live['window_open'], utc=True)
    live['available_time'] = pd.to_datetime(live['available_time'], utc=True)

    series: dict[tuple, list] = {}
    for line in open('data/book_full.jsonl'):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        key = (record['symbol'], str(record['window_open'])[:19])
        series[key] = record['series']
    print(f'{len(series):,} backfilled windows, {len(live):,} live ladder rows')

    rows = []
    for record in live.to_dict('records'):
        key = (record['symbol'], record['window_open'].isoformat()[:19])
        packed = series.get(key)
        if not packed:
            continue
        poll_ms = int(record['available_time'].timestamp() * 1000)
        stamps = [s[IX['ts']] or 0 for s in packed]
        i = bisect.bisect_right(stamps, poll_ms) - 1
        if i < 0:
            continue
        snap = packed[i]
        try:
            yes = json.loads(record['yes_levels'] or '[]')
            no = json.loads(record['no_levels'] or '[]')
        except (TypeError, ValueError):
            continue
        if not yes or not no:
            continue
        live_bid = max(float(p) for p, _ in yes)
        live_ask = 1.0 - max(float(p) for p, _ in no)
        b_bid = snap[IX['best_bid']]
        b_ask = snap[IX['best_ask']]
        if b_bid is None or b_ask is None:
            continue
        rows.append({
            'lag_s': (poll_ms - stamps[i]) / 1000.0,
            'bid_live': live_bid, 'bid_back': float(b_bid) / 100.0,
            'ask_live': live_ask, 'ask_back': float(b_ask) / 100.0,
            'vol_live': sum(float(s) for _, s in yes),
            'vol_back': float(snap[IX['bid_vol']] or 0),
            'lev_live': float(len(yes)),
            'lev_back': float(snap[IX['bid_levels']] or 0),
        })

    if len(rows) < 30:
        print(f'only {len(rows)} matched observations — let the backfill run on')
        return 1
    f = pd.DataFrame(rows)
    print(f'{len(f):,} observations matched to the exact poll instant  '
          f'(median lag to the last book change {f["lag_s"].median():.2f}s)\n')

    print('=' * 78)
    print('SAME BOOK? — matched in time, so nothing here is the clock')
    print('=' * 78)
    for label, a, b in (('best bid', 'bid_live', 'bid_back'),
                        ('best ask', 'ask_live', 'ask_back')):
        d = (f[a] - f[b]).abs()
        print(f'  {label:9s} exact {100*(d < 1e-9).mean():5.1f}%   '
              f'within 1c {100*(d <= 0.0101).mean():5.1f}%   '
              f'median |diff| {d.median():.4f}   p95 {d.quantile(.95):.4f}')
    for label, a, b in (('resting size', 'vol_live', 'vol_back'),
                        ('level count', 'lev_live', 'lev_back')):
        r = (f[b] + 1) / (f[a] + 1)
        print(f'  {label:12s} median backfill/live {r.median():.3f}   '
              f'p10 {r.quantile(.10):.3f}   p90 {r.quantile(.90):.3f}')
    print('\n  Price agreement decides whether the economics can be trusted;')
    print('  resting size decides whether the depth can. A level count that')
    print('  disagrees while both of those agree is the two sources counting')
    print('  the ladder differently, not seeing a different book.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
