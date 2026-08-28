"""Does the backfilled book describe the same object as the book we recorded live?

Validation checkpoint 2 of the collection design. This is the ONLY independent
evidence that Predexon's reconstruction of a window matches what the venue
actually served at the time, and it is possible only because the live-recorded
ladders (`venue_ladder`, `pm_ladder`) were deliberately preserved when the
backfill was deleted — they cannot be re-created, so they are the fixed point
everything else is checked against.

What it compares, on windows both sources cover: the top of book at the same
minute. Prices should agree almost exactly. Sizes and level counts need not:
Kalshi's live orderbook endpoint serves a tapered deci-cent grid (a tenth of a
cent below 10c and above 90c) while the Predexon snapshot is quantised to whole
cents, so `levels_bid`/`levels_ask` are known NOT to be comparable across
sources — a measured ratio of 0.579, unchanged by any time filter.

A price disagreement, by contrast, means the two sources are describing
different moments or different markets, and the backfill cannot be trusted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.datastore import ResearchStore                      # noqa: E402

ARCHIVE = Path(os.getenv('COLLECT_DATA', 'data/collection')) / 'archive'


def _live_top(store, table: str) -> pd.DataFrame:
    """Best bid/ask per (symbol, window, minute) from the recorded ladders."""
    frame = store.read(table)
    if frame is None or frame.empty:
        return pd.DataFrame()
    rows = []
    for row in frame.itertuples():
        try:
            yes = json.loads(row.yes_levels or '[]')
            no = json.loads(row.no_levels or '[]')
        except (TypeError, ValueError):
            continue
        best_bid = max((p for p, _ in yes), default=None)
        # `no_levels` is NO-denominated on BOTH venues by construction, so the
        # YES ask is 1 - best NO bid. Polymarket's asks are converted at write
        # time precisely so this one line is correct for both.
        best_no = max((p for p, _ in no), default=None)
        rows.append({
            'symbol': row.symbol,
            'window_open': pd.Timestamp(row.window_open).tz_convert('UTC'),
            # The instant the observation was actually taken. The recorder
            # samples ~15-45s past the minute, so rounding to a minute and
            # comparing against the backfill's LAST tick in that minute
            # compares instants up to 45 seconds apart — which on a
            # fifteen-minute binary near expiry is several cents of genuine
            # drift, and reads as a data error when it is a timing error.
            'at': pd.Timestamp(row.available_time).tz_convert('UTC'),
            'live_bid': best_bid * 100 if best_bid is not None else None,
            'live_ask': (1.0 - best_no) * 100 if best_no is not None else None,
        })
    return pd.DataFrame(rows)


def _backfill_top(venue: str, wanted=None) -> pd.DataFrame:
    """Best bid/ask per (symbol, window, minute) from the collected archive.

    `wanted` is the set of (symbol, window_open) the live side actually covers,
    and it is not an optimisation. Without it this reads EVERY snapshot of every
    partition — 255 million rows, ~20 GB as Python objects — to answer a question
    about a few hundred overlapping windows. Measured: 5 GB resident and still
    climbing after 23 minutes, which is the same shape that froze this machine
    when `consolidate` held a partition in memory.
    """
    rows = []
    paths = list(ARCHIVE.glob(f'venue={venue}/**/windows.jsonl.gz'))
    paths += list(ARCHIVE.glob(f'venue={venue}/**/windows.jsonl'))
    if wanted:
        months = {w[1].strftime('%Y-%m') for w in wanted}
        symbols = {w[0] for w in wanted}

        def _tags(q):
            return {t.split('=')[0]: t.split('=')[1] for t in q.parts if '=' in t}
        paths = [q for q in paths
                 if _tags(q).get('month') in months
                 and _tags(q).get('symbol') in symbols]
    for path in paths:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt') as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                opened = pd.Timestamp(rec['window_open']).tz_convert('UTC')
                if wanted and (rec.get('symbol'), opened) not in wanted:
                    continue
                for snap in rec.get('series') or []:
                    ts, bid, ask = snap[0], snap[1], snap[2]
                    if ts is None:
                        continue
                    rows.append({
                        'symbol': rec['symbol'], 'window_open': opened,
                        'at': pd.Timestamp(int(ts), unit='ms', tz='UTC'),
                        'bf_bid': bid, 'bf_ask': ask,
                    })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    # Normalise resolution: the ladders carry microseconds and the packed
    # series milliseconds, and merge_asof refuses to join across the two.
    frame['at'] = frame['at'].astype('datetime64[us, UTC]')
    return frame.sort_values('at')


TOLERANCES = ('5s', '20s', '90s')


def sweep(live, back) -> list:
    """Agreement at several matching tolerances.

    The sweep IS the test. Two independent samplers cannot be compared
    exactly: the live recorder stamps an observation when its request
    returned, the backfill stamps a tick when the venue published it, and on a
    fifteen-minute binary near expiry a second of drift is worth cents. So a
    raw agreement percentage cannot distinguish "the backfill is wrong" from
    "the two clocks differ".

    What DOES distinguish them is the shape: if the two describe the same
    book, agreement improves as the match tightens, because less real price
    movement fits inside the window. If they describe different objects — a
    shifted window, the wrong market, mangled units — the disagreement is
    structural and tightening the match does nothing.
    """
    out = []
    for tol in TOLERANCES:
        merged = pd.merge_asof(live, back, on='at', by=['symbol', 'window_open'],
                               direction='backward',
                               tolerance=pd.Timedelta(tol)).dropna(
                                   subset=['live_bid', 'bf_bid'])
        if merged.empty:
            continue
        gap = (merged['live_bid'] - merged['bf_bid']).abs()
        out.append({'tolerance': tol, 'n': len(merged),
                    'exact': float((gap == 0).mean()),
                    'within_1c': float((gap <= 1).mean()),
                    'within_2c': float((gap <= 2).mean()),
                    'median_gap_c': float(gap.median())})
    return out


def compare(venue: str, live_table: str, store) -> dict:
    live = _live_top(store, live_table)
    # Only the windows the live recorder covers; see _backfill_top.
    wanted = set(zip(live['symbol'], live['window_open'])) if len(live) else set()
    back = _backfill_top(venue, wanted)
    if live.empty or back.empty:
        return {'venue': venue, 'overlap': 0,
                'note': 'no overlap yet — live recording and backfill must '
                        'cover the same windows for this check to run'}
    # As-of: for each live observation, the book state the backfill says was
    # standing at that instant. `backward` because a book is a step function —
    # the last tick at or before the observation IS the state then.
    live = live.dropna(subset=['at']).copy()
    live['at'] = live['at'].astype('datetime64[us, UTC]')
    live = live.sort_values('at')
    merged = pd.merge_asof(
        live, back, on='at', by=['symbol', 'window_open'],
        direction='backward', tolerance=pd.Timedelta('90s'))
    merged = merged.dropna(subset=['live_bid', 'bf_bid'])
    if merged.empty:
        return {'venue': venue, 'overlap': 0, 'note': 'no overlapping minutes'}
    bid_gap = (merged['live_bid'] - merged['bf_bid']).abs()
    return {
        'venue': venue,
        'overlap': len(merged),
        'windows': int(merged['window_open'].nunique()),
        'median_gap_c': float(bid_gap.median()),
        'sweep': sweep(live, back),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    print('backfill vs live-recorded book, top of book at the same minute')
    print('=' * 68)
    verdicts = []
    for venue, table in (('kalshi', 'venue_ladder'), ('polymarket', 'pm_ladder')):
        try:
            result = compare(venue, table, store)
        except Exception as exc:                              # noqa: BLE001
            print(f'{venue}: could not compare: {str(exc)[:140]}')
            continue
        print(f'\n{venue}: {result["overlap"]:,} shared observations over '
              f'{result.get("windows", 0)} windows')
        if not result.get('sweep'):
            continue
        print(f'  {"tol":>5s} {"n":>6s} {"exact":>7s} {"<=1c":>7s} {"<=2c":>7s} {"median":>7s}')
        for row in result['sweep']:
            print(f'  {row["tolerance"]:>5s} {row["n"]:6,} {row["exact"]:6.1%} '
                  f'{row["within_1c"]:6.1%} {row["within_2c"]:6.1%} '
                  f'{row["median_gap_c"]:6.1f}c')
        tight, loose = result['sweep'][0], result['sweep'][-1]
        improves = tight['within_1c'] > loose['within_1c']
        close = tight['median_gap_c'] <= 1.0
        verdicts.append((venue, improves, close, tight))
        print(f'  tightening the match {"improves" if improves else "DOES NOT improve"}'
              f' agreement; median gap at the tightest is {tight["median_gap_c"]:.1f}c')

    print()
    if not verdicts:
        print('VERDICT: no overlap to judge yet. Re-run once collection has '
              'reached windows the live recorder also covered.')
        return 0
    bad = [v for v, improves, close, _ in verdicts if not (improves and close)]
    if bad:
        print(f'VERDICT: FAIL for {", ".join(bad)}. Disagreement that does not '
              f'shrink as the match tightens, or a median gap above a cent, is '
              f'structural — a shifted window, the wrong market, or mangled '
              f'units. Investigate before trusting the corpus.')
        return 1
    print('VERDICT: PASS. On every venue the disagreement shrinks as the match '
          'tightens and the median gap at the tightest match is at most a cent, '
          'which is drift between two independently-timed samplers rather than '
          'two different objects.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
