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
            'minute': int(round(float(row.minute_into_window))),
            'live_bid': best_bid * 100 if best_bid is not None else None,
            'live_ask': (1.0 - best_no) * 100 if best_no is not None else None,
        })
    return pd.DataFrame(rows)


def _backfill_top(venue: str) -> pd.DataFrame:
    """Best bid/ask per (symbol, window, minute) from the collected archive."""
    rows = []
    for path in ARCHIVE.glob(f'venue={venue}/**/windows.jsonl'):
        with open(path) as handle:
            for line in handle:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                opened = pd.Timestamp(rec['window_open']).tz_convert('UTC')
                for snap in rec.get('series') or []:
                    ts, bid, ask = snap[0], snap[1], snap[2]
                    if ts is None:
                        continue
                    when = pd.Timestamp(int(ts), unit='ms', tz='UTC')
                    rows.append({
                        'symbol': rec['symbol'], 'window_open': opened,
                        'minute': int((when - opened).total_seconds() // 60),
                        'bf_bid': bid, 'bf_ask': ask,
                    })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    # Many ticks per minute; take the last, which is the state a decision at
    # that minute would have seen.
    return (frame.sort_values('minute')
                 .groupby(['symbol', 'window_open', 'minute'], as_index=False).last())


def compare(venue: str, live_table: str, store) -> dict:
    live = _live_top(store, live_table)
    back = _backfill_top(venue)
    if live.empty or back.empty:
        return {'venue': venue, 'overlap': 0,
                'note': 'no overlap yet — live recording and backfill must '
                        'cover the same windows for this check to run'}
    merged = live.merge(back, on=['symbol', 'window_open', 'minute'], how='inner')
    merged = merged.dropna(subset=['live_bid', 'bf_bid'])
    if merged.empty:
        return {'venue': venue, 'overlap': 0, 'note': 'no overlapping minutes'}
    bid_gap = (merged['live_bid'] - merged['bf_bid']).abs()
    ask_gap = (merged['live_ask'] - merged['bf_ask']).abs()
    return {
        'venue': venue,
        'overlap': len(merged),
        'windows': merged['window_open'].nunique(),
        'bid_agree_1c': float((bid_gap <= 1).mean()),
        'ask_agree_1c': float((ask_gap <= 1).mean()),
        'bid_median_gap_c': float(bid_gap.median()),
        'ask_median_gap_c': float(ask_gap.median()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    print('backfill vs live-recorded book, top of book at the same minute')
    print('=' * 68)
    worst = None
    for venue, table in (('kalshi', 'venue_ladder'), ('polymarket', 'pm_ladder')):
        try:
            result = compare(venue, table, store)
        except Exception as exc:                              # noqa: BLE001
            print(f'{venue}: could not compare: {str(exc)[:120]}')
            continue
        print()
        for key, value in result.items():
            print(f'  {key:18s} {value}')
        if result.get('overlap'):
            agree = min(result['bid_agree_1c'], result['ask_agree_1c'])
            worst = agree if worst is None else min(worst, agree)
    print()
    if worst is None:
        print('VERDICT: no overlap to judge yet. Re-run once collection has '
              'reached windows the live recorder also covered.')
        return 0
    print(f'VERDICT: worst side agrees within 1c on {worst:.1%} of shared minutes.')
    if worst < 0.90:
        print('  BELOW 90% — the backfill may not describe the same object. '
              'Investigate before trusting the corpus.')
        return 1
    print('  The two sources describe the same book.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
