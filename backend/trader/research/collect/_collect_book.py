"""The complete Kalshi order book across each traded window, from Predexon.

Pages each window to exhaustion and packs every snapshot into thirteen numbers
(FIELDS), because Kalshi deletes books on settlement. Predexon returns book
changes forward from `start_time`, 2,000 per page, at 1 req/s on an org-wide
bucket. Run: `python research/collect/_collect_book.py` from backend/trader.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

import aiohttp
import pandas as pd

BASE = 'https://api.predexon.com'
PAUSE = 1.15
OUT = os.getenv('BOOK_OUT', 'data/book_full.jsonl')
# Every minute, because the offset grid is itself under test. The index is a
# convenience; `series` holds every book change.
OFFSETS = tuple(int(x) for x in
                os.getenv('BOOK_OFFSETS', '1,2,3,4,5,6,7,8,9,10,11,12,13,14')
                .split(','))
WINDOW_TARGET = int(os.getenv('BOOK_WINDOWS', '25000'))
REQUEST_BUDGET = int(os.getenv('BOOK_BUDGET', '60000'))
TAIL_SECONDS = 90
# The first day Predexon serves a KXBTC15M book with resting levels. Older
# windows return placeholders with empty ladders. A probe of this boundary must
# send `start_time` and `end_time`, or every window looks empty; check a
# known-good window first.
BOOK_COVERAGE_START = os.getenv('BOOK_COVERAGE_START', '2026-01-08')
MAX_PAGES = 30

FIELDS = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')


def ms(stamp: pd.Timestamp) -> int:
    return int(stamp.timestamp() * 1000)


def pack(book: dict) -> list:
    """A snapshot as thirteen numbers, in FIELDS order."""
    bids = book.get('yes_bids') or []
    asks = book.get('yes_asks') or []
    best_bid = max((b['price'] for b in bids), default=None)
    best_ask = min((a['price'] for a in asks), default=None)

    def within(side, best, sign, cents):
        if best is None:
            return 0
        return sum(x['size'] for x in side
                   if 0 <= sign * (best - x['price']) <= cents)

    return [
        book.get('timestamp'), best_bid, best_ask,
        sum(b['size'] for b in bids if b['price'] == best_bid) if best_bid else 0,
        sum(a['size'] for a in asks if a['price'] == best_ask) if best_ask else 0,
        within(bids, best_bid, 1, 1), within(asks, best_ask, -1, 1),
        within(bids, best_bid, 1, 5), within(asks, best_ask, -1, 5),
        len(bids), len(asks),
        sum(b['size'] for b in bids), sum(a['size'] for a in asks),
    ]


async def fetch_all(session, ticker, lo, hi):
    """Every snapshot in the span, following pagination to exhaustion."""
    out, cursor, pages, calls = [], None, 0, 0
    while pages < MAX_PAGES:
        await asyncio.sleep(PAUSE)
        params = {'ticker': ticker, 'start_time': ms(lo), 'end_time': ms(hi),
                  'limit': 2000}
        if cursor:
            params['pagination_key'] = cursor
        try:
            async with session.get(f'{BASE}/v2/kalshi/orderbooks',
                                   params=params) as r:
                calls += 1
                if r.status == 429:
                    await asyncio.sleep(2.5)
                    continue
                text = await r.text()
                if r.status >= 400:
                    return out, calls, f'{r.status}:{text[:70]}'
                payload = json.loads(text or '{}')
        except Exception as exc:                      # noqa: BLE001
            return out, calls, str(exc)[:70]
        rows = payload.get('snapshots') or payload.get('data') or []
        out += rows
        pages += 1
        page = payload.get('pagination') or {}
        cursor = page.get('pagination_key')
        if not page.get('has_more') or not cursor or not rows:
            break
    return out, calls, None


async def main() -> int:
    key = os.getenv('PREDEXON_API_KEY', '').strip()
    if not key:
        print('PREDEXON_API_KEY is not set.')
        return 1

    # Put backend/trader on sys.path, so `core` imports when run as a script.
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
    from core.datastore import ResearchStore
    store = ResearchStore(os.getenv('RESEARCH_STORE'))

    # The window list comes from `venue_settlements`, which runs close to now,
    # so backfilled windows overlap the live recorder. No price filter: the
    # book features must not be selected on the price they explain.
    settled = store.read('venue_settlements')
    settled = settled[settled['venue'] == 'kalshi']
    settled['window_open'] = pd.to_datetime(settled['window_open'], utc=True)
    rows = settled[['symbol', 'window_open', 'market_ticker']].dropna(
    ).drop_duplicates()

    # Older windows cost a request each and return nothing.
    rows = rows[rows['window_open'] >= pd.Timestamp(BOOK_COVERAGE_START, tz='UTC')]

    # Stop two hours back, so the backfill overlaps the live recorder. Two
    # hours lets the venue's own history settle.
    rows = rows[rows['window_open']
                < pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=2)]
    rows = rows.sort_values('window_open').reset_index(drop=True)
    step = max(1, len(rows) // WINDOW_TARGET)
    rows = rows.iloc[::step].head(WINDOW_TARGET)
    rows = rows.sample(frac=1.0, random_state=20260826).reset_index(drop=True)

    # Windows the live recorder also saw go first, for the live-vs-backfill
    # check. The rest stay shuffled, so any prefix is a representative sample.
    try:
        recorded = store.read('venue_ladder')
        first_live = pd.to_datetime(recorded['window_open'], utc=True).min()
    except Exception:                                     # noqa: BLE001
        first_live = None
    if first_live is not None and pd.notna(first_live):
        overlap = rows['window_open'] >= first_live
        rows = pd.concat([rows[overlap], rows[~overlap]], ignore_index=True)
        print(f'{int(overlap.sum())} windows overlap the live recorder and go '
              f'first, for the live-vs-backfill check', flush=True)

    done = set()
    if os.path.exists(OUT):
        with open(OUT) as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)['market_ticker'])
                except Exception:
                    pass
    todo = rows[~rows['market_ticker'].isin(done)]
    print(f'{len(rows):,} windows, full book, offsets {OFFSETS}. '
          f'{len(done):,} done, {len(todo):,} to go', flush=True)

    headers = {'x-api-key': key, 'Accept': 'application/json'}
    written = errors = empty = truncated = 0
    calls_total = 0
    async with aiohttp.ClientSession(headers=headers) as session:
        with open(OUT, 'a') as handle:
            for _, row in todo.iterrows():
                if calls_total >= REQUEST_BUDGET:
                    print('  budget reached'); break
                w = row['window_open']
                books, calls, err = await fetch_all(
                    session, row['market_ticker'], w,
                    w + pd.Timedelta(minutes=OFFSETS[-1], seconds=TAIL_SECONDS))
                calls_total += calls
                if err:
                    errors += 1
                    if errors <= 3:
                        print(f'  err {row["market_ticker"]}: {err}', flush=True)
                    continue
                if not books:
                    empty += 1
                    continue
                books.sort(key=lambda b: b.get('timestamp', 0))
                series = [pack(b) for b in books]
                # Index of the last snapshot at or before each offset: the state
                # a decision there would price against.
                marks = {}
                for offset in OFFSETS:
                    mark = ms(w + pd.Timedelta(minutes=offset))
                    prior = [i for i, s in enumerate(series) if s[0] <= mark]
                    marks[str(offset)] = prior[-1] if prior else None
                if calls >= MAX_PAGES:
                    truncated += 1
                handle.write(json.dumps({
                    'symbol': row['symbol'],
                    'window_open': w.isoformat(),
                    'market_ticker': row['market_ticker'],
                    'fields': list(FIELDS),
                    'n': len(series),
                    'offset_index': marks,
                    'series': series,
                }) + '\n')
                handle.flush()
                written += 1
                if written % 50 == 0:
                    print(f'  {written:,} windows, {calls_total:,} calls, '
                          f'{empty} empty, {errors} errors, '
                          f'{truncated} hit the page cap', flush=True)
    print(f'done: {written:,} windows, {calls_total:,} calls, {empty} empty, '
          f'{errors} errors, {truncated} truncated')
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
