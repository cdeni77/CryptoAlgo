"""The complete Kalshi order book across each traded window, from Predexon.

**Complete, not sampled.** An earlier version took two capped requests per market
and accepted truncation. Re-fetching later costs exactly what fetching properly
costs now, and the book is the one dataset Kalshi destroys on settlement — so
this pages to exhaustion and keeps every snapshot.

**Why the whole window and not just the offsets that trade.** Measured on the
five-year Coinbase history at one-minute resolution, log-loss skill over the
baseline DECLINES with offset and +12m sits near the bottom of the grid: +4m
scores +0.002486 (6/6 folds) against +12m's +0.001042, and eleven of thirteen
alternatives beat it. So the offset grid is itself under test, and a book sampled
only where the model currently scores would foreclose the question. Keeping every
snapshot also answers ones nobody has asked yet — queue position, ladder slope,
how depth rebuilds after a sweep — which any fixed sample cannot.

**Storage.** Each snapshot is reduced to thirteen numbers and stored as a packed
array rather than a dict, with the field order recorded once per row. The raw
ladders are ~1.1MB per window; this is ~60 bytes per snapshot, so a full window
costs ~100KB instead of ~1MB.

Established by probing, so it is not rediscovered:
  * snapshots are book CHANGES returned FORWARD from `start_time`, capped at
    2,000 per page, with `pagination.pagination_key` to continue
  * update rate runs a median 1.8/s, 4.13/s at p90, 30/s at worst — so a full
    window is 1,600 to 27,000 snapshots and pagination is mandatory
  * the endpoint reports `zero_credit_endpoint`: free, limited only to 1 req/s on
    an ORG-wide bucket, so nothing else may hit the API concurrently
  * coverage for KXBTC15M begins ~2026-06-19, not at the series open

Work order is shuffled (seeded) so an interrupted run is a representative sample
of the 70 days rather than the oldest slice — iterating chronologically made a
75% hit rate look like 25%, because the early windows are the uncovered ones.
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
# Every minute, not the four the model happens to score. The series already
# holds every book change (~1.8/s), so this index is pure convenience — but a
# four-entry index implies a privilege the data does not have, and the offset
# grid is itself under test. Recomputable from `series` for any offset.
OFFSETS = tuple(int(x) for x in
                os.getenv('BOOK_OFFSETS', '1,2,3,4,5,6,7,8,9,10,11,12,13,14')
                .split(','))
WINDOW_TARGET = int(os.getenv('BOOK_WINDOWS', '25000'))
REQUEST_BUDGET = int(os.getenv('BOOK_BUDGET', '60000'))
TAIL_SECONDS = 90
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

    from core.datastore import ResearchStore
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    at12 = quotes[(quotes['offset_minutes'] == 12)
                  & quotes['usable'].astype(bool)
                  & (quotes['market_probability'] > 0.05)
                  & (quotes['market_probability'] < 0.95)]
    rows = at12[['symbol', 'window_open', 'market_ticker']].dropna().drop_duplicates()
    # **Stop two hours back, not a day.** A one-day cutoff meant the backfill
    # ended exactly where the live ladder recorder began, so the two never
    # described the same minute and could not be checked against each other.
    # That cross-check is the only independent evidence that the history every
    # book feature will be trained on is the same object we record live. Two
    # hours is enough for the venue's own history to settle.
    rows = rows[rows['window_open']
                < pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=2)]
    rows = rows.sort_values('window_open').reset_index(drop=True)
    step = max(1, len(rows) // WINDOW_TARGET)
    rows = rows.iloc[::step].head(WINDOW_TARGET)
    rows = rows.sample(frac=1.0, random_state=20260826).reset_index(drop=True)

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
                # Index of the last snapshot at or before each offset — the state
                # a decision there would actually have priced against. Stored as
                # an index rather than a copy so the series stays the only truth.
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
