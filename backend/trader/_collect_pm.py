"""Polymarket short-dated crypto: enumerate the markets, then price them.

**Why this is the most valuable dataset available.** Every result in this project
rests on 70 days of ONE venue. Polymarket runs the same instrument — 15-minute
BTC/ETH/SOL up/down — with different participants, a different settlement source
(Binance rather than CF Benchmarks BRTI), and tick coverage from 2026-03-02,
which is ~178 days against Kalshi's 70. If the offset structure replicates there,
the mechanism is about crypto. If it does not, the Kalshi result is about Kalshi.

Two stages, because discovery and pricing have different shapes:

    stage 1   page `/v2/polymarket/markets` with tags=crypto, sort=created_asc,
              following `pagination_key`, keeping every `*-updown-15m-*` slug
    stage 2   for each market found, pull its price history

Notes established by probing, so they are not rediscovered:
  * every endpoint used here reports `zero_credit_endpoint` — free, and the only
    limit is 1 req/s, enforced on an ORG-wide bucket (so nothing else may run
    against the API at the same time)
  * offset pagination was removed; `pagination_key` is the only way through
  * `search` is the ONE endpoint that costs quota, so it is not used
  * slug pattern is `{asset}-updown-15m-{unix_ts}`, and the timestamp is the
    window, which is what makes the offsets computable without extra calls

Resumable: both stages skip what is already on disk.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys

import aiohttp

BASE = 'https://api.predexon.com'
PAUSE = 1.15
MARKETS_OUT = 'data/pm_markets.jsonl'
PRICES_OUT = 'data/pm_prices.jsonl'
MAX_PAGES = int(os.getenv('PM_PAGES', '1200'))
MAX_PRICED = int(os.getenv('PM_PRICED', '16000'))
# Polymarket tick coverage begins 2026-03-02; older markets have no
# price history, so discovery stops there rather than paging to 2023.
COVERAGE_START = int(dt.datetime(2026, 3, 2, tzinfo=dt.timezone.utc).timestamp())
# Only BTC for now: it is the deepest book and the direct counterpart to
# KXBTC15M. Widening to eth/sol is a matter of this one predicate.
ASSET_PREFIX = os.getenv('PM_ASSET', 'btc-')


async def get(session, path, params, *, tries=4):
    for _ in range(tries):
        await asyncio.sleep(PAUSE)
        try:
            async with session.get(f'{BASE}{path}', params=params) as r:
                if r.status == 429:
                    await asyncio.sleep(2.0)
                    continue
                text = await r.text()
                if r.status >= 400:
                    return None, f'{r.status}:{text[:110]}'
                return json.loads(text or '{}'), None
        except Exception as exc:                      # noqa: BLE001
            return None, str(exc)[:110]
    return None, '429'


def is_short(slug: str) -> bool:
    return 'updown-15m' in (slug or '') and (slug or '').startswith(ASSET_PREFIX)


async def discover(session) -> int:
    seen = set()
    if os.path.exists(MARKETS_OUT):
        with open(MARKETS_OUT) as handle:
            for line in handle:
                try:
                    seen.add(json.loads(line)['market_slug'])
                except Exception:
                    pass
    print(f'stage 1: discovery ({len(seen):,} already known)', flush=True)

    cursor, pages, kept = None, 0, 0
    with open(MARKETS_OUT, 'a') as handle:
        while pages < MAX_PAGES:
            # `tags=15M` returns 50/50 fifteen-minute markets, ~90 minutes of
            # them per page across ~8 assets. `sort=created` is newest-first,
            # which is what we want: walk backwards until the tick coverage
            # boundary. Comma-ANDing tags returns zero, so this is the filter.
            params = {'limit': 50, 'tags': '15M', 'sort': 'created'}
            if cursor:
                params['pagination_key'] = cursor
            payload, err = await get(session, '/v2/polymarket/markets', params)
            if err:
                print(f'  discovery stopped: {err}', flush=True)
                break
            markets = payload.get('markets') or payload.get('data') or []
            if not markets:
                break
            # Stop once the page has walked back past Polymarket tick coverage
            # (2026-03-02). Anything older has no price history to fetch.
            stamps = [int(s.split('-')[-1]) for s in
                      (m.get('market_slug') or '' for m in markets)
                      if s.split('-')[-1].isdigit()]
            if stamps and max(stamps) < COVERAGE_START:
                print(f'  reached coverage boundary at '
                      f'{dt.datetime.fromtimestamp(max(stamps), dt.timezone.utc):%Y-%m-%d}',
                      flush=True)
                break
            for m in markets:
                slug = m.get('market_slug') or ''
                if is_short(slug) and slug not in seen:
                    seen.add(slug)
                    handle.write(json.dumps({
                        'market_slug': slug,
                        'condition_id': m.get('condition_id'),
                        'market_id': m.get('market_id'),
                        'title': m.get('title'),
                        'start_time': m.get('start_time'),
                        'end_time': m.get('end_time'),
                        'close_time': m.get('close_time'),
                        'status': m.get('status'),
                        'winning_side': m.get('winning_side'),
                        'outcomes': m.get('outcomes'),
                        'total_volume_usd': m.get('total_volume_usd'),
                    }) + '\n')
                    kept += 1
            handle.flush()
            pages += 1
            cursor = (payload.get('pagination') or {}).get('pagination_key')
            if not cursor:
                break
            if pages % 25 == 0:
                print(f'  page {pages}, {kept:,} short-dated found', flush=True)
    print(f'stage 1 done: {pages} pages, {kept:,} new, {len(seen):,} total',
          flush=True)
    return kept


FIELDS = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')
OFFSETS = (3, 6, 9, 12)


def pack(book: dict) -> list:
    """One Polymarket snapshot in the SAME thirteen fields as the Kalshi book.

    Deliberately identical to `_collect_book.pack` so the two venues are directly
    comparable — the whole point of collecting Polymarket is to ask whether the
    offset structure replicates, and that question dies if the two datasets are
    shaped differently.

    Prices arrive as dollar floats here and as integer cents on Kalshi, so they
    are converted to cents to match.
    """
    bids = book.get('bids') or []
    asks = book.get('asks') or []

    def cents(x):
        return int(round(float(x) * 100))

    bid_px = [cents(b['price']) for b in bids]
    ask_px = [cents(a['price']) for a in asks]
    best_bid = max(bid_px, default=None)
    best_ask = min(ask_px, default=None)

    def within(side, prices, best, sign, c):
        if best is None:
            return 0
        return sum(float(x['size']) for x, p in zip(side, prices)
                   if 0 <= sign * (best - p) <= c)

    return [
        book.get('timestamp'), best_bid, best_ask,
        sum(float(b['size']) for b, p in zip(bids, bid_px) if p == best_bid)
        if best_bid is not None else 0,
        sum(float(a['size']) for a, p in zip(asks, ask_px) if p == best_ask)
        if best_ask is not None else 0,
        within(bids, bid_px, best_bid, 1, 1), within(asks, ask_px, best_ask, -1, 1),
        within(bids, bid_px, best_bid, 1, 5), within(asks, ask_px, best_ask, -1, 5),
        len(bids), len(asks),
        sum(float(b['size']) for b in bids), sum(float(a['size']) for a in asks),
    ]


async def price(session) -> int:
    """The order book across each window, not candlesticks.

    Candlesticks come back at FIVE-minute granularity — eight rows for a
    fifteen-minute market — which cannot resolve offsets at 3, 6, 9 and 12.
    `interval` takes an integer and did not change it. The order book gives ~35
    snapshots across the window, which is ample for a mid at each offset, and it
    carries depth and imbalance as well.

    The slug's trailing unix timestamp is the window CLOSE, so the window opens
    fifteen minutes earlier and offset `m` is `close - 15m + m`.
    """
    if not os.path.exists(MARKETS_OUT):
        print('stage 2: nothing discovered'); return 0
    markets = [json.loads(l) for l in open(MARKETS_OUT)]
    markets = [m for m in markets
               if m.get('outcomes') and m.get('winning_side') in ('A', 'B')]
    done = set()
    if os.path.exists(PRICES_OUT):
        with open(PRICES_OUT) as handle:
            for line in handle:
                try:
                    done.add(json.loads(line)['market_slug'])
                except Exception:
                    pass
    todo = [m for m in markets if m['market_slug'] not in done][:MAX_PRICED]
    print(f'\nstage 2: book for {len(todo):,} settled markets '
          f'({len(done):,} already done)', flush=True)

    written = errors = empty = 0
    with open(PRICES_OUT, 'a') as handle:
        for m in todo:
            close = int(m['market_slug'].split('-')[-1])
            open_ts = close - 15 * 60
            token = (m['outcomes'][0] or {}).get('token_id')
            if not token:
                errors += 1
                continue
            payload, err = await get(session, '/v2/polymarket/orderbooks', {
                'token_id': token,
                'start_time': open_ts * 1000,
                'end_time': (close + 90) * 1000,
                'limit': 2000})
            if err:
                errors += 1
                if errors <= 3:
                    print(f'  err {m["market_slug"]}: {err}', flush=True)
                if errors > 60 and written == 0:
                    print('  stage 2 abandoned: the book endpoint returns nothing '
                          'for these markets', flush=True)
                    break
                continue
            books = payload.get('snapshots') or payload.get('data') or []
            if not books:
                empty += 1
                continue
            books.sort(key=lambda b: b.get('timestamp', 0))
            series = [pack(b) for b in books]
            marks = {}
            for offset in OFFSETS:
                mark = (open_ts + offset * 60) * 1000
                prior = [i for i, s in enumerate(series)
                         if s[0] is not None and s[0] <= mark]
                marks[str(offset)] = prior[-1] if prior else None
            handle.write(json.dumps({
                'venue': 'polymarket',
                'market_slug': m['market_slug'],
                'condition_id': m['condition_id'],
                'token_id_up': token,
                'window_open': dt.datetime.fromtimestamp(
                    open_ts, dt.timezone.utc).isoformat(),
                'winning_side': m['winning_side'],
                'outcome_labels': [o.get('label') for o in m['outcomes']],
                'fields': list(FIELDS),
                'n': len(series),
                'offset_index': marks,
                'series': series,
            }) + '\n')
            handle.flush()
            written += 1
            if written % 100 == 0:
                print(f'  {written:,} priced, {empty} empty, {errors} errors',
                      flush=True)
    print(f'stage 2 done: {written:,} books, {empty} empty, {errors} errors',
          flush=True)
    return written


async def main() -> int:
    key = os.getenv('PREDEXON_API_KEY', '').strip()
    if not key:
        print('PREDEXON_API_KEY is not set.')
        return 1
    async with aiohttp.ClientSession(
            headers={'x-api-key': key, 'Accept': 'application/json'}) as session:
        if os.getenv('PM_STAGE', 'both') in ('both', 'discover'):
            await discover(session)
        if os.getenv('PM_STAGE', 'both') in ('both', 'price'):
            await price(session)
    return 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
