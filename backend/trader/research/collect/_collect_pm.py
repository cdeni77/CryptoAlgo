"""Polymarket 15-minute crypto up/down: enumerate the markets, then fetch their books.

Stage 1 pages `/v2/polymarket/markets` (tags=15M) by `pagination_key`; stage 2
fetches each settled market's order book into the same thirteen fields as
Kalshi. Predexon allows 1 req/s on an org-wide bucket. Both stages resume.
Run: `python research/collect/_collect_pm.py` (PM_STAGE=discover|price|both).
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
# Where the discovery walk stops going backwards. Set before the first
# endpoint-era market, so both eras are enumerated and labelled by `era`.
COVERAGE_START = int(dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc).timestamp())
# Empty means every asset. Each page mixes all assets, so filtering by asset
# does not reduce the page count.
ASSET_PREFIX = os.getenv('PM_ASSET', '')
# Where an interrupted walk saves its `pagination_key`, so a resume continues
# mid-stream.
CURSOR_FILE = 'data/pm_markets_cursor.json'


def _load_cursor():
    if not os.path.exists(CURSOR_FILE):
        return None
    try:
        with open(CURSOR_FILE) as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cursor(cursor, pages: int) -> None:
    with open(CURSOR_FILE, 'w') as handle:
        json.dump({'cursor': cursor, 'pages': pages}, handle)


def _clear_cursor() -> None:
    if os.path.exists(CURSOR_FILE):
        os.remove(CURSOR_FILE)


async def get(session, path, params, *, tries=4):
    """One request, retried on transient failure.

    Retries 429 and 5xx with backoff. Any other 4xx fails at once.
    """
    last_err = None
    for attempt in range(tries):
        await asyncio.sleep(PAUSE)
        try:
            async with session.get(f'{BASE}{path}', params=params) as r:
                if r.status == 429:
                    await asyncio.sleep(2.0)
                    continue
                if r.status >= 500:
                    last_err = f'{r.status}:{(await r.text())[:110]}'
                    await asyncio.sleep(2.0 * (attempt + 1))
                    continue
                text = await r.text()
                if r.status >= 400:
                    return None, f'{r.status}:{text[:110]}'
                return json.loads(text or '{}'), None
        except Exception as exc:                      # noqa: BLE001
            last_err = str(exc)[:110]
            await asyncio.sleep(2.0 * (attempt + 1))
    return None, last_err or '429'


# Polymarket has run two settlement rules, told apart only by the slug:
#
#   ENDPOINT_ERA  `{asset}-up-or-down-15m-{ts}`: the Chainlink spot stream
#       read at the end of the range, against the price at its start.
#   TWAP_ERA      `{asset}-updown-15m-{ts}`: the Chainlink TWAP-60s stream
#       over the range, against the price at its start.
#
# These are different instruments. Only TWAP_ERA matches `core/windows.py`.
# Discovery keeps and labels both; consumers filter on `era`.
TWAP_ERA = 'twap'
ENDPOINT_ERA = 'endpoint'


def era_of(slug):
    """The settlement era of a slug, or None for a non-15m-up/down market.

    Dash-delimited, so neither pattern matches inside the other.
    """
    text = slug or ''
    if '-updown-15m-' in text:
        return TWAP_ERA
    if '-up-or-down-15m-' in text:
        return ENDPOINT_ERA
    return None


def is_short(slug: str, asset_prefix: str) -> bool:
    return era_of(slug) is not None and (slug or '').startswith(asset_prefix)


# The first window whose order book Predexon returns. Earlier markets exist
# but return no snapshots. Discovery runs past this for settlements; pricing
# does not, because each earlier market wastes a request.
BOOK_START = int(dt.datetime(2025, 12, 31, tzinfo=dt.timezone.utc).timestamp())


def worth_pricing(market) -> bool:
    """Whether a discovered market can plausibly return a book.

    Local checks only, to save a request. Excludes the endpoint era.
    """
    slug = (market or {}).get('market_slug') or ''
    if era_of(slug) != TWAP_ERA:
        return False
    stamp = slug.rsplit('-', 1)[-1]
    if not stamp.isdigit():
        return False
    return int(stamp) >= BOOK_START


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

    saved = _load_cursor()
    cursor = saved.get('cursor') if saved else None
    pages = saved.get('pages') if saved else 0
    if saved:
        print(f'  resuming from a saved cursor at page {pages:,}', flush=True)
    kept = 0
    with open(MARKETS_OUT, 'a') as handle:
        while pages < MAX_PAGES:
            # `sort=created` is newest first, so the walk goes backwards.
            # Comma-joined tags return zero rows.
            params = {'limit': 50, 'tags': '15M', 'sort': 'created'}
            if cursor:
                params['pagination_key'] = cursor
            payload, err = await get(session, '/v2/polymarket/markets', params)
            if err:
                # Save the cursor that produced this request, so a resume
                # retries this page.
                print(f'  discovery stopped: {err}', flush=True)
                _save_cursor(cursor, pages)
                break
            markets = payload.get('markets') or payload.get('data') or []
            if not markets:
                _clear_cursor()
                break
            # Stop once the page is older than COVERAGE_START.
            stamps = [int(s.split('-')[-1]) for s in
                      (m.get('market_slug') or '' for m in markets)
                      if s.split('-')[-1].isdigit()]
            if stamps and max(stamps) < COVERAGE_START:
                print(f'  reached coverage boundary at '
                      f'{dt.datetime.fromtimestamp(max(stamps), dt.timezone.utc):%Y-%m-%d}',
                      flush=True)
                _clear_cursor()
                break
            for m in markets:
                slug = m.get('market_slug') or ''
                if is_short(slug, ASSET_PREFIX) and slug not in seen:
                    seen.add(slug)
                    handle.write(json.dumps({
                        'market_slug': slug,
                        'era': era_of(slug),
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
                _clear_cursor()
                break
            if pages % 25 == 0:
                print(f'  page {pages}, {kept:,} short-dated found', flush=True)
        else:
            # MAX_PAGES ran out before the boundary: save the cursor.
            _save_cursor(cursor, pages)
    print(f'stage 1 done: {pages} pages, {kept:,} new, {len(seen):,} total',
          flush=True)
    return kept


FIELDS = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')
OFFSETS = (3, 6, 9, 12)


def pack(book: dict) -> list:
    """One Polymarket snapshot in the SAME thirteen fields as the Kalshi book.

    Matches `_collect_book.pack`, so the venues are comparable. Polymarket
    prices are dollars; they are converted to integer cents.
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
    """The order book across each window.

    Not candlesticks: they are five-minute bars and cannot resolve the offsets.
    The slug's trailing unix stamp is the window open.
    """
    if not os.path.exists(MARKETS_OUT):
        print('stage 2: nothing discovered'); return 0
    markets = [json.loads(l) for l in open(MARKETS_OUT)]
    settled = [m for m in markets
               if m.get('outcomes') and m.get('winning_side') in ('A', 'B')]
    # Skip markets that cannot return a book (see `worth_pricing`), and say so.
    markets = [m for m in settled if worth_pricing(m)]
    if len(settled) != len(markets):
        print(f'  skipping {len(settled) - len(markets):,} markets with no '
              f'retrievable book (pre-2025-12-31, or the endpoint era)',
              flush=True)
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
            # The slug's trailing stamp is the window open, not the close.
            open_ts = int(m['market_slug'].split('-')[-1])
            close = open_ts + 15 * 60
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
