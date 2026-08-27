"""Polymarket short-dated crypto: enumerate the markets, then price them.

**Why this is the most valuable dataset available.** Polymarket runs the same
instrument — 15-minute BTC/ETH/SOL up/down — with different participants and a
different settlement source (Chainlink's TWAP-60s data stream rather than CF
Benchmarks BRTI — confirmed from a live market's own `resolutionSource`, not
assumed). If the offset structure replicates there, the mechanism is about
crypto. If it does not, the Kalshi result is about Kalshi.

It also reaches back further than Kalshi does. Measured rather than assumed
(see COVERAGE_START): the TWAP instrument runs from **2025-10-10** for BTC and
ETH and **2025-10-28** for SOL, so ~320 days, against Kalshi order books from
2026-06-19 and Kalshi settlements from 2026-01-06. Two separate numbers used to
be conflated as "Kalshi's 70 days" — 70 days is the ORDER BOOK; the venue's own
settlement record goes back more than three times further.

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
# Where the discovery walk stops going backwards.
#
# This said 2026-03-02 and described it as where "Polymarket tick coverage
# begins". That was never measured and it is wrong by nearly five months.
# Measured by binary search against `?market_slug=` (an exact-match existence
# check, so a day costs two requests rather than a page walk):
#
#     btc  TWAP era from 2025-10-10   clean edge, continues after
#     eth  TWAP era from 2025-10-10   clean edge, continues after
#     sol  TWAP era from 2025-10-28   clean edge, 18 days after btc/eth
#     any  endpoint era from 2025-09-12, the earlier and different instrument
#
# So the usable history is ~320 days rather than the ~178 this constant
# implied, and the walk was stopping ~4.5 months early. September is used here
# rather than the 2025-10-10 TWAP edge so the endpoint era is enumerated too:
# it is the same pages either way, `era` labels every row, and having the rows
# is what lets the boundary be measured instead of assumed a second time.
COVERAGE_START = int(dt.datetime(2025, 9, 1, tzinfo=dt.timezone.utc).timestamp())
# Empty means every asset. Verified live: one unfiltered page of
# `/v2/polymarket/markets?tags=15M` already mixes BTC, ETH, SOL, BNB, DOGE,
# ZEC, XRP and HYPE — filtering to one asset does not shrink the page count,
# it just re-walks the identical ~4,000-page stream once per asset, keeping a
# different eighth of it each time. Set PM_ASSET to narrow it deliberately.
ASSET_PREFIX = os.getenv('PM_ASSET', '')
# Where an interrupted walk's pagination_key is parked so a resume can pick
# up mid-stream instead of re-fetching every already-known page from page 1.
# Measured cost of not having this: a resume after a bare Predexon 500 took
# over two hours to re-walk 844 pages it already had, before it could make
# any forward progress at all.
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

    429 (rate limit) and 5xx (the venue's own problem, not ours) both get a
    retry with backoff. A 500 truncated overnight discovery at page 909 of a
    walk that needed ~4,000 to reach the coverage boundary — `discover()`
    treated it as fatal and stopped 84 days short, silently, because this
    function returned it as an ordinary error on the first try. A genuine
    4xx (400, 404, ...) still fails immediately: retrying a malformed
    request produces the same malformed request.
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


# Polymarket has run 15-minute crypto up/down under two different settlement
# rules, and the slug is the only thing that tells them apart. Both read live
# from the venue's own `description` and `resolutionSource`:
#
#   ENDPOINT_ERA  `{asset}-up-or-down-15m-{ts}`, from 2025-09-12
#       settles on the Chainlink spot stream (data.chain.link/streams/eth-usd)
#       read AT THE END of the range, against the price at its start.
#       The earliest one measured $23.66 volume against $0.00 liquidity.
#
#   TWAP_ERA      `{asset}-updown-15m-{ts}`, the current instrument
#       settles on the Chainlink TWAP-60s stream
#       (data.chain.link/streams/btc-usd-twap-60s-streams): the TWAP OF THE
#       RANGE against the price at its start. A live market measured $21,873
#       of liquidity.
#
# These are not two spellings of one instrument. An endpoint reading and a
# 60-second time-average are different random variables — this repository's own
# invariant is that a time-average over an interval carries a THIRD of its
# endpoint's variance — so pooling the eras would silently train on two
# instruments at once. Only TWAP_ERA matches what `core/windows.py` builds.
#
# `is_short` used to test `'updown-15m' in slug`, which excludes the old era by
# accident: 'updown-15m' is not a substring of 'up-or-down-15m'. Right outcome,
# wrong reason — and it also meant nothing could measure where the boundary
# falls, because the rows were never written. Discovery now keeps both and
# labels each one, so choosing an era is a deliberate downstream filter rather
# than a substring that happens not to match.
TWAP_ERA = 'twap'
ENDPOINT_ERA = 'endpoint'


def era_of(slug):
    """Which settlement era a slug belongs to, or None if it is not a 15m
    up/down market at all. Dash-delimited so neither pattern can match inside
    the other."""
    text = slug or ''
    if '-updown-15m-' in text:
        return TWAP_ERA
    if '-up-or-down-15m-' in text:
        return ENDPOINT_ERA
    return None


def is_short(slug: str, asset_prefix: str) -> bool:
    return era_of(slug) is not None and (slug or '').startswith(asset_prefix)


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
            # `tags=15M` returns 50/50 fifteen-minute markets, ~90 minutes of
            # them per page across ~8 assets. `sort=created` is newest-first,
            # which is what we want: walk backwards until the tick coverage
            # boundary. Comma-ANDing tags returns zero, so this is the filter.
            params = {'limit': 50, 'tags': '15M', 'sort': 'created'}
            if cursor:
                params['pagination_key'] = cursor
            payload, err = await get(session, '/v2/polymarket/markets', params)
            if err:
                # `cursor`/`pages` here are still the ones that produced this
                # request, not the next one — a resume retries this exact
                # page rather than restarting the whole walk from page 1.
                print(f'  discovery stopped: {err}', flush=True)
                _save_cursor(cursor, pages)
                break
            markets = payload.get('markets') or payload.get('data') or []
            if not markets:
                _clear_cursor()
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
            # MAX_PAGES ran out without an error or the boundary — this is
            # progress, not a finish, so save where to pick back up.
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

    Deliberately identical to `_collect_book.pack` (in this same directory) so the two venues are directly
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

    The slug's trailing unix timestamp is the window OPEN, so offset `m` is
    simply `slug_ts + m` and the market closes at `slug_ts + 15m`.
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
            # The slug's trailing stamp is the window OPEN — verified against
            # the venue's `end_time` (stamp + 15 minutes) and its title. Read as
            # a close, every window shifts by fifteen minutes: the books are
            # real and the offsets are wrong, which nothing raises on.
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
