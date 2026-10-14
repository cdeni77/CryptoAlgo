"""Rebuild the market catalogs from each venue's market list, then seed the ledger.

Tickers and slugs are read from the venue, not built. Each catalog row also
carries the venue's settlement (`result` / `winning_side`).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from typing import Iterable, Optional

import requests

from research.collect.fetchers import kalshi_window_open, pm_window_open, verify_window
from research.collect.orchestrator import RateLimiter

BASE = 'https://api.predexon.com/v2'
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}
PM_ASSETS = {'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD'}
# The first day both venues serve books. Earlier markets exist but return no
# book.
COLLECT_FROM = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)


class Predexon:
    def __init__(self, key: str, limiter: Optional[RateLimiter] = None):
        self.limiter = limiter or RateLimiter(1.0)
        # Counted, to tell retry backoff apart from a slow API.
        self.calls = 0
        self.throttled = 0
        self.failed = 0
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': key, 'Accept': 'application/json',
            # gzip, not brotli: brotli fails to decode large responses.
            'Accept-Encoding': 'gzip',
            'User-Agent': 'Mozilla/5.0 (quarter research collector)'})

    def get(self, path: str, params: dict, *, tries: int = 6):
        """(payload, ok). `ok=False` means the request failed, not an empty answer."""
        for attempt in range(tries):
            self.limiter.wait()
            self.calls += 1
            try:
                r = self.session.get(f'{BASE}{path}', params=params, timeout=45)
                if r.status_code == 429 or r.status_code >= 500:
                    self.throttled += 1
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    return None, True
                return r.json(), True
            except Exception:                                 # noqa: BLE001
                self.throttled += 1
                time.sleep(1.5 * (attempt + 1))
        self.failed += 1
        return None, False


def _time(value) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def kalshi_catalog(api: Predexon, out_path: str, *, log=print) -> int:
    """Every KX*15M market Predexon holds, with its window and settlement."""
    written = 0
    with open(out_path, 'w') as handle:
        for series, symbol in SERIES.items():
            cursor, pages = None, 0
            while pages < 800:
                params = {'series_ticker': series, 'limit': 100}
                if cursor:
                    params['pagination_key'] = cursor
                payload, ok = api.get('/kalshi/markets', params)
                if not ok:
                    log(f'  {series}: request failed, stopping this series')
                    break
                markets = (payload or {}).get('markets') or (payload or {}).get('data') or []
                if not markets:
                    break
                pages += 1
                for m in markets:
                    ticker = str(m.get('ticker') or '')
                    try:
                        opened = kalshi_window_open(ticker)
                    except ValueError:
                        continue
                    # Cross-check the decoded window against the venue's times.
                    problem = verify_window(
                        opened, venue_open=_time(m.get('open_time')),
                        venue_close=_time(m.get('close_time')))
                    if problem:
                        continue
                    # The venue serves `volume`/`open_interest` in contracts
                    # and `dollar_*` in dollars. Keys name the unit.
                    #
                    # `status` tells a missing `result` apart from an unsettled
                    # market.
                    handle.write(json.dumps({
                        'venue': 'kalshi', 'symbol': symbol,
                        'market_id': ticker,
                        'window_open': opened.isoformat(),
                        'result': str(m.get('result') or '').strip().lower(),
                        'status': str(m.get('status') or '').strip().lower(),
                        'volume_contracts': m.get('volume'),
                        'volume_dollars': m.get('dollar_volume'),
                        'open_interest_contracts': m.get('open_interest'),
                        'open_interest_dollars': m.get('dollar_open_interest'),
                        'last_price': m.get('last_price'),
                    }) + '\n')
                    written += 1
                cursor = ((payload or {}).get('pagination') or {}).get('pagination_key')
                if not cursor:
                    break
                if pages % 50 == 0:
                    log(f'  {series}: {pages} pages, {written:,} markets')
            log(f'  {series}: done, {written:,} markets so far')
    return written


def pm_catalog(api: Predexon, out_path: str, *, max_pages: int = 6000,
               log=print) -> int:
    """Every 15-minute Polymarket market back to the collection floor.

    Keeps only the TWAP era (`{asset}-updown-15m-{ts}`). The earlier
    `up-or-down-15m` era settles on a Chainlink spot read, a different
    instrument.
    """
    # Resumable: the cursor is saved beside the catalog and the file appended.
    state_path = out_path + '.cursor'
    written, pages, cursor = 0, 0, None
    mode = 'w'
    if os.path.exists(state_path) and os.path.exists(out_path):
        try:
            with open(state_path) as sh:
                saved = json.load(sh)
            cursor, pages, mode = saved.get('cursor'), saved.get('pages', 0), 'a'
            log(f'  resuming from page {pages:,}')
        except (OSError, ValueError):
            cursor, pages, mode = None, 0, 'w'
    floor = COLLECT_FROM.timestamp()
    with open(out_path, mode) as handle:
        while pages < max_pages:
            # 100 is the endpoint's ceiling. A larger limit returns zero rows,
            # not an error.
            params = {'limit': 100, 'tags': '15M', 'sort': 'created'}
            if cursor:
                params['pagination_key'] = cursor
            payload, ok = api.get('/polymarket/markets', params)
            if not ok:
                log('  request failed, stopping discovery')
                with open(state_path, 'w') as sh:
                    json.dump({'cursor': cursor, 'pages': pages}, sh)
                break
            markets = (payload or {}).get('markets') or (payload or {}).get('data') or []
            if not markets:
                break
            pages += 1
            stamps = []
            for m in markets:
                slug = str(m.get('market_slug') or '')
                if '-updown-15m-' not in slug:
                    continue
                asset = slug.split('-')[0]
                if asset not in PM_ASSETS:
                    continue
                try:
                    opened = pm_window_open(slug)
                except ValueError:
                    continue
                stamps.append(opened.timestamp())
                if opened < COLLECT_FROM:
                    continue
                outcomes = m.get('outcomes') or []
                token = (outcomes[0] or {}).get('token_id') if outcomes else None
                if not token:
                    continue
                # Polymarket's `start_time` is when the market was CREATED,
                # not when its window opens — checking the slug against it
                # would reject every market. `end_time` IS the window close
                # (the stamp plus fifteen minutes), so it is the field that
                # can actually corroborate the decoded open.
                problem = verify_window(opened, venue_open=None,
                                        venue_close=_time(m.get('end_time')))
                if problem:
                    continue
                handle.write(json.dumps({
                    'venue': 'polymarket', 'symbol': PM_ASSETS[asset],
                    'market_id': slug, 'token_id': str(token),
                    'window_open': opened.isoformat(),
                    'result': str(m.get('winning_side') or ''),
                    'volume': m.get('total_volume_usd'),
                }) + '\n')
                written += 1
            if stamps and max(stamps) < floor:
                log(f'  reached the collection floor at '
                    f'{dt.datetime.fromtimestamp(max(stamps), dt.timezone.utc):%Y-%m-%d}')
                break
            cursor = ((payload or {}).get('pagination') or {}).get('pagination_key')
            if not cursor:
                break
            with open(state_path, 'w') as sh:
                json.dump({'cursor': cursor, 'pages': pages}, sh)
            if pages % 100 == 0:
                log(f'  page {pages}, {written:,} in-range markets, '
                    f'{api.throttled:,} throttled of {api.calls:,} calls')
    # A clean finish leaves no cursor, so the next run starts fresh.
    if os.path.exists(state_path):
        os.remove(state_path)
    return written


def seed_from_catalogs(ledger, paths: Iterable[str], *, log=print) -> int:
    """Seed a `pending` row for every catalogued market at or after COLLECT_FROM."""
    items = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    opened = dt.datetime.fromisoformat(row['window_open'])
                except Exception:                             # noqa: BLE001
                    continue
                if opened < COLLECT_FROM:
                    continue
                items.append((row['venue'], row['symbol'], opened, row['market_id']))
    ledger.seed(items)
    log(f'  seeded {len(items):,} work items')
    return len(items)


# -- targeted lookup, instead of paging the whole venue ----------------------
#
# Asking for the wanted windows is much cheaper than paging every asset.
#
# Building a slug is safe here only: the grid reproduces every slug pagination
# found, and a wrong slug returns nothing rather than a different market. Every
# returned market is still checked against the venue's `end_time`.
SLUGS_PER_REQUEST = 50            # 100 returns HTTP 422


def window_grid(start: dt.datetime, end: dt.datetime, assets=('btc', 'eth', 'sol')):
    """(slug, symbol, window_open) for every quarter-hour in [start, end)."""
    step = dt.timedelta(minutes=15)
    when = start
    while when < end:
        stamp = int(when.timestamp())
        for asset in assets:
            yield f'{asset}-updown-15m-{stamp}', PM_ASSETS[asset], when
        when += step


def batched(items, size: int):
    items = list(items)
    for i in range(0, len(items), size):
        yield items[i:i + size]


def pm_catalog_by_grid(api: Predexon, out_path: str, *, start=None, end=None,
                       log=print) -> int:
    """Ask the venue about each window we want, 50 slugs at a time.

    Resumable on the batch index. Every grid window is asked about, so an
    absent window means "no market".
    """
    start = start or COLLECT_FROM
    end = end or dt.datetime.now(dt.timezone.utc).replace(
        second=0, microsecond=0, minute=0)
    grid = list(window_grid(start, end))
    chunks = list(batched(grid, SLUGS_PER_REQUEST))

    state_path = out_path + '.grid'
    done_batches, mode = 0, 'w'
    if os.path.exists(state_path) and os.path.exists(out_path):
        try:
            with open(state_path) as sh:
                done_batches = json.load(sh).get('batches', 0)
            mode = 'a'
            log(f'  resuming after batch {done_batches:,}')
        except (OSError, ValueError):
            done_batches, mode = 0, 'w'

    log(f'  {len(grid):,} windows in {len(chunks):,} batches of '
        f'{SLUGS_PER_REQUEST}')
    written = 0
    with open(out_path, mode) as handle:
        for index, chunk in enumerate(chunks):
            if index < done_batches:
                continue
            by_slug = {slug: (symbol, opened) for slug, symbol, opened in chunk}
            payload, ok = api.get('/polymarket/markets',
                                  [('market_slug', s) for s in by_slug]
                                  + [('limit', str(SLUGS_PER_REQUEST))])
            if not ok:
                log(f'  batch {index}: request failed, stopping')
                break
            for m in (payload or {}).get('markets') or (payload or {}).get('data') or []:
                slug = str(m.get('market_slug') or '')
                if slug not in by_slug:
                    continue
                symbol, opened = by_slug[slug]
                outcomes = m.get('outcomes') or []
                token = (outcomes[0] or {}).get('token_id') if outcomes else None
                if not token:
                    continue
                if verify_window(opened, venue_open=None,
                                 venue_close=_time(m.get('end_time'))):
                    continue
                handle.write(json.dumps({
                    'venue': 'polymarket', 'symbol': symbol,
                    'market_id': slug, 'token_id': str(token),
                    'window_open': opened.isoformat(),
                    'result': str(m.get('winning_side') or ''),
                    'status': str(m.get('status') or '').strip().lower(),
                    # The key names the unit, to match Kalshi's.
                    'volume_dollars': m.get('total_volume_usd'),
                    'liquidity_dollars': m.get('liquidity_usd'),
                }) + '\n')
                written += 1
            handle.flush()
            with open(state_path, 'w') as sh:
                json.dump({'batches': index + 1}, sh)
            if (index + 1) % 100 == 0:
                log(f'  batch {index + 1:,}/{len(chunks):,}, {written:,} markets, '
                    f'{api.throttled:,} throttled of {api.calls:,} calls')
    if os.path.exists(state_path):
        os.remove(state_path)
    return written
