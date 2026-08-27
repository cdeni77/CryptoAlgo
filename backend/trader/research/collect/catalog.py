"""Phase 0: rebuild the market catalogs, then seed the ledger from them.

The ledger needs one row per window that SHOULD exist, and the only honest
source for that is each venue's own list of markets. Tickers and slugs are
read from the venue, never constructed — a constructed identifier is a guess
that keeps working until the venue renames a series, and then it silently
finds nothing or, worse, the wrong contract.

Both catalogs also carry the venue's own settlement (`result` / `winning_side`),
which is what `research/validate/_validate_label.py` scores our Coinbase-derived
label against. Collecting it here costs nothing extra — it is on the same
records — and it is the only independent check that the target is built right.
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
# Books are retrievable from 2026-01-08 on Kalshi and from January on
# Polymarket; markets exist earlier on both (Kalshi 2025-12-10, Polymarket
# 2025-10-10) but return no book. Measured, see the design doc.
COLLECT_FROM = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)


class Predexon:
    def __init__(self, key: str, limiter: Optional[RateLimiter] = None):
        self.limiter = limiter or RateLimiter(1.0)
        # Counted, because a walk running at 5s a page when every endpoint
        # answers in 0.04s is entirely retry backoff — and with no counter
        # there is no way to tell that from a slow API.
        self.calls = 0
        self.throttled = 0
        self.failed = 0
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': key, 'Accept': 'application/json',
            # gzip, not brotli. A live probe against a multi-megabyte
            # orderbook response failed with "brotli: decoder process called
            # with data when can_accept_more_data() is False"; inside the
            # collector that would surface only as a silent retry, costing a
            # rate-limit slot for no data.
            'Accept-Encoding': 'gzip',
            'User-Agent': 'Mozilla/5.0 (quarter research collector)'})

    def get(self, path: str, params: dict, *, tries: int = 6):
        """(payload, ok). `ok=False` means the REQUEST failed — never conflated
        with an empty answer."""
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
                    # Cross-check the decoded window against the venue's own
                    # times. A ticker that decodes to a different window than
                    # the venue reports is the failure mode that shifted every
                    # Polymarket window by fifteen minutes, undetected.
                    problem = verify_window(
                        opened, venue_open=_time(m.get('open_time')),
                        venue_close=_time(m.get('close_time')))
                    if problem:
                        continue
                    handle.write(json.dumps({
                        'venue': 'kalshi', 'symbol': symbol,
                        'market_id': ticker,
                        'window_open': opened.isoformat(),
                        'result': str(m.get('result') or '').strip().lower(),
                        'volume': m.get('dollar_volume'),
                        'open_interest': m.get('dollar_open_interest'),
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

    Only the TWAP era (`{asset}-updown-15m-{ts}`) is kept. The earlier
    `up-or-down-15m` era settles on a Chainlink SPOT read at the range's end
    rather than a 60-second TWAP of it — a different random variable, so
    pooling the two would train on two instruments at once.
    """
    # Resumable: a killed walk must not cost the whole thing again. The
    # pagination cursor is parked beside the catalog and the file is appended,
    # so a restart continues mid-stream instead of re-walking from today.
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
            # 100 is the endpoint's ceiling, measured: 100 returns 100, while
            # 200 and 500 both return zero rather than erroring. At 50 this
            # walk needed ~3,550 pages to reach the collection floor and ran
            # at ~1.5 days of history a minute; the page count halves here,
            # and page count is the whole cost on a 1 req/s bucket.
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
    """Every catalogued market at or after the collection floor becomes a
    `pending` row, so 'never asked' is a state rather than an inference."""
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
# Pagination degrades with depth (1.8 days of history a minute at the start,
# 0.55 four hundred pages in) and walks eight-plus assets to extract three.
# Asking for the windows we actually want is flat and ~15x cheaper.
#
# Constructing a slug is safe HERE and nowhere else, for two measured reasons:
# every one of the 10,762 slugs pagination had already found is reproduced
# exactly by this grid with zero off-grid windows; and a wrong slug returns
# nothing rather than something, so the failure mode is a market we miss, not
# a different market we mistake for this one. Every market that comes back is
# still cross-checked against the venue's own end_time before it is written.
SLUGS_PER_REQUEST = 50            # 100 returns HTTP 422; this is a ceiling


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

    Resumable on the batch index, so a killed run continues rather than
    re-asking. Windows the venue does not know about are simply absent from
    the catalog, which is the honest answer: we asked about every window in
    the grid, so an absence is "no market", not "not looked at".
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
                    'volume': m.get('total_volume_usd'),
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
