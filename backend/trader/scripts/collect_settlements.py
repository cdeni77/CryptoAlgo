"""The venue's own settlement for every 15-minute market, both venues, one table.

**Why this is the most important thing missing.** Every target in this repository
is built by `core/windows.py` from Coinbase one-minute bars: the strike is the
mean over [t0-1min, t0), the settlement value the mean over [t1-1min, t1), and
`>=` decides. But the market settles on CF Benchmarks BRTI, and `CLAUDE.md` lists
that basis as an **unmeasured risk** — "a close proxy, not the same number".

It stayed unmeasured because nothing here held the venue's answer. Predexon
serves `result` on every settled Kalshi market, and Polymarket publishes
`winning_side`, so the proxy can now be scored against the thing it proxies on
every window rather than on the ~6% we traded. If the two disagree materially,
every label in five years of training data carries that error and no amount of
modelling fixes it.

Uniform by construction. Kalshi's `result` ('yes'/'no') and Polymarket's
`winning_side` ('A'/'B') both reduce to `settled_up`, and both land in
`venue_settlements` with the same columns — so the two venues can be compared to
each other as well as to us. Two independent settlement sources disagreeing on
the same fifteen minutes is itself a measurement.

Cheap: the markets endpoint caps `limit` at 100, so ~70 days of three series is
roughly 200 requests. It is not the book — it is one row per market.

Both endpoints refuse a bare `Python-urllib`/`aiohttp` default User-Agent with a
Cloudflare 1010, which is why the header is set explicitly.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
from typing import Optional

import pandas as pd

from core.datastore import ResearchStore
from core.config import series_to_symbol

logger = logging.getLogger('settlements')

PREDEXON = 'https://api.predexon.com'
GAMMA = 'https://gamma-api.polymarket.com'
PAUSE = 1.15
# The one series<->symbol mapping — see core/config.SERIES_BY_SYMBOL.
# This used to hardcode its own copy with no env read, so pointing
# KALSHI_SERIES_BTC at a demo series moved what the trader traded while
# this kept scraping production.
SERIES = series_to_symbol()
PM_ASSETS = {'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD'}
HEADERS = {'Accept': 'application/json',
           'User-Agent': 'Mozilla/5.0 (quarter research collector)'}


def _time(value) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        return pd.Timestamp(value).tz_convert('UTC')
    except (TypeError, ValueError):
        try:
            return pd.Timestamp(value, tz='UTC')
        except Exception:                                 # noqa: BLE001
            return None


async def _get(session, url: str, params: dict, key: Optional[str] = None):
    headers = dict(HEADERS)
    if key:
        headers['x-api-key'] = key
    for _ in range(4):
        await asyncio.sleep(PAUSE)
        async with session.get(url, params=params, headers=headers) as response:
            if response.status == 429:
                await asyncio.sleep(2.5)
                continue
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f'{response.status}: {text[:140]}')
            return json.loads(text or '{}')
    raise RuntimeError('rate limited')


async def kalshi_direct(now, store=None) -> list[dict]:
    """Settled markets straight from Kalshi, using the trading credential.

    The DEFAULT path for keeping the label current. Predexon remains the only
    way to reach HISTORY — Kalshi purges older markets — so `--source predexon`
    is still there for a backfill, and the 196-day 62,097-row archive came
    through it. But nothing about ongoing collection needs a research key in
    the container that holds the trading credential.

    Pages until it reaches windows already stored, so an hourly run is a couple
    of requests rather than a walk.
    """
    from core.config import SERIES_BY_SYMBOL
    from data_collection.kalshi_client import KalshiClient

    known: set = set()
    if store is not None:
        try:
            existing = store.read('venue_settlements')
            known = set(existing.loc[existing['venue'] == 'kalshi',
                                     'market_ticker'].astype(str))
        except Exception:                                     # noqa: BLE001
            known = set()

    rows: list[dict] = []
    async with KalshiClient() as client:
        if not client.configured:
            logger.error('Kalshi credentials are not configured; '
                         'cannot collect settlements directly')
            return []
        for symbol, series in SERIES_BY_SYMBOL.items():
            cursor, pages = None, 0
            while pages < 20:
                params = {'series_ticker': series, 'status': 'settled',
                          'limit': 200}
                if cursor:
                    params['cursor'] = cursor
                try:
                    payload = await client._request(  # noqa: SLF001
                        'GET', '/markets', params=params)
                except Exception as exc:                      # noqa: BLE001
                    logger.warning('%s settled markets: %s', series,
                                   str(exc)[:120])
                    break
                markets = payload.get('markets') or []
                fresh = [m for m in markets
                         if str(m.get('ticker') or '') not in known]
                rows += rows_from_kalshi_markets(fresh, symbol=symbol, now=now)
                cursor = payload.get('cursor')
                pages += 1
                # Every market on this page is already stored, so everything
                # older is too — the listing is ordered by close.
                if not cursor or not markets or not fresh:
                    break
            logger.info('%s: %d new settlements', series,
                        sum(1 for r in rows if r['symbol'] == symbol))
    return rows


def rows_from_kalshi_markets(markets, *, symbol: str, now) -> list[dict]:
    """`venue_settlements` rows from Kalshi's OWN settled-market listing.

    `GET /markets?status=settled` carries `result` per market, so the ongoing
    label needs no research key — the live container is already authenticated to
    Kalshi. Predexon was only ever required for HISTORY, because Kalshi purges
    older markets; that is what the 196-day, 62,097-row backfill went through.

    Removing the extra credential is not cosmetic. `venue_outcome` closes a
    measured 43% label leak — training on our Coinbase label while pricing
    against BRTI-based quotes let the model bet the index disagreement (72.77%
    win rate where the labels differ against 56.17% where they agree). A label
    path gated on a key nobody remembered to pass is that leak waiting to
    reopen, which is precisely what happened for the six days after
    2026-08-27.

    A market without a `result` is SKIPPED, never defaulted: `status=settled`
    can return one mid-finalisation, and reading a missing result as 'no' would
    fabricate half the labels it touched.
    """
    rows = []
    for market in markets or []:
        result = str(market.get('result') or '').strip().lower()
        if result not in ('yes', 'no'):
            continue
        close = _time(market.get('close_time'))
        if close is None or pd.isna(close):
            continue
        # The window opens WINDOW_MINUTES before it closes; the settlement is
        # keyed on the open, as everywhere else in the store.
        open_time = close - pd.Timedelta(minutes=15)
        rows.append({
            'venue': 'kalshi', 'symbol': symbol,
            'event_time': open_time, 'available_time': now,
            'quality': 'valid',
            'market_ticker': str(market.get('ticker') or ''),
            'window_open': open_time,
            'close_time': close,
            'settlement_time': _time(market.get('settlement_time')),
            'result': result,
            'settled_up': result == 'yes',
            'volume': float(market.get('volume') or 0.0),
            'open_interest': float(market.get('open_interest') or 0.0),
            'last_price': float(market.get('last_price') or 0.0),
            'source': 'kalshi_direct',
        })
    return rows


async def kalshi(session, key: str, now: pd.Timestamp, store=None,
                 args_full: bool = False) -> list[dict]:
    """One row per settled market, from the venue's own `result`.

    Writes after each series rather than at the end. Three series against a
    1 req/s bucket shared with the book backfill is a long job, and a long job
    that only persists on success loses everything to one timeout.
    """
    known = set()
    if store is not None:
        try:
            existing = store.read('venue_settlements')
            known = set(existing.loc[existing['venue'] == 'kalshi',
                                     'market_ticker'].astype(str))
        except Exception:                                 # noqa: BLE001 - first run
            known = set()
    rows: list[dict] = []
    for series, symbol in SERIES.items():
        before = len(rows)
        cursor, pages, stale_pages = None, 0, 0
        while pages < 400:
            params = {'series_ticker': series, 'limit': 100}
            if cursor:
                params['pagination_key'] = cursor
            try:
                payload = await _get(session, f'{PREDEXON}/v2/kalshi/markets',
                                     params, key)
            except Exception as exc:                      # noqa: BLE001
                logger.warning('%s: %s', series, str(exc)[:110])
                break
            markets = payload.get('markets') or payload.get('data') or []
            if not markets:
                break
            for market in markets:
                result = str(market.get('result') or '').strip().lower()
                if result not in ('yes', 'no'):
                    continue          # not settled, or settled void
                open_time = _time(market.get('open_time'))
                if open_time is None:
                    continue
                rows.append({
                    'venue': 'kalshi', 'symbol': symbol,
                    'event_time': open_time, 'available_time': now,
                    'quality': 'valid',
                    'market_ticker': str(market.get('ticker') or ''),
                    'window_open': open_time,
                    'close_time': _time(market.get('close_time')),
                    'settlement_time': _time(market.get('settlement_time')),
                    'result': result,
                    'settled_up': result == 'yes',
                    'volume': float(market.get('volume') or 0.0),
                    'open_interest': float(market.get('open_interest') or 0.0),
                    'last_price': float(market.get('last_price') or 0.0),
                })
            pages += 1
            # **Stop once the page is entirely already known.** The endpoint
            # serves newest first, so a full page of markets we hold means the
            # rest is older and equally held. Without this, keeping settlements
            # current means re-walking ~200 pages of a 1 req/s bucket shared with
            # the book backfill; with it, a routine run is a page or two. Two
            # consecutive stale pages rather than one, so a single page of
            # already-seen markets straddling a boundary does not end it early.
            fresh = [m for m in markets
                     if str(m.get('ticker') or '') not in known]
            stale_pages = 0 if fresh else stale_pages + 1
            if stale_pages >= 2 and not args_full:
                logger.info('%s: reached known history after %d pages',
                            series, pages)
                break
            cursor = (payload.get('pagination') or {}).get('pagination_key')
            if not cursor:
                break
        logger.info('%s: %d settled markets (%d total)',
                    series, len(rows) - before, len(rows))
        if store is not None and len(rows) > before:
            fresh = pd.DataFrame(rows[before:]).drop_duplicates(
                subset=['venue', 'market_ticker'], keep='last')
            store.write('venue_settlements', fresh)
            logger.info('%s: persisted', series)
    return rows


async def polymarket(session, now: pd.Timestamp) -> list[dict]:
    """The same, from `winning_side` on the discovered 15-minute markets.

    Reads whatever `_collect_pm.py` has discovered rather than re-walking the
    venue: discovery is the expensive half and it is already resumable.
    """
    path = 'data/pm_markets.jsonl'
    if not os.path.exists(path):
        logger.info('no %s yet; skipping polymarket', path)
        return []
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            try:
                market = json.loads(line)
            except ValueError:
                continue
            slug = str(market.get('market_slug') or '')
            side = str(market.get('winning_side') or '').strip().upper()
            stamp = slug.rsplit('-', 1)[-1]
            if side not in ('A', 'B') or not stamp.isdigit():
                continue
            asset = slug.split('-', 1)[0]
            symbol = PM_ASSETS.get(asset)
            if symbol is None:
                continue
            # The slug's trailing stamp is the window OPEN, not its close —
            # verified against the venue's own `end_time` and title. Read as a
            # close it shifts every Polymarket window fifteen minutes and drags
            # agreement with our label to a coin flip.
            opened = pd.Timestamp(int(stamp), unit='s', tz='UTC')
            close = opened + pd.Timedelta(minutes=15)
            # `outcomes` is ordered, and A is the first — "Up". Carried through
            # rather than assumed: a market whose labels are not Up/Down is not
            # the instrument this is about and is dropped.
            labels = [str(o.get('label') or '').strip().lower()
                      for o in (market.get('outcomes') or [])]
            if len(labels) < 2 or labels[0] != 'up':
                continue
            rows.append({
                'venue': 'polymarket', 'symbol': symbol,
                'event_time': opened,
                'available_time': now, 'quality': 'valid',
                'market_ticker': slug,
                'window_open': opened,
                'close_time': close,
                'settlement_time': _time(market.get('close_time')),
                'result': 'yes' if side == 'A' else 'no',
                'settled_up': side == 'A',
                'volume': float(market.get('total_volume_usd') or 0.0),
                'open_interest': 0.0, 'last_price': 0.0,
            })
    logger.info('polymarket: %d settled markets', len(rows))
    return rows


async def run(args) -> int:
    import aiohttp

    key = os.getenv('PREDEXON_API_KEY', '').strip()
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    now = pd.Timestamp.now(tz='UTC')
    rows: list[dict] = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)) as session:
        if args.venue in ('kalshi', 'both'):
            # Kalshi's own listing by default: it carries `result` per market
            # and needs no key beyond the trading credential already present.
            # Predexon is for HISTORY, which Kalshi purges.
            if args.source == 'predexon':
                if not key:
                    logger.error('PREDEXON_API_KEY is not set; skipping kalshi')
                else:
                    rows += await kalshi(session, key, now, store,
                                         args_full=args.full)
            else:
                rows += await kalshi_direct(now, store)
        if args.venue in ('polymarket', 'both'):
            rows += await polymarket(session, now)

    if not rows:
        logger.error('nothing collected')
        return 1
    frame = pd.DataFrame(rows).drop_duplicates(
        subset=['venue', 'market_ticker'], keep='last')
    store.write('venue_settlements', frame)
    per = frame.groupby(['venue', 'symbol'])['settled_up'].agg(['size', 'mean'])
    logger.info('wrote %d settlements\n%s', len(frame), per.to_string())
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--venue', choices=('kalshi', 'polymarket', 'both'),
                        default='both')
    parser.add_argument('--source', choices=('kalshi', 'predexon'),
                        default='kalshi',
                        help='where Kalshi results come from. `kalshi` uses the '
                             'venue\'s own settled-market listing and needs no '
                             'extra key; `predexon` reaches history Kalshi has '
                             'purged')
    parser.add_argument('--full', action='store_true',
                        help='walk the whole history rather than stopping at '
                             'markets already stored')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
