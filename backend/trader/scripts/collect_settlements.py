"""Collect the venue's own settlement for every 15-minute market, both venues.

Our labels come from Coinbase bars; the venues settle on other indices. Both
venues' results reduce to `settled_up` in `venue_settlements`, so each can be
compared with our label and with the other venue.

Predexon and Polymarket refuse a default `Python-urllib`/`aiohttp` User-Agent
with a Cloudflare 1010, so the header is set explicitly.
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
from data_collection.kalshi_client import _money, _quantity

logger = logging.getLogger('settlements')

PREDEXON = 'https://api.predexon.com'
GAMMA = 'https://gamma-api.polymarket.com'
PAUSE = 1.15
# The one series<->symbol mapping: core/config.SERIES_BY_SYMBOL.
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


async def kalshi_direct(now, store=None, *, args_full: bool = False) -> list[dict]:
    """Return settled markets straight from Kalshi, using the trading credential.

    The default path for keeping the label current. Kalshi purges older
    markets, so history needs `--source predexon`. Pages until it reaches
    windows already stored.
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
            cursor, pages, stale_pages = None, 0, 0
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
                # Stop after two consecutive stale pages, not one. The cursor
                # walks backwards, so stopping at one stale page in front of a
                # hole would leave the hole forever.
                stale_pages = 0 if fresh else stale_pages + 1
                if not cursor or not markets:
                    break
                if stale_pages >= 2 and not args_full:
                    logger.info('%s: reached known history after %d pages',
                                series, pages)
                    break
            logger.info('%s: %d new settlements', series,
                        sum(1 for r in rows if r['symbol'] == symbol))
    return rows


def rows_from_kalshi_markets(markets, *, symbol: str, now) -> list[dict]:
    """Return `venue_settlements` rows from Kalshi's settled-market listing.

    A market without a `result` is skipped, never defaulted: `status=settled`
    can return one mid-finalisation.
    """
    rows = []
    for market in markets or []:
        result = str(market.get('result') or '').strip().lower()
        if result not in ('yes', 'no'):
            continue
        close = _time(market.get('close_time'))
        if close is None or pd.isna(close):
            continue
        # The settlement is keyed on the window open, WINDOW_MINUTES before
        # the close.
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
            # Read through the `_fp`/`_dollars` parsers: Kalshi serves
            # `volume_fp`, `open_interest_fp` and `last_price_dollars` with the
            # bare fields null.
            'volume': _quantity(market, 'volume') or 0.0,
            'open_interest': _quantity(market, 'open_interest') or 0.0,
            'last_price': _money(market, 'last_price') or 0.0,
            'source': 'kalshi_direct',
        })
    return rows


async def kalshi(session, key: str, now: pd.Timestamp, store=None,
                 args_full: bool = False) -> list[dict]:
    """Return one row per settled market, from the venue's own `result`.

    Writes after each series, so one timeout does not lose a long job.
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
            # Stop once pages are entirely known: the endpoint serves newest
            # first. Two consecutive stale pages, so one page straddling a
            # boundary does not end it early.
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
    """Return Polymarket settlement rows from the discovered 15-minute markets.

    Reads the catalog `_collect_pm.py` writes instead of re-walking the venue.
    Polymarket settles on Chainlink's BTC-USD TWAP-60s, `>=` against the open:
    an oracle independent of Kalshi's BRTI.
    """
    path = os.getenv('PM_CATALOG', 'data/collection/pm_catalog.jsonl')
    if not os.path.exists(path):
        logger.info('no %s yet; skipping polymarket', path)
        return []
    rows: list[dict] = []
    seen: set = set()
    with open(path) as handle:
        for line in handle:
            try:
                market = json.loads(line)
            except ValueError:
                continue
            if str(market.get('status') or '').lower() != 'closed':
                continue
            result = str(market.get('result') or '').strip().upper()
            symbol = str(market.get('symbol') or '')
            opened = market.get('window_open')
            if result not in ('A', 'B') or not symbol or not opened:
                continue
            try:
                opened = pd.Timestamp(opened)
            except (ValueError, TypeError):
                continue
            if opened.tzinfo is None:
                opened = opened.tz_localize('UTC')
            opened = opened.tz_convert('UTC')
            key = (symbol, opened)
            if key in seen:
                continue
            seen.add(key)
            # `A` is up. This was measured against Kalshi's settlement, not
            # read from the letter.
            rows.append({
                'venue': 'polymarket', 'symbol': symbol,
                'event_time': opened,
                'available_time': now, 'quality': 'valid',
                'market_ticker': str(market.get('market_id') or ''),
                'window_open': opened.to_pydatetime(),
                'close_time': (opened + pd.Timedelta(minutes=15)).to_pydatetime(),
                'settlement_time': None,
                'result': 'yes' if result == 'A' else 'no',
                'settled_up': result == 'A',
                'volume': float(market.get('volume_dollars') or 0.0),
                'open_interest': float(market.get('liquidity_dollars') or 0.0),
                'last_price': 0.0,
            })
    logger.info('polymarket: %d settled window(s)', len(rows))
    return rows



async def run(args) -> int:
    import aiohttp

    key = os.getenv('PREDEXON_API_KEY', '').strip()
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    now = pd.Timestamp.now(tz='UTC')
    rows: list[dict] = []
    # Sources that answered, so "nothing new" can be told from "never ran".
    reached: list[str] = []
    async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)) as session:
        if args.venue in ('kalshi', 'both'):
            # Kalshi's own listing by default. Predexon is for history, which
            # Kalshi purges.
            if args.source == 'predexon':
                if not key:
                    logger.error('PREDEXON_API_KEY is not set; skipping kalshi')
                else:
                    rows += await kalshi(session, key, now, store,
                                         args_full=args.full)
                    reached.append('predexon')
            else:
                rows += await kalshi_direct(now, store)
                reached.append('kalshi')
        if args.venue in ('polymarket', 'both'):
            pm_rows = await polymarket(session, now)
            rows += pm_rows
            # Record that Polymarket answered, as the Kalshi branches do.
            reached.append('polymarket')

    if not rows:
        # "Nothing new" from a source that answered is success: the store is
        # current. A false failure trains the reader to ignore real ones.
        if reached:
            logger.info('no new settlements from %s; the store is already '
                        'current', ', '.join(reached))
            return 0
        logger.error('nothing collected, and no source answered — this is a '
                     'reachability failure, not an empty result')
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
