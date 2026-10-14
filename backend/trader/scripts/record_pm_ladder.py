"""Archive Polymarket's raw order book every minute, in Kalshi's schema.

Writes `pm_ladder` with the columns `venue_ladder` uses, so the venues join on
(symbol, window_open, minute). No venue serves historical depth, so a minute
not recorded is lost.

Uses the public gamma and CLOB endpoints, which need no key. Both refuse a
bare `Python-urllib` User-Agent with a 403. The slug
`{asset}-updown-15m-{unix_open}` is asked of the venue, not trusted: a market
that does not come back is an abstention.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from core import heartbeat
from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore

logger = logging.getLogger('pm-ladder')

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
ASSETS = {'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD'}
# Both hosts answer a browser agent and 403 a bare urllib one.
HEADERS = {'User-Agent': 'Mozilla/5.0 (quarter research collector)',
           'Accept': 'application/json'}


# The latest Polymarket touch per symbol, for the trading loop in this process.
# Published in cents, as `core.book_features` reads. The YES ask is
# `1 - best_no_bid`, because `no_levels` is NO-denominated.
CACHE: dict = {}


def window_of(slug: str) -> pd.Timestamp:
    """Return the window a slug names. Its trailing unix stamp is the open.

    The title and `end_time` (the stamp plus fifteen minutes) agree. Reading
    the stamp as the close shifts every window silently.
    """
    return pd.Timestamp(int(str(slug).rsplit('-', 1)[-1]), unit='s', tz='UTC')


def slug_for(asset: str, now: datetime) -> str:
    """Return the slug of the window `now` is inside: floor, never ceil.

    `ceil` names the next window, which already trades and returns a healthy
    book under the wrong window.
    """
    opened = pd.Timestamp(now).tz_convert('UTC').floor(
        f'{DEFAULT_CONFIG.window_minutes}min')
    return f'{asset}-updown-15m-{int(opened.timestamp())}'


def _levels(raw) -> list:
    """Convert `[{price, size}, ...]` to `[[price, size], ...]`, best first.

    Polymarket serves the touch last on both sides. Stored order differs by
    venue: `venue_ladder.yes_levels` is ascending, `pm_ladder.yes_levels` is
    best first. Never index either by position.
    """
    out = []
    for entry in raw or []:
        try:
            out.append([float(entry['price']), float(entry['size'])])
        except (TypeError, ValueError, KeyError):
            continue
    return list(reversed(out))


def _no_levels(raw) -> list:
    """Convert Polymarket's YES asks to NO-denominated bids, as Kalshi stores them.

    Kalshi's book is two bid stacks, so its `no_levels` holds NO prices.
    Polymarket's asks are YES prices on one token. Converting at write time
    keeps `no_levels` meaning the same on both venues.
    """
    return [[round(1.0 - price, 6), size] for price, size in _levels(raw)]


async def _get(session, url: str, *, tries: int = 3):
    """Send one GET, retried on a transient failure.

    403 and 429 get a short retry with backoff: the CLOB host returns a
    transient 403 under an edge throttle. Any other 4xx fails at once.
    """
    for attempt in range(tries):
        try:
            async with session.get(url, headers=HEADERS) as response:
                # 403/429 retry up to the limit; any other status is decided
                # at once by raise_for_status.
                if response.status in (403, 429) and attempt < tries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return json.loads(await response.text() or 'null')
        except (asyncio.TimeoutError, ConnectionError, OSError):
            # A network failure, not an HTTP status: always retry.
            if attempt < tries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f'unreachable: retries exhausted for {url}')


async def _post(session, url: str, payload, *, tries: int = 3):
    """Send one POST with the same retry rules as `_get`."""
    for attempt in range(tries):
        try:
            async with session.post(url, headers=HEADERS, json=payload) as response:
                if response.status in (403, 429) and attempt < tries - 1:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                return json.loads(await response.text() or 'null')
        except (asyncio.TimeoutError, ConnectionError, OSError):
            if attempt < tries - 1:
                await asyncio.sleep(1.5 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f'unreachable: retries exhausted for {url}')


# A real CTF token id is a ~256-bit number (70+ decimal digits). A new market
# can briefly serve a short placeholder id, which the CLOB always refuses.
# 40 sits well between the two.
MIN_TOKEN_LEN = 40


def _valid_token(token) -> bool:
    return bool(token) and len(str(token)) >= MIN_TOKEN_LEN


def _match_books(wanted: list[tuple[str, str]], books: list[dict]) -> dict[str, dict]:
    """Match a batch `/books` response back to each symbol.

    Matches on the echoed `asset_id`, never on response order, which is not
    guaranteed.
    """
    by_token = {token: symbol for symbol, token in wanted}
    out = {}
    for book in books or []:
        symbol = by_token.get(book.get('asset_id'))
        if symbol:
            out[symbol] = book
    return out


async def run(args, gate=None) -> int:
    import aiohttp

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    rows: list[dict] = []

    def _flush_on_exit() -> None:
        """Write the buffer before this task is cancelled.

        `asyncio.CancelledError` is a BaseException, so `except Exception`
        never sees a shutdown. Writes synchronously, because the event loop is
        shutting down.
        """
        if not rows:
            return
        try:
            store.write('pm_ladder', pd.DataFrame(rows))
            logger.info('flushed %d buffered row(s) on shutdown', len(rows))
            rows.clear()
        except Exception as exc:                          # noqa: BLE001
            logger.error('could not flush %d buffered row(s): %s',
                         len(rows), str(exc)[:160])


    while True:
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=20)) as session:
                while True:
                    if gate is not None:
                        await gate.idle()
                    now = datetime.now(timezone.utc)
                    window_open = pd.Timestamp(now).tz_convert('UTC').floor(
                        f'{config.window_minutes}min')
                    minute = (now - window_open.to_pydatetime()).total_seconds() / 60.0

                    # One batched gamma call and one batched CLOB call for all
                    # assets, to ask the throttled CLOB host less often.
                    slugs = {asset: slug_for(asset, now) for asset in ASSETS}
                    qs = '&'.join(f'slug={slug}' for slug in slugs.values())
                    try:
                        found = await _get(session, f'{GAMMA}/markets?{qs}') or []
                    except Exception as exc:              # noqa: BLE001
                        logger.warning('gamma batch: %s', str(exc)[:90])
                        found = []
                    by_slug = {m.get('slug'): m for m in found}

                    # Token 0 is "Up" and token 1 is "Down". One book mirrors
                    # the other, so the Up side is the whole market. `outcomes`
                    # is carried to prove it.
                    wanted: list[tuple[str, str]] = []       # (symbol, token)
                    market_by_symbol: dict[str, dict] = {}
                    slug_by_symbol: dict[str, str] = {}
                    for asset, symbol in ASSETS.items():
                        slug = slugs[asset]
                        market = by_slug.get(slug)
                        if not market:
                            continue
                        tokens = json.loads(market.get('clobTokenIds') or '[]')
                        if not tokens or not _valid_token(tokens[0]):
                            continue
                        wanted.append((symbol, str(tokens[0])))
                        market_by_symbol[symbol] = market
                        slug_by_symbol[symbol] = slug

                    books_by_symbol: dict[str, dict] = {}
                    if wanted:
                        try:
                            resp = await _post(
                                session, f'{CLOB}/books',
                                [{'token_id': token} for _, token in wanted]) or []
                        except Exception as exc:          # noqa: BLE001
                            logger.warning('books batch: %s', str(exc)[:90])
                            resp = []
                        books_by_symbol = _match_books(wanted, resp)

                    for symbol, token in wanted:
                        book = books_by_symbol.get(symbol)
                        if not book:
                            continue
                        yes = _levels(book.get('bids'))
                        no = _no_levels(book.get('asks'))
                        if not yes and not no:
                            continue
                        CACHE[symbol] = {
                            'window_open': window_open,
                            'best_bid': (yes[0][0] * 100.0) if yes else float('nan'),
                            'best_ask': ((1.0 - no[0][0]) * 100.0) if no else float('nan'),
                            'at': pd.Timestamp(now, tz='UTC') if pd.Timestamp(now).tzinfo is None
                                  else pd.Timestamp(now),
                        }
                        rows.append({
                            'venue': 'polymarket', 'symbol': symbol,
                            'event_time': pd.Timestamp(now).floor('min'),
                            'available_time': pd.Timestamp(now),
                            'quality': 'valid',
                            # Never null: `transport` is part of the `pm_ladder`
                            # event key, so each transport's rows stay distinct.
                            'transport': 'rest',
                            'book_age_ms': float('nan'),
                            'market_ticker': slug_by_symbol[symbol],
                            'window_open': window_open.to_pydatetime(),
                            'minute_into_window': round(minute, 3),
                            'yes_levels': json.dumps(yes),
                            'no_levels': json.dumps(no),
                            'yes_total': sum(s for _, s in yes),
                            'no_total': sum(s for _, s in no),
                            'outcomes': str(market_by_symbol[symbol].get('outcomes') or ''),
                            'token_id_up': token,
                        })
                    if len(rows) >= args.batch_rows:
                        await asyncio.to_thread(
                            store.write, 'pm_ladder', pd.DataFrame(rows))
                        logger.info('wrote %d ladder rows (%d bid levels last)',
                                    len(rows), len(json.loads(rows[-1]['yes_levels'])))
                        rows.clear()
                    heartbeat.touch('pm_ladder')
                    await asyncio.sleep(args.interval)
        except asyncio.CancelledError:
            _flush_on_exit()
            raise
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('pm ladder recorder: %s; retrying in 20s', str(exc)[:160])
            await asyncio.sleep(20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--batch-rows', type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
