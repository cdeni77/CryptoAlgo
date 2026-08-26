"""Archive Polymarket's raw order book every minute, in Kalshi's schema.

The counterpart to `scripts/record_ladder.py`, and it exists for the same
reason: resting size at a price is not served historically by anyone, so a day
not recorded is a day gone. It writes to `pm_ladder` with the same columns
`venue_ladder` uses, so the two venues join on (symbol, window_open, minute).

**Why record a venue we do not trade.** Every result in this project rests on
one venue's 70 days. Polymarket runs the same instrument — 15-minute BTC/ETH/SOL
up/down — with different participants and a different settlement source (Binance
rather than CF Benchmarks BRTI). If the offset structure replicates there, the
mechanism is about crypto. If it does not, the Kalshi result is about Kalshi.
That question cannot be asked at all without the data, and it cannot be asked
honestly if the two datasets are shaped differently.

**Public endpoints, not Predexon.** `gamma-api.polymarket.com` and
`clob.polymarket.com` need no key. That matters: Predexon's 1 req/s is an
ORG-WIDE bucket, so a live recorder running through it would throttle the
backfill it is meant to complement. Both refuse a bare `Python-urllib`
User-Agent with a 403 — the header below is not decoration.

**The slug is verified, not assumed.** `{asset}-updown-15m-{unix_open}` is
computable from the clock, and this asks the venue for that slug rather than
trusting it: a market that does not come back is an abstention, and a renamed
series then records nothing instead of recording the wrong thing.

Cost: 3 assets x 2 calls a minute is ~8,600 calls a day, unauthenticated and
unmetered.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore

logger = logging.getLogger('pm-ladder')

GAMMA = 'https://gamma-api.polymarket.com'
CLOB = 'https://clob.polymarket.com'
ASSETS = {'btc': 'BTC-USD', 'eth': 'ETH-USD', 'sol': 'SOL-USD'}
# Both hosts answer a browser agent and 403 a bare urllib one.
HEADERS = {'User-Agent': 'Mozilla/5.0 (quarter research collector)',
           'Accept': 'application/json'}


def window_of(slug: str) -> pd.Timestamp:
    """The window a slug names. Its trailing unix stamp is the OPEN.

    Assumed to be the close at first, which shifted every Polymarket window by
    fifteen minutes and was invisible: every window is a valid window and
    nothing raised. It showed up only when the venue's settlement was scored
    against our Coinbase label — Kalshi agreed 96.98%, Polymarket 49.85%, and
    Kalshi against Polymarket 50.0%, which places the error in the mapping
    rather than in either venue's data.

    The venue says it three ways and they agree: the slug stamp, the title
    ("9:30PM-9:45PM ET"), and `end_time`, which is the stamp plus fifteen
    minutes.
    """
    return pd.Timestamp(int(str(slug).rsplit('-', 1)[-1]), unit='s', tz='UTC')


def slug_for(asset: str, now: datetime) -> str:
    """The slug of the window `now` is INSIDE — floor, never ceil.

    `ceil` names the next window. That market already exists and already trades,
    so the request succeeded and returned a healthy book, which was then stamped
    with this window's `window_open`. A wrong answer that looks entirely right.
    """
    opened = pd.Timestamp(now).tz_convert('UTC').floor(
        f'{DEFAULT_CONFIG.window_minutes}min')
    return f'{asset}-updown-15m-{int(opened.timestamp())}'


def _levels(raw) -> list:
    """`[{price, size}, ...]` as `[[price, size], ...]`, ordered best first.

    Polymarket serves bids ascending and asks descending, so the touch is the
    LAST entry on both sides. Kalshi's ladder is stored best-first, and a schema
    that agrees on columns while disagreeing on order is worse than one that
    disagrees openly. The same argument applies to denomination — see
    `_no_levels`, which is why the ask side is not stored as served.
    """
    out = []
    for entry in raw or []:
        try:
            out.append([float(entry['price']), float(entry['size'])])
        except (TypeError, ValueError, KeyError):
            continue
    return list(reversed(out))


def _no_levels(raw) -> list:
    """Polymarket's YES asks as NO-denominated bids, to match Kalshi exactly.

    Kalshi's orderbook is two BID stacks — `yes_dollars` and `no_dollars` — so
    `no_levels` there is NO-side prices and the YES ask is `1 - best_no_bid`.
    Polymarket serves `bids`/`asks` on one token, so its asks are YES-denominated.
    Storing them unchanged would put a 0.51 YES ask in the column that holds a
    0.51 NO bid on the other venue: same name, opposite meaning, and every shared
    aggregate silently wrong by the spread with imbalance inverted.

    Converting here rather than at read time means the stored row is correct on
    its own terms, which is the only version of "uniform" worth having.
    """
    return [[round(1.0 - price, 6), size] for price, size in _levels(raw)]


async def _get(session, url: str):
    async with session.get(url, headers=HEADERS) as response:
        response.raise_for_status()
        return json.loads(await response.text() or 'null')


async def run(args) -> int:
    import aiohttp

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    rows: list[dict] = []

    while True:
        try:
            async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=20)) as session:
                while True:
                    now = datetime.now(timezone.utc)
                    window_open = pd.Timestamp(now).tz_convert('UTC').floor(
                        f'{config.window_minutes}min')
                    minute = (now - window_open.to_pydatetime()).total_seconds() / 60.0
                    for asset, symbol in ASSETS.items():
                        slug = slug_for(asset, now)
                        try:
                            found = await _get(session, f'{GAMMA}/markets?slug={slug}')
                        except Exception as exc:          # noqa: BLE001
                            logger.warning('%s: %s', slug, str(exc)[:90])
                            continue
                        if not found:
                            continue
                        market = found[0]
                        tokens = json.loads(market.get('clobTokenIds') or '[]')
                        if not tokens:
                            continue
                        # Token 0 is "Up" and token 1 is "Down"; one book is the
                        # mirror of the other, so recording the Up side is the
                        # whole market. `outcomes` is carried to prove it.
                        try:
                            book = await _get(
                                session, f'{CLOB}/book?token_id={tokens[0]}')
                        except Exception as exc:          # noqa: BLE001
                            logger.warning('%s book: %s', slug, str(exc)[:90])
                            continue
                        yes = _levels((book or {}).get('bids'))
                        no = _no_levels((book or {}).get('asks'))
                        if not yes and not no:
                            continue
                        rows.append({
                            'venue': 'polymarket', 'symbol': symbol,
                            'event_time': pd.Timestamp(now).floor('min'),
                            'available_time': pd.Timestamp(now),
                            'quality': 'valid',
                            'market_ticker': slug,
                            'window_open': window_open.to_pydatetime(),
                            'minute_into_window': round(minute, 3),
                            'yes_levels': json.dumps(yes),
                            'no_levels': json.dumps(no),
                            'yes_total': sum(s for _, s in yes),
                            'no_total': sum(s for _, s in no),
                            'outcomes': str(market.get('outcomes') or ''),
                            'token_id_up': str(tokens[0]),
                        })
                    if len(rows) >= args.batch_rows:
                        store.write('pm_ladder', pd.DataFrame(rows))
                        logger.info('wrote %d ladder rows (%d bid levels last)',
                                    len(rows), len(json.loads(rows[-1]['yes_levels'])))
                        rows.clear()
                    await asyncio.sleep(args.interval)
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('pm ladder recorder: %s; retrying in 20s', str(exc)[:160])
            await asyncio.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--batch-rows', type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
