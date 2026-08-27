"""Archive the raw order book every minute, for training we have not designed yet.

Read-only. One REST call per market per minute; the ladder is stored verbatim.

**The whole argument is irrecoverability.** `GET /markets/{ticker}/orderbook`
returns the full ladder while a market is open and
`{"no_dollars": [], "yes_dollars": []}` the moment it settles — verified. No
historical endpoint carries resting size at any price. So unlike every other
dataset here, this one cannot be rebuilt later at any cost, and a day not recorded
is a day gone.

Stored raw rather than summarised. `venue_depth` keeps cumulative size within 1c
and 5c of the touch, which answers exactly one question — would a `fill_or_kill`
have filled. The measured finding of this project is that the market holds
information the feature set cannot express, and book imbalance, ladder slope,
depth asymmetry and level counts are all plausible carriers of it that appear in
no feature group today. A projection chosen for one question forecloses the rest.

Cost: 3 symbols x ~15 minutes x 96 windows a day is ~4,300 calls, about 0.05 req/s
against a Basic-tier ceiling of 20.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from core.config import DEFAULT_CONFIG, series_to_symbol
from core.datastore import ResearchStore
from scripts.record_stream import CACHE

logger = logging.getLogger('ladder')
# The one series<->symbol mapping — see core/config.SERIES_BY_SYMBOL.
# This used to hardcode its own copy with no env read, so pointing
# KALSHI_SERIES_BTC at a demo series moved what the trader traded while
# this kept scraping production.
SERIES = series_to_symbol()


def ws_row(cache, *, ticker, symbol, now, open_time, minute, read_at=None):
    """The same ladder, sampled from the stream cache instead of a REST call.

    Returns None when the cache holds no book for the ticker. An empty ladder
    and no ladder are different claims, and writing an empty one would record a
    dead subscription as a market with nothing resting in it.

    **A STALE book is still recorded, with its age.** For the archive a
    four-second-old book is data, not a fault: a feature can filter on
    `book_age_ms` later, and discarding it forecloses that question. The trading
    path makes the opposite choice — it refuses a book it cannot date — and that
    asymmetry is deliberate rather than an oversight.
    """
    ladder = cache.ladder(ticker)
    if ladder is None:
        return None
    yes = [[p, s] for p, s in ladder.yes]
    no = [[p, s] for p, s in ladder.no]
    # **`available_time` is when the BOOK was knowable, not when we looked.**
    # The cache is read after the REST call returns, so the read instant is up
    # to a round trip later than the book it describes. Stamping the read
    # instant overstates how late this information arrived, and — because the
    # REST row is stamped from the same cycle clock — made the two rows look
    # simultaneous when they are ~100-150ms apart. Measured against the captured
    # fixture, that much skew alone drops top-of-book agreement from 100% to
    # ~92%, which is the whole of the shortfall the live comparison was showing.
    #
    # The cache knows exactly when its book last changed, so say that.
    # `read_at` is when the cache was actually consulted (`now` is the cycle's
    # clock, taken before the REST round trip, and the two are not the same
    # instant); the age is measured from the read, so the frame landed at
    # `read_at - age`. `event_time` still comes from `now` so this row pairs
    # with the REST row for the same minute.
    read_at = pd.Timestamp(now if read_at is None else read_at)
    event_time = pd.Timestamp(now).floor('min')
    observed = read_at - pd.Timedelta(seconds=ladder.age_seconds)
    return {
        'venue': 'kalshi', 'symbol': symbol,
        'event_time': event_time,
        'available_time': max(observed, event_time),
        'quality': 'valid',
        'market_ticker': ticker, 'window_open': open_time,
        'minute_into_window': round(minute, 3),
        'yes_levels': json.dumps(yes), 'no_levels': json.dumps(no),
        'yes_total': sum(s for _, s in yes),
        'no_total': sum(s for _, s in no),
        'transport': 'ws',
        'book_age_ms': round(ladder.age_seconds * 1000.0, 1),
    }


def _levels(raw) -> list:
    out = []
    for entry in raw or []:
        try:
            out.append([float(entry[0]), float(entry[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


async def run(args, gate=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    rows: list[dict] = []

    while True:
        try:
            async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                    private_key_pem=pem) as client:
                while True:
                    # Never start a cycle while a trading decision is in flight:
                    # this is a 60s cadence and a few seconds of deferral
                    # costs nothing, while a delayed decision costs edge.
                    if gate is not None:
                        await gate.idle()
                    now = datetime.now(timezone.utc)
                    for series, symbol in SERIES.items():
                        payload = await client._request(  # noqa: SLF001
                            'GET', '/markets',
                            params={'series_ticker': series, 'status': 'open',
                                    'limit': 5})
                        for market in payload.get('markets', []):
                            if not market.get('open_time'):
                                continue
                            open_time = datetime.strptime(
                                market['open_time'], '%Y-%m-%dT%H:%M:%SZ'
                            ).replace(tzinfo=timezone.utc)
                            minute = (now - open_time).total_seconds() / 60.0
                            if not (0 <= minute <= config.window_minutes):
                                continue
                            try:
                                book = await client._request(  # noqa: SLF001
                                    'GET', f"/markets/{market['ticker']}/orderbook")
                            except Exception as exc:      # noqa: BLE001
                                logger.warning('%s: %s', market['ticker'], str(exc)[:90])
                                continue
                            # When the response actually landed. The cycle clock
                            # `now` was taken before the round trip, so using it
                            # claims this book was knowable earlier than it was
                            # — and made the REST and stream rows look
                            # simultaneous when they are ~100-150ms apart.
                            rest_at = datetime.now(timezone.utc)
                            ladder = (book.get('orderbook_fp')
                                      or book.get('orderbook') or {})
                            yes = _levels(ladder.get('yes_dollars') or ladder.get('yes'))
                            no = _levels(ladder.get('no_dollars') or ladder.get('no'))
                            if not yes and not no:
                                continue
                            rows.append({
                                'venue': 'kalshi', 'symbol': symbol,
                                'event_time': pd.Timestamp(now).floor('min'),
                                'available_time': max(
                                    pd.Timestamp(rest_at),
                                    pd.Timestamp(now).floor('min')),
                                'quality': 'valid',
                                'market_ticker': market['ticker'],
                                'window_open': open_time,
                                'minute_into_window': round(minute, 3),
                                'yes_levels': json.dumps(yes),
                                'no_levels': json.dumps(no),
                                'yes_total': sum(s for _, s in yes),
                                'no_total': sum(s for _, s in no),
                                # The REST call is instantaneous by definition:
                                # it IS the sample, so there is no cache age.
                                'transport': 'rest', 'book_age_ms': 0.0,
                            })
                            # The same minute, sampled from the stream. Both
                            # rows survive a read because `transport` is part of
                            # the event key — see EVENT_KEY_EXTRA. Comparing
                            # them is the only evidence the stream reproduces
                            # the book, and nothing flips until they agree.
                            paired = ws_row(
                                CACHE, ticker=market['ticker'], symbol=symbol,
                                now=now, open_time=open_time, minute=minute,
                                read_at=datetime.now(timezone.utc))
                            if paired is not None:
                                rows.append(paired)
                    if len(rows) >= args.batch_rows:
                        await asyncio.to_thread(
                            store.write, 'venue_ladder', pd.DataFrame(rows))
                        logger.info('wrote %d ladder rows (%d levels last)',
                                    len(rows), len(json.loads(rows[-1]['yes_levels'])))
                        rows.clear()
                    await asyncio.sleep(args.interval)
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('ladder recorder: %s; retrying in 20s', str(exc)[:160])
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
