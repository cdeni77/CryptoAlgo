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

from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore

logger = logging.getLogger('ladder')
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}


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
                            ladder = (book.get('orderbook_fp')
                                      or book.get('orderbook') or {})
                            yes = _levels(ladder.get('yes_dollars') or ladder.get('yes'))
                            no = _levels(ladder.get('no_dollars') or ladder.get('no'))
                            if not yes and not no:
                                continue
                            rows.append({
                                'venue': 'kalshi', 'symbol': symbol,
                                'event_time': pd.Timestamp(now).floor('min'),
                                'available_time': pd.Timestamp(now),
                                'quality': 'valid',
                                'market_ticker': market['ticker'],
                                'window_open': open_time,
                                'minute_into_window': round(minute, 3),
                                'yes_levels': json.dumps(yes),
                                'no_levels': json.dumps(no),
                                'yes_total': sum(s for _, s in yes),
                                'no_total': sum(s for _, s in no),
                            })
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
