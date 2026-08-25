"""Record order-book depth at each decision offset, from the websocket.

**This is the only data in the project that cannot be backfilled at any price.**
Verified against the API: `GET /markets/{ticker}/orderbook` on a settled market
returns `{"no_dollars": [], "yes_dollars": []}`. The candlestick history gives
top-of-book bid and ask and nothing behind them. So depth exists only while a
market is open, and only if something is listening.

It is the missing half of the economic question. `scripts/retro_economics.py` can
now price a trade at the venue's real ask — a genuine improvement over the
backtest, where `price_source` substituted `F(x/sigma)` for the market — but it
still has to assume the order filled at the touch, and no historical endpoint can
ever say whether it would have. Every day this is not running is a day of fill
evidence that is gone.

The schemas below were read off the wire, because the documentation has none:

    orderbook_snapshot  {"type","sid","seq","msg":{"market_ticker","market_id",
                         "yes_dollars_fp":[["0.0010","751606.00"], ...]}}
    orderbook_delta     {"type","sid","seq","msg":{"market_ticker","price_dollars",
                         "delta_fp","side","ts","ts_ms"}}

**`seq` is per-SUBSCRIPTION, not per-market**, and getting that wrong is the first
thing this recorder did. One subscription carrying three markets increments a
single counter, so consecutive messages for any one market are almost never
consecutive in `seq`, and tracking it per-book reported a gap on nearly every
message. The counter sat around 37,000 within a minute of connecting, which is the
tell: no single 15-minute market generates that.

A gap in the subscription-level sequence means the connection lost messages, so
every book on it is suspect. The honest response is to drop them all and wait for
fresh snapshots rather than guess what changed — a book reconstructed across a gap
looks perfectly reasonable and is wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore

logger = logging.getLogger('depth')

WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}
CENT = 0.01


class Book:
    """One market's ladder, rebuilt from a snapshot and ordered deltas.

    Kalshi quotes a single book from the YES side: `yes` levels are bids for YES,
    `no` levels are bids for NO — which is the same as offers on YES at
    `1 - price`. So the best ask on YES is `1 - best_no_bid`.
    """

    def __init__(self) -> None:
        self.yes: dict[float, float] = {}
        self.no: dict[float, float] = {}
        self.gaps = 0
        self.stale = True
        self.seq_at_write = 0

    def snapshot(self, msg: dict) -> None:
        self.yes = {float(p): float(s) for p, s in msg.get('yes_dollars_fp', []) or []}
        self.no = {float(p): float(s) for p, s in msg.get('no_dollars_fp', []) or []}
        self.stale = False

    def delta(self, msg: dict) -> None:
        if self.stale:
            return
        side = self.no if msg.get('side') == 'no' else self.yes
        price = float(msg.get('price_dollars', 'nan'))
        size = side.get(price, 0.0) + float(msg.get('delta_fp', 0.0))
        if size > 1e-9:
            side[price] = size
        else:
            side.pop(price, None)

    def top(self) -> dict:
        """Best bid/ask on the YES side, plus cumulative depth behind each."""
        if self.stale or not self.yes or not self.no:
            return {}
        best_bid = max(self.yes)
        best_ask = 1.0 - max(self.no)
        return {
            'yes_bid': best_bid, 'yes_ask': best_ask,
            'yes_bid_size': self.yes[best_bid],
            'yes_ask_size': self.no[max(self.no)],
            'depth_bid_1c': sum(s for p, s in self.yes.items() if p >= best_bid - CENT),
            'depth_bid_5c': sum(s for p, s in self.yes.items() if p >= best_bid - 5 * CENT),
            'depth_ask_1c': sum(s for p, s in self.no.items()
                                if (1.0 - p) <= best_ask + CENT),
            'depth_ask_5c': sum(s for p, s in self.no.items()
                                if (1.0 - p) <= best_ask + 5 * CENT),
            'levels_bid': len(self.yes), 'levels_ask': len(self.no),
            'seq': float(self.seq_at_write), 'gaps': float(self.gaps),
        }


async def open_markets(client, series: str) -> list[dict]:
    payload = await client._request(  # noqa: SLF001
        'GET', '/markets', params={'series_ticker': series, 'status': 'open',
                                   'limit': 20})
    return [m for m in payload.get('markets', []) if m.get('open_time')]


async def run(args) -> int:
    import websockets

    from data_collection.kalshi_client import KalshiClient

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    key_id = os.environ['KALSHI_KEY_ID']

    while True:
        try:
            client = KalshiClient(key_id=key_id, private_key_pem=pem)
            headers = client._headers('GET', '/trade-api/ws/v2')  # noqa: SLF001
            async with client:
                wanted: dict[str, tuple[str, datetime]] = {}
                for series, symbol in SERIES.items():
                    for m in await open_markets(client, series):
                        wanted[m['ticker']] = (symbol, datetime.strptime(
                            m['open_time'], '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc))
                if not wanted:
                    await asyncio.sleep(20)
                    continue
                logger.info('subscribing to %d markets', len(wanted))
                books: dict[str, Book] = defaultdict(Book)
                written: set[tuple[str, int]] = set()
                rows: list[dict] = []
                last_seq: int | None = None       # per SUBSCRIPTION, not per market
                gaps = 0

                async with websockets.connect(WS_URL, extra_headers=headers,
                                              open_timeout=15) as ws:
                    await ws.send(json.dumps({'id': 1, 'cmd': 'subscribe', 'params': {
                        'channels': ['orderbook_delta'],
                        'market_tickers': sorted(wanted)}}))
                    # Resubscribe at the next window boundary, not on a fixed
                    # 16-minute timer. A market becomes `open` when its window
                    # starts, so a fixed timer that began mid-window subscribes to
                    # a market whose offsets have already passed and then sits
                    # idle — which is exactly what the first run did: it
                    # subscribed at +12.6m to a window whose last offset was 38
                    # seconds behind it, and recorded nothing for 16 minutes.
                    now = datetime.now(timezone.utc)
                    boundary = now.replace(second=0, microsecond=0, minute=(
                        now.minute // config.window_minutes) * config.window_minutes)
                    deadline = boundary + timedelta(minutes=config.window_minutes,
                                                    seconds=5)
                    while datetime.now(timezone.utc) < deadline:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        except asyncio.TimeoutError:
                            continue
                        msg = json.loads(raw)
                        kind, body = msg.get('type'), msg.get('msg', {}) or {}
                        ticker = body.get('market_ticker')
                        seq = msg.get('seq')
                        if seq is not None and kind in ('orderbook_snapshot',
                                                        'orderbook_delta'):
                            seq = int(seq)
                            if last_seq is not None and seq > last_seq + 1:
                                # The subscription lost messages, so every book on
                                # it has diverged — not just this market's.
                                gaps += 1
                                logger.warning(
                                    'subscription seq gap %s -> %s; dropping all %d '
                                    'books until fresh snapshots arrive',
                                    last_seq, seq, len(books))
                                for book in books.values():
                                    book.stale = True
                                    book.gaps += 1
                            last_seq = max(seq, last_seq or seq)
                        if kind == 'orderbook_snapshot' and ticker in wanted:
                            books[ticker].snapshot(body)
                        elif kind == 'orderbook_delta' and ticker in wanted:
                            books[ticker].delta(body)
                        else:
                            continue
                        books[ticker].seq_at_write = last_seq or 0

                        symbol, open_time = wanted[ticker]
                        elapsed = (datetime.now(timezone.utc) - open_time).total_seconds() / 60.0
                        for offset in config.decision_offsets:
                            # The first message at or after the offset instant.
                            if not (offset <= elapsed < offset + 0.5):
                                continue
                            if (ticker, offset) in written:
                                continue
                            top = books[ticker].top()
                            if not top:
                                continue
                            written.add((ticker, offset))
                            event_time = open_time + timedelta(minutes=offset)
                            rows.append({'venue': 'kalshi', 'symbol': symbol,
                                         'event_time': event_time,
                                         'available_time': event_time,
                                         'quality': 'valid', 'market_ticker': ticker,
                                         'window_open': open_time,
                                         'offset_minutes': offset, **top})
                        if len(rows) >= args.batch_rows:
                            store.write('venue_depth', pd.DataFrame(rows))
                            logger.info('wrote %d depth rows', len(rows))
                            rows.clear()
                if rows:
                    store.write('venue_depth', pd.DataFrame(rows))
                    logger.info('wrote %d depth rows', len(rows))
        except Exception as exc:                   # noqa: BLE001 - reconnect and continue
            logger.error('depth recorder: %s; reconnecting in 15s', str(exc)[:200])
            await asyncio.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    # One 15-minute window yields 3 markets x 4 offsets = 12 rows, so a larger batch
    # just delays the first write by whole windows. Flush per window instead.
    parser.add_argument('--batch-rows', type=int, default=12)
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
