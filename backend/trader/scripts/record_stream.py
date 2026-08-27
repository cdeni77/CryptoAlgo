"""Hold the venue books open, and archive every frame that builds them.

The cache this maintains is process-wide by design: `record_ladder` samples it,
and in a later phase so does the trading loop. One book, read by everyone who
needs one, is the whole point — two samplers of one object was the defect this
replaces.

**Never gated.** Every other recorder awaits `TradingGate.idle()` before a cycle
because their work is a bursty Parquet write. A stream reader that pauses is a
stream reader that goes stale, and staleness is precisely the failure the cache
exists to prevent. Only the spool flush takes the gate.

**Resubscribing is the repair for everything.** Kalshi's `seq` is contiguous per
SUBSCRIPTION, so one missed frame makes every book on the socket suspect and
there is no per-market fix. The same action also picks up the next window's
markets, so one code path covers both.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from pathlib import Path

from core.config import series_to_symbol
from core.spool import FrameSpool, event_rows
from core.stream_book import BookCache
from data_collection.stream.kalshi import VENUE, KalshiStream

logger = logging.getLogger('stream')

# Process-wide, so `record_ladder` and (later) the trading loop read the same
# book the stream is maintaining.
CACHE = BookCache()


def retire(cache: BookCache, keep: set[str]) -> None:
    """Drop books for markets that are no longer open.

    A settled Kalshi market serves an empty ladder rather than an error, so a
    book left in the cache would go quietly stale and still look like a market
    with nothing resting in it.
    """
    for ticker in [t for t in cache.tickers() if t not in keep]:
        cache.forget(ticker)


async def open_tickers(client) -> dict[str, str]:
    """ticker -> symbol, for every open market on the traded series.

    Asks the venue rather than building a ticker from a pattern — the live
    format has already gained a `-15` suffix that no documented pattern predicts.
    """
    out: dict[str, str] = {}
    for series, symbol in series_to_symbol().items():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'status': 'open', 'limit': 5})
        for market in payload.get('markets', []):
            if market.get('ticker'):
                out[market['ticker']] = symbol
    return out


async def consume(stream, cache, spool, symbols, *, gate=None,
                  until: float, flush_every: float = 5.0) -> str:
    """Fold frames until the subscription needs rebuilding.

    Returns the reason, so the caller logs why rather than reconnecting
    silently. **Does not await the gate** — see the module docstring.
    """
    next_flush = time.monotonic() + flush_every
    async for event in stream.events():
        cache.apply(event)
        spool.extend(event_rows(event, symbols.get(event.market_ticker, 'UNKNOWN')))
        now = time.monotonic()
        if now >= next_flush:
            if gate is not None:
                await gate.idle()
            await asyncio.to_thread(spool.flush)
            next_flush = now + flush_every
        if cache.any_gapped():
            return 'sequence gap'
        if now >= until:
            return 'subscription refresh'
    return 'socket closed'


async def run(args, gate=None, cache=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    cache = CACHE if cache is None else cache
    spool = FrameSpool(args.spool_root, VENUE)
    pem = (os.getenv('KALSHI_PRIVATE_KEY')
           or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read())

    async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                            private_key_pem=pem) as client:
        try:
            while True:
                symbols = await open_tickers(client)
                retire(cache, set(symbols))
                if not symbols:
                    logger.warning('no open markets; retrying')
                    await asyncio.sleep(args.refresh_seconds)
                    continue
                stream = KalshiStream(client)
                await stream.connect()
                # The venue restarts `seq` at 1 for a new subscription. Without
                # this the first frame reads as an enormous backwards jump.
                cache.reset_sequence(VENUE)
                await stream.subscribe(list(symbols))
                logger.info('streaming %d markets: %s', len(symbols),
                            ', '.join(sorted(symbols)))
                try:
                    reason = await consume(
                        stream, cache, spool, symbols, gate=gate,
                        until=time.monotonic() + args.refresh_seconds)
                finally:
                    await stream.close()
                    spool.flush()
                logger.info('resubscribing (%s)', reason)
        finally:
            spool.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--spool-root', default='data/spool')
    # Shorter than a window, so a new market is picked up well before it is
    # decided on, and long enough that resubscribing is not the main traffic.
    parser.add_argument('--refresh-seconds', type=float, default=120.0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
