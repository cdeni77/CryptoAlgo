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
from core.spool import DEFAULT_SPOOL_ROOT, FrameSpool, event_rows
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
                  until: float, flush_every: float = 5.0,
                  idle_timeout: float = 15.0) -> str:
    """Fold frames until the subscription needs rebuilding.

    Returns the reason, so the caller logs why rather than reconnecting
    silently. **Does not await the gate** around the read — see the module
    docstring; only the flush is gated.

    **Every exit condition is enforced on a timeout, not on frame arrival.**
    The first version checked the refresh deadline inside the loop body, which
    deadlocks the moment the socket goes quiet: observed live at 23:30:00, a
    window boundary, when the subscribed markets settled and the venue simply
    stopped sending. The loop that would have resubscribed to the new window
    could only run if a frame arrived, and no frame was ever coming. Nothing
    raised, `supervise` saw a coroutine still legitimately awaiting, and the
    recorder sat dead behind a healthy container — the exact failure the
    staleness contract exists to prevent, one layer further down.

    So a silent socket IS a condition: at ~436 frames a second on a live market,
    fifteen seconds of nothing means the markets settled or the connection died,
    and both are repaired the same way.
    """
    iterator = stream.events().__aiter__()
    next_flush = time.monotonic() + flush_every
    while True:
        remaining = until - time.monotonic()
        if remaining <= 0:
            return 'subscription refresh'
        try:
            event = await asyncio.wait_for(
                iterator.__anext__(), timeout=min(remaining, idle_timeout))
        except asyncio.TimeoutError:
            # The wait is capped by whichever came first. Re-check, so the
            # reason names the condition that actually fired — an operator
            # reading "silent for 15s" when it was a scheduled refresh would
            # go looking for a dead socket that is not there.
            if time.monotonic() >= until:
                return 'subscription refresh'
            return f'silent for {idle_timeout:.0f}s'
        except StopAsyncIteration:
            return 'socket closed'

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


async def run(args, gate=None, cache=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    cache = CACHE if cache is None else cache
    spool = FrameSpool(args.spool_root, VENUE)
    pem = (os.getenv('KALSHI_PRIVATE_KEY')
           or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read())

    # **The socket is kept unless something actually requires a new one.**
    # Resubscribing is how both failure modes are repaired, but it is not free:
    # rebuilding on every refresh would open ~80 connections an hour and pull a
    # fresh snapshot of every market each time, which is the kind of traffic
    # that gets an API key throttled. So the deadline only prompts a CHECK, and
    # the connection is rebuilt when the market set changed or the stream told
    # us it was broken.
    REBUILD = ('socket closed', 'sequence gap')
    stream = None
    subscribed: set[str] = set()

    async def teardown():
        nonlocal stream, subscribed
        if stream is not None:
            await stream.close()
        stream, subscribed = None, set()

    try:
        async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                private_key_pem=pem) as client:
            while True:
                symbols = await open_tickers(client)
                if not symbols:
                    logger.warning('no open markets; retrying')
                    await teardown()
                    await asyncio.sleep(args.refresh_seconds)
                    continue
                if stream is None or set(symbols) != subscribed:
                    await teardown()
                    retire(cache, set(symbols))
                    stream = KalshiStream(client)
                    await stream.connect()
                    # The venue restarts `seq` at 1 for a new subscription.
                    # Without this the first frame reads as a huge jump back.
                    cache.reset_sequence(VENUE)
                    await stream.subscribe(list(symbols))
                    subscribed = set(symbols)
                    logger.info('streaming %d markets: %s', len(symbols),
                                ', '.join(sorted(symbols)))
                reason = await consume(
                    stream, cache, spool, symbols, gate=gate,
                    until=time.monotonic() + args.refresh_seconds,
                    idle_timeout=args.idle_timeout)
                await asyncio.to_thread(spool.flush)
                if reason in REBUILD or reason.startswith('silent'):
                    logger.info('rebuilding the subscription (%s)', reason)
                    await teardown()
    finally:
        await teardown()
        spool.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--spool-root', default=str(DEFAULT_SPOOL_ROOT))
    # Well under the 3-minute mark of a window, so the next window's market is
    # subscribed long before anything decides on it. Resubscribing costs one
    # snapshot per market, which is nothing against 400+ frames a second.
    parser.add_argument('--refresh-seconds', type=float, default=45.0)
    # At ~436 frames/s on a live market, this much silence means the markets
    # settled or the socket died. Both are repaired by resubscribing.
    parser.add_argument('--idle-timeout', type=float, default=15.0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
