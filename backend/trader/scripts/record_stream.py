"""Hold the Kalshi books open over the websocket and archive every frame.

`CACHE` is process-wide, so `record_ladder` and the trading loop read one book.

The read loop is never gated, because a paused stream goes stale. Only the
spool flush checks the gate, and it defers instead of waiting.

Resubscribing repairs every fault. Kalshi's `seq` is contiguous per
subscription, so one missed frame makes every book on the socket suspect. A
resubscribe also picks up the next window's markets.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from core import heartbeat
from core.config import series_to_symbol
from core.spool import DEFAULT_SPOOL_ROOT, FrameSpool, event_rows
from core.stream_book import BookCache
from data_collection.stream.kalshi import VENUE, KalshiStream

logger = logging.getLogger('stream')

# Process-wide, so `record_ladder` and the trading loop read the same book.
CACHE = BookCache()


def retire(cache: BookCache, keep: set[str]) -> None:
    """Drop books for markets that are no longer open.

    A settled market serves an empty ladder, not an error, so a kept book would
    look like a market with nothing resting.
    """
    for ticker in [t for t in cache.tickers() if t not in keep]:
        cache.forget(ticker)


def _stamp(raw) -> datetime | None:
    try:
        return datetime.strptime(raw, '%Y-%m-%dT%H:%M:%SZ').replace(
            tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return None


def _trading(market: dict, now: datetime) -> bool:
    """Return whether `now` is inside this market's own window.

    Judged on `open_time`/`close_time`, not `status`, because `status` lags the
    market by up to forty seconds. A market missing either timestamp is kept.
    """
    opens, closes = _stamp(market.get('open_time')), _stamp(market.get('close_time'))
    if opens is None or closes is None:
        return True
    return opens <= now < closes


async def open_tickers(client, *, now: datetime | None = None) -> dict[str, str]:
    """Return ticker -> symbol for every traded-series market now trading.

    Asks the venue; never builds a ticker from a pattern. Queries by close
    time, not status, so a new market is found as soon as its `open_time`
    passes.
    """
    now = now or datetime.now(timezone.utc)
    stamp = int(now.timestamp())
    out: dict[str, str] = {}
    for series, symbol in series_to_symbol().items():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'min_close_ts': stamp,
                    'max_close_ts': stamp + 1800, 'limit': 10})
        for market in payload.get('markets', []):
            if market.get('ticker') and _trading(market, now):
                out[market['ticker']] = symbol
    return out


async def consume(stream, cache, spool, symbols, *, gate=None,
                  until: float, flush_every: float = 5.0,
                  idle_timeout: float = 15.0) -> str:
    """Fold frames until the subscription needs rebuilding. Returns the reason.

    Never awaits the gate; a flush due during a decision is deferred. Every
    exit is enforced on a timeout, not on frame arrival, because a silent
    socket never delivers a frame to check on. Silence means the markets
    settled or the connection died.
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
            # reason names the condition that fired.
            if time.monotonic() >= until:
                return 'subscription refresh'
            return f'silent for {idle_timeout:.0f}s'
        except StopAsyncIteration:
            return 'socket closed'

        cache.apply(event)
        spool.extend(event_rows(event, symbols.get(event.market_ticker, 'UNKNOWN')))
        now = time.monotonic()
        # Defer the flush while a decision is in flight; never wait for it.
        # Awaiting the gate would freeze the book the decision reads.
        if now >= next_flush and (gate is None or gate.is_idle):
            heartbeat.touch('stream')
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

    # Keep the socket unless something requires a new one. The deadline only
    # prompts a check; the connection is rebuilt when the market set changes or
    # the stream reports a fault. Rebuilding every time risks a throttle.
    REBUILD = ('socket closed', 'sequence gap')
    stream = None
    subscribed: set[str] = set()

    async def teardown():
        nonlocal stream, subscribed
        if stream is not None:
            await stream.close()
            # Every book now lags the venue until its new snapshot lands.
            cache.invalidate(VENUE)
        stream, subscribed = None, set()

    try:
        async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                private_key_pem=pem) as client:
            while True:
                symbols = await open_tickers(client)
                if not symbols:
                    # Retry fast, not on the refresh cadence: at a window
                    # boundary the next market appears within seconds.
                    logger.warning('no open markets; retrying in %.0fs',
                                   args.empty_retry_seconds)
                    await teardown()
                    await asyncio.sleep(args.empty_retry_seconds)
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
                # Re-check an incomplete market set soon, not on the refresh
                # cadence: the venue lists each series' next market at a
                # different instant.
                covered = set(symbols.values())
                wait = (args.empty_retry_seconds
                        if len(covered) < len(series_to_symbol())
                        else args.refresh_seconds)
                reason = await consume(
                    stream, cache, spool, symbols, gate=gate,
                    until=time.monotonic() + wait,
                    idle_timeout=args.idle_timeout)
                heartbeat.touch('stream')
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
    # Well under the first decision offset, so the next window's market is
    # subscribed before any decision on it.
    parser.add_argument('--refresh-seconds', type=float, default=45.0)
    # At hundreds of frames a second, this much silence means the markets
    # settled or the socket died. Both are repaired by resubscribing.
    parser.add_argument('--idle-timeout', type=float, default=15.0)
    # How long to wait when every market is filtered out. This happens for a
    # few seconds at each window boundary, so it is a fast retry, not a backoff.
    parser.add_argument('--empty-retry-seconds', type=float, default=3.0)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(build_parser().parse_args()))


if __name__ == '__main__':
    raise SystemExit(main())
