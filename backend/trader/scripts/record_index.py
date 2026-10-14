"""Hold the index constituents' books open and record a per-second index proxy.

One reconnecting socket per exchange (Coinbase, Kraken, Bitstamp, Gemini), each
folding its own messages into a `core.index_proxy.ExchangeState`, and one
sampler that every second takes the median fresh mid per symbol, records it in
`CACHE` for the live process and spools it to `index_proxy_1s`.

**Coinbase gets its own `level2` socket here rather than reusing
`record_spot_stream`.** That stream carries trades only — there is no book on
it to take a mid from — and it feeds the bars the trading loop decides on. Its
own socket means an index failure, a 10MB level2 snapshot or a sequence gap here
can never touch the tape trading reads.

**Fails soft, by construction.** Every exchange task catches its own failures
and reconnects with backoff; a dead venue is a NaN column, and fewer than two
fresh venues is a NaN proxy, never a crash. The spool cannot raise into the
sampler (`SafeSpool`). Nothing here is read by the trading loop today.

**Every exit is on a timeout**, the pattern `record_stream.consume` was rewritten
around: the connect, and each receive, so a venue that goes silent is noticed
in `idle_timeout` seconds rather than never.

Never gated except for the spool flush, which is deferred (not awaited) while a
decision is in flight.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import time
from typing import Callable, Optional, Sequence

import aiohttp

from core import heartbeat
from core.config import DEFAULT_CONFIG
from core.index_proxy import (
    EXCHANGES, HANDLERS, URLS, ExchangeState, IndexProxyCache, proxy_row,
    subscribe_messages,
)
from core.spool import DEFAULT_SPOOL_ROOT, SafeSpool, table_spool

logger = logging.getLogger('index')

# Process-wide, like `record_stream.CACHE`: the trading loop reads the proxy
# this recorder maintains.
CACHE = IndexProxyCache()

VENUE = 'index_proxy'
DATASET = 'index_proxy_1s'


def _decode(raw) -> Optional[dict]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


async def consume(ws, exchange: str, state: ExchangeState, *, idle_timeout: float,
                  large_bytes: int = 1_000_000, now: Callable[[], float] = time.time,
                  seqs=None) -> str:
    """Fold messages until the connection needs rebuilding; returns why.

    A message over `large_bytes` (a Coinbase level2 snapshot runs to megabytes)
    is DECODED in a thread so the parse cannot stall the trading loop's event
    loop. It is still APPLIED here: folding it in a thread would let the
    sampler read a book mid-rebuild.
    """
    handler = HANDLERS[exchange]
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=idle_timeout)
            except asyncio.TimeoutError:
                return f'silent for {idle_timeout:.0f}s'
            if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING):
                return 'socket closed'
            if msg.type == aiohttp.WSMsgType.ERROR:
                return f'socket error: {ws.exception()}'
            if msg.type not in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                continue
            received = now()
            if len(msg.data) > large_bytes:
                payload = await asyncio.to_thread(_decode, msg.data)
            else:
                payload = _decode(msg.data)
            if payload is None:
                continue
            # A lost Coinbase message is a lost book delta, so the book is
            # wrong until a fresh snapshot: rebuild rather than serve it.
            if (seqs is not None and 'sequence_num' in payload
                    and seqs.observe(str(payload.get('channel') or ''),
                                     payload.get('sequence_num'))):
                return 'sequence gap'
            reason = handler(payload, state, received)
            if reason:
                return reason
    finally:
        # Nothing on a dead connection is current.
        state.disconnect()


async def run_exchange(session, exchange: str, state: ExchangeState, args) -> None:
    """One venue, forever: connect, subscribe, consume, back off, repeat."""
    backoff = args.backoff_start
    url = (getattr(args, 'urls', None) or {}).get(exchange, URLS[exchange])
    while True:
        started = time.monotonic()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(url, heartbeat=20, max_msg_size=64 * 2**20),
                timeout=args.connect_timeout)
            async with ws:
                state.disconnect()
                for message in subscribe_messages(exchange, args.symbols):
                    await ws.send_json(message)
                logger.info('%s: streaming %s', exchange, ', '.join(args.symbols))
                seqs = None
                if exchange == 'coinbase':
                    from scripts.record_spot_stream import SequenceTracker
                    seqs = SequenceTracker()
                reason = await consume(ws, exchange, state,
                                       idle_timeout=args.idle_timeout,
                                       large_bytes=args.large_message_bytes,
                                       seqs=seqs)
        except asyncio.CancelledError:
            state.disconnect()
            raise
        except Exception as exc:                          # noqa: BLE001
            state.disconnect()
            reason = f'{type(exc).__name__}: {str(exc)[:160]}'
        if time.monotonic() - started >= 60.0:
            backoff = args.backoff_start
        logger.warning('%s: stream ended (%s); reconnecting in %.0fs',
                       exchange, reason, backoff)
        await asyncio.sleep(backoff)
        backoff = min(backoff * 2.0, args.backoff_max)


def sample(states: Sequence[ExchangeState], symbols: Sequence[str], now: float, *,
           max_staleness: float, min_exchanges: int) -> list[dict]:
    """One row per symbol, stamped at the second `now` falls in.

    `t` is `floor(now)` and the row describes the books at `now`, so everything
    in a row stamped `t` was known before `t + 1`. A decision at `d` that reads
    rows stamped strictly before `d` therefore never sees information from `d`
    onward.
    """
    stamp = float(math.floor(now))
    rows = []
    for symbol in symbols:
        row = proxy_row(states, symbol, now, max_staleness=max_staleness,
                        min_exchanges=min_exchanges)
        row.update({'t': stamp, 'venue': VENUE, 'computed': now})
        rows.append(row)
    return rows


def spool_record(row: dict) -> dict:
    """JSON-safe: NaN becomes null."""
    out = {}
    for key, value in row.items():
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        out[key] = value
    return out


async def sample_loop(states, cache: IndexProxyCache, spool, args, *, gate=None,
                      clock: Callable[[], float] = time.time,
                      sleep=asyncio.sleep, flush_every: float = 5.0) -> None:
    last_stamp: Optional[float] = None
    next_flush = next_beat = 0.0
    while True:
        now = clock()
        # Just past the next whole second, so each row lands in its own second.
        await sleep(max(0.0, math.floor(now) + 1.0 - now) + 0.005)
        now = clock()
        rows = sample(list(states.values()), args.symbols, now,
                      max_staleness=args.max_staleness,
                      min_exchanges=args.min_exchanges)
        if rows and rows[0]['t'] == last_stamp:
            continue
        last_stamp = rows[0]['t'] if rows else last_stamp
        for row in rows:
            cache.record(row)
        spool.extend(spool_record(r) for r in rows)
        # Liveness means a proxy is being PRODUCED, not that the loop ticks:
        # with every venue down this is exactly the component not doing its job.
        if now >= next_beat and any(r['n_exchanges'] >= args.min_exchanges
                                    for r in rows):
            heartbeat.touch('index')
            next_beat = now + 5.0
        if now >= next_flush and (gate is None or gate.is_idle):
            await asyncio.to_thread(spool.flush)
            next_flush = now + flush_every


async def run(args, gate=None, cache: Optional[IndexProxyCache] = None,
              states: Optional[dict] = None) -> int:
    cache = CACHE if cache is None else cache
    symbols = tuple(args.symbols)
    states = states or {e: ExchangeState(e, symbols) for e in args.exchanges}
    spool = SafeSpool(lambda: table_spool(args.spool_root, DATASET, VENUE),
                      name='index spool')
    async with aiohttp.ClientSession(
            headers={'User-Agent': 'quarter-index-proxy/1.0'}) as session:
        tasks = [asyncio.create_task(run_exchange(session, e, states[e], args),
                                     name=f'index-{e}') for e in states]
        tasks.append(asyncio.create_task(
            sample_loop(states, cache, spool, args, gate=gate),
            name='index-sampler'))
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            spool.flush()
            spool.close()
    return 0


def _exchanges_default() -> list[str]:
    raw = os.getenv('INDEX_PROXY_EXCHANGES', '')
    chosen = [e.strip().lower() for e in raw.split(',') if e.strip()]
    return chosen or list(EXCHANGES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--exchanges', nargs='+', choices=EXCHANGES,
                        default=_exchanges_default())
    parser.add_argument('--symbols', nargs='+', default=list(DEFAULT_CONFIG.symbols))
    parser.add_argument('--spool-root', default=str(DEFAULT_SPOOL_ROOT))
    parser.add_argument('--max-staleness', type=float,
                        default=DEFAULT_CONFIG.index_proxy_max_staleness_seconds)
    parser.add_argument('--min-exchanges', type=int,
                        default=DEFAULT_CONFIG.index_proxy_min_exchanges)
    # Kraken and Coinbase heartbeat every second and Gemini every five; this
    # much silence is a dead socket on any of the four.
    parser.add_argument('--idle-timeout', type=float, default=15.0)
    parser.add_argument('--connect-timeout', type=float, default=15.0)
    parser.add_argument('--backoff-start', type=float, default=1.0)
    parser.add_argument('--backoff-max', type=float, default=60.0)
    parser.add_argument('--large-message-bytes', type=int, default=1_000_000)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
