"""Hold the Coinbase trade stream open and fold it into one-minute bars.

The trading loop's bar fetch was a ~3s REST round trip on the decision path,
read ~1s after the minute closed — early enough that the venue may not have
finished the candle. This keeps `TAPE` (a `core.trade_bars.TradeTape`) current
from the public `market_trades` channel, so `scripts.live` can take the newest
minutes from memory where the tape proves it saw all of them, and REST
everywhere else.

**Every loss of coverage is recorded, never smoothed over.** A reconnect, an
idle socket and a sequence gap all end the tape's coverage at the last instant
we know we were receiving; minutes overlapping the hole are never served.

Protocol (Coinbase Advanced Trade WebSocket, public, no auth):

    subscribe  {"type": "subscribe", "product_ids": [...], "channel": "market_trades"}
               {"type": "subscribe", "product_ids": [...], "channel": "heartbeats"}
    message    {"channel", "client_id", "timestamp", "sequence_num",
                "events": [{"type": "snapshot"|"update",
                            "trades": [{"trade_id", "product_id", "price",
                                        "size", "side", "time"}]}]}

Read from Coinbase's own SDK (`coinbase-advanced-py`). Whether `sequence_num`
counts per connection or per channel is not something the documentation
settles, so a gap is declared only when the number jumps BOTH across the
connection and within the channel — which is the correct test under either.

Never gated: it writes nothing to the store, and a reader that pauses goes stale.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Optional, Sequence

import aiohttp

from core import heartbeat
from core.config import DEFAULT_CONFIG
from core.trade_bars import TradeTape

logger = logging.getLogger('spot-stream')

WS_URL = 'wss://advanced-trade-ws.coinbase.com'

# Process-wide, so the trading loop reads the tape this recorder maintains.
TAPE = TradeTape(grace_seconds=DEFAULT_CONFIG.spot_bar_grace_seconds)


class SequenceTracker:
    """A lost message, judged on both readings of `sequence_num`."""

    def __init__(self) -> None:
        self.previous: Optional[int] = None
        self.by_channel: dict[str, int] = {}

    def observe(self, channel: str, seq) -> bool:
        """True when a message was lost before this one.

        A connection-wide counter is contiguous message to message unless one
        was lost; a per-channel counter is contiguous within its channel. A
        loss breaks whichever one is real, and the other is never contiguous
        anyway — so requiring both is right under either reading.
        """
        try:
            seq = int(seq)
        except (TypeError, ValueError):
            return False
        across = self.previous is not None and seq != self.previous + 1
        last = self.by_channel.get(channel)
        within = last is not None and seq > last + 1
        self.previous = seq
        # A lower number within the channel is an out-of-order delivery.
        if last is None or seq > last:
            self.by_channel[channel] = seq
        return across and within


def handle(payload: dict, tape: TradeTape, seqs: SequenceTracker, *,
           received: float, last_received: Optional[float]) -> Optional[str]:
    """Fold one message. Returns a reason when the connection must be rebuilt."""
    if not isinstance(payload, dict):
        return None
    if payload.get('type') == 'error':
        return f"venue error: {str(payload.get('message') or payload)[:160]}"
    channel = str(payload.get('channel') or '')
    # Every channel is observed, `subscriptions` included: under a
    # connection-wide counter an unobserved message would read as a loss.
    if 'sequence_num' in payload and seqs.observe(
            channel, payload.get('sequence_num')):
        logger.warning('sequence gap on %s; minutes since %.1fs ago are not '
                       'served from the stream', channel,
                       received - (last_received or received))
        tape.gap(since=last_received if last_received is not None else received,
                 at=received)
    if channel != 'market_trades':
        return None
    for event in payload.get('events') or []:
        snapshot = str(event.get('type') or '') == 'snapshot'
        if not tape.is_connected:
            # Coverage begins with the first trade message on this connection.
            tape.connected(at=received)
        for trade in event.get('trades') or []:
            tape.add(str(trade.get('product_id') or ''), trade.get('trade_id'),
                     trade.get('price'), trade.get('size'), trade.get('time'),
                     received=received, snapshot=snapshot)
    return None


async def consume(ws, tape: TradeTape, *, idle_timeout: float,
                  now=time.time) -> str:
    """Read until the connection needs rebuilding; every exit is on a timeout."""
    seqs = SequenceTracker()
    last: Optional[float] = None
    next_beat = 0.0
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
            if msg.type != aiohttp.WSMsgType.TEXT:
                continue
            received = now()
            try:
                payload = json.loads(msg.data)
            except (TypeError, ValueError):
                continue
            reason = handle(payload, tape, seqs, received=received,
                            last_received=last)
            last = received
            if reason:
                return reason
            if received >= next_beat:
                heartbeat.touch('spot_stream')
                next_beat = received + 5.0
    finally:
        # The last instant we KNOW we were receiving, not the moment we noticed.
        tape.disconnected(at=last if last is not None else now())


async def run(args, gate=None, tape: Optional[TradeTape] = None) -> int:
    tape = TAPE if tape is None else tape
    products = list(args.products)
    backoff = args.backoff_start
    async with aiohttp.ClientSession(
            headers={'User-Agent': 'quarter-spot-stream/1.0'}) as session:
        while True:
            started = time.monotonic()
            try:
                async with session.ws_connect(
                        args.url, heartbeat=20, max_msg_size=16 * 2**20) as ws:
                    for channel in ('heartbeats', 'market_trades'):
                        await ws.send_json({'type': 'subscribe',
                                            'product_ids': products,
                                            'channel': channel})
                    logger.info('streaming trades for %s', ', '.join(products))
                    reason = await consume(ws, tape,
                                           idle_timeout=args.idle_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as exc:                          # noqa: BLE001
                tape.disconnected()
                reason = f'{type(exc).__name__}: {str(exc)[:160]}'
            if time.monotonic() - started >= 60.0:
                backoff = args.backoff_start
            logger.warning('trade stream ended (%s); reconnecting in %.0fs',
                           reason, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2.0, args.backoff_max)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--url', default=WS_URL)
    # The products the live loop fetches REST candles for.
    parser.add_argument('--products', nargs='+',
                        default=list(DEFAULT_CONFIG.symbols))
    # Heartbeats arrive every second, so this much silence is a dead socket.
    parser.add_argument('--idle-timeout', type=float, default=10.0)
    parser.add_argument('--backoff-start', type=float, default=1.0)
    parser.add_argument('--backoff-max', type=float, default=30.0)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
