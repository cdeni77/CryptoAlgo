"""Kalshi's `orderbook_delta`, normalized.

The handshake reuses `KalshiClient._headers` verbatim: that method takes the
already-prefixed path and signs `timestamp + METHOD + path`, while `_request` is
what adds `/trade-api/v2`. So the correct WebSocket signature is
`_headers('GET', '/trade-api/ws/v2')` with no change to the signing code.
`Content-Type: application/json` is dropped from the handshake headers — it
describes a body an upgrade request does not have.

**Every field name below was read off the live venue, not the documentation.**
The published guide implies `price` and `delta`; the wire says:

    orderbook_snapshot  msg: market_ticker, market_id, yes_dollars_fp, no_dollars_fp
    orderbook_delta     msg: market_ticker, market_id, price_dollars, delta_fp,
                             side, ts, ts_ms
    both          top-level: type, sid, seq, msg

Note the naming inversion against REST, which is a live trap: REST serves
`orderbook_fp.yes_dollars`, the stream serves `msg.yes_dollars_fp`. Reading one
shape against the other yields an empty book and no exception.

`delta_fp` is a SIGNED CHANGE ("-5.00"), not a resulting size, so delta events
carry `absolute=False` and the cache adds them to what is resting. Prices and
sizes are fixed-point strings, the same convention `KalshiClient` already parses
for REST quotes.
"""
from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Optional, Sequence

from data_collection.stream.base import BookEvent, Level

logger = logging.getLogger('kalshi-stream')

WS_URL = 'wss://api.elections.kalshi.com/trade-api/ws/v2'
WS_PATH = '/trade-api/ws/v2'
VENUE = 'kalshi'


def _levels(raw) -> list[Level]:
    out: list[Level] = []
    for entry in raw or []:
        try:
            price, size = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        if size > 0:
            out.append((price, size))
    return sorted(out)


def rest_levels(book: dict) -> tuple[list[Level], list[Level]]:
    """The same ladder from `GET /markets/{t}/orderbook`, for cross-checking.

    Note `orderbook_fp` here against `_dollars_fp` on the stream — the two
    surfaces spell the same content differently.
    """
    ladder = book.get('orderbook_fp') or book.get('orderbook') or {}
    return (_levels(ladder.get('yes_dollars') or ladder.get('yes')),
            _levels(ladder.get('no_dollars') or ladder.get('no')))


def parse_frame(payload: dict, received: float) -> Optional[BookEvent]:
    """A book message as a BookEvent, or None for anything else.

    **Never raises.** A subscribe acknowledgement, a heartbeat and an error
    frame all arrive on this socket, and a parser that crashed on one would take
    the whole stream down over a message carrying no book.
    """
    if not isinstance(payload, dict):
        return None
    kind_raw = payload.get('type')
    if kind_raw == 'orderbook_snapshot':
        kind = 'snapshot'
    elif kind_raw == 'orderbook_delta':
        kind = 'delta'
    else:
        return None

    msg = payload.get('msg') or {}
    ticker = msg.get('market_ticker')
    if not ticker:
        return None

    if kind == 'snapshot':
        yes = _levels(msg.get('yes_dollars_fp') or msg.get('yes'))
        no = _levels(msg.get('no_dollars_fp') or msg.get('no'))
        absolute = True
    else:
        try:
            price = float(msg['price_dollars'])
            change = float(msg['delta_fp'])
        except (KeyError, TypeError, ValueError):
            return None
        side = str(msg.get('side') or '').lower()
        if side not in ('yes', 'no'):
            return None
        level = [(price, change)]
        yes, no = (level, []) if side == 'yes' else ([], level)
        absolute = False

    seq = payload.get('seq')
    try:
        seq = int(seq) if seq is not None else None
    except (TypeError, ValueError):
        seq = None

    return BookEvent(venue=VENUE, market_ticker=str(ticker), kind=kind,
                     received=received, seq=seq, yes=yes, no=no,
                     absolute=absolute)


class KalshiStream:
    """Connect, subscribe, and yield normalized events."""

    def __init__(self, client, url: str = WS_URL) -> None:
        self._client = client
        self._url = url
        self._ws = None
        self._next_id = 1

    async def connect(self) -> None:
        headers = self._client._headers('GET', WS_PATH)  # noqa: SLF001
        headers.pop('Content-Type', None)
        self._ws = await self._client._session.ws_connect(  # noqa: SLF001
            self._url, headers=headers, heartbeat=10)

    async def subscribe(self, tickers: Sequence[str]) -> None:
        if not tickers:
            return
        await self._ws.send_json({
            'id': self._next_id, 'cmd': 'subscribe',
            'params': {'channels': ['orderbook_delta'],
                       'market_tickers': list(tickers)}})
        self._next_id += 1

    async def events(self) -> AsyncIterator[BookEvent]:
        async for msg in self._ws:
            try:
                payload = json.loads(msg.data)
            except (ValueError, TypeError):
                continue
            event = parse_frame(payload, time.time())
            if event is not None:
                yield event

    async def close(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
