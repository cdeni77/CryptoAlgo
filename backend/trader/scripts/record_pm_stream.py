"""Hold one Polymarket CLOB subscription and keep a live book per symbol.

Keeps the peer book contemporaneous with Kalshi's, as training assumes.

* Every exit from `consume` is a timeout. A silent socket cannot wake itself:
  silence means the market settled or the connection died, and both are
  repaired by resubscribing.
* Token ids change with each fifteen-minute window, so the tokens are
  re-checked on a deadline.
* `CACHE` has the shape `cross_venue_row` reads, the same as
  `record_pm_ladder.CACHE`: best bid and ask in cents with an `at` stamp.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import pandas as pd

from core import heartbeat
from core.pm_stream_book import PmBookCache

logger = logging.getLogger('pm-stream')

WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'
SILENCE_SECONDS = 15.0
REFRESH_SECONDS = 60.0

# symbol -> {'best_bid': cents, 'best_ask': cents, 'at': Timestamp}
CACHE: dict = {}

# symbol -> reason, set by the trading loop (`request_resubscribe`) and read by
# `consume` on every frame and every timeout.
_RESUBSCRIBE: dict = {}

# The heartbeat is a file write, so it is written at most this often.
HEARTBEAT_EVERY_SECONDS = 5.0
_LAST_BEAT = {'at': 0.0}


def request_resubscribe(symbol: str, reason: str = '') -> None:
    """Ask the recorder to drop `symbol`'s state and subscribe again.

    Safe to call when no recorder runs: nothing reads the request then.
    """
    _RESUBSCRIBE[str(symbol)] = str(reason or 'requested')


def take_resubscribe_requests() -> dict:
    """Return and clear the {symbol: reason} requests since the last call."""
    out = dict(_RESUBSCRIBE)
    _RESUBSCRIBE.clear()
    return out


def forget(symbols) -> None:
    """Drop the published book and subscribed slug for each symbol."""
    for symbol in symbols:
        CACHE.pop(symbol, None)
        _WINDOWS.pop(symbol, None)


def current_window(now=None) -> pd.Timestamp:
    """Return the open of the window `now` is inside."""
    from core.config import DEFAULT_CONFIG
    now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    return now.floor(f'{int(DEFAULT_CONFIG.window_minutes)}min')


def _beat(clock=time.monotonic) -> None:
    at = clock()
    if at - _LAST_BEAT['at'] >= HEARTBEAT_EVERY_SECONDS:
        _LAST_BEAT['at'] = at
        heartbeat.touch('pm_stream')


def publish(book: PmBookCache, tokens: dict, *, window=None, now=None) -> None:
    """Publish the live book in the shape `cross_venue_row` reads.

    Prices are in cents. A one-sided book publishes nothing. The heartbeat is
    touched only for a two-sided book of the current window.
    """
    windows = window if isinstance(window, dict) else {}
    current = None
    for token, symbol in tokens.items():
        bid, ask = book.best_bid(token), book.best_ask(token)
        if bid is None or ask is None:
            CACHE.pop(symbol, None)
            continue
        # Stamp with the book's own last change, not the socket's last frame.
        # The channel has no sequence number, so a quiet book and a stopped
        # feed look the same.
        stamp = book._stamp.get(token)  # noqa: SLF001 — same module family
        for_window = (pd.Timestamp(windows[symbol])
                      if symbol in windows and windows[symbol] is not None
                      else (pd.Timestamp(window)
                            if window is not None and not isinstance(window, dict)
                            else None))
        if for_window is not None:
            current = current_window(now) if current is None else current
            if for_window == current:
                _beat()
        CACHE[symbol] = {
            'best_bid': bid * 100.0,
            'best_ask': ask * 100.0,
            'at': stamp if stamp is not None else pd.Timestamp.now(tz='UTC'),
            # The window this book is for. An age check alone cannot catch a
            # rollover: just after a boundary the previous window's book still
            # looks fresh.
            'window': for_window,
        }


def _window_of(tokens: dict) -> dict:
    """Return {symbol: window} for the subscribed tokens, per symbol.

    Per symbol, so one failed lookup cannot disable the rollover guard for all.
    """
    live = {sym for sym in tokens.values()}
    return {sym: w for sym, (_slug, w) in subscribed_windows().items()
            if sym in live}


async def consume(stream, *, until: Optional[float], silence_seconds: float,
                  book: Optional[PmBookCache] = None,
                  tokens: Optional[dict] = None, window=None) -> str:
    """Fold frames until a deadline or a silence. Returns the reason.

    `until` is seconds from now, or None for no deadline. Both exits are
    timeouts, because a silent socket never delivers a frame to check on.
    """
    deadline = (time.monotonic() + until) if until is not None else None
    iterator = stream.events().__aiter__()
    while True:
        if _RESUBSCRIBE:
            return ('resubscribe requested: ' + ', '.join(
                f'{sym} ({why})' for sym, why in sorted(_RESUBSCRIBE.items())))
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return 'subscription refresh (window rollover)'
        else:
            remaining = silence_seconds
        try:
            event = await asyncio.wait_for(
                iterator.__anext__(), timeout=min(remaining, silence_seconds))
        except asyncio.TimeoutError:
            if _RESUBSCRIBE:
                continue
            if deadline is not None and time.monotonic() >= deadline:
                return 'subscription refresh (window rollover)'
            return (f'silence for {silence_seconds:.0f}s — the market settled '
                    f'or the connection died')
        except StopAsyncIteration:
            return 'stream ended'
        if book is not None:
            book.apply(event)
            if tokens:
                publish(book, tokens, window=window)


class _Socket:
    """The live websocket, yielding decoded events one at a time."""

    def __init__(self, session, tokens: list[str]) -> None:
        self._session, self._tokens, self._ws = session, tokens, None

    async def __aenter__(self):
        self._ws = await self._session.ws_connect(WS_URL, heartbeat=10).__aenter__()
        await self._ws.send_json({'assets_ids': self._tokens, 'type': 'market'})
        return self

    async def __aexit__(self, *exc):
        if self._ws is not None:
            await self._ws.close()

    def events(self):
        ws = self._ws

        async def _gen():
            import aiohttp
            async for msg in ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                try:
                    payload = json.loads(msg.data)
                except ValueError:
                    continue
                for event in (payload if isinstance(payload, list) else [payload]):
                    yield event
        return _gen()


async def resolve_tokens(session, now) -> dict:
    """Return {token_id: symbol} for the window `now` is inside.

    Reuses `record_pm_ladder`'s slug resolution. Token 0 is "Up"; one book
    mirrors the other.
    """
    from scripts.record_pm_ladder import ASSETS, GAMMA, slug_for, _get

    slugs = {asset: slug_for(asset, now) for asset in ASSETS}
    qs = '&'.join(f'slug={slug}' for slug in slugs.values())
    try:
        found = await _get(session, f'{GAMMA}/markets?{qs}') or []
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('gamma batch: %s', str(exc)[:90])
        return {}
    by_slug = {m.get('slug'): m for m in found}
    tokens: dict = {}
    for asset, symbol in ASSETS.items():
        market = by_slug.get(slugs[asset])
        if not market:
            continue
        try:
            ids = json.loads(market.get('clobTokenIds') or '[]')
        except ValueError:
            continue
        if ids:
            tokens[str(ids[0])] = symbol
            # Record the window too, and log it, so alignment with the Kalshi
            # ticker (keyed on the settle) can be checked.
            _WINDOWS[symbol] = slugs[asset]
    return tokens


# symbol -> the slug currently subscribed, for the log line and for tests.
_WINDOWS: dict = {}


def subscribed_windows() -> dict:
    """Return {symbol: (slug, window_open)} for the current subscription."""
    from scripts.record_pm_ladder import window_of
    return {sym: (slug, window_of(slug)) for sym, slug in _WINDOWS.items()}


async def run(args=None, gate=None) -> int:
    """Hold a subscription, refreshing it as the window rolls.

    A failure never stops trading: the loop logs, backs off and reconnects.
    `cross_venue_row` refuses a stale reading, so an outage makes the feature
    absent, not wrong.
    """
    import aiohttp
    from datetime import datetime, timezone
    from scripts.record_pm_ladder import HEADERS

    backoff = 1.0
    while True:
        try:
            async with aiohttp.ClientSession(
                    headers=HEADERS,
                    timeout=aiohttp.ClientTimeout(total=None)) as session:
                while True:
                    requested = take_resubscribe_requests()
                    if requested:
                        forget(requested)
                        logger.warning('resubscribing at the trading loop\'s '
                                       'request: %s', ', '.join(
                                           f'{sym} ({why})' for sym, why
                                           in sorted(requested.items())))
                    now = datetime.now(timezone.utc)
                    tokens = await resolve_tokens(session, now)
                    if not tokens:
                        logger.warning('no Polymarket markets resolved; retrying')
                        await asyncio.sleep(10)
                        continue
                    windows = subscribed_windows()
                    logger.info(
                        'streaming %d Polymarket book(s): %s',
                        len(tokens),
                        ', '.join(f'{sym}@{w.strftime("%H:%M")}'
                                  for sym, (_slug, w) in sorted(windows.items())))
                    # Hold the socket across token checks. The deadline only
                    # re-checks the tokens; only a change reconnects, because a
                    # reconnect empties the cache until the first `book` frame.
                    # A new connection resends every book.
                    book = PmBookCache()
                    async with _Socket(session, list(tokens)) as sock:
                        while True:
                            reason = await consume(
                                sock, until=REFRESH_SECONDS,
                                silence_seconds=SILENCE_SECONDS,
                                book=book, tokens=tokens,
                                window=_window_of(tokens))
                            if 'refresh' not in reason:
                                logger.info('reconnecting: %s', reason)
                                break
                            fresh = await resolve_tokens(
                                session, datetime.now(timezone.utc))
                            if fresh and set(fresh) != set(tokens):
                                logger.info(
                                    'window rolled to %s; resubscribing',
                                    ', '.join(
                                        f'{sym}@{w.strftime("%H:%M")}' for sym, (_s, w)
                                        in sorted(subscribed_windows().items())))
                                break
                    backoff = 1.0
        except asyncio.CancelledError:
            raise
        except Exception as exc:                              # noqa: BLE001
            logger.warning('polymarket stream: %s; reconnecting in %.0fs',
                           str(exc)[:120], backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)


def main() -> int:
    import argparse
    from scripts._common import setup_logging

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    setup_logging(args.verbose)
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
