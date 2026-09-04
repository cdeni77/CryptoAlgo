"""Hold one Polymarket CLOB subscription and keep a live book per symbol.

The counterpart to `scripts/record_stream.py`, written to the same discipline
because the Kalshi port paid for these lessons:

**A silent socket cannot wake itself.** `record_stream.consume` originally
checked its refresh deadline inside the loop body, so when the subscribed
markets settled at a window boundary the venue stopped sending and the loop
waited forever for a frame that was never coming. Nothing raised, `supervise`
saw a coroutine legitimately awaiting, and the recorder sat dead behind a
container reporting healthy. Here every exit is enforced on a TIMEOUT: at ~300
frames a second, fifteen seconds of silence means the market settled or the
connection died, and both are repaired by resubscribing.

**The window rolls every fifteen minutes** and the token ids change with it, so
the subscription is refreshed on a deadline rather than held.

**What this publishes is what `cross_venue_row` reads** — the same shape as
`record_pm_ladder.CACHE`: best bid and ask in CENTS with an `at` stamp, so the
staleness guard keeps working and the two sources are swappable.

Why it exists at all: `cross_venue` is the only load-bearing group in the model
(leave-one-out takes skill +0.00282 -> -0.00015), and the REST recorder wrote at
a median 32s past the minute against decisions at ~+2s. Measured over 516,680
rows, a gap computed against a peer one minute stale correlates only 0.672 with
the contemporaneous one, and 59.2% of its variance is the peer's own drift. The
socket makes the two books contemporaneous, which is what training assumes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

import pandas as pd

from core.pm_stream_book import PmBookCache

logger = logging.getLogger('pm-stream')

WS_URL = 'wss://ws-subscriptions-clob.polymarket.com/ws/market'
SILENCE_SECONDS = 15.0
REFRESH_SECONDS = 60.0

# symbol -> {'best_bid': cents, 'best_ask': cents, 'at': Timestamp}
CACHE: dict = {}


def publish(book: PmBookCache, tokens: dict, *, window=None) -> None:
    """Fold the live book into the shape `cross_venue_row` reads.

    CENTS, matching `core.book_features`. A one-sided book publishes nothing: a
    lone bid says the probability is at LEAST something, which is not a
    probability — the same rule `_two_sided_mid` applies downstream.
    """
    for token, symbol in tokens.items():
        bid, ask = book.best_bid(token), book.best_ask(token)
        if bid is None or ask is None:
            CACHE.pop(symbol, None)
            continue
        stamp = book._stamp.get(token)  # noqa: SLF001 — same module family
        CACHE[symbol] = {
            'best_bid': bid * 100.0,
            'best_ask': ask * 100.0,
            'at': stamp if stamp is not None else pd.Timestamp.now(tz='UTC'),
            # **Which window this book is FOR.** Staleness alone cannot catch a
            # rollover: observed live, Kalshi rebuilt its subscription at
            # 22:45:16 and Polymarket at 22:45:46, because each venue's silence
            # begins when ITS markets settle. For those thirty seconds the cache
            # held the 22:30 book, stamped recently enough to pass the 30s
            # guard — fresh-looking data about the wrong fifteen minutes.
            'window': pd.Timestamp(window) if window is not None else None,
        }


def _window_of(tokens: dict):
    """The window the subscribed tokens belong to, or None."""
    windows = {w for _slug, w in subscribed_windows().values()}
    return next(iter(windows)) if len(windows) == 1 else None


async def consume(stream, *, until: Optional[float], silence_seconds: float,
                  book: Optional[PmBookCache] = None,
                  tokens: Optional[dict] = None, window=None) -> str:
    """Fold frames until a deadline or a silence. Returns the reason.

    `until` is seconds from now, or None for no refresh deadline. Both exits are
    timeouts — see the module docstring for why frame-driven exits deadlock.
    """
    deadline = (time.monotonic() + until) if until is not None else None
    iterator = stream.events().__aiter__()
    while True:
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
    """{token_id: symbol} for the window `now` is inside.

    Reuses `record_pm_ladder`'s resolution rather than restating it: the slug's
    trailing stamp is the window OPEN and reading it as a close shifts every
    window fifteen minutes, which this repo has already been bitten by once.
    Token 0 is "Up"; one book is the mirror of the other.
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
            # The WINDOW, not just the symbol. Kalshi logs its exact ticker
            # (KXBTC15M-26SEP032245-45, keyed on the SETTLE) while this is keyed
            # on the slug's unix OPEN, so the two align only if both name the
            # same fifteen minutes — and a misalignment here returns a healthy
            # book for the wrong window, which is "a wrong answer that looks
            # entirely right". Logged so it can be checked rather than assumed.
            _WINDOWS[symbol] = slugs[asset]
    return tokens


# symbol -> the slug currently subscribed, for the log line and for tests.
_WINDOWS: dict = {}


def subscribed_windows() -> dict:
    """{symbol: (slug, window_open)} for what is on the wire right now."""
    from scripts.record_pm_ladder import window_of
    return {sym: (slug, window_of(slug)) for sym, slug in _WINDOWS.items()}


async def run(args=None, gate=None) -> int:
    """Hold a subscription, refreshing it as the window rolls.

    A failure here must never stop trading: the loop logs, backs off and
    reconnects, and `cross_venue_row` already refuses a stale reading, so an
    outage degrades the feature to absent rather than to wrong.
    """
    import aiohttp
    from datetime import datetime, timezone
    from scripts.record_pm_ladder import HEADERS

    book = PmBookCache()
    backoff = 1.0
    while True:
        try:
            async with aiohttp.ClientSession(
                    headers=HEADERS,
                    timeout=aiohttp.ClientTimeout(total=None)) as session:
                while True:
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
                    # **Hold the socket across window CHECKS, not just across
                    # frames.** The first version reconnected every
                    # REFRESH_SECONDS whether or not anything had changed, so
                    # the book was dropped and re-snapshotted once a minute
                    # while the window only rolls every fifteen. Between the
                    # reconnect and the first `book` frame the cache is empty,
                    # which is the staleness this recorder exists to remove —
                    # self-inflicted, once a minute, forever.
                    #
                    # So the deadline is a moment to RE-CHECK the tokens, and
                    # only a change tears the socket down.
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
