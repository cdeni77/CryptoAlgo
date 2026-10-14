"""One-minute bars for the live loop: REST fetch, rolling cache, stream splice."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config
from data_collection.coinbase_connector import CoinbaseRESTClient

logger = logging.getLogger('live')

# History fetched per cycle. The longest feature lookback is 1,440 minutes and
# the seasonal factor is a fitted lookup, so a day plus a margin suffices.
FETCH_MINUTES = 1_500

# The rolling window, kept between cycles. See `fetch_bars`.
_BAR_CACHE: dict[str, pd.DataFrame] = {}

# How far back to re-ask on an incremental fetch. The newest cached bar is the
# minute still forming, so it is refetched to get its final values; the extra
# minutes cover a bar the venue revised or filled in late.
BAR_REFETCH_MINUTES = 3

# When each symbol's REST cache was last asked for (epoch seconds), and how long
# after its close a REST candle must have been read to count as settled. Only a
# settled candle can anchor the trade tape: one read a second after its minute
# closed may still be missing that minute's last prints.
_BAR_FETCHED_AT: dict[str, float] = {}
BAR_SETTLED_SECONDS = 10.0


def _merge_bars(cached: pd.DataFrame, fresh: pd.DataFrame,
                *, floor: pd.Timestamp) -> pd.DataFrame:
    """Cached bars plus fresh ones, the fresh copy winning on any overlap.

    `keep='last'` matters: the overlap is deliberately the minutes that were
    still forming when they were last read, and the newer read is the finished
    one. Keeping the cached copy would freeze a partial candle into the window
    forever, and every `log_rv_*` feature is built off these closes.
    """
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.sort_values('event_time', ignore_index=True)
    combined = combined.drop_duplicates(subset='event_time', keep='last')
    return combined[combined['event_time'] >= floor].reset_index(drop=True)


async def fetch_bars(config: Config, minutes: int = FETCH_MINUTES, *,
                     prefer_stream: bool = False) -> dict[str, pd.DataFrame]:
    """One-minute bars for the universe, straight from the venue.

    **Only the tail is fetched.** The window is kept between cycles and only the
    newest few minutes are re-asked; a full 1,500-minute window is ~6
    sequential 300-candle pages per symbol (measured 3.2s). A symbol with no
    usable cache — first cycle, or a gap wider than the window — falls back to
    the full fetch, so this never loses history.

    `prefer_stream` (decision cycles) first tries the Coinbase trade tape for
    the newest minutes — see `stream_bars` — and asks REST only for symbols
    the tape cannot vouch for. That takes the REST round trip off the decision
    path, and with it the risk of reading a candle Coinbase has not finished.
    """
    out: dict[str, pd.DataFrame] = {}
    symbols = list(config.symbols)
    if prefer_stream:
        out = await _bars_from_stream(config, minutes)
        symbols = [s for s in symbols if s not in out]
        if not symbols:
            logger.debug('bars from the trade stream for every symbol')
            return out
        if out:
            logger.info('bars from the trade stream for %s, REST for %s',
                        sorted(out), symbols)
    out.update(await _fetch_rest_bars(config, minutes, symbols))
    return out


async def _fetch_rest_bars(config: Config, minutes: int,
                           symbols: list[str]) -> dict[str, pd.DataFrame]:
    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(minutes=minutes)
    floor = pd.Timestamp(start, tz='UTC')
    asked_at = time.time()
    client = CoinbaseRESTClient(
        api_key=os.getenv('COINBASE_API_KEY'),
        api_secret=os.getenv('COINBASE_API_SECRET'),
    )
    out: dict[str, pd.DataFrame] = {}
    try:
        # Symbols concurrently; `return_exceptions` so one symbol's failure
        # costs only that symbol — the loop already handles a missing one.
        spans: dict[str, datetime] = {}
        for symbol in symbols:
            cached = _BAR_CACHE.get(symbol)
            spans[symbol] = start
            if cached is None or cached.empty:
                continue
            last = pd.Timestamp(cached['event_time'].iloc[-1])
            gap_start = (last.tz_convert(None).to_pydatetime()
                         - timedelta(minutes=BAR_REFETCH_MINUTES))
            # Only incremental if the cache still reaches back far enough to
            # cover the window on its own; otherwise the tail would leave a hole
            # in the middle, which no feature would notice and every `rv` would
            # be computed across.
            if gap_start > start and pd.Timestamp(cached['event_time'].iloc[0]) <= floor:
                spans[symbol] = gap_start

        results = await asyncio.gather(
            *(client.get_candles_range(symbol, '1m', spans[symbol], end)
              for symbol in symbols),
            return_exceptions=True)
        for symbol, bars in zip(symbols, results):
            if isinstance(bars, BaseException):
                logger.error('%s: the venue refused one-minute bars (%s)',
                             symbol, str(bars)[:120])
                continue
            if not bars:
                logger.error('%s: the venue returned no one-minute bars', symbol)
                continue
            frame = pd.DataFrame([{
                'event_time': pd.Timestamp(b.event_time, tz='UTC'),
                'open': b.open, 'high': b.high, 'low': b.low, 'close': b.close,
                'volume': b.volume, 'quote_volume': getattr(b, 'quote_volume', np.nan),
                'trade_count': getattr(b, 'trade_count', np.nan),
            } for b in bars]).sort_values('event_time', ignore_index=True)
            cached = _BAR_CACHE.get(symbol)
            if cached is not None and spans[symbol] > start:
                frame = _merge_bars(cached, frame, floor=floor)
            else:
                frame = frame[frame['event_time'] >= floor].reset_index(drop=True)
            _BAR_CACHE[symbol] = frame
            _BAR_FETCHED_AT[symbol] = asked_at
            out[symbol] = frame
            logger.debug('%s: %d bars to %s (%s)', symbol, len(frame),
                         frame['event_time'].iloc[-1],
                         'incremental' if spans[symbol] > start else 'full')
    finally:
        close = getattr(client, 'close', None)
        if close is not None:
            result = close()
            if asyncio.iscoroutine(result):
                await result
    return out


def _spot_tape():
    """The trade-stream recorder's tape, when this process is running one."""
    try:
        from scripts.record_spot_stream import TAPE
        return TAPE
    except Exception:                                         # noqa: BLE001
        return None


def _epoch_minutes(frame: pd.DataFrame) -> np.ndarray:
    stamps = pd.to_datetime(frame['event_time'], utc=True)
    return ((stamps - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(seconds=1)
            ).to_numpy(dtype='int64')


def _same_bar(rest_row, bar: dict) -> bool:
    for key in ('open', 'high', 'low', 'close'):
        a, b = float(rest_row[key]), float(bar[key])
        if abs(a - b) > 1e-9 * max(1.0, abs(a)):
            return False
    a, b = float(rest_row['volume']), float(bar['volume'])
    return abs(a - b) <= 1e-6 * max(1.0, abs(a))


def stream_bars(rest: Optional[pd.DataFrame], tape, symbol: str, *,
                fetched_at: Optional[float], now: float,
                floor: pd.Timestamp) -> Optional[pd.DataFrame]:
    """REST bars up to a verified anchor and the trade tape after it, or None.

    The anchor is the newest REST candle read at least `BAR_SETTLED_SECONDS`
    after it closed. The tape is used only when it

      * covered every minute from the anchor through the newest closed minute,
        continuously and untainted (`TradeTape.reliable`), and
      * reproduces every settled REST candle inside that coverage — same OHLC,
        same volume, and no bar where REST has none.

    Anything less returns None and the caller uses REST exactly as before. The
    frame has REST's shape and its newest row is still the minute before the
    decision, so no feature definition moves.
    """
    if tape is None or rest is None or rest.empty or fetched_at is None:
        return None
    from core.trade_bars import MINUTE, minute_of

    last = minute_of(now) - MINUTE
    if not tape.final(last, now=now):
        return None
    epochs = _epoch_minutes(rest)
    settled = epochs[epochs + MINUTE + BAR_SETTLED_SECONDS <= fetched_at]
    if not len(settled):
        return None
    anchor = int(settled.max())
    if not tape.reliable(symbol, anchor, now=now):
        return None
    by_minute = {int(m): i for i, m in enumerate(epochs)}
    first = anchor
    while first - MINUTE >= int(epochs.min()) and tape.reliable(
            symbol, first - MINUTE, now=now):
        first -= MINUTE
    for minute in range(first, anchor + MINUTE, MINUTE):
        bar = tape.bar(symbol, minute)
        index = by_minute.get(minute)
        if index is None and bar is None:
            continue
        if index is None or bar is None or not _same_bar(rest.iloc[index], bar):
            logger.warning(
                '%s: the trade stream disagrees with REST at %s (REST %s, '
                'stream %s); using REST bars this cycle', symbol,
                pd.Timestamp(minute, unit='s', tz='UTC'),
                None if index is None else rest.iloc[index][
                    ['open', 'high', 'low', 'close', 'volume']].to_dict(), bar)
            return None
    tail = tape.span(symbol, anchor + MINUTE, last, now=now) if last > anchor else []
    if tail is None:
        return None
    head = rest[epochs <= anchor]
    rows = pd.DataFrame([{
        'event_time': pd.Timestamp(b['minute'], unit='s', tz='UTC'),
        'open': b['open'], 'high': b['high'], 'low': b['low'],
        'close': b['close'], 'volume': b['volume'],
        # REST carries neither and nothing reads them.
        'quote_volume': None, 'trade_count': None,
    } for b in tail])
    frame = pd.concat([head, rows], ignore_index=True) if len(rows) else head.copy()
    frame = frame[frame['event_time'] >= floor].reset_index(drop=True)
    logger.debug('%s: REST to %s, stream for %d minute(s) after, verified on %d',
                 symbol, pd.Timestamp(anchor, unit='s', tz='UTC'), len(tail),
                 (anchor - first) // MINUTE + 1)
    return frame


async def _bars_from_stream(config: Config, minutes: int) -> dict[str, pd.DataFrame]:
    tape = _spot_tape()
    if tape is None or not getattr(tape, 'is_connected', False):
        return {}
    from core.trade_bars import MINUTE, minute_of

    tape.grace_seconds = float(config.spot_bar_grace_seconds)
    now = time.time()
    # The decision wakes ~1s past the minute and the newest bar it needs is
    # final `grace` after its close: wait out any remainder rather than refuse.
    wait = minute_of(now) + config.spot_bar_grace_seconds - now
    if 0 < wait <= config.spot_bar_grace_seconds:
        await asyncio.sleep(wait)
        now = time.time()
    floor = pd.Timestamp(now - minutes * MINUTE, unit='s', tz='UTC')
    out: dict[str, pd.DataFrame] = {}
    for symbol in config.symbols:
        try:
            frame = stream_bars(_BAR_CACHE.get(symbol), tape, symbol,
                                fetched_at=_BAR_FETCHED_AT.get(symbol),
                                now=now, floor=floor)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('%s: stream bars unavailable (%s)', symbol, str(exc)[:120])
            frame = None
        if frame is not None and len(frame):
            out[symbol] = frame
    return out
