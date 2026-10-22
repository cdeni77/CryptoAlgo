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

# The rolling window of bars per symbol, kept between cycles.
_BAR_CACHE: dict[str, pd.DataFrame] = {}

# How far back to re-ask on an incremental fetch. The newest cached bar is the
# minute still forming, so it is refetched to get its final values; the extra
# minutes cover a bar the venue revised or filled in late.
BAR_REFETCH_MINUTES = 3

# When each symbol's REST cache was last asked for (epoch seconds), and how long
# after its close a REST candle must be read to count as settled. Only a
# settled candle can anchor the trade tape, because an earlier read can miss
# the minute's last prints.
_BAR_FETCHED_AT: dict[str, float] = {}
BAR_SETTLED_SECONDS = 10.0


def _merge_bars(cached: pd.DataFrame, fresh: pd.DataFrame,
                *, floor: pd.Timestamp) -> pd.DataFrame:
    """Merge cached and fresh bars; the fresh copy wins on any overlap.

    The overlap holds minutes that were still forming at the last read, so the
    newer read is the final one.
    """
    combined = pd.concat([cached, fresh], ignore_index=True)
    combined = combined.sort_values('event_time', ignore_index=True)
    combined = combined.drop_duplicates(subset='event_time', keep='last')
    return combined[combined['event_time'] >= floor].reset_index(drop=True)


async def fetch_bars(config: Config, minutes: int = FETCH_MINUTES, *,
                     prefer_stream: bool = False) -> dict[str, pd.DataFrame]:
    """Return one-minute bars for the universe from the venue.

    Only the newest minutes are re-fetched; a symbol with no usable cache gets
    the full window. With `prefer_stream`, the Coinbase trade tape supplies
    the newest minutes and REST serves only the symbols the tape cannot verify.

    **`config.venue` decides the PRICE, never the activity.** On
    `cfbenchmarks` the prices come from the settlement index and the volume is
    carried from these spot bars, because the index has none -- verified on
    both transports, the frame carries only a value. Spot is therefore fetched
    either way; the index is an overlay on top of it, not a replacement for
    it.
    """
    out: dict[str, pd.DataFrame] = {}
    symbols = list(config.symbols)
    if prefer_stream:
        out = await _bars_from_stream(config, minutes)
        symbols = [s for s in symbols if s not in out]
        if not symbols:
            logger.debug('bars from the trade stream for every symbol')
            return await _overlay_index(config, out, minutes)
        if out:
            logger.info('bars from the trade stream for %s, REST for %s',
                        sorted(out), symbols)
    out.update(await _fetch_rest_bars(config, minutes, symbols))
    return await _overlay_index(config, out, minutes)


#: Index history read from the store, kept between cycles. The store holds
#: everything `scripts/backfill_index` and `scripts/record_index_stream` have
#: written; only the minutes since the last flush need splicing from memory.
_INDEX_CACHE: dict[str, pd.DataFrame] = {}
INDEX_VENUE = 'cfbenchmarks'


async def _overlay_index(config: Config, spot: dict[str, pd.DataFrame],
                         minutes: int) -> dict[str, pd.DataFrame]:
    """Replace spot prices with the settlement index, keeping spot's activity.

    Returns `spot` untouched unless `config.venue` asks for the index, so the
    default path is exactly what it was.

    **A symbol with no index history keeps its spot bars and says so.** Half a
    universe priced on one index and half on another is the cross-index basis
    C6 exists to prevent, so the fallback is loud rather than silent.
    """
    if getattr(config, 'venue', None) != INDEX_VENUE or not spot:
        return spot
    try:
        return await asyncio.to_thread(_index_overlay_sync, config, spot, minutes)
    except Exception as exc:                                   # noqa: BLE001
        logger.error('index overlay failed (%s); REFUSING to trade spot prices '
                     'against a model fitted on the index', str(exc)[:160])
        return {}


def _index_overlay_sync(config: Config, spot: dict[str, pd.DataFrame],
                        minutes: int) -> dict[str, pd.DataFrame]:
    from core.datastore import ResearchStore
    from core.index_bars import carry_activity, rows_for_store
    from scripts import record_index_stream

    store = ResearchStore()
    live = record_index_stream.live_bars()
    now = time.time()
    floor = pd.Timestamp(now - minutes * 60, unit='s', tz='UTC').floor('min')

    out: dict[str, pd.DataFrame] = {}
    for symbol, spot_bars in spot.items():
        history = _INDEX_CACHE.get(symbol)
        newest = (history['event_time'].max()
                  if history is not None and len(history) else None)
        # Re-read only what the cache is missing. The store grows by one
        # flush at a time, so this is a few minutes after the first cycle.
        start = floor if newest is None else newest - pd.Timedelta(minutes=2)
        fresh = store.read('minute_bars', venue=INDEX_VENUE, symbols=[symbol],
                           start=start, min_quality=None)
        if fresh is not None and len(fresh):
            fresh['event_time'] = pd.to_datetime(fresh['event_time'], utc=True)
            history = (fresh if history is None
                       else _merge_bars(history, fresh, floor=floor))
        # Then the minutes folded in this process but not yet flushed.
        if live is not None:
            after = -1 if history is None or not len(history) else int(
                history['event_time'].max().timestamp())
            spliced = rows_for_store(
                live.since(symbol, after=after, now=now),
                available_time=pd.Timestamp(now, unit='s', tz='UTC'))
            if spliced:
                frame = pd.DataFrame(spliced)
                history = (frame if history is None
                           else _merge_bars(history, frame, floor=floor))
        if history is None or not len(history):
            logger.error('%s: no %s history; keeping SPOT prices for it, which '
                         'mixes two indices across the universe', symbol,
                         INDEX_VENUE)
            out[symbol] = spot_bars
            continue
        _INDEX_CACHE[symbol] = history
        out[symbol] = carry_activity(
            history[history['event_time'] >= floor].reset_index(drop=True),
            spot_bars)
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
        # Fetch symbols concurrently. One symbol's failure costs only that
        # symbol.
        spans: dict[str, datetime] = {}
        for symbol in symbols:
            cached = _BAR_CACHE.get(symbol)
            spans[symbol] = start
            if cached is None or cached.empty:
                continue
            last = pd.Timestamp(cached['event_time'].iloc[-1])
            gap_start = (last.tz_convert(None).to_pydatetime()
                         - timedelta(minutes=BAR_REFETCH_MINUTES))
            # Fetch incrementally only if the cache covers the window start.
            # Otherwise the result has a silent hole in the middle.
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
    """Return the trade-stream recorder's tape, if this process runs one."""
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
    """Return REST bars up to a verified anchor and tape bars after it, or None.

    The anchor is the newest REST candle read at least `BAR_SETTLED_SECONDS`
    after its close. The tape is used only when it covers every minute from the
    anchor to the newest closed minute (`TradeTape.reliable`), and reproduces
    every settled REST candle in that span. Otherwise returns None and the
    caller uses REST. The frame has the REST shape.
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
    # The newest bar is final `grace` seconds after its close. Wait out the
    # remainder instead of refusing.
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
