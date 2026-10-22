"""The Kalshi book for the live loop: market resolution, touch and ladder.

Resolves each window's market by asking the venue. Quotes come from the
in-process stream book, else from one REST `/orderbook` read. Also holds the
ladder and depth helpers and the top-of-book archive.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG, SERIES_BY_SYMBOL
from data_collection.kalshi_client import (
    KalshiClient, Quote, market_meta, quote_from_levels,
)
from livetrade import state

logger = logging.getLogger('live')

# symbol -> (settle_time, market metadata), resolved once per window. The
# metadata carries no touch, so a price can only come from a book.
_MARKETS: dict[str, tuple[pd.Timestamp, Quote]] = {}

_UNSET = object()


async def _resolve_one(kalshi, symbol: str, settle_time: pd.Timestamp, *,
                       refresh: bool) -> Optional[Quote]:
    series = SERIES_BY_SYMBOL.get(symbol)
    if not series:
        logger.warning('%s has no Kalshi series configured, no quote', symbol)
        return None
    cached = _MARKETS.get(symbol)
    known = cached is not None and cached[0] == settle_time
    if known and not refresh:
        return cached[1]
    try:
        if known:
            raw = await kalshi.market(cached[1].ticker)
        else:
            raw = await kalshi.resolve_window_market(
                series, pd.Timestamp(settle_time).to_pydatetime())
    except Exception as exc:                                  # noqa: BLE001
        if known:
            logger.warning('%s: market refresh failed (%s); keeping what is known',
                           symbol, str(exc)[:120])
            return cached[1]
        raise
    if not raw:
        return cached[1] if known else None
    meta = market_meta(raw)
    if not meta.ticker:
        return cached[1] if known else None
    _MARKETS[symbol] = (settle_time, meta)
    return meta


def cached_floor_strikes(symbols, settle_time: pd.Timestamp) -> dict:
    """Each symbol's venue `floor_strike` for this window, from the cache.

    **No network call.** `resolve_markets(refresh=True)` already refreshes this
    on non-decision cycles, precisely "so a decision never pays to resolve it",
    so the decision path reads what is already there. Fetching it here instead
    would put a round trip between the book read and `decided_at`, which is
    charged to `quote_age_seconds` and judged against a 5-second bound.

    A symbol the venue has not published a strike for is absent, never zero.
    """
    out = {}
    want = pd.Timestamp(settle_time)
    for symbol in symbols:
        known = _MARKETS.get(symbol)
        if not known:
            continue
        cached_settle, meta = known
        # The strike belongs to ONE window; a stale entry from the previous
        # window would silently price this one against the wrong barrier.
        if pd.Timestamp(cached_settle) != want:
            continue
        value = getattr(meta, 'floor_strike', None)
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if value > 0:
            out[symbol] = value
    return out


async def resolve_markets(kalshi, symbols, settle_time: pd.Timestamp, *,
                          refresh: bool = False) -> dict[str, Quote]:
    """Return each symbol's market for this window, cached or resolved.

    `refresh` re-reads `/markets/{ticker}` for status, shard and
    `floor_strike`, which the venue publishes after the window opens.
    Only non-decision cycles refresh.
    """
    symbols = list(symbols)
    results = await asyncio.gather(
        *(_resolve_one(kalshi, s, settle_time, refresh=refresh) for s in symbols),
        return_exceptions=True)
    out: dict[str, Quote] = {}
    for symbol, result in zip(symbols, results):
        if isinstance(result, BaseException):
            logger.error('%s: could not resolve the market (%s)', symbol,
                         str(result)[:160])
        elif result is not None:
            out[symbol] = result
    return out


def stream_quote(cache, meta: Quote, max_age_seconds: float) -> Optional[Quote]:
    """Price `meta` from the in-process stream book, or return None."""
    if cache is None or meta is None or not meta.ticker:
        return None
    try:
        ladder = cache.current(meta.ticker, max_age_seconds)
    except Exception:                                         # noqa: BLE001
        return None
    if ladder is None:
        return None
    # The book is known correct as of the connection's last frame.
    return quote_from_levels(meta, ladder.yes, ladder.no, source='stream',
                             at=time.time() - max(0.0, float(ladder.age_seconds)))


async def fetch_quotes(
    kalshi: Optional[KalshiClient],
    symbols: list[str],
    settle_time: pd.Timestamp,
    *,
    cache=_UNSET,
    max_age_seconds: Optional[float] = None,
) -> dict[str, tuple[Quote, str]]:
    """Return each symbol's touch for this window.

    Uses the stream book while it is present, not gapped, and younger than
    `stream_quote_max_age_seconds`. Otherwise reads `/orderbook` once per
    symbol. Never uses the `/markets` touch, because it is cached and can be
    cents wrong; a failed book read gives no quote.
    """
    if kalshi is None:
        return {}
    if max_age_seconds is None:
        max_age_seconds = float(
            (state._RUNNING_CONFIG or DEFAULT_CONFIG).stream_quote_max_age_seconds)
    metas = await resolve_markets(kalshi, symbols, settle_time)
    stream = _stream_cache() if cache is _UNSET else cache
    found: dict[str, Quote] = {}
    rest: list[str] = []
    for symbol in symbols:
        meta = metas.get(symbol)
        if meta is None:
            continue
        quote = stream_quote(stream, meta, max_age_seconds)
        if quote is None:
            rest.append(symbol)
        else:
            found[symbol] = quote
    if rest:
        results = await asyncio.gather(
            *(kalshi.book_quote(metas[s]) for s in rest), return_exceptions=True)
        for symbol, result in zip(rest, results):
            if isinstance(result, BaseException):
                logger.error('%s: could not read the book (%s); no quote, so no '
                             'decision for it this cycle', symbol, str(result)[:160])
                continue
            found[symbol] = result
    quotes: dict[str, tuple[Quote, str]] = {}
    for symbol, quote in found.items():
        if not quote.tradeable():
            logger.info('%s %s is not tradeable (%s), abstaining',
                        symbol, quote.ticker, quote.status)
            continue
        quotes[symbol] = (quote, quote.ticker)
        logger.debug('%s %s: %.4f / %.4f from %s', symbol, quote.ticker,
                     quote.yes_bid, quote.yes_ask, quote.book_source)
    return quotes


def _stream_cache():
    """Return the stream task's cache, if this process runs one."""
    try:
        from scripts.record_stream import CACHE
        return CACHE
    except Exception:                                         # noqa: BLE001
        return None


async def fresh_quote(kalshi, quote, config: Config):
    """Re-read one market's touch: the stream book first, else one REST read."""
    fresh = stream_quote(_stream_cache(), quote, config.stream_quote_max_age_seconds)
    if fresh is not None:
        return fresh
    if kalshi is None or not hasattr(kalshi, 'book_quote'):
        return None
    try:
        return await kalshi.book_quote(quote)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('%s: could not re-read the book (%s)',
                       getattr(quote, 'ticker', '?'), str(exc)[:120])
        return None


def _ladder_touch(levels, *, invert: bool, fallback: float) -> float:
    """Return the best price the ladder shows, in the quote's denomination.

    Depth buckets must use the touch of the same snapshot as the levels.
    `fallback` is used only when the ladder is empty.
    """
    best = None
    for entry in levels or []:
        try:
            price = float(entry[0])
        except (TypeError, ValueError, IndexError):
            continue
        effective = (1.0 - price) if invert else price
        # YES bids: higher is better. YES asks (from NO bids): lower is better.
        if best is None or (effective < best if invert else effective > best):
            best = effective
    return float(fallback) if best is None else best


def _cumulative(levels, *, best: float, within: float, invert: bool) -> float:
    """Return resting size at prices at least as good as `best +/- within`."""
    total = 0.0
    for entry in levels or []:
        try:
            price, size = float(entry[0]), float(entry[1])
        except (TypeError, ValueError, IndexError):
            continue
        effective = (1.0 - price) if invert else price
        if (effective <= best + within) if invert else (effective >= best - within):
            total += size
    return total


def _resting_total(levels) -> float:
    """Return all resting contracts on one side, or NaN with no levels."""
    total, seen = 0.0, False
    for entry in levels or []:
        try:
            total += float(entry[1])
            seen = True
        except (TypeError, ValueError, IndexError):
            continue
    return total if seen else float('nan')


def ladder_from_cache(cache, ticker: str):
    """Return (yes_levels, no_levels) from the in-process stream book, or empty.

    Returns empty when there is no cache, or the book is stale or gapped.
    `seq` is global per subscription, so a gap condemns every book on the
    connection.
    """
    if cache is None or not ticker:
        return [], []
    try:
        if cache.gapped(ticker):
            return [], []
        ladder = cache.ladder(ticker)
    except Exception:                                         # noqa: BLE001
        return [], []
    if ladder is None or getattr(ladder, 'stale', True):
        return [], []
    return list(ladder.yes or []), list(ladder.no or [])


def quote_ladder(quote) -> tuple[list, list]:
    """Return (yes_levels, no_levels) carried on a quote, or empty."""
    return (list(getattr(quote, 'yes_levels', ()) or ()),
            list(getattr(quote, 'no_levels', ()) or ()))


def depth_dollars(quote, yes_levels, no_levels):
    """Return (depth_up, depth_down) in dollars, or (None, None) with no ladder.

    Kalshi's book is two bid stacks. The YES ask size comes from the NO stack,
    and the NO ask size from the YES stack. None means unmeasured, so
    `decide()` uses the standing cap; zero would refuse every trade.
    """
    def _touch(levels):
        best = None
        for entry in levels or []:
            try:
                price, size = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                continue
            if size > 0 and (best is None or price > best[0]):
                best = (price, size)
        return best

    if not yes_levels and not no_levels:
        return None, None
    ask = float(getattr(quote, 'yes_ask', np.nan))
    bid = float(getattr(quote, 'yes_bid', np.nan))
    no_touch, yes_touch = _touch(no_levels), _touch(yes_levels)
    up = (no_touch[1] * ask) if (no_touch and np.isfinite(ask)) else None
    down = (yes_touch[1] * (1.0 - bid)) if (yes_touch and np.isfinite(bid)) else None
    return up, down


def quote_age_seconds(quote, decided_at) -> float:
    """Seconds from the book's as-of time to the decision, or NaN if unknown."""
    at = getattr(quote, 'book_time', None)
    if at is None or decided_at is None:
        return float('nan')
    age = (pd.Timestamp(decided_at) - pd.Timestamp(float(at), unit='s', tz='UTC')
           ).total_seconds()
    return float(max(0.0, age))


async def _record_touch(scored: pd.DataFrame, quotes: dict, window_open, offset: int,
                        config: Config, kalshi, *, decided_at=None) -> None:
    """Write the decision's book, peer book and implied-vol fit to the store.

    Best effort: a failure here never stops a trading cycle. No historical
    endpoint serves depth, so a book not written now is lost.
    """
    try:
        from core.datastore import ResearchStore

        rows = []
        for symbol in scored['symbol']:
            quote = quotes.get(symbol, (None, None))[0]
            if quote is None or quote.yes_bid is None or quote.yes_ask is None:
                continue
            ticker = quotes[symbol][1]
            # Prefer the ladder the decision priced from, then the stream
            # cache. A REST fetch runs after our own fill changed the book, so
            # its row is flagged `rest_after_order`.
            yes_levels, no_levels = quote_ladder(quote)
            ladder_source = getattr(quote, 'book_source', None) or 'stream'
            if not yes_levels and not no_levels:
                yes_levels, no_levels = ladder_from_cache(_stream_cache(), ticker)
                ladder_source = 'stream'
            if not yes_levels and not no_levels and kalshi is not None and ticker:
                ladder_source = 'rest_after_order'
                try:
                    book = await kalshi._request(  # noqa: SLF001
                        'GET', f'/markets/{ticker}/orderbook')
                    ladder = book.get('orderbook_fp') or book.get('orderbook') or {}
                    yes_levels = ladder.get('yes_dollars') or ladder.get('yes') or []
                    no_levels = ladder.get('no_dollars') or ladder.get('no') or []
                except Exception:                  # noqa: BLE001 - top of book still lands
                    pass
            event_time = pd.Timestamp(window_open) + pd.Timedelta(minutes=offset)
            rows.append({
                # Age of the priced book at the decision. NaN when unknown;
                # never invent a value, because the backtest reads NULL as fresh.
                'quote_age_seconds': quote_age_seconds(quote, decided_at),
                'venue': 'kalshi', 'symbol': symbol, 'event_time': event_time,
                # **The DECISION instant, not the write instant.** This row
                # is written after the orders, so `now()` here would charge the
                # book for order-placement time. `decision_age` bounds a
                # `live_touch` row by `available_time - event_time` when the row
                # carries no `quote_age_seconds` of its own, and that bound must
                # not include latency the book never had. The book was read at
                # or before `decided_at`, so it is the conservative instant.
                'available_time': pd.Timestamp(decided_at) if decided_at is not None
                else pd.Timestamp.now(tz='UTC'),
                'quality': 'valid',
                # Which sampler produced the ladder half of this row.
                'transport': ladder_source,
                'market_ticker': quotes[symbol][1], 'window_open': window_open,
                'offset_minutes': offset,
                'yes_bid': float(quote.yes_bid), 'yes_ask': float(quote.yes_ask),
                'yes_bid_size': float(quote.yes_bid_size or 0.0),
                'yes_ask_size': float(quote.yes_ask_size or 0.0),
                # Measure depth from the ladder's own touch, as in
                # `book_feature_row`.
                'depth_bid_1c': _cumulative(
                    yes_levels, within=0.01, invert=False,
                    best=_ladder_touch(yes_levels, invert=False,
                                       fallback=float(quote.yes_bid))),
                'depth_bid_5c': _cumulative(
                    yes_levels, within=0.05, invert=False,
                    best=_ladder_touch(yes_levels, invert=False,
                                       fallback=float(quote.yes_bid))),
                'depth_ask_1c': _cumulative(
                    no_levels, within=0.01, invert=True,
                    best=_ladder_touch(no_levels, invert=True,
                                       fallback=float(quote.yes_ask))),
                'depth_ask_5c': _cumulative(
                    no_levels, within=0.05, invert=True,
                    best=_ladder_touch(no_levels, invert=True,
                                       fallback=float(quote.yes_ask))),
                # The whole resting ladder, read by `depth_ratio` and
                # `book_convexity`.
                'depth_bid_total': _resting_total(yes_levels),
                'depth_ask_total': _resting_total(no_levels),
                'levels_bid': float(len(yes_levels)),
                'levels_ask': float(len(no_levels)),
                'seq': float('nan'), 'gaps': 0.0,
                # The observer. `source` is part of the `venue_depth` event key,
                # so each observer's row for one minute survives a read. 'live'
                # is the ladder recorder's REST poll.
                'source': 'live_touch',
            })
        # Record the peer book and implied-vol fit this decision read, so the
        # backtest can join the inputs live used.
        from livetrade.features import decision_input_rows, mark_fits_written
        peer_rows, fit_rows = decision_input_rows(
            list(scored['symbol']), window_open, offset, decided_at=decided_at)
        rows.extend(peer_rows)
        store = ResearchStore()
        if rows:
            # Synchronous Parquet work runs off the event loop the stream
            # readers share.
            await asyncio.to_thread(store.write, 'venue_depth', pd.DataFrame(rows))
        if fit_rows:
            await asyncio.to_thread(store.write, 'venue_implied_vol',
                                    pd.DataFrame(fit_rows))
            mark_fits_written(fit_rows)
    except Exception as exc:                       # noqa: BLE001 - never break a cycle
        logger.warning('could not record top of book (%s)', str(exc)[:120])
