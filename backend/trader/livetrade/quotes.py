"""The Kalshi book for the live loop: which market, its touch, its ladder.

Markets are resolved by asking the venue, once per window. A quote comes from
the in-process stream book when it can vouch for it, else one REST
`/orderbook` read. Also here: the ladder and depth helpers every book feature is
built from, and the top-of-book archive written after each decision.
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

# symbol -> (settle_time, market metadata): each window's market is resolved
# once, not on every read. The metadata carries NO touch (`market_meta`), so a
# price can only come from a book.
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


async def resolve_markets(kalshi, symbols, settle_time: pd.Timestamp, *,
                          refresh: bool = False) -> dict[str, Quote]:
    """Each symbol's market for this window: cached, or resolved concurrently.

    `refresh` re-reads `/markets/{ticker}` for every cached market — status,
    shard and `floor_strike`, which the venue publishes after the window opens.
    Non-decision cycles do it, so a decision never pays for it.
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
    """`meta` priced off the in-process stream book, or None if it cannot vouch."""
    if cache is None or meta is None or not meta.ticker:
        return None
    try:
        ladder = cache.current(meta.ticker, max_age_seconds)
    except Exception:                                         # noqa: BLE001
        return None
    if ladder is None:
        return None
    # As of the connection's last frame: until then the venue had told us
    # every change, so that is when this book was last known to be right.
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
    """Each symbol's touch for this window: the stream book, else one REST read.

    **The stream first.** `record_stream.CACHE` is folded from the venue's own
    socket in this process, ~34ms behind the venue, so pricing from it costs no
    round trip. It is used only while that book is present, not gapped and its
    connection delivered a frame within `stream_quote_max_age_seconds`.

    **Otherwise one `/orderbook` read per symbol, all concurrently.** Never the
    `/markets` touch: it is cached and measured up to 6.3c wrong, so a failed
    book read is no quote — `decide()` refuses `no_quote` — not a fallback.

    Each `Quote` carries `book_source` and `book_time`, stamped at its own read.
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
    """The stream task's cache, when this process is running one."""
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
    """The best price the LADDER actually shows, in the quote's denomination.

    Depth buckets must be measured against the touch of the same snapshot that
    supplies the levels; a separately-fetched touch a cent better makes the 1c
    bucket read zero.

    `fallback` is used only when the ladder is empty, where there is nothing to
    measure and the NaN is the honest answer.
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
    """Resting size at prices at least as good as `best +/- within`.

    This, not the touch, is what decides a fill: our limit sits a cent or two
    past the touch, so the order fills only if the cumulative size up to that
    limit covers it.
    """
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
    """Every resting contract on one side, at any price (`depth_*_total`)."""
    total, seen = 0.0, False
    for entry in levels or []:
        try:
            total += float(entry[1])
            seen = True
        except (TypeError, ValueError, IndexError):
            continue
    return total if seen else float('nan')


def ladder_from_cache(cache, ticker: str):
    """(yes_levels, no_levels) from the stream's in-process book, or empty.

    `run_live` runs the stream and trade tasks in one process, so
    `record_stream.CACHE` already holds every subscribed ladder: no fetch, and
    fresher than REST (~34ms against ~73ms).

    Refused in three cases, all of which mean the book in hand may not be the
    book on the venue:

      * no cache — `scripts.live` also runs standalone, without the stream task;
      * stale — ten seconds of silence at 400+ frames a second is a sick
        transport, not a quiet market;
      * gapped — `seq` is global per SUBSCRIPTION, so a miss condemns every book
        on the connection, this one included.

    Empty levels give NaN depth features, which `_warn_unscoreable_features`
    reports. That is the honest answer, not a silent substitution.
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
    """(yes_levels, no_levels) carried on a quote, or empty."""
    return (list(getattr(quote, 'yes_levels', ()) or ()),
            list(getattr(quote, 'no_levels', ()) or ()))


def depth_dollars(quote, yes_levels, no_levels):
    """(depth_up, depth_down) in DOLLARS, or (None, None) with no ladder.

    `decide()` caps the stake at `measured_depth * depth_fraction`; this is
    where `depth_up`/`depth_down` come from.

    Kalshi's book is two BID stacks: buying YES crosses the YES ask, whose size
    comes from the NO stack; buying NO crosses the NO ask, whose size comes from
    the YES stack. Dollars, because that is what `decide()` compares against the
    stake.

    None, not zero, when there is no ladder — zero would refuse every trade,
    where the honest answer is "unmeasured, fall back to the standing cap".
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
    """Write the venue's book — top of book plus cumulative depth — to the store,
    with the peer book and implied-vol fit the same decision read.

    Best effort: a failure here must never stop a trading cycle, because this is
    measurement and the cycle is money.

    Depth is in no historical endpoint and `/orderbook` returns empty once a
    market settles, so a book not written down now is gone.
    """
    try:
        from core.datastore import ResearchStore

        rows = []
        for symbol in scored['symbol']:
            quote = quotes.get(symbol, (None, None))[0]
            if quote is None or quote.yes_bid is None or quote.yes_ask is None:
                continue
            ticker = quotes[symbol][1]
            # The ladder the decision priced from, the same snapshot as the
            # touch. Then the stream cache. A REST fetch is the last resort:
            # it runs after our own fill has removed size from the traded side,
            # so the row is flagged `rest_after_order`.
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
                # How old the priced book was when the decision was taken: the
                # stream's last frame or the REST response, to the decision
                # instant. NaN only when the quote carries no read time — the
                # backtest reads a NULL age as fresh, so never invent one.
                'quote_age_seconds': quote_age_seconds(quote, decided_at),
                'venue': 'kalshi', 'symbol': symbol, 'event_time': event_time,
                'available_time': pd.Timestamp.now(tz='UTC'), 'quality': 'valid',
                # Which sampler produced the LADDER half of this row: an
                # observation at the decision and a reconstruction after our
                # own fill are different claims.
                'transport': ladder_source,
                'market_ticker': quotes[symbol][1], 'window_open': window_open,
                'offset_minutes': offset,
                'yes_bid': float(quote.yes_bid), 'yes_ask': float(quote.yes_ask),
                'yes_bid_size': float(quote.yes_bid_size or 0.0),
                'yes_ask_size': float(quote.yes_ask_size or 0.0),
                # ONE sampler, as in `book_feature_row`: the ladder's own
                # touch, not the quote's.
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
                # The whole resting ladder: `depth_ratio` and `book_convexity`
                # read these, and the next retrain trains on what is recorded.
                'depth_bid_total': _resting_total(yes_levels),
                'depth_ask_total': _resting_total(no_levels),
                'levels_bid': float(len(yes_levels)),
                'levels_ask': float(len(no_levels)),
                'seq': float('nan'), 'gaps': 0.0,
                # The observer. `source` is in `EVENT_KEY_EXTRA['venue_depth']`
                # so independent observations of one minute survive a read.
                # Not 'live': that is the ladder recorder's REST poll, a
                # different sample at a different instant.
                'source': 'live_touch',
            })
        # The peer book and the implied-vol fit the same decision read, so
        # the backtest can join the inputs live actually used.
        from livetrade.features import decision_input_rows, mark_fits_written
        peer_rows, fit_rows = decision_input_rows(
            list(scored['symbol']), window_open, offset, decided_at=decided_at)
        rows.extend(peer_rows)
        store = ResearchStore()
        if rows:
            # Synchronous Parquet work: off the event loop, so the stream
            # readers sharing it keep folding frames meanwhile.
            await asyncio.to_thread(store.write, 'venue_depth', pd.DataFrame(rows))
        if fit_rows:
            await asyncio.to_thread(store.write, 'venue_implied_vol',
                                    pd.DataFrame(fit_rows))
            mark_fits_written(fit_rows)
    except Exception as exc:                       # noqa: BLE001 - never break a cycle
        logger.warning('could not record top of book (%s)', str(exc)[:120])
