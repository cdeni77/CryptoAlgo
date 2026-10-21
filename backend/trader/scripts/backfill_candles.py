"""Kalshi's own per-minute candlesticks into `venue_depth`, at the TRUE tick.

**Why this exists.** 96% of what the deployed model trains on is Predexon
backfill, and the compose file already calls that "not an executable price".
Measured 2026-10-21, the reason is not staleness and not a mapping error --
both were tested and ruled out. It is QUANTISATION:

```
share of quotes sitting exactly on a whole cent
source     <10c     10-90c    >90c
backfill   100.0%   100.0%   100.0%
live_ws     10.3%   100.0%     7.4%
```

Kalshi's `price_level_structure` is `tapered_deci_cent`: a whole cent between
10c and 90c, a TENTH of a cent outside it. In the middle band both sources are
100% on-cent because that is the real grid and they agree exactly. In the tails
the venue quotes deci-cents and Predexon rounds every one to a whole cent --
27% of rows, and precisely the rows log loss is most sensitive to, because a
0.5c rounding at a 4c quote is a 12.5% relative error in the probability. That
is the +0.0425 nat forecast handicap, against a model edge of +0.00079.

These candles carry the venue's real grid (`0.0410`, `0.0070`), so they do not
have the defect.

**Alignment, which is the one thing that must not be got wrong.** Verified
against the venue's own metadata on live markets: the ticker names the CLOSE in
New York local time, `close - 15min` reproduces `open_time` exactly, and each
market returns 16 candles at offsets +1..+16 from the window open. A candle
covers the minute ENDING at its `end_period_ts`, so:

    offset_minutes m  <-  the candle with end_period_ts == window_open + m*60

for m in 1..15. There is no candle at offset 0, and +16 lies past the close and
belongs to no window. This is "last at or before the mark", the same rule the
Predexon ASOF join uses, so it is point-in-time with no lookahead.

**`source='candles'`, not `'backfill'`.** `source` is in
`EVENT_KEY_EXTRA['venue_depth']`, so these land BESIDE the Predexon rows rather
than replacing them -- which is what makes the two comparable, exactly as
`live` and `live_ws` already sit beside each other. Nothing is overwritten and
re-running is idempotent.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import sys
from collections import deque
from zoneinfo import ZoneInfo

#: Which of the OHLC fields each block actually carries. `price` is the only
#: one the venue sends a `previous` for; the quote blocks have no close-to-
#: close notion, so asking for one would store a column of nulls.
CANDLE_COLUMNS = (
    'price_open', 'price_high', 'price_low', 'price_close', 'price_mean',
    'price_previous',
    'yes_bid_open', 'yes_bid_high', 'yes_bid_low', 'yes_bid_close',
    'yes_ask_open', 'yes_ask_high', 'yes_ask_low', 'yes_ask_close',
)

logger = logging.getLogger('candles')

WINDOW_MINUTES = 15
MONTHS = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6,
              JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}

#: Kalshi's documented read budget is 200 tokens/s against a cost of 10 for a
#: default endpoint -- 20 requests a second. This takes a small fraction of it
#: ON PURPOSE: the live trader shares the bucket, and starving the loop that is
#: placing real orders to speed up a backfill is a bad trade at any speed.
DEFAULT_RATE = 3.0


def window_from_ticker(ticker: str):
    """`KXBTC15M-26OCT202245-45` -> (open_utc, close_utc), or None.

    Returns None rather than guessing on a shape it does not recognise: a
    ticker parsed wrong puts a whole window's book on the wrong minutes, which
    is the failure this module exists to avoid.
    """
    try:
        body = ticker.split('-')[1]
        yy, mon, dd = int(body[:2]), MONTHS[body[2:5]], int(body[5:7])
        hh, mm = int(body[7:9]), int(body[9:11])
    except (IndexError, KeyError, ValueError):
        return None
    close = dt.datetime(2000 + yy, mon, dd, hh, mm,
                        tzinfo=ZoneInfo('America/New_York')).astimezone(dt.timezone.utc)
    return close - dt.timedelta(minutes=WINDOW_MINUTES), close


def _money(block, key='close_dollars'):
    """A price out of a candle block, under EITHER spelling.

    **The two candle endpoints disagree on field names and neither errors.**
    The batch `/markets/candlesticks` sends `close_dollars`; the per-ticker
    `/historical/markets/{ticker}/candlesticks` sends plain `close`. Reading
    one shape against the other yields no rows and no exception -- verified,
    `rows_for_market` returned 0 of 15 candles on a historical payload. That is
    the same silent-failure shape CLAUDE.md records for the websocket's
    `yes_dollars_fp` against REST's `orderbook_fp.yes_dollars`.
    """
    if not isinstance(block, dict):
        return None
    raw = block.get(key)
    if raw in (None, '') and key.endswith('_dollars'):
        raw = block.get(key[:-len('_dollars')])
    if raw in (None, ''):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _fp(raw):
    """A fixed-point string like "12077.70" as a float, or None."""
    if raw in (None, ''):
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _candle_fp(candle, key):
    """`volume_fp` on the batch endpoint, plain `volume` on the per-ticker one.

    Same divergence as `_money`; see its docstring.
    """
    if not isinstance(candle, dict):
        return None
    got = _fp(candle.get(key))
    if got is None and key.endswith('_fp'):
        got = _fp(candle.get(key[:-len('_fp')]))
    return got


def candle_rows_for_market(ticker: str, symbol: str, candles, *, fetched_at):
    """`venue_candles` rows for one market -- the WHOLE payload.

    `rows_for_market` keeps the closing bid and ask, which is the book summary
    `venue_depth` exists to hold. Everything else the venue sends was being
    discarded: the traded-price OHLC, volume, and open interest.

    `price_low` is the one to notice. It is the cheapest PRINT inside the
    minute, so a `price_low` at or below a resting bid is evidence that bid
    would have been hit -- the question the maker thesis turns on, answerable
    over every market this endpoint serves rather than only where the tape has
    been backfilled. It is a bound and not the tape: a candle cannot say which
    side crossed, so it cannot distinguish a fill from an adverse one. That
    needs `venue_trades.taker_book_side`.

    `available_time` is the FETCH, not the mark, because open interest is a
    revised series -- the venue restates it, and a row claiming to have been
    knowable at the mark would be a lookahead this store is built to refuse.
    """
    window = window_from_ticker(ticker)
    if window is None:
        return []
    opened, _close = window
    base = int(opened.timestamp())
    out = []
    for candle in candles or []:
        end = candle.get('end_period_ts')
        if end is None:
            continue
        offset, remainder = divmod(int(end) - base, 60)
        if remainder or not (1 <= offset <= WINDOW_MINUTES):
            continue
        price = candle.get('price') if isinstance(candle.get('price'), dict) else {}
        bid = candle.get('yes_bid') if isinstance(candle.get('yes_bid'), dict) else {}
        ask = candle.get('yes_ask') if isinstance(candle.get('yes_ask'), dict) else {}
        row = {
            'venue': 'kalshi', 'symbol': symbol,
            'event_time': opened + dt.timedelta(minutes=offset),
            'available_time': fetched_at,
            'quality': 'valid', 'market_ticker': ticker,
            'window_open': opened, 'offset_minutes': offset,
            'volume': _candle_fp(candle, 'volume_fp'),
            'open_interest': _candle_fp(candle, 'open_interest_fp'),
        }
        for pre, block in (('price', price), ('yes_bid', bid), ('yes_ask', ask)):
            for field in ('open', 'high', 'low', 'close', 'mean',
                          'previous'):
                key = '%s_%s' % (pre, field)
                if key in CANDLE_COLUMNS:
                    row[key] = _money(block, field + '_dollars')
        # A candle with no quote and no print is the venue saying nothing
        # happened; keeping it would pad the table with rows that carry no
        # measurement.
        if all(row.get(c) is None for c in CANDLE_COLUMNS) and \
                row['volume'] is None and row['open_interest'] is None:
            continue
        out.append(row)
    return out


def rows_for_market(ticker: str, symbol: str, candles, *, fetched_at):
    """`venue_depth` rows for one market. Offsets 1..15 only."""
    window = window_from_ticker(ticker)
    if window is None:
        logger.warning('ticker %s does not parse; skipped rather than guessed', ticker)
        return []
    opened, _close = window
    base = int(opened.timestamp())
    out = []
    for candle in candles or []:
        end = candle.get('end_period_ts')
        if end is None:
            continue
        offset, remainder = divmod(int(end) - base, 60)
        # Off-grid, or outside the window. +16 exists and is past the close.
        if remainder or not (1 <= offset <= WINDOW_MINUTES):
            continue
        bid = _money(candle.get('yes_bid'))
        ask = _money(candle.get('yes_ask'))
        if bid is None or ask is None:
            continue
        mark = opened + dt.timedelta(minutes=offset)
        out.append({
            'venue': 'kalshi', 'symbol': symbol,
            'event_time': mark, 'available_time': fetched_at,
            'quality': 'valid', 'market_ticker': ticker,
            'window_open': opened, 'offset_minutes': offset,
            'yes_bid': bid, 'yes_ask': ask,
            'yes_bid_size': None, 'yes_ask_size': None,
            'depth_bid_1c': None, 'depth_bid_5c': None,
            'depth_ask_1c': None, 'depth_ask_5c': None,
            'depth_bid_total': None, 'depth_ask_total': None,
            'levels_bid': None, 'levels_ask': None,
            'seq': None, 'gaps': None,
            'source': 'candles',
            # **Zero, and it means mark-ALIGNED, not freshly observed.** The
            # candle's close is by construction the book as of the period end,
            # and the period ends exactly at the mark. What a candle cannot say
            # is whether the book moved without printing inside that minute --
            # but that is unobservable for every source, not staleness of this
            # one. `levels_*` stay NULL rather than 0: a level COUNT from a
            # candle does not exist, and `levels_bid`/`levels_ask` are already
            # documented as not comparable across sources.
            'quote_age_seconds': 0.0,
        })
    return out


async def _markets(client, series: str, limit: int):
    """Every market in a series, newest first, from whichever tier serves it.

    The live tier's `status=settled` listing stops at the historical cutoff;
    `/historical/markets` reaches a month further back. Both are walked, and
    tickers are de-duplicated.
    """
    seen: dict[str, dict] = {}
    for base, path, params in (
            (None, '/markets', {'series_ticker': series, 'status': 'settled'}),
            (client.historical_base_url, '/historical/markets',
             {'series_ticker': series})):
        cursor = None
        while len(seen) < limit:
            query = dict(params, limit=200)
            if cursor:
                query['cursor'] = cursor
            try:
                payload = await client._request('GET', path, params=query, base=base)
            except Exception as exc:                    # noqa: BLE001
                logger.warning('%s listing stopped: %s', path, str(exc)[:160])
                break
            rows = payload.get('markets') or []
            for row in rows:
                ticker = row.get('ticker')
                if ticker:
                    seen.setdefault(str(ticker), row)
            cursor = payload.get('cursor')
            if not cursor or not rows:
                break
    return list(seen)[:limit]


#: **`GET /markets/candlesticks` takes many markets in ONE request**, and its
#: only limit is `markets x range_minutes <= 10000` -- counted on what is
#: REQUESTED, not what comes back (measured 2026-10-21: the error names both
#: numbers, e.g. "requested candlesticks across all markets: 12000, max
#: candlesticks: 10000"). Markets outside the range simply return nothing.
#:
#: So the efficient shape is to batch by TIME: W consecutive windows across all
#: three symbols is `3W` markets over `15W` minutes, and `3W * 15W <= 10000`
#: gives W = 14 -- 42 markets a request. Per-market fetching needed 68,257
#: requests; this needs ~1,625.
WINDOWS_PER_REQUEST = 14
MAX_CANDLES_PER_REQUEST = 10000


async def run(args) -> int:
    import pandas as pd
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore()
    fetched_at = dt.datetime.now(dt.timezone.utc)
    gap = 1.0 / max(0.1, float(args.rate))

    done: set[str] = set()
    if not args.force:
        # Keyed on `venue_candles`, not on the `venue_depth` rows this also
        # writes. A market fetched before `venue_candles` existed has its
        # closing quote stored and its volume, open interest and traded-price
        # OHLC thrown away; counting it as done would make that loss
        # permanent and invisible. Re-fetching is idempotent -- the depth
        # write merges on its own event key.
        try:
            held = store.read('venue_candles', min_quality=None)
            if len(held):
                done = set(held['market_ticker'].dropna().astype(str))
        except Exception as exc:                        # noqa: BLE001
            logger.info('no stored candles yet (%s); fetching everything',
                        str(exc)[:100])
    logger.info('%d market(s) already have candles; skipped (--force to redo)',
                len(done))

    client = KalshiClient()                 # not live=True: cannot place orders
    async with client:
        # One catalogue across all three series, keyed by window so a request
        # can carry every symbol for the same minutes.
        by_window: dict[dt.datetime, list[tuple[str, str]]] = {}
        for series, symbol in SERIES.items():
            for ticker in await _markets(client, series, args.limit):
                if ticker in done:
                    continue
                window = window_from_ticker(ticker)
                if window is None:
                    continue
                by_window.setdefault(window[0], []).append((ticker, symbol))
        windows = sorted(by_window)
        logger.info('%d window(s) to fetch, %d market(s)',
                    len(windows), sum(len(v) for v in by_window.values()))
        if not windows:
            return 0

        written = requests = 0
        batch: list[dict] = []
        candle_batch: list[dict] = []
        for chunk, markets, lo, hi in plan_requests(windows, by_window):
            try:
                payload = await client._request(
                    'GET', '/markets/candlesticks',
                    params={'market_tickers': ','.join(t for t, _ in markets),
                            'period_interval': 1,
                            'start_ts': int(lo.timestamp()),
                            'end_ts': int(hi.timestamp())})
            except Exception as exc:                    # noqa: BLE001
                logger.warning('chunk at %s failed: %s', lo, str(exc)[:160])
                await asyncio.sleep(gap)
                continue
            requests += 1
            symbol_of = dict(markets)
            for entry in payload.get('markets') or []:
                ticker = entry.get('ticker') or entry.get('market_ticker')
                if ticker not in symbol_of:
                    continue
                batch.extend(rows_for_market(
                    str(ticker), symbol_of[ticker],
                    entry.get('candlesticks'), fetched_at=fetched_at))
                candle_batch.extend(candle_rows_for_market(
                    str(ticker), symbol_of[ticker],
                    entry.get('candlesticks'), fetched_at=fetched_at))
            await asyncio.sleep(gap)
            if len(batch) >= args.batch:
                written += _flush(store, batch, candle_batch, args.dry_run)
                logger.info('  %d request(s), %d row(s) %s', requests, written,
                            'ready' if args.dry_run else 'written')
                batch, candle_batch = [], []
        written += _flush(store, batch, candle_batch, args.dry_run)

    logger.info('%d request(s), %d row(s) %s', requests, written,
                'ready (dry run, nothing written)' if args.dry_run else 'written')
    return 0


def _flush(store, depth_rows, candle_rows, dry_run: bool) -> int:
    """Write both tables. Returns the `venue_depth` count, which is what the
    progress line has always reported; `venue_candles` is logged separately so
    a silent failure on one cannot hide behind the other's total."""
    # Imported here, not at module scope: this script keeps pandas out of the
    # import path so `--help` and the ticker parser stay cheap, and `run()`
    # imports it locally for the same reason. A module-level helper cannot see
    # that local name.
    import pandas as pd

    if dry_run:
        return len(depth_rows)
    written = 0
    if depth_rows:
        written = store.write('venue_depth', pd.DataFrame(depth_rows))
    if candle_rows:
        n = store.write('venue_candles', pd.DataFrame(candle_rows))
        logger.info('  venue_candles: %d row(s) written', n)
    return written


def plan_requests(windows, by_window, *,
                  max_candles: int = MAX_CANDLES_PER_REQUEST,
                  per_request: int = WINDOWS_PER_REQUEST):
    """Yield `(chunk, markets, lo, hi)` for every request to make.

    A work QUEUE, not a fixed stride, because an oversized chunk has to be
    SPLIT rather than dropped. The previous loop logged "split it" and then
    `continue`d -- so it skipped the request and every window in it, and the
    chunks that overflow are by definition the dense ones. The holes landed
    exactly where the data was richest, and nothing downstream could tell
    them from windows that never traded.

    Every window handed in comes back out in exactly one chunk, which is the
    property worth testing and the one the old loop broke.
    """
    pending = deque(windows[i:i + per_request]
                    for i in range(0, len(windows), per_request))
    while pending:
        chunk = pending.popleft()
        if not chunk:
            continue
        markets = [m for w in chunk for m in by_window[w]]
        lo = min(chunk)
        hi = max(chunk) + dt.timedelta(minutes=WINDOW_MINUTES + 1)
        span = int((hi - lo).total_seconds() // 60)
        if len(markets) * span > max_candles and len(chunk) > 1:
            # Halving cuts BOTH factors, so the product quarters and this
            # converges in a couple of passes. Pushing to the FRONT keeps the
            # fetch in window order.
            half = len(chunk) // 2
            pending.appendleft(chunk[half:])
            pending.appendleft(chunk[:half])
            continue
        if len(markets) * span > max_candles:
            # One window is as small as a request gets. Ask anyway and let the
            # venue truncate: a short answer is a measurement, a skipped
            # window is a hole nothing records.
            logger.warning('single window at %s asks for %d candles; '
                           'requesting it anyway', lo, len(markets) * span)
        yield chunk, markets, lo, hi


async def run_historical(args) -> int:
    """Per-ticker candles over a date range, for markets the LISTING cannot reach.

    `/markets?status=settled` and `/historical/markets` only enumerate about 69
    days back, which is why the batch path bottoms out at 2026-08-13. The
    per-ticker endpoint serves far older markets -- the wall was the listing,
    never the data. Tickers are CONSTRUCTED from the window, so no enumeration
    is needed at all.

    Verified 2026-10-21: full 15 candles per market from 2025-12 through
    2026-08, and EMPTY from 2026-09 (recent markets are not on the historical
    tier). So this path and the batch path tile rather than overlap.
    """
    import pandas as pd
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient
    from scripts.backfill_markets import ticker_for

    store = ResearchStore(getattr(args, 'research_store', None))
    start = pd.Timestamp(args.start, tz='UTC').floor('15min')
    end = pd.Timestamp(args.end, tz='UTC').floor('15min')

    done: set[str] = set()
    if not args.force:
        try:
            held = store.read('venue_candles', min_quality=None)
            if len(held):
                done = set(held['market_ticker'].dropna().astype(str))
        except Exception as exc:                              # noqa: BLE001
            logger.info('no stored candles yet (%s)', str(exc)[:90])
    logger.info('%d market(s) already held', len(done))

    windows = pd.date_range(start, end, freq='15min', inclusive='left')
    jobs = []
    for series, symbol in SERIES.items():
        for w in windows:
            tk = ticker_for(series, w.to_pydatetime() + dt.timedelta(minutes=WINDOW_MINUTES))
            if tk not in done:
                jobs.append((tk, symbol, w.to_pydatetime()))
    logger.info('%d market(s) to fetch', len(jobs))
    if args.dry_run or not jobs:
        return 0

    fetched_at = dt.datetime.now(dt.timezone.utc)
    sem = asyncio.Semaphore(max(1, args.concurrency))

    async def one(client, tk, symbol, opened):
        close = opened + dt.timedelta(minutes=WINDOW_MINUTES)
        async with sem:
            for attempt in range(6):
                try:
                    payload = await client._request(      # noqa: SLF001
                        'GET', '/historical/markets/%s/candlesticks' % tk,
                        base=client.historical_base_url,
                        params={'start_ts': int(opened.timestamp()),
                                'end_ts': int(close.timestamp()),
                                'period_interval': 1})
                except Exception as exc:                  # noqa: BLE001
                    msg = str(exc)
                    if '429' in msg or 'too_many_requests' in msg:
                        await asyncio.sleep(1.5 * (2 ** attempt))
                        continue
                    return [], []
                cs = payload.get('candlesticks') or []
                return (rows_for_market(tk, symbol, cs, fetched_at=fetched_at),
                        candle_rows_for_market(tk, symbol, cs, fetched_at=fetched_at))
        return [], []

    written = asked = 0
    async with KalshiClient(live=False) as client:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            got = await asyncio.gather(*[one(client, tk, sym, w) for tk, sym, w in chunk])
            depth = [r for d, _ in got for r in d]
            cand = [r for _, c in got for r in c]
            written += _flush(store, depth, cand, args.dry_run)
            asked += len(chunk)
            logger.info('  %d/%d markets, %d depth row(s) written',
                        asked, len(jobs), written)
    logger.info('done: %d depth row(s)', written)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true',
                        help='fetch and map, write nothing')
    parser.add_argument('--limit', type=int, default=200,
                        help='markets per series (default 200; ~23,000 exist)')
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE,
                        help='requests per second (default %.1f)' % DEFAULT_RATE)
    parser.add_argument('--batch', type=int, default=4000)
    parser.add_argument('--force', action='store_true',
                        help='re-fetch markets already held')
    parser.add_argument('--historical', action='store_true',
                        help='per-ticker historical path: constructs tickers over '
                             '[--start, --end) instead of enumerating markets. '
                             'Reaches 2025-12 where the listing stops at ~69 days.')
    parser.add_argument('--start', default='2025-12-16')
    parser.add_argument('--end', default='2026-08-13')
    parser.add_argument('--concurrency', type=int, default=8)
    parser.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    args = build_parser().parse_args(argv)
    # Two tiles, not two options: the per-ticker historical path serves
    # 2025-12..2026-08 and is EMPTY after it; the batch path serves ~69 days
    # back from today. Together they cover everything; neither covers it alone.
    return asyncio.run(run_historical(args) if args.historical else run(args))


if __name__ == '__main__':
    sys.exit(main())
