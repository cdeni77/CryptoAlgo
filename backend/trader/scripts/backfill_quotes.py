"""Backfill the venue's own quote at each decision offset, from Kalshi candles.

Read-only. Enumerates settled markets, makes one candlestick call per market,
and writes `venue_quotes` to the research store.

* `open_time` comes from the enumeration response, never the ticker: the
  ticker names the close in US Eastern.
* `end_period_ts` is the inclusive end of a candle. The candle stamped
  `open + offset*60` is what a live decision sees; the next one would leak, so
  a missing offset stays missing.
* `yes_ask = 1.0000` means "no offer" only when the spread is also wide.
* Fields are `_dollars` / `_fp` strings.

Rows without a two-sided mid are written with `usable=False` and a reason, so
coverage can count what was lost.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from core.datastore import ResearchStore
from core.config import series_to_symbol

logger = logging.getLogger('backfill_quotes')

# The one series<->symbol mapping: core/config.SERIES_BY_SYMBOL.
SERIES = series_to_symbol()
OFFSETS = (3, 6, 9, 12)
# +14m is not traded; it is a leak control. Near settlement the market must
# score very well but not perfectly; a `market_ll` near 0 means the candle
# contains the settlement. Pulled as a separate sampled pass.
CONTROL_OFFSET = 14
WINDOW_MINUTES = 15
# Stay under the Basic-tier ceiling of 20 req/s.
DEFAULT_RATE = 12.0
NO_OFFER_SPREAD = 0.05      # ask >= 1.0 is only "no offer" when the book is wide


def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def _field(block: dict, *names: str) -> float:
    """Return the first named field present, as a float.

    The two endpoints name the same values differently:

    Live  `/series/{s}/markets/{t}/candlesticks` -> `close_dollars`, `volume_fp`
    Hist. `/historical/markets/{t}/candlesticks` -> `close`,         `volume`
    """
    for name in names:
        if name in block:
            return _f(block[name])
    return float('nan')


def parse_open_time(raw: str) -> datetime:
    return datetime.strptime(raw, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def classify(bid: float, ask: float) -> tuple[bool, str]:
    """Return whether the quote is usable, and the reason if not.

    The exclusion list is fixed; do not add to it here.
    """
    import math

    if not (math.isfinite(bid) and math.isfinite(ask)):
        return False, 'missing_side'
    spread = ask - bid
    if ask >= 1.0 and spread > NO_OFFER_SPREAD:
        return False, 'no_offer_ask_one'
    if bid <= 0.0 and spread > NO_OFFER_SPREAD:
        return False, 'no_offer_bid_zero'
    mid = (bid + ask) / 2.0
    if not (0.0 < mid < 1.0):
        return False, 'mid_out_of_range'
    return True, ''


def rows_from_candles(candles: list[dict], *, symbol: str, ticker: str,
                      open_time: datetime,
                      offsets: tuple[int, ...] = OFFSETS) -> list[dict]:
    open_ts = int(open_time.timestamp())
    by_offset: dict[int, dict] = {}
    for candle in candles:
        end = int(candle.get('end_period_ts', 0))
        delta, rem = divmod(end - open_ts, 60)
        if rem == 0:
            by_offset[delta] = candle

    out: list[dict] = []
    for offset in offsets:
        event_time = open_time + timedelta(minutes=offset)
        candle = by_offset.get(offset)
        if candle is None:
            out.append({
                'venue': 'kalshi', 'symbol': symbol, 'event_time': event_time,
                'available_time': event_time, 'quality': 'valid',
                'window_open': open_time, 'offset_minutes': offset,
                'market_ticker': ticker,
                'yes_bid': float('nan'), 'yes_ask': float('nan'),
                'market_probability': float('nan'), 'spread': float('nan'),
                'volume': float('nan'), 'open_interest': float('nan'),
                'usable': False, 'exclude_reason': 'no_candle',
            })
            continue
        bid = _field(candle.get('yes_bid', {}) or {}, 'close_dollars', 'close')
        ask = _field(candle.get('yes_ask', {}) or {}, 'close_dollars', 'close')
        usable, reason = classify(bid, ask)
        out.append({
            'venue': 'kalshi', 'symbol': symbol, 'event_time': event_time,
            'available_time': event_time, 'quality': 'valid',
            'window_open': open_time, 'offset_minutes': offset,
            'market_ticker': ticker,
            'yes_bid': bid, 'yes_ask': ask,
            'market_probability': (bid + ask) / 2.0 if usable else float('nan'),
            'spread': ask - bid,
            'volume': _field(candle, 'volume_fp', 'volume'),
            'open_interest': _field(candle, 'open_interest_fp', 'open_interest'),
            'usable': usable, 'exclude_reason': reason,
        })
    return out


class Throttle:
    def __init__(self, per_second: float):
        self.interval = 1.0 / max(per_second, 0.1)
        self._last = 0.0

    async def wait(self) -> None:
        gap = self.interval - (time.monotonic() - self._last)
        if gap > 0:
            await asyncio.sleep(gap)
        self._last = time.monotonic()


async def fetch_cutoff(client) -> Optional[datetime]:
    """Return the live/historical boundary from `GET /historical/cutoff`.

    Uses `market_settled_ts`. No cutoff field names candlesticks, so both
    paths are tried and merged; a mis-routed market costs one extra request.
    The cutoff moves forward, so it is fetched per run.
    """
    try:
        payload = await client._request('GET', '/historical/cutoff')  # noqa: SLF001
    except Exception as exc:                       # noqa: BLE001 - degrade to live-first
        logger.warning('could not read /historical/cutoff (%s); trying both '
                       'endpoints for every market', str(exc)[:100])
        return None
    raw = payload.get('market_settled_ts') if isinstance(payload, dict) else None
    if not raw:
        return None
    try:
        cutoff = datetime.strptime(str(raw), '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning('unparseable cutoff %r; trying both endpoints', raw)
        return None
    logger.info('historical cutoff (market_settled_ts): %s', cutoff)
    return cutoff


def merge_candles(*payloads: Optional[dict]) -> list[dict]:
    """Union the candles from several responses, keyed on `end_period_ts`.

    First writer wins per timestamp; both endpoints serve the same values.
    """
    out: dict[int, dict] = {}
    for payload in payloads:
        for candle in (payload or {}).get('candlesticks', []) or []:
            key = int(candle.get('end_period_ts', 0))
            out.setdefault(key, candle)
    return [out[k] for k in sorted(out)]


async def enumerate_settled(client, series: str) -> list[dict]:
    """Return every settled market for a series, paging the cursor to the end."""
    out, cursor = [], None
    while True:
        params = {'series_ticker': series, 'status': 'settled', 'limit': 1000}
        if cursor:
            params['cursor'] = cursor
        payload = await client._request('GET', '/markets', params=params)  # noqa: SLF001
        batch = payload.get('markets', [])
        if not batch:
            break
        out.extend(batch)
        cursor = payload.get('cursor')
        if not cursor:
            break
    return [m for m in out if m.get('ticker') and m.get('open_time')]


async def run(args) -> int:
    from data_collection.kalshi_client import KalshiClient

    key_id = os.getenv('KALSHI_KEY_ID')
    pem = os.getenv('KALSHI_PRIVATE_KEY')
    path = os.getenv('KALSHI_PRIVATE_KEY_PATH')
    if not pem and path and Path(path).exists():
        pem = Path(path).read_text()
    if not (key_id and pem):
        print('KALSHI_KEY_ID and a private key are required (read-only calls).')
        return 2

    store = ResearchStore(args.store or os.getenv('RESEARCH_STORE'))
    checkpoint = Path(args.checkpoint)
    done: set[str] = set()
    if checkpoint.exists():
        done = set(json.loads(checkpoint.read_text()).get('tickers', []))
        # Discard bare-ticker entries: they do not say which offsets were
        # pulled. Re-requesting is cheap and idempotent.
        stale = {k for k in done if '@' not in k}
        if stale:
            logger.warning('discarding %d checkpoint entry/entries written '
                           'before offsets were part of the key; those tickers '
                           'will be re-requested', len(stale))
            done -= stale
        logger.info('resuming: %d markets already pulled', len(done))

    throttle = Throttle(args.rate)
    written = total = failed = straddled = 0
    # No `live=True`: this client is structurally incapable of placing an order.
    async with KalshiClient(key_id=key_id, private_key_pem=pem) as client:
        cutoff = await fetch_cutoff(client)
        for series, symbol in SERIES.items():
            markets = await enumerate_settled(client, series)
            markets.sort(key=lambda m: m['open_time'])
            todo = [m for m in markets
                    if _checkpoint_key(m['ticker'], args.offsets) not in done]
            if args.limit_per_series:
                # Evenly spaced across the whole span, not the first N.
                step = max(1, len(todo) // args.limit_per_series)
                todo = todo[::step][:args.limit_per_series]
            logger.info('%s: %d settled markets, %d to pull (%d already done)',
                        series, len(markets), len(todo), len(markets) - len(todo))
            batch: list[dict] = []
            for i, market in enumerate(todo, 1):
                ticker = market['ticker']
                open_time = parse_open_time(market['open_time'])
                open_ts = int(open_time.timestamp())
                await throttle.wait()
                params = {'start_ts': open_ts - 60,
                          'end_ts': open_ts + (WINDOW_MINUTES + 1) * 60,
                          'period_interval': 1}
                live = f'/series/{series}/markets/{ticker}/candlesticks'
                hist = f'/historical/markets/{ticker}/candlesticks'
                # Route by the cutoff, then fall back to the other side.
                order = ([hist, live] if (cutoff and open_time < cutoff)
                         else [live, hist])
                payloads, last_error = [], 'no candlesticks'
                for path in order:
                    try:
                        payloads.append(await client._request('GET', path, params=params))  # noqa: SLF001
                    except Exception as exc:          # noqa: BLE001 - try the other side
                        last_error = str(exc)[:120]
                        continue
                    if len(merge_candles(*payloads)) >= WINDOW_MINUTES:
                        break                         # complete; no need for the other
                    await throttle.wait()
                candles = merge_candles(*payloads)
                if not candles:
                    failed += 1
                    logger.warning('%s: %s', ticker, last_error)
                    continue
                if len(payloads) > 1:
                    straddled += 1
                batch.extend(rows_from_candles(
                    candles, symbol=symbol, ticker=ticker, open_time=open_time,
                    offsets=args.offsets))
                # Key on the offsets too, so a control pass does not mark a
                # ticker done for the traded offsets.
                done.add(_checkpoint_key(ticker, args.offsets))
                total += 1
                if len(batch) >= args.batch_rows:
                    written += store.write('venue_quotes', pd.DataFrame(batch))
                    batch.clear()
                    checkpoint.write_text(json.dumps({'tickers': sorted(done)}))
                    logger.info('%s: %d/%d markets, %d rows written',
                                series, i, len(todo), written)
            if batch:
                written += store.write('venue_quotes', pd.DataFrame(batch))
                checkpoint.write_text(json.dumps({'tickers': sorted(done)}))
    logger.info('done: %d markets, %d rows, %d failures, %d needed both endpoints',
                total, written, failed, straddled)
    print(f'\n{total:,} markets pulled, {written:,} rows written, {failed} failures, '
          f'{straddled} needed both endpoints')
    return 0


def _checkpoint_key(ticker: str, offsets) -> str:
    """Return the checkpoint key: this ticker at these offsets, in sorted order."""
    return f"{ticker}@{','.join(str(int(o)) for o in sorted(offsets))}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=None, help='defaults to $RESEARCH_STORE')
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE,
                        help='requests per second (Basic tier ceiling is 20)')
    parser.add_argument('--batch-rows', type=int, default=4000)
    parser.add_argument('--checkpoint', default='/app/data/quote_backfill.json')
    parser.add_argument('--offsets', default=','.join(str(o) for o in OFFSETS),
                        help='minutes after open to store; 14 is the control')
    parser.add_argument('--limit-per-series', type=int, default=0,
                        help='sample this many markets, evenly spaced (0 = all)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    args.offsets = tuple(int(x) for x in str(args.offsets).split(',') if x.strip())
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
