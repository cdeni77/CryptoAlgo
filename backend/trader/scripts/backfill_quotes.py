"""Backfill the venue's own quote at each decision offset, from Kalshi candles.

**Why this exists.** A backtest built from Coinbase bars has no order book, so
`price_source` stands the calibrated baseline in for the market — which makes
"beat the price" and "beat the baseline" the same question answered twice with
the same number. `model_minus_market` can only be computed where a real quote was
recorded, and recording forward was going to take ~70 days.

It turns out not to be necessary. Kalshi's candlestick history reaches the series
origin: `GET /series/{s}/markets/{t}/candlesticks` returns 16 one-minute candles
for the oldest settled market in KXBTC15M, opened 2026-06-17T23:45Z. Sampled
across the full span, 100% of markets returned candles and 100% of the four
offsets were present and two-sided, at a 1.0c median spread.

Read-only. Enumerates settled markets, pulls one candlestick call per market, and
writes `venue_quotes` to the research store.

Correctness notes, each of which was a real trap:

* **`open_time` comes from the enumeration response, never from the ticker.**
  `KXBTC15M-26JUN172000-00` opens at 2026-06-17T**23:45**Z — the ticker names the
  close in US Eastern. Building times from tickers is the pattern
  `CLAUDE.md` already warns about for market resolution.
* **`end_period_ts` is the inclusive END of a candle.** The candle stamped
  `open + offset*60` closes at exactly the offset minute, which is the instant a
  live decision sees. The next candle would leak, so a missing offset is recorded
  as missing rather than filled from its neighbour.
* **`yes_ask = "1.0000"` can mean "no offer".** Observed as an ask of 1.0000
  against a 0.53 bid in a market's first candle. But a tight 0.999/1.000 book is
  a real near-certain market — of 46 such rows in the live store, every one had a
  spread of 0.3c or less. So the rule is ask >= 1.0 AND spread > 5c, not ask
  >= 1.0 alone.
* Everything is `_dollars` / `_fp` strings. The integer-cent fields the older
  documentation describes are not present on this endpoint at all.

Rows that cannot yield a two-sided mid are written with `usable=False` and a
reason rather than dropped, so the coverage report can count what was lost and
`DECISION_RULE.md`'s 5% exclusion ceiling can be checked against data instead of
against a claim.
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

logger = logging.getLogger('backfill_quotes')

SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}
OFFSETS = (3, 6, 9, 12)
WINDOW_MINUTES = 15
# Measured 12.1-12.6 req/s without throttling against a Basic-tier ceiling of
# 20/s (200 read tokens/s at 10 tokens a request). Stay under it deliberately.
DEFAULT_RATE = 12.0
NO_OFFER_SPREAD = 0.05      # ask >= 1.0 is only "no offer" when the book is wide


def _f(value) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float('nan')
    return out


def _field(block: dict, *names: str) -> float:
    """Read the first name present. **The two endpoints disagree on names.**

    Live  `/series/{s}/markets/{t}/candlesticks` -> `close_dollars`, `volume_fp`
    Hist. `/historical/markets/{t}/candlesticks` -> `close`,         `volume`

    Same values, verified against the same market: the historical response's
    `yes_bid.close "0.7600"` is the live response's fourth candle exactly. Reading
    only the `_dollars` names off a historical response yields NaN for every row
    and no error, which is the same class of failure as the `yes_bid_dollars`
    trap this project was already bitten by — in the opposite direction.

    It matters because `GET /historical/cutoff` reads 2026-06-25 and "will be
    regularly updated, advancing forward over time". Candlesticks do not appear
    to be governed by it today (the live path serves 2026-06-17, a week before
    the cutoff, and the cutoff's fields are all positions/orders/trades/
    settlements). But "does not appear to be governed by it today" is not a
    guarantee, and the failure mode is silent.
    """
    for name in names:
        if name in block:
            return _f(block[name])
    return float('nan')


def parse_open_time(raw: str) -> datetime:
    return datetime.strptime(raw, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def classify(bid: float, ask: float) -> tuple[bool, str]:
    """Is this quote usable, and if not, why? The exclusion list is fixed by
    DECISION_RULE.md Appendix A and must not grow here."""
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
                      open_time: datetime) -> list[dict]:
    open_ts = int(open_time.timestamp())
    by_offset: dict[int, dict] = {}
    for candle in candles:
        end = int(candle.get('end_period_ts', 0))
        delta, rem = divmod(end - open_ts, 60)
        if rem == 0:
            by_offset[delta] = candle

    out: list[dict] = []
    for offset in OFFSETS:
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
    """The live/historical boundary, per Kalshi's own migration guidance.

    `GET /historical/cutoff` returns a set of timestamps and the venue's advice is
    to route by them and merge across the boundary rather than guess. Measured
    2026-08-25 it reads:

        market_positions_last_updated_ts  2026-06-25T00:00:00Z
        market_settled_ts                 2026-06-25T00:00:00Z
        orders_updated_ts                 2026-06-25T00:00:00Z
        trades_created_ts                 2026-06-25T00:00:00Z

    **None of those fields names candlesticks**, and empirically the live path
    serves 2026-06-17 — a week older than every one of them. So candlesticks
    either are not governed by this cutoff or have a longer window. `market_
    settled_ts` is used as the routing boundary because it is the market-shaped
    one, and because being wrong about it is cheap: both paths are tried and
    merged, so a mis-routed market costs one extra request rather than a hole.

    The cutoff "will be regularly updated, advancing forward over time", which is
    why this is fetched per run instead of hardcoded.
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

    Kalshi's guidance is to query both sides and combine when a range spans the
    boundary. A 15-minute market almost never straddles it, but "almost never" is
    the wrong thing to build on, and the merge costs nothing when only one side
    answered. First writer wins per timestamp; the two endpoints serve identical
    values under different key names, verified on the same market.
    """
    out: dict[int, dict] = {}
    for payload in payloads:
        for candle in (payload or {}).get('candlesticks', []) or []:
            key = int(candle.get('end_period_ts', 0))
            out.setdefault(key, candle)
    return [out[k] for k in sorted(out)]


async def enumerate_settled(client, series: str) -> list[dict]:
    """Every settled market for a series. Cursor to exhaustion."""
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
        logger.info('resuming: %d markets already pulled', len(done))

    throttle = Throttle(args.rate)
    written = total = failed = straddled = 0
    # No `live=True`: this client is structurally incapable of placing an order.
    async with KalshiClient(key_id=key_id, private_key_pem=pem) as client:
        cutoff = await fetch_cutoff(client)
        for series, symbol in SERIES.items():
            markets = await enumerate_settled(client, series)
            markets.sort(key=lambda m: m['open_time'])
            todo = [m for m in markets if m['ticker'] not in done]
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
                # Route by the cutoff, as the venue advises, then fall back to the
                # other side. Preferred first so the common case is one request.
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
                    candles, symbol=symbol, ticker=ticker, open_time=open_time))
                done.add(ticker)
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
    print('next: python -m scripts.quote_coverage')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=None, help='defaults to $RESEARCH_STORE')
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE,
                        help='requests per second (Basic tier ceiling is 20)')
    parser.add_argument('--batch-rows', type=int, default=4000)
    parser.add_argument('--checkpoint', default='/app/data/quote_backfill.json')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if not args.verbose else logging.DEBUG,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
