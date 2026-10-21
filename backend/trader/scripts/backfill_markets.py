"""The venue's OWN strike and settlement value, for every 15-minute market.

**Tickers are CONSTRUCTED, not enumerated, and that is the whole unlock.**
`/markets?status=settled` and `/historical/markets` only list about 69 days
back, which is why `backfill_candles` bottomed out at 2026-08-13 and why this
project believed Kalshi history could not be extended. But the per-ticker
endpoints serve far older markets — the limit was the LISTING, never the data.
A ticker is `series + close in New York local + minute suffix`, fully
determined by the window, so we can ask for any market that ever existed.

Measured 2026-10-21 by bisection against the live venue:

```
KXBTC15M  first market ~2025-12-16    expiration_value from ~2026-01-14
KXETH15M  first market ~2025-12-16    expiration_value from ~2026-01-14
KXSOL15M  first market ~2025-12-31    expiration_value from ~2026-01-14
```

**What this fixes.**

* `floor_strike` is the number the market actually settles against — the
  60-second mean of CF Benchmarks' BRTI at the window open. `core/windows.py`
  stands a Coinbase OHLC mean in for it, and measured over 300 windows that
  proxy is off by a median 0.55bp, p90 1.40bp. At +12m against ~8bp of
  remaining sigma that is ~2.7pp of probability error on `z_score`, the feature
  the whole barrier framing rests on, against a target edge of 1.5pp.
  Reconstructing it from BRTI bars does NOT help — measured, a BRTI OHLC mean
  is 0.73bp off against Coinbase's 0.81bp, because the error is the OHLC-mean
  approximation of a time-average, not the price source. Only the venue's own
  number is exact.
* `expiration_value` is the venue's own settlement. It removes the standing
  "~3% of every training label is wrong" caveat outright, for every window from
  2026-01-14, rather than reducing it.

Read-only. Writes `venue_strikes`; nothing here places or cancels anything.

    python -m scripts.backfill_markets --start 2026-01-14 --dry-run
    python -m scripts.backfill_markets --start 2026-01-14 --concurrency 8
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger('backfill_markets')

NY = ZoneInfo('America/New_York')
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
WINDOW_MINUTES = 15
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}


def ticker_for(series: str, close_utc: dt.datetime) -> str:
    """`KXBTC15M-26OCT202245-45` — series, CLOSE in New York local, minute.

    The close, not the open: verified against `window_from_ticker`, which this
    is the inverse of. Getting it backwards shifts every window by fifteen
    minutes and nothing raises, which is the Polymarket-slug trap wearing a
    different hat.
    """
    local = close_utc.astimezone(NY)
    return '%s-%02d%s%02d%02d%02d-%02d' % (
        series, local.year % 100, MONTHS[local.month - 1], local.day,
        local.hour, local.minute, local.minute)


def _money(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_for(market: dict, symbol: str, window_open, close_utc, fetched_at):
    """One `venue_strikes` row, or None when the venue returned nothing usable."""
    floor = _money(market.get('floor_strike'))
    expiry = _money(market.get('expiration_value'))
    result = market.get('result') or None
    if floor is None and expiry is None and not result:
        return None
    return {
        'venue': 'kalshi', 'symbol': symbol,
        # Keyed on the window OPEN, like every other per-window dataset here.
        'event_time': window_open, 'available_time': fetched_at,
        'quality': 'valid',
        'market_ticker': market.get('ticker'),
        'window_open': window_open, 'close_time': close_utc,
        'floor_strike': floor, 'expiration_value': expiry, 'result': result,
    }


async def fetch_one(client, sem, series, symbol, window_open, fetched_at):
    close = window_open + dt.timedelta(minutes=WINDOW_MINUTES)
    tk = ticker_for(series, close)
    async with sem:
        # **BOTH tiers, and this is not optional.** The live route refuses
        # anything past a moving ~3-month cutoff and the historical host
        # refuses anything inside it, so each serves exactly what the other
        # will not. Asking only the historical host returned 0 rows from 2,220
        # requests on a recent range -- every one a 404 that looked like "this
        # market never existed" rather than "wrong host". Ordered newest-first
        # because the backfill walks forward and most calls are recent.
        for base in (None, client.historical_base_url):
            path = ('/markets/%s' % tk if base is None
                    else '/historical/markets/%s' % tk)
            for attempt in range(5):
                try:
                    payload = await client._request('GET', path, base=base)  # noqa: SLF001
                except Exception as exc:                  # noqa: BLE001
                    msg = str(exc)
                    if '429' in msg or 'too_many_requests' in msg:
                        await asyncio.sleep(1.5 * (2 ** attempt))
                        continue
                    # A 404 here means "not on THIS tier"; try the other one
                    # before concluding the market never existed.
                    break
                market = payload.get('market') or payload
                return row_for(market, symbol, window_open, close, fetched_at)
    return None


async def run(args) -> int:
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore(args.research_store)
    start = pd.Timestamp(args.start, tz='UTC').floor('15min')
    end = (pd.Timestamp(args.end, tz='UTC') if args.end
           else pd.Timestamp.now(tz='UTC')).floor('15min')
    wanted = {s: sym for s, sym in SERIES.items()
              if not args.symbols or sym in args.symbols}

    done: set[tuple] = set()
    if not args.force:
        try:
            held = store.read('venue_strikes', min_quality=None)
            if len(held):
                held['window_open'] = pd.to_datetime(held['window_open'], utc=True)
                done = set(zip(held['symbol'].astype(str), held['window_open']))
        except Exception as exc:                          # noqa: BLE001
            logger.info('nothing stored yet (%s)', str(exc)[:90])
    logger.info('%d window(s) already held', len(done))

    windows = pd.date_range(start, end, freq='%dmin' % WINDOW_MINUTES, inclusive='left')
    jobs = [(s, sym, w) for s, sym in wanted.items() for w in windows
            if (sym, w) not in done]
    logger.info('%d window(s) x %d series -> %d request(s)',
                len(windows), len(wanted), len(jobs))
    if args.dry_run or not jobs:
        return 0

    fetched_at = dt.datetime.now(dt.timezone.utc)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    written = batch = 0
    rows: list[dict] = []
    async with KalshiClient(live=False) as client:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            got = await asyncio.gather(*[
                fetch_one(client, sem, s, sym, w, fetched_at) for s, sym, w in chunk])
            rows = [r for r in got if r]
            if rows:
                written += store.write('venue_strikes', pd.DataFrame(rows))
            batch += len(chunk)
            logger.info('  %d/%d asked, %d row(s) written', batch, len(jobs), written)
    logger.info('done: %d row(s) in venue_strikes', written)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--start', default='2026-01-14')
    p.add_argument('--end', default=None)
    p.add_argument('--symbols', nargs='*', default=None)
    p.add_argument('--concurrency', type=int, default=8)
    p.add_argument('--batch', type=int, default=2000)
    p.add_argument('--force', action='store_true')
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
