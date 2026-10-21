"""The venue's OWN strike for every market it still serves.

**What this fixes.** `core/windows.py` builds the strike as the mean of
Coinbase OHLC over the minute before the window opens, standing in for CF
Benchmarks' 60-second mean -- the number the market actually settles against.
Measured 2026-10-21 across 600 markets, the two differ by:

```
symbol      n     mean    median|.|   p90|.|      (basis points)
BTC-USD   200   -0.29       0.92       2.17
ETH-USD   200   -0.28       1.49       3.52
SOL-USD   200   -1.06       1.58       3.54
ALL       600   -0.54       1.33       3.23
```

Mostly random rather than a bias. That is negligible against a price and large
against SIGMA: CLAUDE.md measures ~8bp of remaining sigma at three minutes
left, so a 1.33bp strike error is ~17% of it. `z_score = (price - strike)/sigma`
is the feature the barrier framing is built on -- "the displacement x is known
exactly" is only true once the strike is the venue's. The error bites hardest
at offsets 9 and 12, where sigma is smallest and the window is nearly decided.

**Coverage is the good news.** `/historical/markets` carries `floor_strike` on
**99.9%** of rows (11,984 of 12,000 sampled) and reached back to at least
2026-04-15 without exhausting. That is four months further than candlesticks,
which do not exist before 2026-08-13.

`expiration_value` is stored alongside. It adds magnitude, not sign -- the
LABEL already comes from `result` via `--label-source venue`, which covers
95.9% of book-era windows. It is worth having for the near-tie diagnostics
`research/validate/_validate_label.py` does by hand today.

Writes into `venue_settlements`, which is already keyed on
`(market_ticker, window_open)`. Idempotent: a re-run merges on the event key.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import sys

logger = logging.getLogger('strikes')

SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}
#: Shares a 200 token/s read budget with the live trader. A listing page is one
#: request for 200 markets, so even this is quick.
DEFAULT_RATE = 5.0


def _number(value):
    if value in (None, ''):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_from_markets(markets, symbol: str, *, fetched_at):
    """`venue_settlements` rows carrying the venue's strike and settlement."""
    from scripts.backfill_candles import window_from_ticker

    out = []
    for m in markets or []:
        ticker = str(m.get('ticker') or '')
        window = window_from_ticker(ticker)
        strike = _number(m.get('floor_strike'))
        if window is None or strike is None:
            continue
        opened, close = window
        result = (str(m.get('result')).strip().lower()
                  if m.get('result') not in (None, '') else None)
        out.append({
            'venue': 'kalshi', 'symbol': symbol,
            # The settlement instant, which is what this row is ABOUT. Keeping
            # it equal to the close means a re-run merges onto the row
            # `collect_settlements` already wrote rather than shadowing it.
            'event_time': close, 'available_time': fetched_at,
            'quality': 'valid', 'market_ticker': ticker,
            'window_open': opened, 'close_time': close,
            'settlement_time': close,
            'result': result,
            'settled_up': (None if result not in ('yes', 'no')
                           else result == 'yes'),
            'volume': _number(m.get('volume')),
            'open_interest': _number(m.get('open_interest')),
            'last_price': _number(m.get('last_price_dollars')
                                  or m.get('last_price')),
            'floor_strike': strike,
            'expiration_value': _number(m.get('expiration_value')),
        })
    return out


async def run(args) -> int:
    import pandas as pd
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore()
    fetched_at = dt.datetime.now(dt.timezone.utc)
    gap = 1.0 / max(0.1, float(args.rate))
    client = KalshiClient()                 # not live=True: cannot place orders
    total = pages_done = 0

    async with client:
        for series, symbol in SERIES.items():
            cursor, pages, batch, kept = None, 0, [], 0
            while pages < args.max_pages:
                query = {'series_ticker': series, 'limit': 200}
                if cursor:
                    query['cursor'] = cursor
                try:
                    payload = await client._request(
                        'GET', '/historical/markets', params=query,
                        base=client.historical_base_url)
                except Exception as exc:                # noqa: BLE001
                    logger.warning('%s page %d failed: %s', series, pages,
                                   str(exc)[:160])
                    break
                markets = payload.get('markets') or []
                batch.extend(rows_from_markets(markets, symbol,
                                               fetched_at=fetched_at))
                cursor = payload.get('cursor')
                pages += 1
                pages_done += 1
                await asyncio.sleep(gap)
                if len(batch) >= args.batch:
                    kept += len(batch)
                    if not args.dry_run:
                        store.write('venue_settlements', pd.DataFrame(batch))
                    batch = []
                if not cursor or not markets:
                    break
            if batch:
                kept += len(batch)
                if not args.dry_run:
                    store.write('venue_settlements', pd.DataFrame(batch))
            total += kept
            logger.info('%s: %d page(s), %d row(s) with a strike', series, pages, kept)

    logger.info('%d page(s), %d row(s) %s', pages_done, total,
                'ready (dry run, nothing written)' if args.dry_run else 'written')
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-pages', type=int, default=400,
                        help='listing pages per series (200 markets each)')
    parser.add_argument('--rate', type=float, default=DEFAULT_RATE)
    parser.add_argument('--batch', type=int, default=4000)
    return parser


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    sys.exit(main())
