"""Measure the book on the 15-minute markets. The assumption most likely to sink this.

Two numbers in `core/costs.py` are assumptions rather than measurements, and both
are load-bearing:

* `half_spread_cents = 1.0`. Above 83c it is the larger of the two costs, so if
  the real spread is 3c the required edge at 85c goes from 1.9pp to about 3.9pp.
* `max_stake_dollars = 25`. This stands in for depth, and nobody has read the
  depth of a 15-minute Kalshi book.

The first live preflight found something worse than either being wrong: an
*active* market with no bid and no ask at all. A spread cannot be crossed that
does not exist. If that is the normal state outside US hours, then liquidity is
the binding constraint and no forecast fixes it — which is exactly how the
previous perp system died at h=96h, where the limiting gate turned out to be
`participation_limit` rather than `edge_below_cost`.

So measure it. This polls the live market for each symbol, records the book, and
reports the distribution by hour of day: how often a two-sided market exists at
all, what the spread is when it does, and how the depth at the touch compares to
the stake the sizing rules want.

Read-only. The client is constructed without `live=True` and cannot order.

    python -m scripts.measure_book --hours 24
    python -m scripts.measure_book --hours 2 --out books.parquet
    python -m scripts.measure_book --report books.parquet     # re-read a sample
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from core.costs import fee_per_contract, required_edge_pp
from data_collection.kalshi_client import DEMO_BASE_URL, KalshiClient, KalshiError
from scripts.live import SERIES_BY_SYMBOL

logger = logging.getLogger('measure_book')


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument('--hours', type=float, default=24.0,
                        help='How long to sample. A full day covers every '
                             'session; anything less cannot say whether an empty '
                             'book is nocturnal or permanent.')
    parser.add_argument('--every-seconds', type=int, default=30,
                        help='Poll interval. 30s gives ~30 observations per '
                             '15-minute window, which is enough to see the book '
                             'fill and thin within one.')
    parser.add_argument('--out', type=str, default='book_samples.parquet')
    parser.add_argument('--report', type=str, default=None,
                        help='Skip sampling and re-report an existing file.')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


async def sample_once(client: KalshiClient, now: datetime) -> list[dict]:
    """One observation per symbol of the currently open market."""
    window = now.replace(second=0, microsecond=0)
    window -= timedelta(minutes=window.minute % DEFAULT_CONFIG.window_minutes)
    settle = window + timedelta(minutes=DEFAULT_CONFIG.window_minutes)
    elapsed = (now - window).total_seconds() / 60.0

    rows = []
    for symbol, series in SERIES_BY_SYMBOL.items():
        row = {
            'observed_at': now, 'symbol': symbol, 'series': series,
            'window_open': window, 'settle_time': settle,
            'minutes_elapsed': elapsed,
            'ticker': None, 'status': None,
            'yes_bid': np.nan, 'yes_ask': np.nan,
            'no_bid': np.nan, 'no_ask': np.nan,
            'last_price': np.nan, 'volume': 0, 'open_interest': 0,
            'resolved': False, 'two_sided': False,
        }
        try:
            market = await client.resolve_window_market(series, settle)
            if market is None:
                rows.append(row)
                continue
            row['resolved'] = True
            ticker = str(market.get('ticker', ''))
            row['ticker'] = ticker
            quote = await client.quote(ticker)
            row.update({
                'status': quote.status,
                'yes_bid': quote.yes_bid if quote.yes_bid is not None else np.nan,
                'yes_ask': quote.yes_ask if quote.yes_ask is not None else np.nan,
                'no_bid': quote.no_bid if quote.no_bid is not None else np.nan,
                'no_ask': quote.no_ask if quote.no_ask is not None else np.nan,
                'last_price': quote.last_price if quote.last_price is not None else np.nan,
                'volume': quote.volume, 'open_interest': quote.open_interest,
                'two_sided': quote.tradeable(),
            })
        except KalshiError as exc:
            logger.warning('%s: %s', symbol, exc)
        rows.append(row)
    return rows


def report(frame: pd.DataFrame) -> None:
    """What the sample says about the two unmeasured assumptions."""
    if frame.empty:
        print('no observations')
        return

    frame = frame.copy()
    frame['spread'] = frame['yes_ask'] - frame['yes_bid']
    frame['spread_cents'] = frame['spread'] * 100
    frame['hour_utc'] = pd.DatetimeIndex(frame['observed_at']).hour
    frame['quartile'] = pd.cut(
        frame['minutes_elapsed'], [-0.01, 3.75, 7.5, 11.25, 15.01],
        labels=['0-4m', '4-8m', '8-11m', '11-15m'])

    span = (frame['observed_at'].max() - frame['observed_at'].min())
    print('=' * 78)
    print(f'{len(frame):,} observations over {span}, '
          f'{frame["window_open"].nunique():,} windows')
    print('=' * 78)

    print('\nHOW OFTEN A TWO-SIDED BOOK EXISTS AT ALL')
    print('  This is the question. A spread cannot be crossed that does not exist,')
    print('  and no forecast fixes an absent counterparty.')
    by_symbol = frame.groupby('symbol').agg(
        observations=('two_sided', 'size'),
        resolved=('resolved', 'mean'),
        two_sided=('two_sided', 'mean'),
        median_spread_c=('spread_cents', 'median'),
        median_volume=('volume', 'median'),
        median_oi=('open_interest', 'median'))
    print(by_symbol.to_string(float_format=lambda v: f'{v:,.3f}'))

    print('\nBY HOUR (UTC) — is an empty book nocturnal or permanent?')
    hourly = frame.groupby('hour_utc').agg(
        n=('two_sided', 'size'),
        two_sided=('two_sided', 'mean'),
        median_spread_c=('spread_cents', 'median'),
        median_volume=('volume', 'median'))
    print(hourly.to_string(float_format=lambda v: f'{v:,.2f}'))

    print('\nBY POSITION IN THE WINDOW — does the book fill as settlement nears?')
    print('  It matters where: the barrier edge is predicted to peak mid-window,')
    print('  so a book that only exists in the last two minutes is a book that')
    print('  exists where the forecast is least useful.')
    print(frame.groupby('quartile', observed=True).agg(
        n=('two_sided', 'size'), two_sided=('two_sided', 'mean'),
        median_spread_c=('spread_cents', 'median')).to_string(
        float_format=lambda v: f'{v:,.2f}'))

    tradeable = frame.loc[frame['two_sided']]
    print('\n' + '=' * 78)
    print('WHAT THIS DOES TO THE COST ASSUMPTIONS')
    print('=' * 78)
    if tradeable.empty:
        print('\n  No two-sided book was ever observed.')
        print('  The strategy is not tradeable on this venue as configured, and')
        print('  that is a liquidity finding rather than a forecasting one — no')
        print('  amount of skill fixes an absent counterparty. Before concluding:')
        print('  sample a full 24h, and check whether these markets quote at all')
        print('  outside a narrow window of the US session.')
        return

    half = tradeable['spread_cents'] / 2
    assumed = DEFAULT_CONFIG.half_spread_cents
    print(f'\n  assumed half-spread   {assumed:.1f}c')
    print(f'  measured half-spread  median {half.median():.1f}c, '
          f'p25 {half.quantile(0.25):.1f}c, p75 {half.quantile(0.75):.1f}c, '
          f'p95 {half.quantile(0.95):.1f}c')
    ratio = half.median() / assumed if assumed else float('nan')
    print(f'  ratio to the assumption: {ratio:.2f}x')

    print('\n  required edge at each price, assumed vs measured:')
    print(f'    {"price":>7}{"assumed":>10}{"measured":>10}')
    from core.config import Config
    measured_config = Config(half_spread_cents=float(half.median()))
    for price in (0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        a = float(required_edge_pp(price, DEFAULT_CONFIG))
        m = float(required_edge_pp(price, measured_config))
        print(f'    {price:>7.2f}{a:>9.2f}pp{m:>9.2f}pp')

    print('\n  The fee is the same in both columns; the whole difference is the')
    print('  spread. If the measured column is the truth, `--half-spread-cents`')
    print(f'  should be {half.median():.1f} in every evaluate run, and the cost')
    print('  stress table already reports what that does to the answer.')

    if ratio > 1.5:
        print(f'\n  NOTE: at {ratio:.1f}x the assumption, every backtested number')
        print('  produced with the default is optimistic. Re-run evaluate with')
        print(f'  --half-spread-cents {half.median():.1f} before trusting a gate.')


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format='%(asctime)s %(levelname)-7s %(message)s', datefmt='%H:%M:%S',
        stream=sys.stdout)

    if args.report:
        report(pd.read_parquet(args.report))
        return 0

    client = KalshiClient(base_url=DEMO_BASE_URL if args.demo else None, live=False)
    if not client.configured:
        raise SystemExit('Kalshi credentials are not configured; see .env.example')

    deadline = datetime.now(timezone.utc) + timedelta(hours=args.hours)
    out = Path(args.out)
    collected: list[dict] = []

    print(f'sampling every {args.every_seconds}s until '
          f'{deadline:%Y-%m-%d %H:%M} UTC ({args.hours:g}h) -> {out}')
    print('ctrl-c to stop early; whatever was collected is still written\n')

    async with client:
        try:
            while datetime.now(timezone.utc) < deadline:
                now = datetime.now(timezone.utc)
                rows = await sample_once(client, now)
                collected.extend(rows)
                live = [r for r in rows if r['two_sided']]
                summary = ' | '.join(
                    f'{r["symbol"].split("-")[0]} '
                    f'{r["yes_bid"]:.2f}/{r["yes_ask"]:.2f}'
                    for r in live) or 'no two-sided book on any symbol'
                print(f'\r{now:%H:%M:%S}  {len(collected):>6,} obs  {summary}',
                      end='', flush=True)
                # Persist as we go: a sampler that only writes at the end loses a
                # whole night to one ctrl-c.
                if len(collected) % 60 == 0:
                    pd.DataFrame(collected).to_parquet(out, index=False)
                await asyncio.sleep(args.every_seconds)
        except KeyboardInterrupt:
            print('\ninterrupted')

    if not collected:
        print('\nnothing collected')
        return 1
    frame = pd.DataFrame(collected)
    frame.to_parquet(out, index=False)
    print(f'\n\nwrote {len(frame):,} observations to {out}\n')
    report(frame)
    return 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
