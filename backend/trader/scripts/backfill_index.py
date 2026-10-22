"""CF Benchmarks' own index into `minute_bars`, under its own venue.

**Why.** The markets settle on CF Benchmarks' BRTI, and the market prices on
it. We compute `displacement`, `sigma_remaining` and 35 of the deployed
artifact's 55 features from COINBASE bars instead. Measured 2026-10-21 over 400
consecutive BTC windows, the Coinbase-vs-BRTI strike basis has sd 0.96bp and an
autocorrelation at 15 minutes of **+0.094** — so it is white noise at the
window scale and does NOT cancel between the strike and the decision, contrary
to what `CLAUDE.md` assumed. About 1.29bp of independent noise reaches
displacement, which at +12m against ~8bp of remaining sigma is ~6pp of
probability error, against a target edge of 1.5pp. The signal we are hunting is
smaller than the noise in its own input.

**The design is deliberately boring: same schema, different venue.**
`load_minute_bars` reads `minute_bars` with `venue=config.venue`, so writing
these as `venue='cfbenchmarks'` makes the WHOLE pipeline — windows, baseline,
every price-derived feature — run on the index by changing one config value,
with no code change and a clean A/B against Coinbase.

**Volume is carried across from Coinbase, and that is not a fudge.** An index
has no volume; it is a median of constituent quotes. But `microstructure` reads
`volume_z_15`, `vwap_gap_15` and `signed_volume_15`, and under
`--complete-cases` a null volume column would drop EVERY row and make the
comparison impossible. Volume is also the one quantity the basis cannot
distort: "how much traded on Coinbase" is the same question whichever index
prices it. The PRICES here are the venue's; only the activity columns are
Coinbase's, and `venue` says so.

    python -m scripts.backfill_index --start 2026-01-01 --dry-run
    python -m scripts.backfill_index --start 2026-01-01
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger('backfill_index')

#: The venue these bars are written under. NOT `coinbase_spot`: the whole point
#: is that a run can be pointed at one or the other and compared.
VENUE = 'cfbenchmarks'

#: CF Benchmarks index id per symbol, verified against the live venue.
INDEX_BY_SYMBOL = {
    'BTC-USD': 'BRTI',
    'ETH-USD': 'ETHUSD_RTI',
    'SOL-USD': 'SOLUSD_RTI',
}

#: `/cfbenchmarks/*` costs 50 tokens against a 200/s refill, so ~4 req/s.
#: The token bucket refills at 300/s and is SHARED across endpoints, priced
#: per request: `/cfbenchmarks/*` costs 50 (so 6/s is the ceiling here) while
#: the default is 10 (30/s). A cheap high-rate consumer therefore starves this
#: one, because it needs 50 tokens at once -- measured, `/markets` returned 200
#: at the same instant `/cfbenchmarks` returned 429. Run this alone.
DEFAULT_RATE = 5.0

#: Hours abandoned after exhausting retries. Non-zero means the index is
#: thinner than the venue served, and the run should be repeated.
DROPPED: dict = collections.defaultdict(int)


def minute_bars_from_points(points, symbol: str) -> list[dict]:
    """1-second index values -> one-minute OHLC.

    `time` is epoch MILLISECONDS. The bar is keyed on its START minute, which
    is the convention `core/windows.py` reads: the strike is the mean over
    `[t0 - 1min, t0)`, so a bar stamped at its end would shift every window by
    a minute and nothing would raise.
    """
    rows = []
    for p in points or []:
        try:
            ts = int(p['time']) // 1000
            v = float(p['value'])
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((ts - ts % 60, v))
    if not rows:
        return []
    f = pd.DataFrame(rows, columns=['minute', 'value'])
    g = f.groupby('minute')['value']
    # `mean_price` is the exact quantity the market settles on: the average
    # of the index values inside the minute. `core/windows.bar_mean` has to
    # approximate it as `(O+H+L+C)/4` for OHLC-only sources; here the prints
    # are available, so the real mean is stored instead of re-derived.
    out = pd.DataFrame({'open': g.first(), 'high': g.max(),
                        'low': g.min(), 'close': g.last(),
                        'mean_price': g.mean()}).reset_index()
    out['event_time'] = pd.to_datetime(out['minute'], unit='s', utc=True)
    out['venue'] = VENUE
    out['symbol'] = symbol
    out['quality'] = 'valid'
    return out.drop(columns=['minute']).to_dict('records')


def attach_coinbase_activity(frame: pd.DataFrame, store, symbol: str):
    """Carry `volume`, `quote_volume` and `trade_count` over from Coinbase.

    See the module docstring: an index has no volume, and leaving it null drops
    every row under `--complete-cases`.
    """
    if frame.empty:
        return frame
    lo, hi = frame['event_time'].min(), frame['event_time'].max()
    try:
        cb = store.read('minute_bars', venue='coinbase_spot', symbols=[symbol],
                        start=lo, end=hi, min_quality=None)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('%s: no Coinbase activity to carry (%s)', symbol, str(exc)[:120])
        return frame
    if not len(cb):
        return frame
    cb = cb[['event_time', 'volume', 'quote_volume', 'trade_count']].copy()
    cb['event_time'] = pd.to_datetime(cb['event_time'], utc=True)
    return frame.merge(cb, on='event_time', how='left')


#: An hour of 1-second index values yields 60 minute bars. Fewer means the
#: hour is partial -- a drop, a truncated response, or a run that stopped --
#: so it is re-asked. 59 tolerates a single boundary minute.
COMPLETE_HOUR_MINUTES = 59


def held_hours(store, symbol: str) -> set:
    """Hours already complete for `symbol`, so a re-run fills gaps only.

    Without this the script re-fetched its whole range every time, which made
    a single dropped hour cost a full pass to repair -- about four hours for
    three symbols. A backfill that cannot cheaply fill its own gaps will not
    have them filled.
    """
    try:
        held = store.read('minute_bars', venue=VENUE, symbols=[symbol],
                          min_quality=None)
    except Exception as exc:                                  # noqa: BLE001
        logger.debug('%s: nothing held yet (%s)', symbol, str(exc)[:100])
        return set()
    if held is None or not len(held):
        return set()
    # **An hour missing `mean_price` is NOT complete.** Rows written before
    # that column existed pass the minute count, so without this they are
    # skipped forever and keep a null mean -- a silent hole in the exact
    # quantity the column was added to capture, and one that only a full
    # refetch could close later.
    if 'mean_price' in held.columns:
        held = held[held['mean_price'].notna()]
    at = pd.to_datetime(held['event_time'], utc=True)
    per = at.dt.floor('h').value_counts()
    return set(per[per >= COMPLETE_HOUR_MINUTES].index)


async def run(args) -> int:
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore(args.research_store)
    start = pd.Timestamp(args.start, tz='UTC').floor('h')
    end = (pd.Timestamp(args.end, tz='UTC') if args.end
           else pd.Timestamp.now(tz='UTC')).floor('h')
    symbols = args.symbols or list(INDEX_BY_SYMBOL)
    gap = 1.0 / max(0.1, float(args.rate))

    hours = pd.date_range(start, end, freq='h', inclusive='left')
    logger.info('%d hour(s) x %d symbol(s) = %d request(s), ~%.0f min at %.1f/s',
                len(hours), len(symbols), len(hours) * len(symbols),
                len(hours) * len(symbols) * gap / 60.0, args.rate)
    if args.dry_run:
        return 0

    written = 0
    async with KalshiClient(live=False) as client:
        for symbol in symbols:
            idx = INDEX_BY_SYMBOL.get(symbol)
            if idx is None:
                logger.warning('no index id for %s; skipped', symbol)
                continue
            batch: list[dict] = []
            done = set() if args.force else held_hours(store, symbol)
            todo = [h for h in hours if h not in done]
            logger.info('%s: %d hour(s) to fetch, %d already complete',
                        symbol, len(todo), len(hours) - len(todo))
            # **Concurrent, because this is LATENCY-bound, not rate-bound.**
            # One hour is one request, and a round trip to the historical host
            # is ~0.8s. Sequentially that caps throughput near 1/s -- measured,
            # 57 hours a minute against a 5/s budget the bucket never felt. A
            # small semaphore fills the gap without approaching the limit:
            # `/cfbenchmarks/*` costs 50 tokens of a 300/s refill, so 6/s is
            # the ceiling and `concurrency * (1/gap)` must stay under it.
            sem = asyncio.Semaphore(max(1, int(args.concurrency)))

            async def one(hour):
                async with sem:
                    for attempt in range(6):
                        try:
                            payload = await client._request(  # noqa: SLF001
                                'GET', '/cfbenchmarks/history/values',
                                params={'id': idx, 'timespan': 'HOUR',
                                        'timestamp': hour.strftime('%Y-%m-%dT%H:%M:%SZ')},
                                base=client.historical_base_url)
                        except Exception as exc:              # noqa: BLE001
                            msg = str(exc)
                            if ('429' in msg or 'too_many_requests' in msg) and attempt < 5:
                                await asyncio.sleep(gap * (2 ** attempt))
                                continue
                            if attempt == 5:
                                DROPPED['hours'] += 1
                                logger.warning(
                                    '%s %s DROPPED after 6 attempts (%s); %d hour(s) '
                                    'lost so far', idx, hour, msg[:90], DROPPED['hours'])
                            else:
                                logger.warning('%s %s: %s', idx, hour, msg[:140])
                            return hour, None
                        await asyncio.sleep(gap)
                        pts = (payload.get('data') or {}).get('payload') or []
                        return hour, minute_bars_from_points(pts, symbol)
                    return hour, None

            # Gathered in slices so a long run flushes as it goes rather than
            # holding every row in memory until the end.
            slice_size = max(1, int(args.concurrency) * 20)
            for i in range(0, len(todo), slice_size):
                done = await asyncio.gather(*[one(h) for h in todo[i:i + slice_size]])
                newest = None
                for hour, rows in done:
                    if rows:
                        batch.extend(rows)
                        newest = hour if newest is None else max(newest, hour)
                if len(batch) >= args.batch:
                    written += _flush(store, batch, symbol)
                    logger.info('  %s %s: %d row(s) written', symbol,
                                (newest or todo[i]).date(), written)
                    batch = []
            written += _flush(store, batch, symbol)
    logger.info('%d minute bar(s) written under venue=%r', written, VENUE)
    return 0


def _flush(store, rows, symbol) -> int:
    if not rows:
        return 0
    frame = attach_coinbase_activity(pd.DataFrame(rows), store, symbol)
    return store.write('minute_bars', frame)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--start', default='2026-01-01')
    p.add_argument('--end', default=None)
    p.add_argument('--symbols', nargs='*', default=None)
    p.add_argument('--rate', type=float, default=DEFAULT_RATE)
    p.add_argument('--concurrency', type=int, default=4,
                   help='In-flight requests. concurrency/gap must stay under the 6/s ceiling for this endpoint.')
    p.add_argument('--batch', type=int, default=20000)
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    p.add_argument('--force', action='store_true',
                   help='re-fetch hours already held')
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
