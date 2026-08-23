"""Scrape one-minute Coinbase spot bars for the three traded underlyings.

**One-minute bars are the whole dataset.** A 15-minute window has fifteen of
them, four decision offsets read the closes inside it, and the strike and
settle prices are the opens on its boundaries. Nothing coarser can express the
problem, and nothing finer is needed: the venue settles on a price at a clock
time, and a minute is the resolution at which that price is unambiguous.

**Coinbase spot, not the perp.** This system trades a Kalshi binary on the
*price of Bitcoin*, so the reference is the deepest, cleanest series available
for that price. Spot bars are also materially cleaner than the CDE nano perps
this repo used to trade — a median close-to-open gap of 1.1bp against 14.2bp,
and no flat-OHLC bars at all — and the nano perp's own index is built from this
book anyway.

**The size of the ask, stated plainly.** Five years is 2.63 million minutes per
symbol. At 300 bars a request that is about 8,800 requests per symbol, 26,000
in total, and a few hours of wall clock. It is resumable: the pipeline fetches
only what is missing at either end, so an interrupted run continues rather than
restarting. Run it once, in the background, and check the coverage report
afterwards.

    python -m scripts.scrape --backfill-days 1825
    python -m scripts.sync_store
    python -m scripts.baseline

**`--backfill-days` on a populated store now actually fetches deeper history.**
It did not until 2026-08-22: the backfill set its start to the newest stored bar
whenever anything was stored, discarded the requested start, and logged "already
up to date". Asking a populated 400-day store for 1,825 days fetched the missing
hour. `tests/test_backfill_windows.py` pins both directions.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

from core.config import DEFAULT_CONFIG
from data_collection.pipeline import DataPipeline, PipelineConfig
from data_collection.timeutil import ensure_naive_utc, utc_now

logger = logging.getLogger('scrape')

# The research store keys on venue, and this label must not collide with
# anything else. It is a different series from the CDE perps the old `coinbase`
# label holds, and storing them together made a cross-venue basis a comparison
# between an instrument and itself.
VENUE_LABEL = 'coinbase_spot'
TIMEFRAME = '1m'


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Scrape one-minute Coinbase spot bars into SQLite.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split('**The size of the ask')[1] if '**The size' in __doc__ else '',
    )
    parser.add_argument('--symbols', type=str, default=','.join(DEFAULT_CONFIG.symbols),
                        help='Comma-separated Coinbase spot product ids')
    parser.add_argument('--backfill-days', type=float, default=1825,
                        help='Days of history to fetch (default 1825 = five years). '
                             'On a populated store this fetches only what is missing '
                             'at either end.')
    parser.add_argument('--backfill-hours', type=float, default=None,
                        help='Overrides --backfill-days. The incremental cycle wants '
                             'hours, not a day rounded up.')
    parser.add_argument('--start', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--end', type=str, default=None, help='YYYY-MM-DD')
    parser.add_argument('--db-path', type=str, default='./data/trading.db')
    parser.add_argument('--venue-label', type=str, default=VENUE_LABEL)
    parser.add_argument('--fill-gaps', action='store_true',
                        help='Find minutes the store is missing and re-request '
                             'only those. Cheap — a handful of requests — and the '
                             'right tool after a backfill logs a skipped window: '
                             'those are usually a rate limit rather than absent '
                             'data, and a full re-scrape to recover ten hours is '
                             'not proportionate.')
    parser.add_argument('--min-gap-minutes', type=int, default=2,
                        help='Ignore gaps shorter than this. A single missing '
                             'minute is usually a minute in which nothing traded, '
                             'which no amount of re-requesting will produce.')
    parser.add_argument('--live', action='store_true',
                        help='After backfilling, keep polling for new bars. The '
                             'orchestrator uses this; a one-off backfill does not.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def find_gaps(db_path: str, symbol: str, timeframe: str, venue: str,
              *, min_minutes: int) -> list[tuple[datetime, datetime]]:
    """Runs of consecutive missing minutes in the stored history.

    Read straight from SQLite rather than the research store, because this runs
    before the sync and the scraper's table is the one with the hole in it.

    Single missing minutes are excluded by default: a minute in which nothing
    traded has no candle to fetch, and asking for it again produces the same
    nothing. What this is for is the multi-hour block that appears when a batch
    request failed — those come back on a second ask.
    """
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        rows = connection.execute(
            'SELECT event_time FROM ohlcv WHERE symbol = ? AND timeframe = ? '
            'AND venue = ? ORDER BY event_time',
            (symbol, timeframe, venue),
        ).fetchall()
    except sqlite3.OperationalError as exc:
        logger.error('cannot read %s: %s', db_path, exc)
        return []
    finally:
        connection.close()

    if len(rows) < 2:
        return []

    step = timedelta(minutes=1)
    gaps: list[tuple[datetime, datetime]] = []
    previous = ensure_naive_utc(_as_datetime(rows[0][0]))
    for (raw,) in rows[1:]:
        current = ensure_naive_utc(_as_datetime(raw))
        missing = int((current - previous).total_seconds() // 60) - 1
        if missing >= min_minutes:
            gaps.append((previous + step, current))
        previous = current
    return gaps


def _as_datetime(value) -> datetime:
    if isinstance(value, datetime):
        return value
    text = str(value).replace('Z', '+00:00')
    return datetime.fromisoformat(text)


async def fill_gaps(pipeline: DataPipeline, symbols: list[str], db_path: str,
                    venue: str, *, min_minutes: int) -> int:
    """Re-request every gap, then report what came back."""
    recovered = 0
    for symbol in symbols:
        gaps = find_gaps(db_path, symbol, TIMEFRAME, venue, min_minutes=min_minutes)
        if not gaps:
            logger.info('%s: no gaps of %d+ minutes', symbol, min_minutes)
            continue
        total = sum(int((b - a).total_seconds() // 60) for a, b in gaps)
        logger.info('%s: %d gap(s) totalling %d minutes', symbol, len(gaps), total)
        for start, end in gaps:
            span = int((end - start).total_seconds() // 60)
            logger.info('  refetching %s..%s (%d minutes)',
                        f'{start:%Y-%m-%d %H:%M}', f'{end:%Y-%m-%d %H:%M}', span)
            before = _stored_count(db_path, symbol, venue)
            # A minute either side, so a boundary bar is not missed by an
            # off-by-one in the range the venue considers inclusive.
            await pipeline.backfill(
                start=start - timedelta(minutes=1),
                end=end + timedelta(minutes=1),
                symbols=[symbol], timeframes=[TIMEFRAME])
            gained = _stored_count(db_path, symbol, venue) - before
            recovered += gained
            if gained == 0:
                logger.info('    nothing returned — the venue has no data here')
            else:
                logger.info('    recovered %d bars', gained)
    return recovered


def _stored_count(db_path: str, symbol: str, venue: str) -> int:
    import sqlite3

    connection = sqlite3.connect(db_path)
    try:
        return int(connection.execute(
            'SELECT COUNT(*) FROM ohlcv WHERE symbol = ? AND timeframe = ? '
            'AND venue = ?', (symbol, TIMEFRAME, venue)).fetchone()[0])
    except sqlite3.OperationalError:
        return 0
    finally:
        connection.close()


async def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-20s %(message)s', datefmt='%H:%M:%S',
        stream=sys.stdout)

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    if args.start:
        start = datetime.strptime(args.start, '%Y-%m-%d')
        end = datetime.strptime(args.end, '%Y-%m-%d') if args.end else utc_now()
    else:
        end = utc_now()
        span = (timedelta(hours=args.backfill_hours) if args.backfill_hours
                else timedelta(days=args.backfill_days))
        start = end - span
    start, end = ensure_naive_utc(start), ensure_naive_utc(end)

    minutes = (end - start).total_seconds() / 60.0
    print('=' * 78)
    print('Coinbase spot, one-minute bars')
    print('=' * 78)
    print(f'symbols       {", ".join(symbols)}')
    print(f'span          {start:%Y-%m-%d %H:%M} .. {end:%Y-%m-%d %H:%M} '
          f'({minutes / 1440:,.0f} days, {minutes:,.0f} minutes per symbol)')
    print(f'requests      ~{minutes / 300 * len(symbols):,.0f} at 300 bars each, '
          f'if the store is empty')
    print(f'venue label   {args.venue_label}')
    print(f'database      {args.db_path}')
    print()

    config = PipelineConfig(
        symbols=symbols,
        timeframes=[TIMEFRAME],
        coinbase_api_key=os.environ.get('COINBASE_API_KEY'),
        coinbase_api_secret=os.environ.get('COINBASE_API_SECRET'),
        db_path=args.db_path,
        venue_label=args.venue_label,
        # Spot products have no funding and no open interest. The poller would
        # ask anyway, get nothing, and log a failure per symbol per cycle.
        enable_funding_polling=False,
    )
    pipeline = DataPipeline(config)
    failures: list[str] = []
    try:
        await pipeline.initialize()
        if args.fill_gaps:
            recovered = await fill_gaps(
                pipeline, symbols, args.db_path, args.venue_label,
                min_minutes=args.min_gap_minutes)
            print(f'\nrecovered {recovered:,} bars')
            print('next: python -m scripts.sync_store')
            return 0
        await pipeline.backfill(start=start, end=end, symbols=symbols,
                                timeframes=[TIMEFRAME])
        if args.live:
            logger.info('backfill complete; polling for new bars (ctrl-c to stop)')
            await pipeline.start()
            while True:
                await asyncio.sleep(3600)
    except KeyboardInterrupt:
        logger.info('interrupted; stored bars are kept and the next run resumes')
    except Exception as exc:  # noqa: BLE001 - collected so one symbol cannot abort the rest
        logger.exception('scrape failed: %s', exc)
        failures.append(str(exc))
    finally:
        await pipeline.stop()

    summary = pipeline.get_quality_summary()
    if summary:
        logger.info('quality: %s', summary)
    print('\nnext: python -m scripts.sync_store')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(asyncio.run(main()))
