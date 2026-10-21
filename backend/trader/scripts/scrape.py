"""Scrape one-minute Coinbase spot bars for the three traded underlyings.

One-minute bars are the whole dataset: both ends of every target are
one-minute means.

**The size of the ask.** Five years is about 2.63 million minutes, or 8,800
requests, per symbol. The run is resumable: it fetches only what is missing at
either end. On a populated store, `--backfill-days` also fetches deeper
history.

    python -m scripts.scrape --backfill-days 1825
    python -m scripts.sync_store
    python -m scripts.baseline
"""

from __future__ import annotations

import argparse
from pathlib import Path
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

# The research store keys on venue. This label must not collide with the
# `coinbase` label, which holds a different series.
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
    # Anchored to the trader root, not the cwd: `sqlite3.connect` creates a
    # missing file, so a relative path would silently start a second database.
    parser.add_argument('--db-path', type=str,
                        default=str(Path(__file__).resolve().parents[1]
                                    / 'data' / 'trading.db'))
    parser.add_argument('--venue-label', type=str, default=VENUE_LABEL)
    parser.add_argument('--fill-gaps', action='store_true',
                        help='Find minutes the store is missing and re-request '
                             'only those. Cheap — a handful of requests — and the '
                             'right tool after a backfill logs a skipped window: '
                             'those are usually a rate limit rather than absent '
                             'data, and a full re-scrape to recover ten hours is '
                             'not proportionate.')
    parser.add_argument('--min-gap-minutes', type=int, default=1,
                        help='Ignore gaps shorter than this. Defaults to 1, so '
                             'isolated single minutes ARE re-requested: 86%% of '
                             'them were destroyed by a client-side pagination '
                             'off-by-one, not by an absence of trading, and they '
                             'come back on a second ask.')
    parser.add_argument('--live', action='store_true',
                        help='After backfilling, keep polling for new bars. The '
                             'orchestrator uses this; a one-off backfill does not.')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser


def find_gaps(db_path: str, symbol: str, timeframe: str, venue: str,
              *, min_minutes: int) -> list[tuple[datetime, datetime]]:
    """Return runs of consecutive missing minutes in the stored history.

    Reads SQLite, not the research store, because this runs before the sync.
    The default `min_minutes=1` includes single minutes: most are fetch
    losses, not untraded minutes, and re-requesting is cheap and idempotent.
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


def coalesce(gaps: list[tuple[datetime, datetime]], *, span_minutes: int
             ) -> list[tuple[datetime, datetime]]:
    """Merge nearby gaps into spans one request can cover."""
    if not gaps:
        return []
    merged = [gaps[0]]
    for start, end in gaps[1:]:
        first_start, _ = merged[-1]
        if int((end - first_start).total_seconds() // 60) <= span_minutes:
            merged[-1] = (first_start, end)
        else:
            merged.append((start, end))
    return merged


async def fill_gaps(pipeline: DataPipeline, symbols: list[str], db_path: str,
                    venue: str, *, min_minutes: int) -> int:
    """Re-request every gap, forcing the fetch, and report what came back.

    `force` is required: without it `backfill` resumes from the store's own
    boundaries and skips interior holes. Rows are counted once per symbol,
    not per gap, because a full count is slow.
    """
    from data_collection.coinbase_connector import MAX_CANDLES_PER_REQUEST

    recovered = 0
    for symbol in symbols:
        gaps = find_gaps(db_path, symbol, TIMEFRAME, venue, min_minutes=min_minutes)
        if not gaps:
            logger.info('%s: no gaps of %d+ minutes', symbol, min_minutes)
            continue
        total = sum(int((b - a).total_seconds() // 60) for a, b in gaps)
        # Pad a minute either side, so an inclusive-range off-by-one cannot
        # miss a boundary bar.
        padded = [(a - timedelta(minutes=1), b + timedelta(minutes=1))
                  for a, b in gaps]
        requests = coalesce(padded, span_minutes=MAX_CANDLES_PER_REQUEST - 1)
        logger.info('%s: %d gap(s) totalling %d minutes, in %d request(s)',
                    symbol, len(gaps), total, len(requests))

        before = _stored_count(db_path, symbol, venue)
        for index, (start, end) in enumerate(requests, 1):
            if index == 1 or index % 250 == 0 or index == len(requests):
                logger.info('  [%d/%d] %s..%s', index, len(requests),
                            f'{start:%Y-%m-%d %H:%M}', f'{end:%Y-%m-%d %H:%M}')
            await pipeline.backfill(start=start, end=end, symbols=[symbol],
                                    timeframes=[TIMEFRAME], force=True)
        gained = _stored_count(db_path, symbol, venue) - before
        recovered += gained
        logger.info('%s: recovered %d of %d missing minutes (%.1f%%)',
                    symbol, gained, total, 100.0 * gained / total if total else 0.0)
        if gained < total:
            logger.info('  the remainder is minutes in which nothing traded — the '
                        'venue has no candle and never will')
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
