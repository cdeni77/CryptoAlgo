"""Archive the raw Kalshi order book every minute, stored verbatim.

Read-only. One REST call per market per minute, plus a sample from the stream
cache. `/orderbook` returns an empty ladder once a market settles, and no
historical endpoint serves depth, so a minute not recorded is lost. The ladder
is stored raw, not summarised, so later features are not foreclosed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd

from core import heartbeat
from core.config import DEFAULT_CONFIG, series_to_symbol
from core.datastore import ResearchStore
from scripts.record_stream import CACHE

logger = logging.getLogger('ladder')
# The one series<->symbol mapping: core/config.SERIES_BY_SYMBOL.
SERIES = series_to_symbol()


def ws_row(cache, *, ticker, symbol, now, open_time, minute, read_at=None):
    """Return the ladder row sampled from the stream cache, or None.

    Returns None when the cache holds no book, because an empty ladder would
    record a dead subscription as an empty market. A stale book is recorded
    with its age (`book_age_ms`); the trading path refuses it instead.
    """
    ladder = cache.ladder(ticker)
    if ladder is None:
        return None
    yes = [[p, s] for p, s in ladder.yes]
    no = [[p, s] for p, s in ladder.no]
    # `available_time` is when the book was knowable: `read_at - age`, where
    # `read_at` is when the cache was consulted. `event_time` comes from `now`,
    # so this row pairs with the REST row for the same minute.
    read_at = pd.Timestamp(now if read_at is None else read_at)
    event_time = pd.Timestamp(now).floor('min')
    observed = read_at - pd.Timedelta(seconds=ladder.age_seconds)
    return {
        'venue': 'kalshi', 'symbol': symbol,
        'event_time': event_time,
        'available_time': max(observed, event_time),
        'quality': 'valid',
        'market_ticker': ticker, 'window_open': open_time,
        'minute_into_window': round(minute, 3),
        'yes_levels': json.dumps(yes), 'no_levels': json.dumps(no),
        'yes_total': sum(s for _, s in yes),
        'no_total': sum(s for _, s in no),
        'transport': 'ws',
        'book_age_ms': round(ladder.age_seconds * 1000.0, 1),
    }


def _levels(raw) -> list:
    out = []
    for entry in raw or []:
        try:
            out.append([float(entry[0]), float(entry[1])])
        except (TypeError, ValueError, IndexError):
            continue
    return out


def seconds_to_next_mark(now: datetime, period: float, phase: float) -> float:
    """Return the seconds to sleep so the next sample lands `phase` past the minute.

    A fixed sleep after each cycle makes the period `interval + work`, so the
    phase drifts and minutes go unsampled. Never returns less than a second.
    """
    period = float(period)
    phase = float(phase) % period
    elapsed = (now - now.replace(second=0, microsecond=0)).total_seconds()
    ahead = (phase - elapsed) % period
    return ahead if ahead > 1.0 else ahead + period


async def run(args, gate=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    pem = os.getenv('KALSHI_PRIVATE_KEY') or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read()
    rows: list[dict] = []

    def _flush_on_exit() -> None:
        """Write the buffer before this task is cancelled.

        `asyncio.CancelledError` is a BaseException, so `except Exception`
        never sees a shutdown. Writes synchronously, because the event loop is
        shutting down.
        """
        if not rows:
            return
        try:
            store.write('venue_ladder', pd.DataFrame(rows))
            logger.info('flushed %d buffered row(s) on shutdown', len(rows))
            rows.clear()
        except Exception as exc:                          # noqa: BLE001
            logger.error('could not flush %d buffered row(s): %s',
                         len(rows), str(exc)[:160])


    while True:
        try:
            async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                    private_key_pem=pem) as client:
                while True:
                    # Never start a cycle while a trading decision is in flight.
                    if gate is not None:
                        await gate.idle()
                    now = datetime.now(timezone.utc)
                    for series, symbol in SERIES.items():
                        payload = await client._request(  # noqa: SLF001
                            'GET', '/markets',
                            # Filter on closing time, not on status: `status`
                            # lags the market by up to forty seconds, which
                            # hides the new market early in each window. The
                            # minute check below drops anything outside it.
                            params={'series_ticker': series,
                                    'min_close_ts': int(now.timestamp()),
                                    'limit': 8})
                        for market in payload.get('markets', []):
                            if not market.get('open_time'):
                                continue
                            open_time = datetime.strptime(
                                market['open_time'], '%Y-%m-%dT%H:%M:%SZ'
                            ).replace(tzinfo=timezone.utc)
                            minute = (now - open_time).total_seconds() / 60.0
                            if not (0 <= minute <= config.window_minutes):
                                continue
                            # Take the stream sample first, so a REST failure
                            # below cannot discard it. The transports must be
                            # independent observers.
                            paired = ws_row(
                                CACHE, ticker=market['ticker'], symbol=symbol,
                                now=now, open_time=open_time, minute=minute,
                                read_at=datetime.now(timezone.utc))
                            if paired is not None:
                                rows.append(paired)
                            try:
                                book = await client._request(  # noqa: SLF001
                                    'GET', f"/markets/{market['ticker']}/orderbook")
                            except Exception as exc:      # noqa: BLE001
                                logger.warning('%s: %s', market['ticker'], str(exc)[:90])
                                continue
                            # When the response landed. `now` precedes the round
                            # trip, so it would claim the book was knowable early.
                            rest_at = datetime.now(timezone.utc)
                            ladder = (book.get('orderbook_fp')
                                      or book.get('orderbook') or {})
                            yes = _levels(ladder.get('yes_dollars') or ladder.get('yes'))
                            no = _levels(ladder.get('no_dollars') or ladder.get('no'))
                            if not yes and not no:
                                # Drops the REST half only; a run of these is
                                # a venue problem.
                                logger.debug('%s: empty REST ladder at minute '
                                             '%.1f', market['ticker'], minute)
                                continue
                            rows.append({
                                'venue': 'kalshi', 'symbol': symbol,
                                'event_time': pd.Timestamp(now).floor('min'),
                                'available_time': max(
                                    pd.Timestamp(rest_at),
                                    pd.Timestamp(now).floor('min')),
                                'quality': 'valid',
                                'market_ticker': market['ticker'],
                                'window_open': open_time,
                                'minute_into_window': round(minute, 3),
                                'yes_levels': json.dumps(yes),
                                'no_levels': json.dumps(no),
                                'yes_total': sum(s for _, s in yes),
                                'no_total': sum(s for _, s in no),
                                # The REST call is the sample, so it has no
                                # cache age.
                                'transport': 'rest', 'book_age_ms': 0.0,
                            })
                            # The stream row for this minute was added above.
                            # Both survive a read, because `transport` is part
                            # of the event key.
                    if len(rows) >= args.batch_rows:
                        await asyncio.to_thread(
                            store.write, 'venue_ladder', pd.DataFrame(rows))
                        logger.info('wrote %d ladder rows (%d levels last)',
                                    len(rows), len(json.loads(rows[-1]['yes_levels'])))
                        rows.clear()
                    # Liveness, stamped after a completed cycle. Then sleep to
                    # the next aligned mark, not a fixed interval.
                    heartbeat.touch('ladder')
                    await asyncio.sleep(seconds_to_next_mark(
                        datetime.now(timezone.utc), args.interval, args.phase))
        except asyncio.CancelledError:
            _flush_on_exit()
            raise
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('ladder recorder: %s; retrying in 20s', str(exc)[:160])
            await asyncio.sleep(20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--phase', type=float, default=25.0,
                        help='seconds past the minute to sample at (default 25). '
                             'Held by re-aligning every cycle, not just at startup.')
    parser.add_argument('--batch-rows', type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
