"""Record CF Benchmarks' index live, into `minute_bars` under its own venue.

The same table `scripts/backfill_index` writes, so the backfill and the live
feed are one continuous series and the pipeline switches with `--venue
cfbenchmarks` rather than with a second code path.

**Why this exists.** C1 measured the venue's own strike with the BRTI price
beating our Coinbase close against an OHLC-mean strike by +0.0033 log loss at
t=+5.71 -- about three times what the model itself contributes. But C6 binds
the strike and the price series together, and `scripts/live.py` already
refuses to override the strike alone, because that makes displacement
`Coinbase_close / BRTI - 1` and injects a cross-index basis into a feature
fitted without one. So the migration needs this feed first, or the model is
fitted on one index and served another.

**A silent socket cannot wake itself**, the lesson `scripts/record_stream`
records: every exit is enforced on a timeout, because at about one frame per
second per index, silence means something is wrong rather than nothing is
happening.

    python -m scripts.record_index_stream            # run it
    python -m scripts.record_index_stream --dry-run  # parse, write nothing
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import logging
import os
import time

logger = logging.getLogger('index-stream')

#: Seconds of silence that mean the socket is dead. At ~1 frame/s/index,
#: thirty of them cannot be a quiet market.
SILENCE_SECONDS = 30.0

#: How often to write. Not every minute: a parquet write is the blocking work
#: that emptied `book_flow` when it sat in a decision path.
FLUSH_EVERY_SECONDS = 300.0

#: **Process-wide, like `record_stream.CACHE`.** `run_live` runs this task and
#: the trading loop in one process, so the decision path can splice the
#: minutes this has folded but not yet written. Without that, the newest bar a
#: decision could read would be up to `FLUSH_EVERY_SECONDS` old -- and the bar
#: a decision needs is the one that closed moments ago.
#:
#: A bare `scripts.record_index_stream` fills its own copy and the trading
#: loop in another process sees an empty one. That is the trap already
#: recorded for the book cache.
BARS = None


def live_bars():
    """The in-process fold, or None if this task is not running here."""
    return BARS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--symbols', default='BTC-USD,ETH-USD,SOL-USD')
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    p.add_argument('--dry-run', action='store_true',
                   help='parse and count, write nothing')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


async def run(args=None) -> int:
    """Subscribe, fold, and persist until cancelled."""
    from core import heartbeat
    from core.datastore import ResearchStore
    from core.index_bars import IndexBars, rows_for_store
    from data_collection.kalshi_client import KalshiClient
    from data_collection.stream.cfbenchmarks import (
        parse_value_frame, subscribe_params)
    from data_collection.stream.kalshi import KalshiStream

    global BARS
    args = args or build_parser().parse_args([])
    symbols = [s.strip() for s in str(args.symbols).split(',') if s.strip()]
    store = None if args.dry_run else ResearchStore(args.research_store)
    # Published before the first frame, so a decision that lands during the
    # 60-second warm-up reads an empty fold rather than a missing attribute.
    bars = BARS = IndexBars()
    pending: list[dict] = []
    last_flush = time.time()
    partial_logged = False

    while True:
        try:
            async with KalshiClient(live=False) as client:
                stream = KalshiStream(client)
                await stream.connect()
                await stream._ws.send_json({                  # noqa: SLF001
                    'id': 1, 'cmd': 'subscribe',
                    'params': subscribe_params(symbols)})
                logger.info('subscribed to the index for %s', ', '.join(symbols))
                partial_logged = False

                while True:
                    # Enforced on a timeout, never on the absence of a frame.
                    msg = await asyncio.wait_for(stream._ws.receive(),  # noqa: SLF001
                                                 timeout=SILENCE_SECONDS)
                    try:
                        payload = json.loads(msg.data)
                    except (ValueError, TypeError, AttributeError):
                        continue
                    value = parse_value_frame(payload)
                    if value is None:
                        continue
                    if not value.settles and not partial_logged:
                        # Said once per connection: the venue needs sixty
                        # seconds to fill its averaging window, and until then
                        # `avg_60s_data` is a different quantity that looks
                        # like the right one.
                        logger.info(
                            'the 60s window is still filling (%d/60); the '
                            'venue average is not the settlement quantity yet',
                            value.window_size)
                        partial_logged = True
                    bars.add(value)
                    heartbeat.touch('index_stream')

                    now = time.time()
                    if now - last_flush < FLUSH_EVERY_SECONDS:
                        continue
                    last_flush = now
                    for symbol in symbols:
                        pending.extend(bars.drain(symbol, now=now))
                    if pending and store is not None:
                        written = await asyncio.to_thread(
                            _write, store, pending, now)
                        if written:
                            pending = []
                    elif pending:
                        logger.info('dry run: %d bar(s) folded', len(pending))
                        pending = []
        except asyncio.CancelledError:
            if pending and store is not None:
                _write(store, pending, time.time())
            raise
        except asyncio.TimeoutError:
            logger.warning('no index frame for %.0fs; reconnecting',
                           SILENCE_SECONDS)
        except Exception as exc:                              # noqa: BLE001
            logger.warning('index stream dropped (%s); reconnecting',
                           str(exc)[:140])
        await asyncio.sleep(2.0)


def _write(store, bars, now) -> int:
    """Persist folded bars. Never raises: a store that cannot be written must
    cost a buffered minute, not the connection."""
    from core.index_bars import rows_for_store
    try:
        import pandas as pd
        rows = rows_for_store(
            bars, available_time=dt.datetime.fromtimestamp(now, dt.timezone.utc))
        if not rows:
            return 0
        store.write('minute_bars', pd.DataFrame(rows))
        logger.info('wrote %d index minute bar(s)', len(rows))
        return len(rows)
    except Exception as exc:                                  # noqa: BLE001
        logger.warning('could not write %d index bar(s): %s',
                       len(bars), str(exc)[:140])
        return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-14s %(message)s',
        datefmt='%H:%M:%S')
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
