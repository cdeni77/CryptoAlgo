"""Record the Kalshi public TAPE live, so `venue_trades` stops being a backfill.

**One question motivates the dataset: would a RESTING order have filled?**
`scripts/backfill_trades.py` states the argument in full and this recorder does
not restate it — in short, a resting bid fills when an aggressive SELLER crosses
into it, the quoted ask never moves, and so neither `venue_depth` (minute
snapshots of the book) nor `venue_book_events` (deltas, which cannot distinguish
a trade from a cancellation) can answer it. The tape can, because each print
carries `taker_book_side`.

Until now the only writer was that manual backfill, so the dataset accrued only
when somebody ran it: measured 2026-10-20, the newest row was 2026-10-14 and a
22-day interior hole sat between 2026-09-16 and 2026-10-13. The maker-pricing
result — the largest lever on P&L this project has — rests on this table.

**REST, not the websocket, and the reasons are specific.**

* The existing socket is a BOOK socket. `KalshiStream.subscribe` asks for
  `['orderbook_delta']` and `parse_frame` returns `None` for every frame that is
  not `orderbook_snapshot`/`orderbook_delta`; `events()` yields `BookEvent`, and
  `BookCache.apply`, `core.spool.event_rows` and `record_stream.consume` all
  consume exactly that type. A trade channel is a second event type threaded
  through all of them.
* `seq` is global per SUBSCRIPTION, not per market — `CLAUDE.md` records this and
  `record_stream` is built on it: one gap condemns every book on the connection
  and `cache.any_gapped()` forces a full resubscribe. Trade frames arriving on
  the same subscription advance that same counter. A book cache that does not
  apply them would see a jump and resubscribe, continuously, on the one component
  whose staleness the trading path reads directly. That is a live-trading risk
  taken for no measured gain.
* There is no gain to take. `venue_trades` is **aggregated per (market, minute,
  aggressor side)** — the extreme price each side's aggressor traded at. That
  aggregate is order-independent and time-independent within the minute, so a
  poll that sees the same prints a minute later computes the identical row. The
  tape's value here is historical, not latent: nothing on the decision path reads
  it.
* The cost is small. `market_trades` pages at 1,000, three markets print on the
  order of a few thousand times a minute between them, and a cycle is a handful
  of requests — the same order as `record_ladder`'s ~0.05 req/s against a
  Basic-tier ceiling of 20.
* And `KalshiClient.market_trades` already exists, already handles the live and
  historical tiers, and is already proven by the backfill. The rows this writes
  are built by the same arithmetic, so the two provenances stay comparable.

**Only COMPLETE minutes are written.** A partial minute's `min_price` is an
understatement of how deep an aggressor went, which is exactly the direction that
would make a fill simulation look better than it was. Rows are emitted for a
minute only once that minute has fully elapsed AND the fetch window covered all
of it, and a watermark per market means each minute is written once.

**A market is kept past its close.** The `/markets` listing is queried by close
time, so a settled market drops out of it immediately — and with it the last
minutes of its tape, which this recorder has not fetched yet because it works a
minute or two behind. `--grace-seconds` keeps a market in the poll set after it
closes; without it minute 13 and 14 of every window would be permanently missing,
silently, which is the most valuable part of the window.

**A minute the lookback could not cover is logged, not written.** If a cycle is
missed for longer than `--lookback-seconds`, the uncovered minutes are skipped
with a warning rather than written from a partial fetch. They are recoverable by
`scripts.backfill_trades`; a partial row is not.

    python -m scripts.record_trades            # standalone
    (and as the `trades` component of scripts.run_live)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from core import heartbeat
from core.config import DEFAULT_CONFIG, series_to_symbol
from core.datastore import ResearchStore
# The one place this repo parses a Kalshi timestamp. Importing it rather than
# writing a third copy of `strptime('%Y-%m-%dT%H:%M:%SZ')`: the venue has
# already been seen to serve both that spelling and a fractional-second ISO one,
# and a second parser is a second thing to keep in step.
from data_collection.kalshi_client import _parse_time  # noqa: PLC2701
# The shared cadence helper. `sleep(interval)` at the end of a cycle makes the
# real period `interval + work`, so the sampling phase walks through the minute
# — measured on `venue_ladder` and fixed there; there is no reason to repeat the
# mistake here, and no reason to define the fix twice.
from scripts.record_ladder import seconds_to_next_mark

logger = logging.getLogger('trades')

DATASET = 'venue_trades'
# The one series<->symbol mapping — see core/config.SERIES_BY_SYMBOL. Read at
# call time, not at import, so a test can point it somewhere else.
SERIES = series_to_symbol()


def _utc(stamp) -> pd.Timestamp:
    ts = pd.Timestamp(stamp)
    return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')


def fold_prints(trades, *, ticker: str, symbol: str | None, window_open,
                fetched_at, first_minute: int, last_minute: int) -> list[dict]:
    """Collapse a market's prints to one row per (minute, aggressor side).

    **The extremes are the point, not the average**, and the reasoning is
    `backfill_trades._rows`': a resting bid at P is hit when a SELLING aggressor
    trades at or below P, so `min_price` on the selling side is the fill test and
    `max_price` on the buying side is its mirror for a resting offer. A mean
    answers neither. Deliberately the same arithmetic as the backfill, so a live
    row and a backfilled row for the same market-minute are the same measurement.

    `first_minute`/`last_minute` bound which offsets may be emitted, inclusive.
    Everything outside is dropped rather than clamped — a print at minute 16
    belongs to the next market, and a minute the caller could not fully observe
    must not be written at all.
    """
    import numpy as np

    window = _utc(window_open)
    buckets: dict = {}
    for trade in trades:
        created = getattr(trade, 'created_time', None)
        price = getattr(trade, 'yes_price', None)
        if created is None or price is None:
            continue
        offset = int((_utc(created) - window).total_seconds() // 60)
        if offset < first_minute or offset > last_minute:
            continue
        side = getattr(trade, 'taker_book_side', None) or 'unknown'
        contracts = float(getattr(trade, 'contracts', 0) or 0)
        bucket = buckets.setdefault(
            (offset, side),
            {'n': 0, 'c': 0.0, 'lo': np.inf, 'hi': -np.inf, 'pv': 0.0})
        bucket['n'] += 1
        bucket['c'] += contracts
        bucket['lo'] = min(bucket['lo'], float(price))
        bucket['hi'] = max(bucket['hi'], float(price))
        bucket['pv'] += float(price) * contracts

    stamped = _utc(fetched_at)
    rows = []
    for (offset, side), bucket in sorted(buckets.items()):
        event_time = window + pd.Timedelta(minutes=offset)
        rows.append({
            'venue': 'kalshi', 'symbol': symbol, 'market_ticker': ticker,
            'event_time': event_time,
            # When this could first have been SEEN. The poll runs behind the
            # tape by construction, so it is always the fetch instant — but
            # `max` keeps the invariant explicit rather than assumed.
            'available_time': max(stamped, event_time),
            # **'valid', where the backfill writes 'backfill'.** This is a
            # live-recorded measurement, and `QUALITY_LEVELS` has no 'backfill'
            # level — so `ResearchStore.read` at its default `min_quality='valid'`
            # returns zero backfilled rows today. These rows are readable.
            'quality': 'valid',
            'window_open': window,
            'offset_minutes': offset,
            'taker_book_side': side,
            'n_prints': bucket['n'],
            'contracts': bucket['c'],
            'min_price': bucket['lo'],
            'max_price': bucket['hi'],
            'vwap': (bucket['pv'] / bucket['c']) if bucket['c'] > 0 else None,
        })
    return rows


def complete_minutes(*, window_open, since: datetime, now: datetime,
                     watermark: int | None, window_minutes: int,
                     ) -> tuple[int, int]:
    """The inclusive offset range that is both fully ELAPSED and fully FETCHED.

    Three bounds, and each one exists because dropping it writes a wrong row
    rather than no row:

    * `since` — the fetch asked for prints at or after this instant, so the
      minute containing it was only partly observed. Emitting it would
      understate `n_prints` and could raise `min_price` above the real low,
      which is the direction that makes a fill simulation look better than it
      was.
    * `now` — a minute that has not finished is partial for the same reason.
      It is emitted on a later cycle, complete.
    * `watermark` — the last offset already written for this market. Minutes
      are written once; re-emitting them appends a superseding revision to the
      partition for no new information.

    Returns `(lo, hi)` with `lo > hi` when there is nothing to write.
    """
    window = _utc(window_open)
    covered_from = math.ceil((_utc(since) - window).total_seconds() / 60.0)
    elapsed_to = math.floor((_utc(now) - window).total_seconds() / 60.0) - 1
    lo = max(0, covered_from)
    if watermark is not None:
        lo = max(lo, watermark + 1)
    return lo, min(elapsed_to, window_minutes)


async def poll_markets(client, *, now: datetime, horizon: int = 1800) -> dict:
    """ticker -> {symbol, window_open, close_time} for the current window and next.

    Queried by CLOSE TIME, not by `status`. `record_stream._trading` documents
    why and `record_ladder` was changed for it: `status` lags the market by up to
    forty seconds, so the replacement market — already published, with an
    `open_time` exactly at the boundary — is invisible to a `status=open` query
    for the first part of every window.

    `max_close_ts` is load-bearing. Without an upper bound every future market
    matches, the venue returns them NEWEST first, and `limit` then yields only
    the furthest ones — which is how `venue_ladder` died for 32 days.
    """
    stamp = int(now.timestamp())
    out: dict[str, dict] = {}
    for series, symbol in SERIES.items():
        payload = await client._request(  # noqa: SLF001
            'GET', '/markets',
            params={'series_ticker': series, 'min_close_ts': stamp,
                    'max_close_ts': stamp + horizon, 'limit': 10})
        for market in payload.get('markets', []):
            ticker = market.get('ticker')
            opens = _parse_time(market.get('open_time'))
            if not ticker or opens is None:
                continue
            out[str(ticker)] = {
                'symbol': symbol,
                'window_open': opens,
                'close_time': (_parse_time(market.get('close_time'))
                               or opens + timedelta(
                                   minutes=DEFAULT_CONFIG.window_minutes)),
            }
    return out


def retire(tracked: dict, *, now: datetime, grace: float) -> list[str]:
    """Drop markets whose tape can no longer gain a print we have not fetched.

    Kept past the close on purpose: the listing in `poll_markets` is bounded by
    `min_close_ts`, so a settled market vanishes from it the instant it settles,
    while this recorder is still a minute or two behind its last prints.
    """
    dead = [ticker for ticker, info in tracked.items()
            if (now - info['close_time']).total_seconds() > grace]
    for ticker in dead:
        tracked.pop(ticker, None)
    return dead


async def _tape(client, ticker: str, since: datetime, limit: int):
    """One market's prints since `since`. A seam, so a test can stub the venue.

    It deliberately does not swallow an error. **An ERROR is not an empty
    tape** — the backfill learned that the expensive way, where returning `[]`
    for a failed request made 20 failures print as "20 markets had no prints at
    all (normal)". Here the consequence is worse than cosmetic: an empty result
    would advance the watermark past minutes that were never fetched, and the
    gap would be permanent and silent.
    """
    return await client.market_trades(ticker=ticker, since=since, limit=limit)


async def run(args, gate=None, store=None, client=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    config = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE')) if store is None else store

    def _pem() -> str:
        # Read only when this loop is building its own client. Reading it
        # unconditionally made an injected client — the test seam, and the way
        # a caller shares one session — require credentials it never uses.
        return (os.getenv('KALSHI_PRIVATE_KEY')
                or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read())

    tracked: dict[str, dict] = {}
    watermark: dict[str, int] = {}
    rows: list[dict] = []

    def _write(batch: list[dict]) -> int:
        """Write a batch and drop exactly those rows, both inside the thread.

        **The clear belongs with the write, not after the await.** With
        `await to_thread(store.write, ...)` followed by `rows.clear()`, a
        cancellation landing between the thread finishing and the coroutine
        resuming skips the clear — and `_flush_on_exit` then writes the same
        rows a second time. The store deduplicates them (same event key, same
        `available_time`), so it was never corruption, but a shutdown path that
        writes twice is not one to leave in place. Nothing appends to `rows`
        while this await is suspended, so the slice is exactly the batch.
        """
        written = store.write(DATASET, pd.DataFrame(batch))
        del rows[:len(batch)]
        return written

    def _flush_on_exit() -> None:
        """Write whatever is buffered before this task goes away.

        `asyncio.CancelledError` is a BaseException, so the `except Exception`
        reconnect handler below never sees a shutdown — the same hole already
        found and fixed in `record_ladder`, on a dataset with the same property:
        a minute not recorded is gone unless somebody runs a backfill.
        """
        if not rows:
            return
        try:
            store.write(DATASET, pd.DataFrame(rows))
            logger.info('flushed %d buffered row(s) on shutdown', len(rows))
            rows.clear()
        except Exception as exc:                          # noqa: BLE001
            logger.error('could not flush %d buffered row(s): %s',
                         len(rows), str(exc)[:160])

    while True:
        try:
            owned = client is None
            opened = (KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                   private_key_pem=_pem())
                      if owned else client)
            if owned:
                await opened.__aenter__()
            try:
                while True:
                    # Never start a cycle while a trading decision is in flight.
                    # This is a 60s cadence working a minute behind the tape; a
                    # few seconds of deferral costs nothing, a delayed decision
                    # costs edge.
                    if gate is not None:
                        await gate.idle()
                    now = datetime.now(timezone.utc)
                    try:
                        tracked.update(await poll_markets(opened, now=now))
                    except Exception as exc:              # noqa: BLE001
                        # Not fatal: the markets already tracked still have tape
                        # to collect, and dropping them because the listing
                        # failed would lose the end of the current window.
                        logger.warning('market listing: %s', str(exc)[:120])
                    for ticker in retire(tracked, now=now,
                                         grace=args.grace_seconds):
                        watermark.pop(ticker, None)

                    since = now - timedelta(seconds=args.lookback_seconds)
                    for ticker, info in sorted(tracked.items()):
                        lo, hi = complete_minutes(
                            window_open=info['window_open'], since=since,
                            now=now, watermark=watermark.get(ticker),
                            window_minutes=config.window_minutes)
                        if lo > hi:
                            continue
                        seen = watermark.get(ticker)
                        if seen is not None and lo > seen + 1:
                            # Honest about the hole rather than papering over
                            # it with a partial fetch. Recoverable by
                            # `scripts.backfill_trades`; a partial row is not.
                            logger.warning(
                                '%s: minutes %d-%d fell outside the %.0fs '
                                'lookback and were NOT recorded',
                                ticker, seen + 1, lo - 1, args.lookback_seconds)
                        try:
                            trades = await _tape(opened, ticker, since,
                                                 args.page_limit)
                        except Exception as exc:          # noqa: BLE001
                            # The watermark is deliberately NOT advanced: the
                            # next cycle re-asks for the same minutes.
                            logger.warning('%s tape: %s', ticker, str(exc)[:120])
                            continue
                        rows.extend(fold_prints(
                            trades, ticker=ticker, symbol=info['symbol'],
                            window_open=info['window_open'], fetched_at=now,
                            first_minute=lo, last_minute=hi))
                        # Advanced on a SUCCESSFUL fetch, not on rows produced.
                        # A fifteen-minute market that genuinely never printed
                        # in a minute is normal, and treating that as "not yet
                        # fetched" would re-ask for it every cycle forever.
                        watermark[ticker] = hi

                    if len(rows) >= args.batch_rows:
                        batch = list(rows)
                        await asyncio.to_thread(_write, batch)
                        logger.info('wrote %d tape rows across %d market-minutes',
                                    len(batch), len({(r['market_ticker'],
                                                      r['offset_minutes'])
                                                     for r in batch}))
                    # Liveness, recorded where the loop has demonstrably done
                    # its job. A stamp written at startup or by the supervisor
                    # would prove only that the process exists.
                    heartbeat.touch('trades')
                    await asyncio.sleep(seconds_to_next_mark(
                        datetime.now(timezone.utc), args.interval, args.phase))
            except asyncio.CancelledError:
                # **Flushed BEFORE the client is closed.** Putting this in the
                # outer handler meant it ran after `__aexit__`, and an await in
                # a `finally` during cancellation can raise `CancelledError`
                # again — which would skip the flush entirely, on the one
                # dataset whose whole justification is that an unrecorded
                # minute is gone.
                _flush_on_exit()
                raise
            finally:
                if owned:
                    await opened.__aexit__(None, None, None)
        except asyncio.CancelledError:
            _flush_on_exit()
            raise
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('tape recorder: %s; retrying in 20s', str(exc)[:160])
            await asyncio.sleep(20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--phase', type=float, default=35.0,
                        help='seconds past the minute to poll at (default 35). '
                             'Away from the decision instants at ~1s past '
                             'minutes 3/6/9/12, and away from the other '
                             'recorders at 25/45/50/55.')
    parser.add_argument(
        '--lookback-seconds', type=float, default=240.0,
        help='how far back each poll asks for prints. Must comfortably exceed '
             'the interval: the minute containing the lookback edge is only '
             'partly observed and is never written, so the slack is what lets '
             'a missed cycle be recovered rather than lost.')
    parser.add_argument(
        '--grace-seconds', type=float, default=300.0,
        help='how long a market is still polled after it closes. The listing '
             'is bounded by close time, so without this the last minutes of '
             'every window are never fetched.')
    parser.add_argument('--page-limit', type=int, default=1000,
                        help='prints per page; the venue allows 1000 here.')
    parser.add_argument('--batch-rows', type=int, default=12,
                        help='rows buffered before a store write. Small on '
                             'purpose: a hard crash loses the buffer.')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
