"""Mark the open positions from the in-process stream book, every few seconds.

**The cycle is the wrong clock for this.** `scripts/live.py` marks positions
while it already holds the book, which is correct but happens once per decision
cycle -- so a dashboard polling every ten seconds was reading a number that only
moved every sixty, and a position could swing from 0.80 to 0.20 with nothing on
screen changing. The staleness was the trader's cadence, not the browser's.

**It costs the venue nothing.** `record_stream` holds one WebSocket
subscription and folds every frame into a process-wide `BookCache` -- measured
at 777-862 frames a second across three markets. The book is therefore already
in memory and already current; this reads it. No REST call, no extra
subscription, no rate limit consumed.

Two things it deliberately does not do:

  * **It never prices a decision.** `decide()` reads the book through
    `fetch_quotes`, which vouches for freshness and falls back to REST. This
    writes telemetry only, so a stale or missing cache here means no mark, not
    a worse trade.
  * **It never marks to our own forecast.** The bid for the side held is what a
    sale would fetch. Marking an open binary at the probability the model
    believes books conviction as profit, which is how a losing system draws a
    rising equity curve.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger('marks')

#: How often to re-read the cache. The stream runs ~34ms behind the venue, so
#: this interval is the whole of the staleness and nothing else contributes.
#:
#: **One second, because the dashboard now polls at one.** This sat at five on
#: the reasoning that the dashboard polled at ten, which stopped being true --
#: and a page reading every second off a mark rewritten every five shows the
#: same number four times and then jumps, which reads as a stuck panel rather
#: than a quiet market. The read is from a process-local `BookCache` that is
#: already current, so the cost is one query for the open rows and one write;
#: with a handful of open positions that is nothing.
DEFAULT_INTERVAL = 1.0

#: A book older than this is not priced against. The stream delivers hundreds
#: of frames a second while a market is live, so silence this long means the
#: connection is gone or the market has settled -- either way the last mark
#: should be left to go visibly stale rather than refreshed from a dead book.
DEFAULT_MAX_AGE = 10.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval-seconds', type=float, default=DEFAULT_INTERVAL)
    parser.add_argument('--max-age-seconds', type=float, default=DEFAULT_MAX_AGE)
    return parser


def _best_bid(levels) -> Optional[float]:
    """The highest price anyone is bidding, or None for an empty stack.

    Kalshi's book is two BID stacks -- `yes` and `no` -- so the best bid for a
    side is the maximum price on that side's stack. There is no ask to invert.
    """
    best = None
    for price, size in (levels or []):
        try:
            price = float(price)
            size = float(size)
        except (TypeError, ValueError):
            continue
        if size <= 0:
            continue
        if best is None or price > best:
            best = price
    return best


async def _sample_venue(writer, client) -> None:
    """Record the venue's cash and its OWN mark of the open book.

    **Kalshi serves no history endpoint** -- probed exhaustively 2026-10-20,
    every one of `/portfolio/{summary,pnl,history,balance_history,statements,
    ledger,...}` is a 404. `portfolio_value` on `/portfolio/balance` is the
    current value and nothing else, so an equity series exists only if we
    sample it. This is that sampler.

    **Why not take it from the trading cycle, which already reads the balance.**
    It does, once a cycle. Measured 2026-10-20, that left the venue's mark up
    to $0.95 stale on a book worth $3 -- far worse than the $0.21 marking
    difference it was brought in to remove, because a 15-minute binary repriced
    hard inside one cycle. A number that is exact and a minute old loses to one
    that is live.

    **It costs one read a second.** That is a real, sustained call against the
    venue's rate limit, taken deliberately: this is the single figure the
    dashboard is judged against. It never writes the bankroll -- `our_bankroll`
    is left out so `adopt_venue_balance` stays the only thing that moves the
    number Kelly sizes against, and a telemetry sampler can never resize a bet.
    """
    now = datetime.now(timezone.utc)
    detail = await client.balance_breakdown()
    shards = detail.get('shards') or {}
    index = _trading_shard(shards)
    await asyncio.to_thread(
        writer.write_venue_balance,
        timestamp=now,
        balance=float(shards.get(index, float('nan'))),
        exchange_index=index,
        total_balance=float(detail.get('total', float('nan'))),
        portfolio_value=float(detail.get('portfolio_value', float('nan'))))

    # **And what it holds, which our own table learns a cycle later.** A
    # position exists at the venue the instant it fills; ours exists once the
    # live loop books it. The dashboard showing ours meant a position visible
    # in the Kalshi app was missing from the page until the next cycle -- and
    # permanently missing for an order POST that timed out after being
    # accepted. Replaced wholesale, so a flat book empties the table.
    rows = await client.positions()
    held = []
    for row in rows:
        try:
            size = float(row.get('position_fp') or row.get('position') or 0.0)
        except (TypeError, ValueError):
            continue
        if size == 0:
            continue
        held.append({
            'ticker': row.get('ticker'),
            'exchange_index': row.get('exchange_index'),
            'position': size,
            'market_exposure': _dollars(row, 'market_exposure'),
            'fees_paid': _dollars(row, 'fees_paid'),
            'realized_pnl': _dollars(row, 'realized_pnl'),
        })
    await asyncio.to_thread(writer.replace_venue_positions, held, at=now)


def _dollars(row: dict, name: str) -> Optional[float]:
    """The `_dollars` suffixed field, which is the one that is actually set.

    The venue serves `market_exposure_dollars: "3.330000"` and leaves the
    unsuffixed integer-cent field null on this account -- the same shape that
    once made every quote parse as empty. The bare name is read as a fallback
    and scaled, never preferred.
    """
    value = row.get(f'{name}_dollars')
    if value is not None:
        try:
            return float(value)
        except (TypeError, ValueError):
            pass
    value = row.get(name)
    if value is None:
        return None
    try:
        return float(value) / 100.0
    except (TypeError, ValueError):
        return None


def _trading_shard(shards: dict) -> Optional[int]:
    """Where the crypto cash is: the shard holding the most of it.

    Hardcoding 2 would be a guess that keeps working until the venue reshards,
    which it has done once already on this account.
    """
    if not shards:
        return None
    return max(shards, key=lambda key: shards.get(key) or 0.0)


async def run(args=None, gate=None, cache=None, writer=None, client=None) -> int:
    from core import heartbeat
    from core.pg_writer import PgWriter
    from livetrade.quotes import _stream_cache

    args = args or build_parser().parse_args([])
    if writer is None:
        url = os.getenv('DATABASE_URL')
        if not url:
            logger.warning('DATABASE_URL is unset; not marking')
            return 0
        writer = PgWriter(database_url=url)

    interval = float(getattr(args, 'interval_seconds', DEFAULT_INTERVAL))
    max_age = float(getattr(args, 'max_age_seconds', DEFAULT_MAX_AGE))

    # **Constructed WITHOUT `live=True`, so it cannot place an order even if
    # something in here tried.** This is a telemetry reader and the client
    # refuses to write unless explicitly built to.
    #
    # Absent credentials it simply does not sample the venue and the local
    # marks carry on, which is what paper mode wants.
    owns_client = False
    if client is None and os.getenv('KALSHI_KEY_ID'):
        try:
            from data_collection.kalshi_client import KalshiClient
            client = KalshiClient()
            await client.__aenter__()
            owns_client = True
            logger.info('sampling the venue balance every %.0fs', interval)
        except Exception:                              # noqa: BLE001
            logger.exception('could not open a venue client; marking locally only')
            client = None

    try:
        return await _loop(writer, client, gate, cache, interval, max_age)
    finally:
        if owns_client and client is not None:
            try:
                await client.__aexit__(None, None, None)
            except Exception:                          # noqa: BLE001
                logger.exception('could not close the venue client')


async def _loop(writer, client, gate, cache, interval, max_age) -> int:
    # Imported here rather than at module scope for the same reason `run` does
    # it: these pull in the ORM and the stream cache, and importing them at
    # module load makes this script unimportable without a database.
    from core import heartbeat
    from livetrade.quotes import _stream_cache

    while True:
        try:
            # **Never while a decision is in flight.** Marking is telemetry and
            # a decision is not; a few seconds of deferral costs a stale mark,
            # a delayed decision costs edge.
            if gate is not None:
                await gate.idle()

            # Sampled before the local marks, so a pass that fails here still
            # records what the book says -- the two are independent readings
            # and one being unavailable is not a reason to drop the other.
            if client is not None:
                try:
                    await _sample_venue(writer, client)
                except asyncio.CancelledError:
                    raise
                except Exception:                      # noqa: BLE001
                    logger.warning('could not sample the venue this pass',
                                   exc_info=True)

            book = _stream_cache() if cache is None else cache
            if book is None:
                heartbeat.touch('marks')
                await asyncio.sleep(interval)
                continue

            positions = await asyncio.to_thread(writer.open_positions)
            if not positions:
                heartbeat.touch('marks')
                await asyncio.sleep(interval)
                continue

            # The ticker is on the ticket, not the position: `positions` has no
            # ticker column because the paper engine predates the venue client.
            tickers = await asyncio.to_thread(writer.tickers_for_open_positions)
            marks: dict = {}
            for position in positions:
                key = (str(position.symbol), position.window_open,
                       int(position.offset_minutes))
                ticker = tickers.get(key)
                if not ticker:
                    continue
                try:
                    ladder = book.current(ticker, max_age)
                except Exception:                          # noqa: BLE001
                    ladder = None
                if ladder is None:
                    continue
                price = _best_bid(ladder.yes if position.side == 'up'
                                  else ladder.no)
                if price is None:
                    continue
                marks[(str(position.symbol), position.window_open)] = price

            if marks:
                await asyncio.to_thread(
                    writer.mark_open_positions, marks,
                    at=datetime.now(timezone.utc))
            # **Touched every pass, not only when something was marked.** A
            # window with no open position is the normal state between trades;
            # treating that as silence would have the healthcheck call a
            # working loop dead.
            heartbeat.touch('marks')
        except asyncio.CancelledError:
            raise
        except Exception:                                   # noqa: BLE001
            # Telemetry must not take the process down, and `supervise` would
            # restart this anyway — logging it keeps a repeated failure visible
            # instead of hiding behind a restart loop.
            logger.exception('marking failed this pass')
        await asyncio.sleep(interval)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    return asyncio.run(run(build_parser().parse_args(argv)))


if __name__ == '__main__':
    raise SystemExit(main())
