"""Sample the book at every MINUTE of the window, so `book_flow` has a history.

**Why this exists.** All six `book_flow` features were NaN on every real-money
decision while the deployed artifact was fitted on them.
`core.book_flow.book_flow_features` is called from `core/dataset.py` alone --
the research path -- and `livetrade/features.py` never built it. The live loop
warned about it every cycle and nothing escalated the warning.

**Why a separate minute task and not the decision cycle.** `book_flow`
differences the book over one and three minutes, and decisions happen only at
offsets 3/6/9/12, so the decision cycle can never supply offset-1 or offset-3.
The history has to be sampled every minute, whether or not anything is decided.

**Why PHASE 0 specifically, and why that is not a detail.**
`core.quotes.decision_age` ages a minute-grid source FORWARD --
`latency_seconds - raw` -- so a row observed more than `decision_latency_seconds`
(3s) after its own minute goes NEGATIVE and `book_flow_features` drops it.
Measured 2026-10-21 on a synthetic ladder: a stream row stamped 0s old yields
6/6 features, the same row stamped 7s old yields 0/6. The live decision cycle
starts about 7 seconds into the minute, so sampling there would have reproduced
the all-NaN bug while looking exactly like a fix.

**No network.** `run_live` runs the stream and this task in one process, so
`record_stream.CACHE` already holds every subscribed ladder -- fresher than
REST (~34ms against ~73ms) and free. The tickers come from the cache too, so
nothing here resolves a market or guesses one from a pattern.

Nothing is written to the store: this is an in-process history for scoring the
window in progress. `venue_depth` already records the same minutes for research
through `record_ladder` and the stream archive.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time

logger = logging.getLogger('book-minutes')

#: One window, in minutes. A sample outside 0..WINDOW is not this window's.
WINDOW_MINUTES = 15

#: How often to sample. The task is phase-aligned by `run_live`, so this is the
#: period, not the offset.
PERIOD_SECONDS = 60.0


def _symbol_for(ticker: str):
    """`KXBTC15M-...` -> `BTC-USD`, or None for a series we do not trade.

    Through `core.config.series_to_symbol`, which derives the inverse from
    `SERIES_BY_SYMBOL` so an env override reaches both directions. A second
    hardcoded mapping is the trap that helper exists to replace.
    """
    from core.config import series_to_symbol
    return series_to_symbol().get(str(ticker).split('-')[0])


def sample_once(cache, *, now=None) -> int:
    """Record one minute of every book the stream is holding. Returns the count.

    Never raises: a sampler that can break the trading loop is worse than one
    that misses a minute, and a missed minute shows up honestly as a NaN
    feature rather than as a wrong one.
    """
    from scripts.backfill_candles import window_from_ticker
    from livetrade.features import record_book_minute
    from livetrade.quotes import ladder_from_cache

    if cache is None:
        return 0
    now = now or dt.datetime.now(dt.timezone.utc)
    kept = 0
    try:
        tickers = cache.tickers()
    except Exception as exc:                                  # noqa: BLE001
        logger.debug('could not list the cache: %s', str(exc)[:120])
        return 0
    for ticker in tickers:
        symbol = _symbol_for(ticker)
        window = window_from_ticker(ticker)
        if symbol is None or window is None:
            continue
        opened, _close = window
        elapsed = (now - opened).total_seconds()
        offset, into_minute = divmod(elapsed, 60.0)
        if not (0 <= offset <= WINDOW_MINUTES):
            continue
        try:
            yes, no = ladder_from_cache(cache, ticker)
        except Exception as exc:                              # noqa: BLE001
            logger.debug('%s: %s', ticker, str(exc)[:120])
            continue
        # `into_minute` IS the age: the cache is current to ~34ms, so what
        # makes this row old is how far past the minute mark we sampled.
        if record_book_minute(symbol, window_open=opened, offset=int(offset),
                              yes_levels=yes, no_levels=no,
                              age_seconds=into_minute):
            kept += 1
    return kept


async def run(args=None, gate=None) -> int:
    """Sample every minute, forever."""
    from livetrade.quotes import _stream_cache                # noqa: SLF001
    from core import heartbeat

    late = 0
    while True:
        # Never start while a decision is in flight. This is an in-process read
        # of a dict, so the deferral is short, but the rule is the same one
        # every other recorder follows.
        if gate is not None:
            await gate.idle()
        cache = _stream_cache()
        try:
            kept = sample_once(cache)
        except Exception as exc:                              # noqa: BLE001
            logger.error('sampling failed: %s', str(exc)[:160])
            kept = 0
        if kept:
            late = 0
        else:
            late += 1
            # Loud only once it is a pattern: the first minutes after a restart
            # legitimately have no subscription, and a settled window between
            # subscriptions is normal.
            if late in (5, 30) or late % 120 == 0:
                logger.warning(
                    'no book minute recorded for %d consecutive samples — '
                    'book_flow will score NaN and live is then not the model '
                    'the gates measured', late)
        # Stamped whether or not a row was kept: an idle book is not a wedged
        # task, and a heartbeat that only fires on success measures the market
        # rather than the loop.
        heartbeat.touch('book_minutes')
        # **Sleep to the next MINUTE BOUNDARY, not for a fixed period.**
        # `sleep(60)` runs AFTER the work, so each cycle costs 60s plus
        # however long sampling took, and the phase creeps forward. At ~45ms
        # of work that is 45ms/minute, which crosses the 3-second freshness
        # threshold after ~67 minutes — measured live 2026-10-21, it kept rows
        # for 68 minutes after a restart and then refused every one for six
        # hours while still stamping its heartbeat. Phase is not a detail
        # here: it is the entire reason this component exists (see the module
        # docstring), so it has to be re-established every cycle rather than
        # assumed to hold.
        now = time.time()
        await asyncio.sleep(max(0.5, PERIOD_SECONDS - (now % PERIOD_SECONDS)))
