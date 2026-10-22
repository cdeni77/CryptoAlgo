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

#: `source` for these rows. Its own name, because this sample is taken at a
#: different instant from every other producer and `venue_depth` already
#: proves what happens when two instants share one label.
SOURCE = 'live_minute'

#: Samples waiting to be written, and the cap. ~45 rows a minute across three
#: symbols, so this holds roughly two hours of a store that cannot be written.
_PENDING: list = []
MAX_PENDING = 6000

#: How often to write. Not every minute: the live loop shares this process,
#: and a parquet write is the kind of blocking work that emptied `book_flow`
#: when it sat in the decision path.
FLUSH_EVERY_SAMPLES = 15

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
            _buffer_row(ticker, symbol, opened, int(offset), into_minute,
                        yes, no)
    return kept


def _buffer_row(ticker, symbol, opened, offset, age, yes, no) -> None:
    """Queue this sample for `venue_depth`. Never raises.

    **This is the only book sample taken at the instant a decision could have
    used it.** The ladder recorder runs at +25s past the minute, so
    `decision_age` scores its rows negative -- seen AFTER the decision -- and
    refuses 97.8-99.8% of them. Measured, the three per-minute recorders are
    140,646 rows contributing 1.2% of usable quotes.

    So the pipeline was persisting a sample it cannot use and discarding the
    one it can. Keeping this one is the only route to depth at non-decision
    offsets that a decision could actually have had: the venue serves no
    historical order book, so it exists forward or not at all.
    """
    try:
        from scripts.build_depth import _cumulative, _row            # noqa: PLC0415
        bid = max((float(p) for p, _ in yes or []), default=None)
        ask = min(((1.0 - float(p)) for p, _ in no or []), default=None)
        event_time = opened + dt.timedelta(minutes=offset)
        _PENDING.append(_row(
            venue='kalshi', symbol=symbol, event_time=event_time,
            available_time=event_time + dt.timedelta(seconds=float(age)),
            market_ticker=ticker, window_open=opened, offset_minutes=offset,
            yes_bid=bid, yes_ask=ask,
            yes_bid_size=sum(float(s) for p, s in (yes or []) if float(p) == bid)
            if bid is not None else 0.0,
            yes_ask_size=sum(float(s) for p, s in (no or [])
                             if abs((1.0 - float(p)) - (ask or -1)) < 1e-9)
            if ask is not None else 0.0,
            depth_bid_1c=_cumulative(yes, best=bid, within=0.01, invert=False),
            depth_bid_5c=_cumulative(yes, best=bid, within=0.05, invert=False),
            depth_ask_1c=_cumulative(no, best=ask, within=0.01, invert=True),
            depth_ask_5c=_cumulative(no, best=ask, within=0.05, invert=True),
            depth_bid_total=sum(float(s) for _, s in (yes or [])),
            depth_ask_total=sum(float(s) for _, s in (no or [])),
            levels_bid=float(len(yes or [])), levels_ask=float(len(no or [])),
            source=SOURCE, quote_age_seconds=float(age), transport='ws'))
        # Bounded: a store that cannot be written must not grow the process
        # until it dies. Dropping the oldest is right -- the newest minutes
        # are the ones the next retrain has least of.
        if len(_PENDING) > MAX_PENDING:
            del _PENDING[:len(_PENDING) - MAX_PENDING]
    except Exception as exc:                                      # noqa: BLE001
        logger.debug('could not buffer %s: %s', ticker, str(exc)[:120])


def flush(store=None) -> int:
    """Write the buffered samples. Returns the count. Never raises."""
    if not _PENDING:
        return 0
    batch, del_count = list(_PENDING), len(_PENDING)
    try:
        import pandas as pd                                       # noqa: PLC0415
        from core.datastore import ResearchStore                  # noqa: PLC0415
        (store or ResearchStore()).write('venue_depth', pd.DataFrame(batch))
    except Exception as exc:                                      # noqa: BLE001
        logger.warning('could not write %d book minute(s): %s',
                       len(batch), str(exc)[:140])
        return 0
    del _PENDING[:del_count]
    return len(batch)


async def run(args=None) -> int:
    """Sample every minute, forever.

    Takes no decision gate. Every other recorder calls the venue and so
    waits its turn; this one reads an in-process dict, and waiting would
    only push the sample past `BOOK_SAMPLE_MAX_AGE_SECONDS` and have the
    minute refused.
    """
    from livetrade.quotes import _stream_cache                # noqa: SLF001
    from core import heartbeat

    late = samples = 0
    while True:
        # **This recorder is NOT gated behind a decision, and that is the
        # difference that makes it work.** Every other recorder calls the
        # venue, so it waits its turn. `sample_once` reads an in-process dict
        # and makes no request, so waiting buys nothing -- and it costs the
        # minute: a decision cycle runs far longer than
        # `BOOK_SAMPLE_MAX_AGE_SECONDS`, so the sample lands stale and every
        # row for that minute is refused.
        #
        # The refused minutes are the DECISION minutes, 3/6/9/12, and every
        # `book_flow` feature differences the book over one or three minutes
        # onto exactly those. Offset 6 reads offset 3, offset 9 reads 6,
        # offset 12 reads 9. Gating this loop therefore emptied `book_flow`
        # on most live decisions while the sampler looked healthy.
        #
        # Sampling is synchronous on a single-threaded event loop, so it
        # cannot interleave with a decision regardless.
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
        # **Written off the event loop, and never every minute.** A parquet
        # write is exactly the blocking work that emptied `book_flow` when it
        # sat in the decision path, so it goes to a thread and runs once every
        # `FLUSH_EVERY_SAMPLES` minutes. `flush` never raises: a store that
        # cannot be written must cost a buffered minute, not the trading loop.
        samples += 1
        if samples % FLUSH_EVERY_SAMPLES == 0 and _PENDING:
            try:
                written = await asyncio.to_thread(flush)
                if written:
                    logger.info('wrote %d book minute(s) to venue_depth', written)
            except Exception as exc:                          # noqa: BLE001
                logger.warning('book-minute flush failed: %s', str(exc)[:140])
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
