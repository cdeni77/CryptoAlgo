"""Everything the live system runs, in one process.

The live side was five containers — the trading loop, three recorders and a
shell loop that scraped and rebuilt the store. They were separate because they
were written one at a time, not because they are different: four are async loops
polling an HTTP API on a sixty-second cadence. One process means one restart
policy, one log stream and one thing to check.

**Trading latency is the constraint everything else bends around**, and the
threat is not the polling — that is all `await`. It is `ResearchStore.write`,
which reads a Parquet partition, concatenates, sorts and rewrites it with zstd.
That is synchronous pandas and pyarrow, and on this event loop it blocks
everything, a pending decision included. Three recorders each do it on a timer.

Four defences, in the order they matter:

1. **Store writes run in a thread** (`to_thread` at each recorder's call site),
   so the stall leaves the event loop entirely. Correct regardless of whether
   these share a process.
2. **The trading gate.** A recorder awaits `gate.idle()` before starting a
   cycle, so it never begins work while a decision is in flight. Recorders are
   on a sixty-second cadence and a few seconds of deferral costs them nothing;
   a decision delayed costs edge directly.
3. **Phased cadences.** Decisions land about a second past minutes 3, 6, 9 and
   12 of each window. Recorders are offset to fire near thirty seconds past the
   minute, as far from those instants as a sixty-second period allows, so the
   gate is rarely even consulted.
4. **`store_sync` stays a subprocess.** Scraping and rebuilding Parquet is
   minutes of pure CPU; awaiting it in-process would stall trading no matter how
   the rest is arranged.

Measured headroom: the loop currently decides 5.7s after its offset, against a
information cliff at 45-60s. The point of the gate is not that the slack is
missing — it is that spending it on a Parquet write buys nothing.

**A failure in one component never stops another.** Split across containers,
Docker restarted each and a crash-looping recorder was invisible unless someone
looked. Here `supervise` catches, logs, and restarts with exponential backoff
that resets after a healthy run, while everything else keeps going. Trading
surviving a broken recorder is the case that matters.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Optional, Sequence

logger = logging.getLogger('run-live')

BACKOFF_START = 5.0
BACKOFF_MAX = 300.0
# A component that ran this long before failing gets a fresh backoff: an hour of
# work then a network blip is not the same as failing on startup.
BACKOFF_RESET_AFTER = 120.0


class TradingGate:
    """Closed while a trading decision is in flight; recorders wait on it.

    Re-entrant by count rather than a boolean, so a nested cycle cannot reopen
    the gate early, and released in a `finally` so a cycle that raises does not
    wedge every recorder forever.
    """

    def __init__(self) -> None:
        self._depth = 0
        self._clear = asyncio.Event()
        self._clear.set()

    @property
    def is_idle(self) -> bool:
        return self._depth == 0

    async def idle(self) -> None:
        """Return once no decision is in flight."""
        await self._clear.wait()

    @contextlib.asynccontextmanager
    async def deciding(self):
        self._depth += 1
        self._clear.clear()
        try:
            yield
        finally:
            self._depth -= 1
            if self._depth <= 0:
                self._depth = 0
                self._clear.set()


async def supervise(
    name: str,
    factory: Callable[[], Awaitable[object]],
    *,
    backoff: float = BACKOFF_START,
    max_backoff: float = BACKOFF_MAX,
    reset_after: float = BACKOFF_RESET_AFTER,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    now: Callable[[], float] = time.monotonic,
) -> None:
    """Run `factory()` forever, restarting it on failure with backoff.

    A component that RETURNS is restarted too. These are endless loops; one that
    returns has stopped doing its job, and treating that as success would leave a
    recorder silently dead behind a healthy-looking container.

    `CancelledError` is re-raised untouched — shutdown must stop a component, not
    restart it.
    """
    delay = backoff
    while True:
        started = now()
        try:
            await factory()
            logger.warning('%s returned on its own; restarting', name)
        except asyncio.CancelledError:
            logger.info('%s stopped', name)
            raise
        except Exception as exc:                          # noqa: BLE001
            logger.error('%s failed: %s', name, str(exc)[:200], exc_info=True)
        if now() - started >= reset_after:
            delay = backoff
        await sleep(delay)
        delay = min(delay * 2.0, max_backoff) if delay else backoff


@dataclass(frozen=True)
class Component:
    """One supervised loop, and when in the minute it prefers to run."""

    name: str
    # Seconds past the minute this component aims to start its cycle, chosen to
    # sit away from the decision instants at ~1s past minutes 3/6/9/12.
    phase: float = 30.0

    @staticmethod
    def selected(known: Sequence[str], *,
                 disable: Iterable[str] = ()) -> list[str]:
        """The components to run, refusing an unknown name rather than ignoring it.

        A typo in `--disable` that silently ran everything would be the worst
        outcome — this is the flag someone reaches for to keep a component from
        touching a live account.
        """
        unknown = sorted(set(disable) - set(known))
        if unknown:
            raise ValueError(
                f'unknown component(s) {", ".join(unknown)}; known: '
                f'{", ".join(known)}')
        return [name for name in known if name not in set(disable)]


async def align_to_phase(phase: float, *,
                         period: float = 60.0,
                         sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                         now: Callable[[], float] = time.time) -> None:
    """Wait until `phase` seconds past the next period boundary.

    Called once at startup so a recorder's whole cadence sits away from the
    decision instants, rather than drifting onto them.
    """
    current = now() % period
    wait = (phase - current) % period
    if wait > 0:
        await sleep(wait)


# ---------------------------------------------------------------------------
# The components
# ---------------------------------------------------------------------------

# Phases in seconds past the minute. Decisions land ~1s past minutes 3/6/9/12,
# so the recorders are spread across the far side of the minute: away from the
# decision instant and away from each other, so no two Parquet writes queue on
# the same thread at the same moment.
COMPONENTS: tuple[Component, ...] = (
    Component('trade', phase=0.0),          # keeps its own schedule
    # Continuous, not periodic, and never gated: a stream that pauses goes
    # stale, which is the failure the book cache exists to prevent. Its phase is
    # meaningless and only the spool flush takes the gate.
    Component('stream', phase=0.0),
    Component('ladder', phase=25.0),
    Component('pm_ladder', phase=35.0),
    Component('implied_vol', phase=45.0),
    Component('store_sync', phase=50.0),
)
NAMES = tuple(c.name for c in COMPONENTS)


def _recorder_args(build_parser, **over):
    """Genuine defaults from a recorder's own `build_parser()`, not a copy.

    This used to hand-build an `argparse.Namespace` with literal defaults
    captioned "the defaults each recorder's own parser would have produced" —
    a second, hardcoded copy of numbers that already lived in three other
    files. The two agreed today; nothing enforced that they would keep
    agreeing, and a default changed in one recorder's own parser (used when it
    runs standalone via `python -m scripts.record_X`) would silently diverge
    from what the supervised `live` service actually uses. Parsing `[]`
    against the real parser makes that impossible: there is exactly one
    definition of each default, in the file that owns it.
    """
    ns = build_parser().parse_args([])
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


async def store_sync_loop(*, every: float = 3600.0) -> None:
    """Scrape recent bars and rebuild the store, as an awaited SUBPROCESS.

    `scrape` and `sync_store` used to be one-off `docker compose run` commands,
    so nothing ran them once the live loop started. Measured 2026-08-25: the
    research store's newest bar was 2026-08-23 04:17 while live windows ran to
    2026-08-25 00:00 — so not one traded window could be replayed offline, and
    the Kalshi quote archive was unusable for comparing a candidate model
    against the market. That is the one thing the archive exists for.

    Deliberately not in-process. This is minutes of blocking CPU — HTTP paging
    then a full Parquet rebuild — and `to_thread` would not save it either,
    because the GIL is held through much of pandas. A subprocess is the only
    arrangement where a rebuild cannot touch the trading loop's latency.
    """
    while True:
        # Fold the closed hours of the frame spool into immutable Parquet. This
        # belongs here rather than in the stream recorder for the same reason
        # the rest of this loop does: it is blocking Parquet work, and the
        # stream must not stop reading to do it.
        try:
            from core.datastore import DEFAULT_ROOT
            from core.spool import DEFAULT_SPOOL_ROOT, compact
            rows = await asyncio.to_thread(
                compact, DEFAULT_SPOOL_ROOT, DEFAULT_ROOT)
            if rows:
                logger.info('compacted %d book-event rows', rows)
        except Exception as exc:                      # noqa: BLE001
            logger.warning('spool compaction: %s', str(exc)[:200])
        # FOUR stages, in dependency order. The first two were here from the
        # start; the second two were one-off commands nobody ran once the loop
        # started, which is the same defect this docstring already describes for
        # `scrape` and `sync_store` — and worse, because it is invisible. Live
        # keeps trading and recording while the TRAINING set stops advancing.
        #
        # Measured six days into the live run: minute_bars, kalshi venue_depth
        # and venue_implied_vol were all current, while venue_settlements — the
        # LABEL — was seven days stale and polymarket venue_depth six. So no
        # window after 2026-08-27 could be trained on, and a fresh walk-forward
        # reproduced the previous run exactly. That reads like a stable model
        # and is a stalled pipeline.
        #
        # `collect_settlements` stops at history it already holds, so keeping
        # current is a page or two rather than a full walk. `build_depth` folds
        # every BOOK source into venue_depth at every minute, which is what
        # carries the recorded Polymarket ladders across for `cross_venue`.
        for step in (('scripts.scrape', '--backfill-days', '3'),
                     ('scripts.sync_store',),
                     ('scripts.build_depth',),
                     ('scripts.collect_settlements', '--venue', 'both')):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-m', *step,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE)
            _out, err = await proc.communicate()
            if proc.returncode:
                logger.warning('%s exited %s: %s', step[0], proc.returncode,
                               (err or b'').decode(errors='replace')[:300])
            else:
                logger.info('%s done', step[0])
        await asyncio.sleep(every)


def build_factories(args, gate: TradingGate) -> dict:
    """One coroutine factory per component, all sharing the gate."""
    from scripts import (record_implied_vol, record_ladder, record_pm_ladder,
                         record_stream)
    from scripts import live as live_module

    trade_argv = list(args.trade_args)

    async def trade():
        # The loop holds the gate across each cycle. It is the only component
        # whose timing matters, so it never waits on anything else.
        return await live_module.main(trade_argv, gate=gate)

    async def stream():
        return await record_stream.run(
            _recorder_args(record_stream.build_parser), gate=gate)

    async def ladder():
        await align_to_phase(25.0)
        return await record_ladder.run(
            _recorder_args(record_ladder.build_parser), gate=gate)

    async def pm_ladder():
        await align_to_phase(35.0)
        return await record_pm_ladder.run(
            _recorder_args(record_pm_ladder.build_parser, batch_rows=6), gate=gate)

    async def implied_vol():
        await align_to_phase(45.0)
        return await record_implied_vol.run(
            _recorder_args(record_implied_vol.build_parser, batch_rows=10),
            gate=gate)

    async def store_sync():
        await align_to_phase(50.0)
        return await store_sync_loop()

    return {'trade': trade, 'stream': stream, 'ladder': ladder,
            'pm_ladder': pm_ladder, 'implied_vol': implied_vol,
            'store_sync': store_sync}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--disable', action='append', default=[], metavar='COMPONENT',
        help=f'skip a component; repeatable. One of: {", ".join(NAMES)}')
    parser.add_argument(
        '--trade-args', nargs=argparse.REMAINDER, default=[],
        help='everything after this is passed to scripts.live verbatim')
    parser.add_argument('--verbose', action='store_true')
    return parser


async def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)-7s %(name)-14s %(message)s',
        datefmt='%H:%M:%S', stream=sys.stdout)

    try:
        selected = Component.selected(NAMES, disable=args.disable)
    except ValueError as exc:
        raise SystemExit(str(exc))

    gate = TradingGate()
    factories = build_factories(args, gate)

    print('=' * 78)
    print('Quarter — one process')
    print('=' * 78)
    for name in NAMES:
        state = 'on ' if name in selected else 'off'
        phase = next(c.phase for c in COMPONENTS if c.name == name)
        note = 'holds the gate' if name == 'trade' else f'+{phase:.0f}s past the minute'
        print(f'  {state}  {name:<12} {note}')
    print(f"  trade args: {' '.join(args.trade_args) or '(none)'}")
    print()

    tasks = [asyncio.create_task(supervise(name, factories[name]), name=name)
             for name in selected]
    if not tasks:
        raise SystemExit('every component disabled; nothing to run')
    # **A deliberate exit from any component stops the whole process.**
    # `live.main` raises SystemExit for a config error or a trader lock already
    # held, and both mean "do not run". Restarting would either spin on a bad
    # flag or fight another trader for one account. Everything else is caught
    # and retried inside `supervise`, so reaching here is a decision, not a bug —
    # it is reported as one line rather than an asyncio traceback.
    try:
        await asyncio.gather(*tasks)
        return 0
    except asyncio.CancelledError:
        return 0
    except SystemExit as exc:
        logger.error('stopping: %s', exc)
        return 1
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


if __name__ == '__main__':
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        logger.info('stopped')
        code = 0
    raise SystemExit(code)
