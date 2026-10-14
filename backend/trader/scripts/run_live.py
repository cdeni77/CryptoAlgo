"""Run the trading loop and every recorder in one process.

Trading latency is the main constraint. `ResearchStore.write` is synchronous
Parquet work that blocks the event loop, so:

1. Store writes run in a thread.
2. A recorder awaits `gate.idle()` before a cycle, so it never starts work
   during a decision.
3. Recorder cadences are phased away from the decision instants.
4. `store_sync` runs as a subprocess, because it is minutes of CPU.

A failure in one component never stops another: `supervise` restarts it with
backoff.
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

from core import heartbeat

logger = logging.getLogger('run-live')

BACKOFF_START = 5.0
BACKOFF_MAX = 300.0
# A component that ran this long before failing gets a fresh backoff.
BACKOFF_RESET_AFTER = 120.0


class TradingGate:
    """Closed while a trading decision is in flight; recorders wait on it.

    Counts holders, so a nested cycle cannot reopen it early. Released in a
    `finally`, so a cycle that raises does not block the recorders.
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
    """Run `factory()` forever, restarting it with backoff on failure.

    A component that returns is restarted too, because these are endless loops.
    `CancelledError` is re-raised, so shutdown stops a component.
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
    """One supervised loop and its preferred start within the minute."""

    name: str
    # Seconds past the minute this component starts its cycle, away from the
    # decision instants at ~1s past minutes 3/6/9/12.
    phase: float = 30.0

    @staticmethod
    def selected(known: Sequence[str], *,
                 disable: Iterable[str] = ()) -> list[str]:
        """Return the components to run; refuse an unknown name in `--disable`."""
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

    Called once at startup, so a recorder's cadence stays away from the
    decision instants.
    """
    current = now() % period
    wait = (phase - current) % period
    if wait > 0:
        await sleep(wait)


# ---------------------------------------------------------------------------
# The components
# ---------------------------------------------------------------------------

# Phases in seconds past the minute. Recorders sit away from the decision
# instant and from each other, so no two Parquet writes queue together.
COMPONENTS: tuple[Component, ...] = (
    Component('trade', phase=0.0),          # keeps its own schedule
    # Sockets hold a subscription, so phase has no meaning. They are never
    # gated, because a paused stream goes stale. Only the spool flush takes
    # the gate.
    Component('stream', phase=0.0),
    # The Polymarket socket keeps the peer book contemporaneous with Kalshi's.
    Component('pm_stream', phase=0.0),
    # The Coinbase trade socket. The trading loop reads its bars on the
    # decision path.
    Component('spot_stream', phase=0.0),
    # The settlement-index proxy: constituent-exchange sockets and a
    # per-second sampler.
    Component('index', phase=0.0),
    Component('ladder', phase=25.0),
    # +55s keeps the REST Polymarket reading a few seconds old at the next
    # decision, well inside the backtest's peer age cap.
    Component('pm_ladder', phase=55.0),
    Component('implied_vol', phase=45.0),
    Component('store_sync', phase=50.0),
)
NAMES = tuple(c.name for c in COMPONENTS)


def _recorder_args(build_parser, **over):
    """Return a recorder's defaults from its own `build_parser()`, with overrides.

    Parsing `[]` keeps one definition of each default, in the recorder's file.
    """
    ns = build_parser().parse_args([])
    for key, value in over.items():
        setattr(ns, key, value)
    return ns


async def store_sync_loop(*, every: float = 3600.0) -> None:
    """Keep the research store current by running each step as a subprocess.

    Not in-process, and not in a thread: the work is minutes of CPU and holds
    the GIL, so only a subprocess cannot affect trading latency.
    """
    # Stamp liveness on entry too, because the first pass takes minutes.
    heartbeat.touch('store_sync')
    while True:
        # Fold closed spool hours into Parquet here, not in the stream
        # recorder, so the stream never stops reading. Each raw table is
        # pruned to its retention, then the free-disk guard runs.
        try:
            from core.datastore import DEFAULT_ROOT
            from core.spool import DEFAULT_SPOOL_ROOT, compact_all
            rows = await asyncio.to_thread(
                compact_all, DEFAULT_SPOOL_ROOT, DEFAULT_ROOT)
            for dataset, count in rows.items():
                if count:
                    logger.info('compacted %d %s rows', count, dataset)
        except Exception as exc:                      # noqa: BLE001
            logger.warning('spool compaction: %s', str(exc)[:200])
        # Stages in dependency order. Each one feeds the training set; a
        # missing stage stalls it while live trading looks healthy.
        for step in (('scripts.scrape', '--backfill-days', '3'),
                     ('scripts.sync_store',),
                     ('scripts.build_depth',),
                     ('scripts.collect_settlements', '--venue', 'both'),
                     # After collect_settlements, because it reads what that
                     # step writes. Idempotent.
                     ('scripts.correct_outcomes',)):
            proc = await asyncio.create_subprocess_exec(
                sys.executable, '-m', *step,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE)
            # Bounded: a wedged child would otherwise park this coroutine
            # forever, and `supervise` cannot detect that.
            try:
                _out, err = await asyncio.wait_for(proc.communicate(),
                                                   timeout=STEP_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.error('%s exceeded %.0fs and was killed; the store is '
                             'now BEHIND and a re-run of the evaluation will '
                             'reproduce the previous one',
                             step[0], STEP_TIMEOUT_SECONDS)
                try:
                    proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass
                continue
            if proc.returncode:
                logger.warning('%s exited %s: %s', step[0], proc.returncode,
                               (err or b'').decode(errors='replace')[:300])
            else:
                logger.info('%s done', step[0])
        # Liveness, stamped once a pass has completed.
        heartbeat.touch('store_sync')
        await asyncio.sleep(every)


# How long one store-sync child may run before it is killed. A normal step
# takes minutes and the loop runs hourly, so this catches a wedge before the
# next pass.
STEP_TIMEOUT_SECONDS = 1800.0


def build_factories(args, gate: TradingGate) -> dict:
    """Return one coroutine factory per component, all sharing the gate."""
    from scripts import (record_implied_vol, record_index, record_ladder,
                         record_pm_ladder, record_pm_stream,
                         record_spot_stream, record_stream)
    from scripts import live as live_module

    trade_argv = list(args.trade_args)

    async def trade():
        # The loop holds the gate across each cycle and never waits on it.
        return await live_module.main(trade_argv, gate=gate)

    async def stream():
        return await record_stream.run(
            _recorder_args(record_stream.build_parser), gate=gate)

    async def ladder():
        await align_to_phase(25.0)
        return await record_ladder.run(
            _recorder_args(record_ladder.build_parser), gate=gate)

    async def pm_ladder():
        # Read the phase from the declaration, so the two cannot drift.
        await align_to_phase(
            next(c for c in COMPONENTS if c.name == 'pm_ladder').phase)
        return await record_pm_ladder.run(
            _recorder_args(record_pm_ladder.build_parser, batch_rows=6), gate=gate)

    async def pm_stream():
        # No phase: a socket holds a subscription and has no cadence.
        return await record_pm_stream.run(gate=gate)

    async def spot_stream():
        return await record_spot_stream.run(
            _recorder_args(record_spot_stream.build_parser), gate=gate)

    async def index():
        return await record_index.run(
            _recorder_args(record_index.build_parser), gate=gate)

    async def implied_vol():
        await align_to_phase(45.0)
        return await record_implied_vol.run(
            _recorder_args(record_implied_vol.build_parser, batch_rows=10),
            gate=gate)

    async def store_sync():
        await align_to_phase(50.0)
        return await store_sync_loop()

    return {'trade': trade, 'stream': stream, 'ladder': ladder,
            'pm_ladder': pm_ladder, 'pm_stream': pm_stream,
            'spot_stream': spot_stream, 'index': index,
            'implied_vol': implied_vol,
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
    # A SystemExit from any component stops the whole process. `live.main`
    # raises it for a config error or a held trader lock; a restart would spin
    # or fight another trader. `supervise` retries everything else.
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
