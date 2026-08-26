"""One container, five loops, and a failure in one must not stop the others.

The live side ran as four containers plus a shell loop because they were added
one at a time. They are the same shape — async loops polling an HTTP API on a
sixty-second cadence — so they belong in one process, and a single restart
policy and log stream is easier to trust than five.

**The supervisor is the whole reason this is safe.** Split across containers,
Docker restarted each one and a crash-looping recorder was invisible unless you
went looking. Here each component is wrapped: an exception is logged and the
component restarts with exponential backoff, while everything else keeps
running. Trading must survive a broken recorder, which is the case that matters.

Backoff resets after a component has run for a while, so a loop that works for
an hour and then fails does not inherit the delay from a failure last week.

`store_sync` stays a subprocess rather than an in-process task. It scrapes and
rebuilds Parquet — CPU-bound, blocking work — and awaiting it in this event loop
would stall the trading decision. The measured latency budget has a cliff at
45-60 seconds and a sync takes minutes.
"""

from __future__ import annotations

import asyncio

import pytest

from scripts.run_live import Component, supervise


def test_a_component_that_raises_is_restarted():
    starts = []

    async def flaky():
        starts.append(1)
        if len(starts) < 3:
            raise RuntimeError('boom')
        await asyncio.sleep(3600)

    async def drive():
        task = asyncio.create_task(
            supervise('flaky', flaky, backoff=0.0, sleep=_nosleep))
        for _ in range(40):
            await asyncio.sleep(0)
            if len(starts) >= 3:
                break
        task.cancel()
        return len(starts)

    assert asyncio.run(drive()) == 3


def test_backoff_grows_on_repeated_immediate_failures():
    waits = []

    async def always_fails():
        raise RuntimeError('boom')

    async def sleeper(seconds):
        waits.append(seconds)
        if len(waits) >= 4:
            raise asyncio.CancelledError

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await supervise('bad', always_fails, backoff=1.0, max_backoff=8.0,
                            sleep=sleeper)

    asyncio.run(drive())
    assert waits == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped():
    waits = []

    async def always_fails():
        raise RuntimeError('boom')

    async def sleeper(seconds):
        waits.append(seconds)
        if len(waits) >= 5:
            raise asyncio.CancelledError

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await supervise('bad', always_fails, backoff=4.0, max_backoff=8.0,
                            sleep=sleeper)

    asyncio.run(drive())
    assert waits == [4.0, 8.0, 8.0, 8.0, 8.0]


def test_backoff_resets_after_a_long_healthy_run():
    """An hour of working then failing is not the same as failing instantly."""
    waits = []
    clock = {'t': 0.0}

    async def slow_then_fails():
        clock['t'] += 100.0
        raise RuntimeError('boom')

    async def sleeper(seconds):
        waits.append(seconds)
        if len(waits) >= 3:
            raise asyncio.CancelledError

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await supervise('slow', slow_then_fails, backoff=1.0,
                            max_backoff=64.0, reset_after=10.0, sleep=sleeper,
                            now=lambda: clock['t'])

    asyncio.run(drive())
    assert waits == [1.0, 1.0, 1.0], 'each run lasted past reset_after'


def test_one_failing_component_does_not_stop_the_others():
    """The case that matters: a broken recorder must not halt trading."""
    ticks = []

    async def broken():
        raise RuntimeError('recorder is down')

    async def trading():
        while True:
            ticks.append(1)
            await asyncio.sleep(0)

    async def drive():
        tasks = [
            asyncio.create_task(supervise('broken', broken, backoff=0.0,
                                          sleep=_nosleep)),
            asyncio.create_task(supervise('trading', trading, backoff=0.0,
                                          sleep=_nosleep)),
        ]
        for _ in range(50):
            await asyncio.sleep(0)
        for t in tasks:
            t.cancel()
        return len(ticks)

    assert asyncio.run(drive()) > 5


def test_cancellation_is_not_swallowed_as_a_failure():
    """Shutdown must stop the component, not restart it."""
    starts = []

    async def component():
        starts.append(1)
        await asyncio.sleep(3600)

    async def drive():
        task = asyncio.create_task(
            supervise('c', component, backoff=0.0, sleep=_nosleep))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return len(starts)

    assert asyncio.run(drive()) == 1


def test_a_component_returning_normally_is_still_restarted():
    """These are endless loops. One that returns has stopped doing its job."""
    starts = []

    async def exits():
        starts.append(1)
        return 0

    async def sleeper(_seconds):
        if len(starts) >= 3:
            raise asyncio.CancelledError

    async def drive():
        with pytest.raises(asyncio.CancelledError):
            await supervise('exits', exits, backoff=0.0, sleep=sleeper)

    asyncio.run(drive())
    assert len(starts) == 3


def test_disabled_components_are_not_started():
    enabled = Component.selected(
        ['trade', 'ladder', 'pm_ladder', 'implied_vol', 'store_sync'],
        disable=('pm_ladder', 'store_sync'))
    assert enabled == ['trade', 'ladder', 'implied_vol']


def test_an_unknown_component_name_is_refused_not_ignored():
    """A typo in a flag must not silently run everything."""
    with pytest.raises(ValueError, match='nonsense'):
        Component.selected(['trade'], disable=('nonsense',))


async def _nosleep(_seconds):
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# The trading gate. Recorders must never start a cycle while a decision is in
# flight, because the decision is the only latency-sensitive thing here.
# ---------------------------------------------------------------------------

from scripts.run_live import TradingGate


def test_the_gate_is_open_when_nothing_is_deciding():
    async def drive():
        gate = TradingGate()
        await asyncio.wait_for(gate.idle(), timeout=1.0)
        return True

    assert asyncio.run(drive())


def test_a_recorder_waits_while_a_decision_is_in_flight():
    order = []

    async def drive():
        gate = TradingGate()

        async def decide():
            async with gate.deciding():
                order.append('decision start')
                await asyncio.sleep(0.05)
                order.append('decision end')

        async def record():
            await asyncio.sleep(0.01)
            await gate.idle()
            order.append('recorder')

        await asyncio.gather(decide(), record())

    asyncio.run(drive())
    assert order == ['decision start', 'decision end', 'recorder']


def test_the_gate_reopens_even_if_the_decision_raises():
    """A crashing cycle must not wedge every recorder forever."""
    async def drive():
        gate = TradingGate()
        try:
            async with gate.deciding():
                raise RuntimeError('cycle blew up')
        except RuntimeError:
            pass
        await asyncio.wait_for(gate.idle(), timeout=1.0)
        return True

    assert asyncio.run(drive())


def test_concurrent_decisions_keep_the_gate_shut_until_the_last_one_leaves():
    async def drive():
        gate = TradingGate()
        async with gate.deciding():
            async with gate.deciding():
                pass
            # still inside the outer one
            assert not gate.is_idle
        return gate.is_idle

    assert asyncio.run(drive())


# ---------------------------------------------------------------------------
# Phasing. Decisions land ~1s past minutes 3/6/9/12; recorders aim for ~30s
# past the minute so the gate is rarely even consulted.
# ---------------------------------------------------------------------------

from scripts.run_live import align_to_phase


def _phase_wait(current_second: float, phase: float = 30.0) -> float:
    waited = []

    async def sleeper(seconds):
        waited.append(seconds)

    asyncio.run(align_to_phase(phase, sleep=sleeper,
                               now=lambda: 1_000_000 * 60 + current_second))
    return waited[0] if waited else 0.0


def test_it_waits_forward_to_the_phase_within_the_same_minute():
    assert _phase_wait(10.0) == pytest.approx(20.0)


def test_it_wraps_to_the_next_minute_when_the_phase_has_passed():
    assert _phase_wait(45.0) == pytest.approx(45.0)


def test_it_does_not_wait_when_already_on_the_phase():
    assert _phase_wait(30.0) == 0.0


def test_the_default_phase_is_far_from_every_decision_instant():
    """Decisions fire ~1s past a minute. 30s is the maximum distance from that."""
    from scripts.live import DECISION_LAG_SECONDS
    decision_second = DECISION_LAG_SECONDS % 60.0
    gap = min(abs(30.0 - decision_second), 60.0 - abs(30.0 - decision_second))
    assert gap >= 25.0, 'the recorder phase has drifted onto the decision instant'


# ---------------------------------------------------------------------------
# A deliberate exit is not a failure to retry.
# ---------------------------------------------------------------------------

def test_a_systemexit_stops_the_component_rather_than_restarting_it():
    """`live.main` raises SystemExit for a config error or a held trader lock.

    Both mean "do not run", and retrying forever would either spin on a bad flag
    or fight another trader for the same account. Observed for real: the runner
    started while the old container still held the lock, and the only correct
    response is to stop.
    """
    starts = []

    async def refuses():
        starts.append(1)
        raise SystemExit('another process already holds the trading lock')

    async def drive():
        with pytest.raises(SystemExit):
            await supervise('trade', refuses, backoff=0.0, sleep=_nosleep)

    asyncio.run(drive())
    assert starts == [1], 'it must not have been retried'


def test_a_keyboard_interrupt_is_not_treated_as_a_crash():
    async def interrupted():
        raise KeyboardInterrupt

    async def drive():
        with pytest.raises(KeyboardInterrupt):
            await supervise('c', interrupted, backoff=0.0, sleep=_nosleep)

    asyncio.run(drive())
