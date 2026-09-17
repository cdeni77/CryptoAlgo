"""Buffered rows must survive a shutdown.

Three recorders accumulate into a local list and write when it reaches
`--batch-rows`. Their only handler was `except Exception`, and
`asyncio.CancelledError` is a **BaseException** -- so `run_live`'s
`task.cancel()` discarded whatever was buffered. Up to a full batch per restart,
on datasets whose entire justification is that a minute not recorded is gone.

`record_stream` was already correct; these three were not.
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

import scripts.record_implied_vol as iv
import scripts.record_ladder as ladder
import scripts.record_pm_ladder as pm

MODULES = (ladder, pm, iv)


@pytest.mark.parametrize('mod', MODULES, ids=lambda m: m.__name__.split('.')[-1])
def test_cancellation_is_caught_before_the_broad_handler(mod):
    src = inspect.getsource(mod.run)
    assert 'asyncio.CancelledError' in src, (
        'CancelledError is a BaseException; `except Exception` never sees a '
        'shutdown and the buffer is lost')
    cancel_at = src.index('except asyncio.CancelledError')
    broad_at = src.index('except Exception as exc:                          # noqa: BLE001 - reconnect')
    assert cancel_at < broad_at, 'the cancel handler must come first'


@pytest.mark.parametrize('mod', MODULES, ids=lambda m: m.__name__.split('.')[-1])
def test_the_cancel_handler_flushes_and_reraises(mod):
    src = inspect.getsource(mod.run)
    tail = src[src.index('except asyncio.CancelledError'):]
    block = tail[:tail.index('except Exception')]
    assert '_flush_on_exit()' in block, 'must flush before going away'
    assert 'raise' in block, (
        'a swallowed CancelledError leaves the task running and breaks the '
        'supervisor shutdown')
