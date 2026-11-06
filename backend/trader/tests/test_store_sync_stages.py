"""Tests that `store_sync_loop` runs every stage the training set needs.

Without `build_depth` and `collect_settlements` the live loop keeps running while the
training data (including the settlement label) stops advancing.
"""
from __future__ import annotations

import inspect

from scripts import run_live

REQUIRED = ('scripts.scrape', 'scripts.sync_store',
            'scripts.build_depth', 'scripts.collect_settlements')


def test_every_stage_the_training_set_needs_is_run():
    source = inspect.getsource(run_live.store_sync_loop)
    missing = [s for s in REQUIRED if s not in source]
    assert not missing, (
        f'{missing} are never run once the live loop starts, so the tables they '
        f'feed stop advancing while everything else looks healthy')


def test_a_failing_stage_does_not_stop_the_others():
    source = inspect.getsource(run_live.store_sync_loop)
    # **Both, not either.** `scripts/run_live.py` contains each independently
    # -- `continue` in the timeout handler, `returncode` in the exit check --
    # so the disjunction could not go red for any single deletion.
    assert 'continue' in source, 'the loop must resume after a stage times out'
    assert 'returncode' in source, 'a stage exit code must be inspected'
