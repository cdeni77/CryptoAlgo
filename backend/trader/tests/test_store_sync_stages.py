"""`store_sync_loop` kept only two of the four stages the store needs.

Its own docstring records this bug class being fixed once already: "`scrape`
and `sync_store` used to be one-off `docker compose run` commands, so nothing
ran them once the live loop started." Exactly that was then true of
`build_depth` and `collect_settlements`, and the effect is worse because it is
invisible — the live loop keeps trading and recording while the TRAINING set
silently stops advancing.

Measured 2026-09-03, six days into the live run:

    minute_bars              max 2026-09-03 14:19   current
    venue_depth  [kalshi]    max 2026-09-03 15:15   current
    venue_implied_vol        max 2026-09-03 15:15   current
    venue_settlements        max 2026-08-27 23:15   7 days stale  <- the LABEL
    venue_depth [polymarket] max 2026-08-28 13:30   6 days stale  <- cross_venue

`venue_settlements` is where the target comes from, so no window after Aug 27
could be trained on at all: a fresh walk-forward run produced fold 5 ending
2026-08-27 and skill identical to the run five days earlier, which reads exactly
like a stable model and is actually a stalled pipeline.
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
    """Four stages in one loop: settlements depend on a venue being reachable
    and build_depth on the ladders already landing. One failing must not cost
    the rest their turn, or a transient venue error freezes the training set
    until someone notices."""
    source = inspect.getsource(run_live.store_sync_loop)
    assert 'continue' in source or 'returncode' in source, (
        'the loop must survive one stage failing')
