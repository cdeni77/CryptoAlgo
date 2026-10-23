"""The freshness measurement must be published faster than the limits it feeds.

A limit tighter than the publish interval cannot be evaluated: the age the
dashboard shows is `now - newest row in the snapshot`, so it carries the
snapshot's own lag. Measured 2026-10-23, `venue_ladder`, `pm_ladder` and
`venue_trades` read 39-43 minutes late off a snapshot published hourly, while
every one of them had written inside the last 82 to 202 seconds.

A monitor that is usually wrong is the one nobody reads, which is how
`venue_ladder` went thirty-one days unnoticed.
"""

from __future__ import annotations

import pytest

from core.freshness import EXPECTED_MAX_AGE
from core.heartbeat import MAX_AGE as HEARTBEAT_MAX_AGE
from scripts.run_live import COMPONENTS, FRESHNESS_EVERY, NAMES


def test_the_publish_interval_is_inside_every_limit_it_feeds():
    limits = [v for v in EXPECTED_MAX_AGE.values() if v is not None]
    assert limits, 'no dataset limits to check against'
    tightest = min(limits)
    # Half, not just under: the publish is one point in a cycle, so an age read
    # between two passes is already up to one interval behind.
    assert FRESHNESS_EVERY <= tightest / 2, (
        f'published every {FRESHNESS_EVERY}s against a {tightest}s limit; '
        f'that limit cannot be evaluated')


def test_freshness_is_its_own_component():
    """It rode the hourly store pass, which is where the lag came from."""
    assert 'freshness' in NAMES
    store = next(c for c in COMPONENTS if c.name == 'store_sync')
    fresh = next(c for c in COMPONENTS if c.name == 'freshness')
    assert fresh.phase != store.phase, 'it must not share the store pass slot'


def test_the_loop_has_a_liveness_limit():
    """If it stops, every age on the dashboard freezes and then reads older --
    indistinguishable on the surface from the recorders themselves dying."""
    assert 'freshness' in HEARTBEAT_MAX_AGE
    assert HEARTBEAT_MAX_AGE['freshness'] >= FRESHNESS_EVERY * 2
