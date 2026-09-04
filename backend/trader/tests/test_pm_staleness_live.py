"""Live read the Polymarket price from a cache written ~30 seconds earlier.

Training aligns the two venues: `event_time` is identical on 100% of 693,805
matched pairs, and on the rows the model sees Polymarket is 4.0 SECONDS STALER
on average. So `venue_prob_gap` in training is a contemporaneous disagreement
between two books.

Live it was not. `record_pm_ladder` wrote at a median of 32 seconds past the
minute while decisions land at about +2s, so the freshest reading available to a
decision was ~30 seconds old — and `cross_venue_row` carried no staleness check,
so the model could not tell. A 30-second-old peer price makes the "gap" partly a
measure of KALSHI'S OWN MOVE over those thirty seconds, which is a different
quantity from the one the model was fitted on.

That matters more here than anywhere else: `cross_venue` is the only
load-bearing group in the model — dropping it takes skill from +0.00282 to
-0.00015.

Two fixes, and both are needed. The guard refuses a reading past the same
`max_age_seconds` the backtest applies to both venues, so a stale gap is NaN
rather than wrong. The re-phasing makes the reading fresh enough to survive that
guard most of the time, so the fix buys parity rather than coverage loss.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd
import pytest

from scripts.live import cross_venue_row


class _Quote:
    yes_bid, yes_ask = 0.50, 0.52


NOW = pd.Timestamp('2026-09-03 12:12:02', tz='UTC')


def _pm(age_seconds, bid=40.0, ask=42.0):
    return {'best_bid': bid, 'best_ask': ask,
            'at': NOW - pd.Timedelta(seconds=age_seconds)}


def test_a_fresh_peer_reading_is_used():
    row = cross_venue_row(_pm(5), _Quote(), now=NOW)
    assert not np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 1.0


def test_a_stale_peer_reading_is_refused():
    """Past the bar the backtest applies to both venues, the gap is a different
    quantity — it carries Kalshi's own move over the interval."""
    row = cross_venue_row(_pm(90), _Quote(), now=NOW)
    assert np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 0.0, (
        'a refused reading must read as ABSENT, not as two venues agreeing')


def test_the_boundary_matches_the_backtest_tolerance():
    from core.quotes import DEFAULT_MAX_AGE
    assert not np.isnan(cross_venue_row(_pm(DEFAULT_MAX_AGE - 1), _Quote(),
                                        now=NOW)['venue_prob_gap'])
    assert np.isnan(cross_venue_row(_pm(DEFAULT_MAX_AGE + 1), _Quote(),
                                    now=NOW)['venue_prob_gap'])


def test_a_reading_with_no_timestamp_is_kept():
    """Absence of a stamp is not evidence of staleness, matching the backtest,
    where a peer row without `quote_age_seconds` is kept."""
    row = cross_venue_row({'best_bid': 40.0, 'best_ask': 42.0}, _Quote(), now=NOW)
    assert not np.isnan(row['venue_prob_gap'])


def test_the_recorder_is_phased_to_be_fresh_at_a_decision():
    """Decisions land ~2s past the minute. A recorder firing at +35s leaves the
    freshest reading ~27s old — inside the 30s bar, but only just, and past it
    on the tail. Phased late in the minute it is a few seconds old."""
    from scripts.run_live import COMPONENTS
    pm = next(c for c in COMPONENTS if c.name == 'pm_ladder')
    staleness_at_decision = (60 - pm.phase) + 2
    assert staleness_at_decision < 15, (
        f'pm_ladder at +{pm.phase}s leaves a decision reading a '
        f'{staleness_at_decision:.0f}s-old book')
