"""Tests that the live Polymarket price in `venue_prob_gap` is fresh and from the same window."""
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
    """A stale peer reading is refused: the gap would include Kalshi's own move since."""
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
    """A reading with no timestamp is kept, as in the backtest."""
    row = cross_venue_row({'best_bid': 40.0, 'best_ask': 42.0}, _Quote(), now=NOW)
    assert not np.isnan(row['venue_prob_gap'])


def test_the_recorder_is_phased_to_be_fresh_at_a_decision():
    """The recorder runs late in the minute, so its reading is a few seconds old at the
    decision.
    """
    from scripts.run_live import COMPONENTS
    pm = next(c for c in COMPONENTS if c.name == 'pm_ladder')
    staleness_at_decision = (60 - pm.phase) + 2
    assert staleness_at_decision < 15, (
        f'pm_ladder at +{pm.phase}s leaves a decision reading a '
        f'{staleness_at_decision:.0f}s-old book')


def test_a_peer_reading_from_another_window_is_refused():
    """A peer reading from another window is refused, however fresh.

    At a window rollover the cache can briefly hold the previous window's book with a
    recent timestamp.
    """
    pm = {'best_bid': 40.0, 'best_ask': 42.0,
          'at': NOW - pd.Timedelta(seconds=3),
          'window': pd.Timestamp('2026-09-03 12:00', tz='UTC')}
    row = cross_venue_row(pm, _Quote(), now=NOW,
                          window_open=pd.Timestamp('2026-09-03 12:15', tz='UTC'))
    assert np.isnan(row['venue_prob_gap'])
    assert row['pm_available'] == 0.0


def test_the_matching_window_is_used():
    pm = {'best_bid': 40.0, 'best_ask': 42.0,
          'at': NOW - pd.Timedelta(seconds=3),
          'window': pd.Timestamp('2026-09-03 12:00', tz='UTC')}
    row = cross_venue_row(pm, _Quote(), now=NOW,
                          window_open=pd.Timestamp('2026-09-03 12:00', tz='UTC'))
    assert not np.isnan(row['venue_prob_gap'])


def test_a_reading_with_no_window_is_still_accepted():
    """A reading with no window (the REST fallback) is still accepted."""
    pm = {'best_bid': 40.0, 'best_ask': 42.0, 'at': NOW - pd.Timedelta(seconds=3)}
    row = cross_venue_row(pm, _Quote(), now=NOW,
                          window_open=pd.Timestamp('2026-09-03 12:15', tz='UTC'))
    assert not np.isnan(row['venue_prob_gap'])


def test_the_declared_phase_is_the_one_actually_used():
    """The recorder aligns on the declared `Component.phase`, not a separate hardcoded number."""
    import inspect
    from scripts.run_live import COMPONENTS, build_factories

    source = inspect.getsource(build_factories)
    body = source.split('async def pm_ladder')[1].split('async def')[0]
    # Either it reads the declaration or it repeats the same number. A different
    # literal is the drift this test catches.
    declared = next(c for c in COMPONENTS if c.name == 'pm_ladder').phase
    reads_declaration = 'COMPONENTS' in body and 'phase' in body
    import re
    literals = [float(x) for x in re.findall(r'align_to_phase\(([0-9.]+)\)', body)]
    assert reads_declaration or literals == [declared], (
        f'pm_ladder declares phase={declared} but aligns on {literals}')
