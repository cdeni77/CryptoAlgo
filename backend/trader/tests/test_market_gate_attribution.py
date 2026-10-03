"""The market gates must measure the CANDIDATE, not its predecessor.

`market_measurement` preferred the live `predictions` table, on the stated
ground that those rows carry "the venue's quote at the instant a decision was
actually made". True — and it created a deadlock.

Those rows were produced by whatever model was deployed at the time, and
`model_version` is NULL on every one of them. A new candidate cannot change a
single row. So a candidate was gated on its predecessor's live track record:

    2026-09-20  blocked  calibration_vs_market      (8,411 live windows)
    2026-09-27  blocked  calibration_vs_market, model_minus_market

Both candidates' own numbers passed — skill +0.0027 over 24,247 windows, every
other gate green. The model froze on 20260917T021814Z for sixteen days and
would have stayed there indefinitely: the live figures can only improve if a
new model is deployed, and no new model could be deployed.

The reason for preferring the live table also expired. Since the quote-source
fix of 2026-09-16 the backtest prices against the same `live_touch` quotes
where they exist, so the choice is no longer observation-versus-reconstruction.
`recorded_only` keeps that property while restoring attribution.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from core.metrics import LIVE_OBSERVERS, market_rows_from_scored


def _scored(observers, n_each=3):
    rows = []
    for i, obs in enumerate(observers):
        for k in range(n_each):
            rows.append({
                'symbol': 'BTC-USD',
                'window_open': pd.Timestamp('2026-09-01T12:00Z')
                + pd.Timedelta(minutes=15 * (i * n_each + k)),
                'offset': 12, 'market_probability': 0.55,
                'baseline_probability': 0.50, 'model_probability': 0.60,
                'outcome': 1.0, 'quote_observer': obs,
            })
    return pd.DataFrame(rows)


def test_recorded_only_keeps_the_watched_observers():
    frame = _scored(['live_touch', 'live_ws', 'live', 'backfill'])
    rows = market_rows_from_scored(frame, recorded_only=True)
    assert len(rows) == 9, 'three live observers at three rows each'


def test_recorded_only_drops_the_reconstruction():
    frame = _scored(['backfill'])
    assert market_rows_from_scored(frame, recorded_only=True) == []
    # ...but it is still a legitimate backtest price when not restricted.
    assert len(market_rows_from_scored(frame, recorded_only=False)) == 3


def test_a_frame_with_no_observer_column_is_refused_not_assumed():
    """Absent provenance must not be read as 'recorded'. That assumption is
    how a reconstruction would be graded as an observation."""
    frame = _scored(['live_touch']).drop(columns=['quote_observer'])
    assert market_rows_from_scored(frame, recorded_only=True) == []


def test_every_live_observer_is_a_real_source():
    from core.quotes import QUOTE_SOURCE_PRIORITY

    for name in LIVE_OBSERVERS:
        assert name in QUOTE_SOURCE_PRIORITY
    assert 'backfill' not in LIVE_OBSERVERS


def test_the_candidate_is_preferred_over_the_live_table():
    """A seam test. The bug was a SOURCE preference, so testing the row filter
    alone proves nothing about which one `market_measurement` reaches for."""
    from scripts.promote import market_measurement

    src = inspect.getsource(market_measurement)
    candidate_at = src.index('recorded_only=True')
    live_at = src.index('scored_against_market()')
    assert candidate_at < live_at, (
        "the candidate's own rows must be tried before the deployed system's "
        'live record, or a new model is gated on its predecessor'
    )


def test_the_live_table_remains_the_fallback():
    """It is still the right answer when the candidate has no recorded quotes
    at all — which is every run before 2026-08-25."""
    from scripts.promote import market_measurement

    src = inspect.getsource(market_measurement)
    assert 'scored_against_market()' in src
    assert 'cannot influence' in src, 'the fallback must say what it is'
