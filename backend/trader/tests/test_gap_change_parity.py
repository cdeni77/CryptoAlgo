"""`venue_gap_change_5` live must be the same feature the backtest fitted.

The backtest computes it as `shift(1)` over rows ordered by OFFSET and grouped
by (symbol, window_open): the previous decision offset within the same window,
never reaching across a window boundary. Consecutive windows chain — a window's
strike is the previous window's settlement value — so a difference that crossed
one would look entirely correct and be wrong.

Two consequences the first live implementation got wrong by using a five-minute
wall-clock lookback instead: the step is one OFFSET (three minutes on the
(3,6,9,12) grid, not five), and the first offset of a window has no predecessor
and must be NaN rather than differenced against the previous window's last.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.live import gap_change, reset_gap_history

W1 = pd.Timestamp('2026-08-28 19:00', tz='UTC')
W2 = pd.Timestamp('2026-08-28 19:15', tz='UTC')


@pytest.fixture(autouse=True)
def _clean():
    reset_gap_history()
    yield
    reset_gap_history()


def test_the_first_offset_of_a_window_has_no_predecessor():
    assert np.isnan(gap_change('BTC-USD', 0.02, window_open=W1, offset=3))


def test_a_later_offset_differences_against_the_previous_one():
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    assert gap_change('BTC-USD', 0.05, window_open=W1, offset=6) == pytest.approx(0.03)
    assert gap_change('BTC-USD', 0.04, window_open=W1, offset=9) == pytest.approx(-0.01)


def test_it_never_reaches_across_a_window_boundary():
    """Consecutive windows chain, so differencing across one is a real error
    that would look entirely correct."""
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    gap_change('BTC-USD', 0.09, window_open=W1, offset=12)
    assert np.isnan(gap_change('BTC-USD', 0.01, window_open=W2, offset=3))


def test_symbols_do_not_bleed_into_each_other():
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    assert np.isnan(gap_change('ETH-USD', 0.07, window_open=W1, offset=3))


def test_a_missing_gap_does_not_reach_back_past_the_hole():
    """CORRECTED. This test previously asserted that offset 9 differences
    against offset 3 when offset 6 is a hole — which is what the first
    implementation did, and is NOT what training does.

    Training is `shift(1)` over a panel carrying all four offsets, so a missing
    gap at the previous offset propagates as NaN rather than lengthening the
    horizon. Reaching back returns a six-minute change under a column named for
    three. The test was written from the same wrong premise as the code, so it
    passed while both were wrong — which is the failure mode a test is supposed
    to prevent.
    """
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    assert np.isnan(gap_change('BTC-USD', float('nan'), window_open=W1, offset=6))
    assert np.isnan(gap_change('BTC-USD', 0.05, window_open=W1, offset=9))


def test_a_repeated_offset_is_not_differenced_against_itself():
    """A cycle can score the same offset twice. The second must not report a
    zero change that reads as two venues holding steady."""
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    gap_change('BTC-USD', 0.05, window_open=W1, offset=6)
    assert gap_change('BTC-USD', 0.05, window_open=W1, offset=6) == pytest.approx(0.03)


def test_one_symbol_does_not_evict_another_within_the_same_window():
    """Three symbols are scored in every cycle of the same window.

    The first version cleared the WHOLE history whenever it met a key it had
    not seen, so BTC's reading was destroyed by ETH's arrival and ETH's by
    SOL's. Every symbol then reported NaN on every cycle forever, and the
    symbols-do-not-bleed test above still passed because it only asserted that
    ETH saw nothing — never that BTC kept what it had.
    """
    for symbol in ('BTC-USD', 'ETH-USD', 'SOL-USD'):
        gap_change(symbol, 0.02, window_open=W1, offset=3)
    for symbol in ('BTC-USD', 'ETH-USD', 'SOL-USD'):
        assert gap_change(symbol, 0.05, window_open=W1, offset=6) == pytest.approx(
            0.03), f'{symbol} lost its offset-3 reading to another symbol'


def test_a_new_window_evicts_only_the_windows_that_ended():
    """Memory must not grow forever, but eviction is by WINDOW, not by arrival."""
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    gap_change('ETH-USD', 0.02, window_open=W1, offset=3)
    # BTC moves to the next window; ETH has not been scored there yet.
    assert np.isnan(gap_change('BTC-USD', 0.01, window_open=W2, offset=3))
    assert gap_change('BTC-USD', 0.04, window_open=W2, offset=6) == pytest.approx(0.03)


def test_a_hole_at_the_previous_offset_gives_NaN_not_a_longer_horizon():
    """Training uses `shift(1)` over a panel that always carries all four
    offsets, so a missing gap at the previous offset propagates as NaN. The
    first live version took `max(o for o in recorded if o < offset)`, which
    SKIPS the hole and returns a six- or nine-minute change under a column named
    for three.

    Measured on the store: of 97,416 decision rows past offset 3 that have a
    gap, 19,548 (20.1%) are NaN in training because the previous offset had
    none, and 1,054 of those had an earlier offset with a gap — exactly the rows
    where live would emit a number training never saw.

    The live exposure is larger than that, because a cycle that misses
    DECISION_TOLERANCE_SECONDS records nothing for its offset at all.
    """
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    # offset 6 is a hole: no gap recorded
    assert np.isnan(gap_change('BTC-USD', 0.05, window_open=W1, offset=9)), (
        'offset 9 must difference against offset 6, which is missing — not '
        'reach back to offset 3'
    )


def test_the_immediately_previous_offset_is_the_only_one_used():
    gap_change('BTC-USD', 0.02, window_open=W1, offset=3)
    gap_change('BTC-USD', 0.04, window_open=W1, offset=6)
    assert gap_change('BTC-USD', 0.05, window_open=W1, offset=9) == pytest.approx(0.01)
