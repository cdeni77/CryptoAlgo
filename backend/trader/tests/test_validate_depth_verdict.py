"""Tests for `agreement_verdict`, which judges the backfill against live by trend.

A gap that shrinks as quotes get fresher is timing; a gap flat in staleness is a
different book.
"""

from __future__ import annotations

from research.validate._validate_depth import agreement_verdict


def test_a_gap_that_shrinks_with_fresher_quotes_is_timing():
    bins = [(1, 277, 0.0100), (3, 255, 0.0120), (7, 531, 0.0200),
            (15, 1421, 0.0210), (30, 4172, 0.0400), (60, 6207, 0.0700)]
    ok, why = agreement_verdict(bins)
    assert ok, why
    assert 'timing' in why.lower()


def test_a_gap_that_ignores_freshness_is_a_different_book():
    """A gap that does not change with freshness means a different book."""
    # Held at one tick so the freshness check passes and the flatness check fires.
    bins = [(1, 300, 0.0100), (3, 300, 0.0102), (7, 300, 0.0098),
            (15, 300, 0.0101), (30, 300, 0.0100), (60, 300, 0.0099)]
    ok, why = agreement_verdict(bins)
    assert not ok
    assert 'flat' in why.lower() or 'different' in why.lower()


def test_the_freshest_bin_must_still_land_near_one_tick():
    """The freshest bin must still be near one tick; a trend alone is not enough."""
    bins = [(1, 300, 0.0500), (3, 300, 0.0700), (7, 300, 0.0900),
            (15, 300, 0.1100), (30, 300, 0.1400), (60, 300, 0.1800)]
    ok, why = agreement_verdict(bins)
    assert not ok
    assert 'tick' in why.lower() or 'freshest' in why.lower()


def test_too_few_pairs_is_not_a_pass():
    ok, why = agreement_verdict([(1, 3, 0.0100), (3, 2, 0.0120)])
    assert not ok
    assert 'enough' in why.lower()
