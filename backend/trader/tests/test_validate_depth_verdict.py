"""Judging the backfill by the SHAPE of its disagreement, not its level.

`_validate_depth` compared a live quote and a backfilled one and failed the
backfill on a median 1.8c difference against a fixed ~1c threshold. But the two
sides are not sampled together: the live row is stamped at the poll, a measured
median 33.8s after the minute it is filed under, while the backfill row is the
book AT the mark. The validator's own docstring says this market moves ~8.4pp
per minute, so tens of seconds of lag produce cents of difference on their own.

Measured across 13,412 overlapping minutes, binned by how stale the worse of
the two quotes is:

    0-2s    1.00c      10-20s   2.10c
    2-5s    1.20c      20-40s   4.00c
    5-10s   2.00c      40s+     7.00c

Monotonic. A backfill describing a DIFFERENT book would be flat in this
variable — the disagreement would not care how fresh the quotes were. So the
verdict belongs on the trend, which is the same correction already made to
`_validate_backfill`, whose verdict "tests whether agreement improves as
tolerance tightens rather than a fixed threshold".
"""

from __future__ import annotations

from research.validate._validate_depth import agreement_verdict


def test_a_gap_that_shrinks_with_fresher_quotes_is_timing():
    """The measured shape: same book, sampled at different instants."""
    bins = [(1, 277, 0.0100), (3, 255, 0.0120), (7, 531, 0.0200),
            (15, 1421, 0.0210), (30, 4172, 0.0400), (60, 6207, 0.0700)]
    ok, why = agreement_verdict(bins)
    assert ok, why
    assert 'timing' in why.lower()


def test_a_gap_that_ignores_freshness_is_a_different_book():
    """Flat in staleness means the disagreement is not about the clock."""
    # Held at one tick so the FRESHNESS check passes and the flatness check is
    # what fires. A same-book comparison must degrade as the quotes drift
    # apart; one that does not is not being driven by the clock.
    bins = [(1, 300, 0.0100), (3, 300, 0.0102), (7, 300, 0.0098),
            (15, 300, 0.0101), (30, 300, 0.0100), (60, 300, 0.0099)]
    ok, why = agreement_verdict(bins)
    assert not ok
    assert 'flat' in why.lower() or 'different' in why.lower()


def test_the_freshest_bin_must_still_land_near_one_tick():
    """A trend is necessary and not sufficient: if even simultaneous quotes
    disagree by five cents, the shape does not rescue it."""
    bins = [(1, 300, 0.0500), (3, 300, 0.0700), (7, 300, 0.0900),
            (15, 300, 0.1100), (30, 300, 0.1400), (60, 300, 0.1800)]
    ok, why = agreement_verdict(bins)
    assert not ok
    assert 'tick' in why.lower() or 'freshest' in why.lower()


def test_too_few_pairs_is_not_a_pass():
    """Silence is not agreement — the failure this whole file exists to avoid."""
    ok, why = agreement_verdict([(1, 3, 0.0100), (3, 2, 0.0120)])
    assert not ok
    assert 'enough' in why.lower()
