"""Reconstructing the strike ladder's implied sigma from history.

**Why this dataset matters more than its size suggests.** The barrier framing
says the displacement `x` is known exactly and `sigma_remaining` is the ONLY
quantity requiring a forecast. Every volatility feature in `core/features.py`
is backward-looking realised vol. The KXBTCD/KXETHD/KXSOLD strike ladders
invert to a FORWARD-looking sigma — the market's own estimate of the one
unknown — at R² > 0.95.

It was in the plan as a feature family and marked "already recorded live",
which was wrong: `venue_implied_vol` held 1,256 rows over 3 days, BTC only.
Three days cannot train anything. The ladders themselves reach January, so the
history is recoverable; this rebuilds it.

The inversion is NOT reimplemented here. `scripts/record_implied_vol.implied_sigma`
is the one definition and the live path uses it, so a second copy would be two
definitions that agree until they don't — the same argument as one `decide()`.
What is new is assembling the cross-section: the live recorder reads every
strike's CURRENT quote at once, while history arrives as one tick series per
strike that has to be aligned onto a common instant.
"""

from __future__ import annotations

import datetime as dt

from research.collect.implied_vol_backfill import (
    LADDER_SERIES, cross_section, pack_series, sample_instants,
)


def _packed(by_strike: dict) -> dict:
    """Strikes to packed tick series, through the real code path."""
    return {k: pack_series(v) for k, v in by_strike.items()}

UTC = dt.timezone.utc


def _snap(ts_ms, bid, ask):
    return {'timestamp': ts_ms,
            'yes_bids': [{'price': bid, 'size': 10}] if bid is not None else [],
            'yes_asks': [{'price': ask, 'size': 10}] if ask is not None else []}


def test_the_cross_section_is_the_last_quote_at_or_before_the_instant():
    """A book is a step function: the state at time T is the last tick at or
    before T, not the nearest one. Taking the nearest would let a quote from
    AFTER the instant inform it, which is lookahead."""
    series = {
        70000.0: [_snap(1_000, 30, 31), _snap(5_000, 40, 41)],
        71000.0: [_snap(2_000, 20, 21), _snap(9_000, 25, 26)],
    }
    rungs = cross_section(_packed(series), at_ms=6_000)
    assert dict(rungs) == {70000.0: 0.405, 71000.0: 0.205}


def test_a_strike_with_no_tick_yet_is_omitted_rather_than_guessed():
    series = {70000.0: [_snap(9_000, 30, 31)]}
    assert cross_section(_packed(series), at_ms=1_000) == []


def test_a_one_sided_quote_is_omitted_because_it_has_no_mid():
    """The inversion needs P(above); a single side does not give one, and
    inventing the other side would fabricate a probability."""
    series = {70000.0: [_snap(1_000, 30, None)],
              71000.0: [_snap(1_000, 20, 21)]}
    assert [s for s, _ in cross_section(_packed(series), at_ms=2_000)] == [71000.0]


def test_the_cross_section_is_sorted_by_strike():
    series = {72000.0: [_snap(1, 10, 11)], 70000.0: [_snap(1, 30, 31)],
              71000.0: [_snap(1, 20, 21)]}
    assert [s for s, _ in cross_section(_packed(series), at_ms=5)] == [70000.0, 71000.0, 72000.0]


def test_probabilities_come_back_as_fractions_not_cents():
    series = {70000.0: [_snap(1, 38, 39)]}
    _, p = cross_section(_packed(series), at_ms=5)[0]
    assert 0.0 < p < 1.0 and abs(p - 0.385) < 1e-9


def test_sample_instants_walk_the_quarter_hour_grid_inside_the_ladder():
    """Sigma is wanted where the 15-minute windows decide, so the samples sit
    on the same grid rather than on an arbitrary cadence."""
    o = dt.datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 21, 0, tzinfo=UTC)
    got = sample_instants(o, c, minutes=15)
    assert got == [o, o + dt.timedelta(minutes=15), o + dt.timedelta(minutes=30),
                   o + dt.timedelta(minutes=45)]


def test_sample_instants_excludes_the_close_itself():
    """At the close there is no time left, so sigma is undefined — dividing by
    sqrt(0) minutes."""
    o = dt.datetime(2026, 7, 29, 20, 0, tzinfo=UTC)
    c = dt.datetime(2026, 7, 29, 20, 30, tzinfo=UTC)
    assert c not in sample_instants(o, c, minutes=15)


def test_all_three_assets_have_a_ladder_series():
    """The live recorder only reads KXBTCD. ETH and SOL ladders exist and
    cross_asset was the strongest measured feature group, so leaving them out
    would waste the cheapest part of this dataset."""
    assert set(LADDER_SERIES.values()) == {'BTC-USD', 'ETH-USD', 'SOL-USD'}


def test_the_inversion_itself_is_reused_not_reimplemented():
    from research.collect import implied_vol_backfill as m
    from scripts.record_implied_vol import implied_sigma
    assert m.implied_sigma is implied_sigma
