"""How many basis points does our Coinbase proxy differ from the venue's index?

`_validate_label` answers the binary question -- how often our UP/DOWN differs
from the venue's -- and gets ~97%. That leaves the size of the error unmeasured,
which matters because ~3% of every training label being wrong is not negligible
against a measured skill of +0.002.

Kalshi's `expiration_value` is the price the window actually settled at, so the
question becomes numeric. The venue never publishes a STRIKE, but it does not
need to: a window's strike is the previous window's settlement value, both being
the mean over the same minute. Consecutive settled markets therefore give the
venue's own displacement, which is the quantity the barrier model consumes.

Its limit is retention, not method: `expiration_value` reaches back only to
2026-06 because Kalshi purges older markets, while Predexon's `result` reaches
2025-12 with no price attached. So the bias is measurable on three months and
must be argued -- not assumed -- to hold earlier.
"""

from __future__ import annotations

import datetime as dt

from research.validate._validate_proxy_bias import (
    bias_bp, displacement_bp, venue_moves,
)

UTC = dt.timezone.utc


def _s(minute, value, symbol='BTC-USD'):
    return {'symbol': symbol,
            'window_open': dt.datetime(2026, 7, 1, 12, minute, tzinfo=UTC).isoformat(),
            'expiration_value': value}


def test_consecutive_settlements_chain_into_a_strike_and_a_settle():
    """The venue publishes no strike. It does not have to: a window's strike is
    the previous window's settlement value, both means over the same minute."""
    moves = venue_moves([_s(0, 100.0), _s(15, 101.0), _s(30, 99.0)])
    assert [(m['strike'], m['settle']) for m in moves] == [(100.0, 101.0), (101.0, 99.0)]


def test_a_missing_previous_window_is_skipped_not_bridged():
    """Chaining across a gap would invent a 30-minute move and call it 15."""
    moves = venue_moves([_s(0, 100.0), _s(30, 99.0)])
    assert moves == []


def test_the_chain_does_not_cross_symbols():
    rows = [_s(0, 100.0, 'BTC-USD'), _s(15, 3000.0, 'ETH-USD')]
    assert venue_moves(rows) == []


def test_a_settlement_without_a_price_breaks_the_chain():
    """`expiration_value` is null on some rows; guessing one would fabricate
    both a strike and a settle."""
    rows = [_s(0, 100.0), _s(15, None), _s(30, 99.0)]
    assert venue_moves(rows) == []


def test_bias_is_reported_in_basis_points_and_signed():
    """Signed, because a proxy that reads consistently high is a different
    problem from one that is merely noisy."""
    assert abs(bias_bp(100.01, 100.0) - 1.0) < 1e-9
    assert abs(bias_bp(99.99, 100.0) + 1.0) < 1e-9


def test_displacement_is_measured_from_the_strike_not_from_zero():
    """The barrier model consumes displacement from the strike, so that is the
    quantity whose error matters -- not the absolute price level."""
    assert abs(displacement_bp(settle=100.05, strike=100.0) - 5.0) < 1e-9
    assert abs(displacement_bp(settle=99.95, strike=100.0) + 5.0) < 1e-9


def test_a_zero_or_missing_strike_yields_no_measurement():
    """A zero strike would divide by zero and report an infinite bias."""
    assert displacement_bp(settle=100.0, strike=0.0) is None
    assert displacement_bp(settle=100.0, strike=None) is None
    assert bias_bp(100.0, 0.0) is None
