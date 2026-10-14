"""Tests for measuring our Coinbase proxy's bias against Kalshi's settlement index.

A window's strike is the previous settlement, so consecutive settlements give the
venue's own displacement.
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
    """Consecutive settlements chain: each window's strike is the previous window's settlement."""
    moves = venue_moves([_s(0, 100.0), _s(15, 101.0), _s(30, 99.0)])
    assert [(m['strike'], m['settle']) for m in moves] == [(100.0, 101.0), (101.0, 99.0)]


def test_a_missing_previous_window_is_skipped_not_bridged():
    """A missing previous window breaks the chain instead of bridging a 30-minute move."""
    moves = venue_moves([_s(0, 100.0), _s(30, 99.0)])
    assert moves == []


def test_the_chain_does_not_cross_symbols():
    rows = [_s(0, 100.0, 'BTC-USD'), _s(15, 3000.0, 'ETH-USD')]
    assert venue_moves(rows) == []


def test_a_settlement_without_a_price_breaks_the_chain():
    rows = [_s(0, 100.0), _s(15, None), _s(30, 99.0)]
    assert venue_moves(rows) == []


def test_bias_is_reported_in_basis_points_and_signed():
    """Bias is signed: a proxy that reads high is a different problem from a noisy one."""
    assert abs(bias_bp(100.01, 100.0) - 1.0) < 1e-9
    assert abs(bias_bp(99.99, 100.0) + 1.0) < 1e-9


def test_displacement_is_measured_from_the_strike_not_from_zero():
    """Displacement is measured from the strike, the quantity the barrier model uses."""
    assert abs(displacement_bp(settle=100.05, strike=100.0) - 5.0) < 1e-9
    assert abs(displacement_bp(settle=99.95, strike=100.0) + 5.0) < 1e-9


def test_a_zero_or_missing_strike_yields_no_measurement():
    assert displacement_bp(settle=100.0, strike=0.0) is None
    assert displacement_bp(settle=100.0, strike=None) is None
    assert bias_bp(100.0, 0.0) is None
