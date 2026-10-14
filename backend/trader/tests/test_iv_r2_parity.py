"""Tests for the `--min-r2` gate in `scripts/record_implied_vol.py`.

The backfill applies no R^2 gate, and `iv_r2` is a model feature, so parity
argues for no gate. But live fits the whole open ladder while the backfill had
only the liquid strikes, so ungated live fits differ widely. The gate stays at
0.90 until strike selection matches.
"""
from __future__ import annotations

import datetime as dt

from scripts.record_implied_vol import build_parser, fits_for


def _market(strike, price, close):
    return {'event_ticker': 'KXSOLD-X', 'close_time': close,
            'floor_strike': strike,
            'yes_bid_dollars': f'{max(price - 0.01, 0.0):.4f}',
            'yes_ask_dollars': f'{min(price + 0.01, 1.0):.4f}'}


def test_the_gate_stays_until_the_strike_selection_is_matched():
    """The default stays 0.90 while live and backfill select different strikes.

    Without the gate, ungated ETH and SOL sigmas were far from the training values,
    which corrupts `iv_minus_realised`. A known coverage gap is the smaller error.
    """
    assert build_parser().parse_args([]).min_r2 == 0.90


def test_a_noisy_ladder_is_kept_and_its_r2_reported():
    """With `min_r2=0` a noisy but well-formed ladder is kept and its R^2 reported."""
    now = dt.datetime(2026, 8, 31, 12, 0, tzinfo=dt.timezone.utc)
    close = (now + dt.timedelta(minutes=40)).isoformat()
    # Not monotone on purpose: a real thin book, not a clean curve.
    noisy = ((95.0, 0.88), (100.0, 0.42), (105.0, 0.55), (110.0, 0.11),
             (115.0, 0.19))
    rows, latest = fits_for([_market(s, p, close) for s, p in noisy],
                            now=now, symbol='SOL-USD',
                            min_minutes=2.0, max_minutes=180.0, min_r2=0.0)
    assert rows, 'a noisy but well-formed ladder produced no fit'
    assert 0.0 <= rows[0]['r2'] <= 1.0
    assert latest is not None


def test_a_threshold_can_still_be_asked_for_explicitly():
    """An explicit threshold can still be set."""
    assert build_parser().parse_args(['--min-r2', '0.9']).min_r2 == 0.9
