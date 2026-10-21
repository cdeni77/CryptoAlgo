"""A rate-limited hour must be retried, not recorded as absent data.

`/cfbenchmarks/*` costs 50 tokens against a 200/s refill, so 429 is the
ordinary failure here. Dropping on it left `minute_bars` under
`venue='cfbenchmarks'` at ~7% coverage, which reads as "the venue serves
little history" rather than "we asked too fast".
"""

from __future__ import annotations

import inspect

from scripts import backfill_index as bi


def test_the_request_sits_inside_a_bounded_retry():
    src = inspect.getsource(bi.run)
    assert "'/cfbenchmarks/history/values'" in src
    assert 'for attempt in range(6)' in src
    assert "DROPPED['hours']" in src
    assert 'DROPPED after 6 attempts' in src


def test_the_drop_counter_starts_empty():
    assert bi.DROPPED['hours'] == 0


def test_the_rate_default_respects_the_endpoint_cost():
    """50 tokens against a 200/s refill is ~4 req/s; 3.0 leaves headroom."""
    assert 0 < bi.DEFAULT_RATE <= 4.0


def test_the_true_minute_mean_is_stored_not_approximated():
    """Kalshi settles on the mean of sixty seconds of index values.

    `core/windows.bar_mean` approximates that as `(O+H+L+C)/4` because
    Coinbase serves only OHLC. BRTI is served at one second, so the exact
    quantity exists — and a run that discards it cannot be repaired without
    fetching the whole index again.
    """
    from core.datastore import SCHEMAS

    assert 'mean_price' in SCHEMAS['minute_bars']
    # A path that SPIKES and comes back, not a ramp: on a monotonic ramp the
    # OHLC average equals the true mean by symmetry, so a ramp would pass
    # whichever quantity were stored and prove nothing.
    values = [100.0] * 59 + [160.0]
    pts = [{'time': (1792497600 + i) * 1000, 'value': str(v)}
           for i, v in enumerate(values)]
    rows = bi.minute_bars_from_points(pts, 'BTC-USD')
    assert len(rows) == 1
    row = rows[0]
    assert row['mean_price'] == sum(values) / len(values)
    ohlc = (row['open'] + row['high'] + row['low'] + row['close']) / 4.0
    assert abs(row['mean_price'] - ohlc) > 1.0   # 101.0 against 115.0
