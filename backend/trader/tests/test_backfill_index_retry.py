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
    """The bucket refills at 300/s and `/cfbenchmarks/*` costs 50, so 6/s is
    the ceiling for this endpoint. The bucket is shared and priced per
    request, so a cheap consumer at 30/s starves this one."""
    assert 0 < bi.DEFAULT_RATE <= 6.0


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


def test_a_rerun_fills_gaps_instead_of_refetching_everything():
    """A backfill that cannot cheaply fill its own gaps will not have them
    filled: one dropped hour used to cost a full ~4-hour pass to repair."""
    import inspect

    src = inspect.getsource(bi.run)
    assert 'held_hours(store, symbol)' in src
    assert 'already complete' in src
    assert bi.COMPLETE_HOUR_MINUTES == 59


def test_a_partial_hour_is_re_asked(monkeypatch):
    """60 one-second-sourced minute bars per hour; fewer means partial."""
    import pandas as _pd

    full = _pd.Timestamp('2026-10-01 00:00', tz='UTC')
    part = _pd.Timestamp('2026-10-01 01:00', tz='UTC')
    rows = ([{'event_time': full + _pd.Timedelta(minutes=i)} for i in range(60)]
            + [{'event_time': part + _pd.Timedelta(minutes=i)} for i in range(10)])

    class Store:
        def read(self, *a, **k):
            return _pd.DataFrame(rows)

    held = bi.held_hours(Store(), 'BTC-USD')
    assert full in held
    assert part not in held
