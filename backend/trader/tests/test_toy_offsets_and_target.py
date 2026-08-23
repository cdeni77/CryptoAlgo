"""PROPOSED. Candle boundaries, offset indexing and the target, on hand-computed toy data.

Drop into `backend/trader/tests/`.

Every existing window test asserts a *relationship* between the table and the
bars (`row.strike == means.loc[...]`) — which is a re-statement of the
implementation, and passes if both sides move together. These assert **literal
numbers worked out by hand**, so an indexing change has to be argued with, not
just re-derived.

The bars are constructed so every quantity is exact in binary floating point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.windows import bar_mean, build_windows

# Three windows of 15 one-minute bars, starting on a quarter-hour boundary.
# Prices are chosen so that the one-minute OHLC mean of bar k is exactly
# 100 + k/4, making every average and every ratio hand-checkable.
N = 45
START = pd.Timestamp('2025-03-01 00:00', tz='UTC')


def toy_bars(closes=None) -> pd.DataFrame:
    """`open == high == low == close == 100 + k/4` unless `closes` overrides."""
    k = np.arange(N, dtype=float)
    price = 100.0 + k / 4.0 if closes is None else np.asarray(closes, dtype=float)
    return pd.DataFrame({
        'event_time': pd.date_range(START, periods=N, freq='1min', tz='UTC'),
        'open': price, 'high': price, 'low': price, 'close': price,
        'volume': np.ones(N), 'quote_volume': np.full(N, np.nan),
        'trade_count': np.ones(N),
    })


def test_the_bar_mean_is_the_ohlc_mean_of_that_minute():
    """Both ends of the target are built from this, so it is worth pinning flat."""
    bars = toy_bars()
    means = bar_mean(bars.set_index('event_time'))
    assert float(means.iloc[0]) == pytest.approx(100.0)
    assert float(means.iloc[4]) == pytest.approx(101.0)
    assert float(means.iloc[44]) == pytest.approx(111.0)


def test_the_strike_is_the_previous_windows_last_minute_and_nothing_else():
    """Window 2 opens at 00:15. Its strike is the mean of the 00:14 bar.

    00:14 is index 14, so the mean is 100 + 14/4 = 103.5. Not the 00:15 open
    (103.75), which is what an earlier version used.
    """
    table, _ = build_windows(toy_bars(), 'TOY', offsets=(3,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    assert float(row['strike']) == pytest.approx(103.5)
    assert float(row['strike']) != pytest.approx(103.75), 'read open(t0), not mean(t0-1m)'


def test_the_settlement_value_is_the_last_minute_of_the_settling_window():
    """Window 2 settles at 00:30; the settling minute is 00:29, index 29,
    mean 100 + 29/4 = 107.25."""
    table, _ = build_windows(toy_bars(), 'TOY', offsets=(3,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    assert float(row['settle_price']) == pytest.approx(107.25)
    assert float(row['settle_return']) == pytest.approx(107.25 / 103.5 - 1.0)
    assert row['outcome'] == 1


@pytest.mark.parametrize('offset,expected_close_index', [
    (1, 15), (3, 17), (6, 20), (9, 23), (12, 26), (14, 28),
])
def test_last_price_is_the_close_of_the_bar_before_the_decision(offset,
                                                                expected_close_index):
    """Window 2 opens at index 15. A decision at offset m sees the close of the
    bar covering [m-1, m), i.e. absolute index 15 + m - 1.

    Hand-checked: at offset 3 that is index 17, close 100 + 17/4 = 104.25.
    """
    table, _ = build_windows(toy_bars(), 'TOY', offsets=(offset,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    expected = 100.0 + expected_close_index / 4.0
    assert float(row['last_price']) == pytest.approx(expected), (
        f'offset {offset}: expected the close of index {expected_close_index} '
        f'({expected}), got {row["last_price"]}'
    )
    assert float(row['displacement']) == pytest.approx(expected / 103.5 - 1.0)
    assert row['decision_time'] == START + pd.Timedelta(minutes=15 + offset)


def test_the_excursion_stops_at_the_bar_before_the_decision():
    """The gap the suite has: `excursion_up`/`excursion_down` are never asserted.

    A spike planted in the bar *at* the decision minute must be invisible. With
    `highs[:, :offset+1]` instead of `highs[:, :offset]` it is visible, and all
    230 existing tests still pass.
    """
    closes = 100.0 + np.arange(N, dtype=float) / 4.0
    bars = toy_bars(closes)
    # Window 2 is indices 15..29. Offset 6 sees window-minutes 0..5, i.e. indices
    # 15..20. Plant a huge high at index 21 — window-minute 6, the decision minute.
    bars.loc[21, 'high'] = 500.0
    table, _ = build_windows(bars, 'TOY', offsets=(6,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    # Highs of indices 15..20 are 103.75 .. 105.0; the largest is 105.0.
    assert float(row['excursion_up']) == pytest.approx(105.0 / 103.5 - 1.0), (
        f'excursion_up is {row["excursion_up"]}; a spike planted in the decision '
        f'minute itself leaked into it'
    )
    # And the same spike must be visible at a later offset, or the test is vacuous.
    later, _ = build_windows(bars, 'TOY', offsets=(9,))
    later_row = later.loc[later['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    assert float(later_row['excursion_up']) == pytest.approx(500.0 / 103.5 - 1.0)


def test_the_excursion_down_stops_at_the_same_boundary():
    bars = toy_bars()
    bars.loc[21, 'low'] = 1.0
    table, _ = build_windows(bars, 'TOY', offsets=(6,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    assert float(row['excursion_down']) == pytest.approx(103.75 / 103.5 - 1.0)


def test_a_dead_flat_window_resolves_up_not_down():
    """`strike_type` is `greater_or_equal`. The docstring of
    `test_windows.py::test_the_base_rate_is_near_a_half` still says "an exactly
    flat window resolves down", which is the old convention — stated here so the
    two cannot both be believed."""
    flat = toy_bars(np.full(N, 100.0))
    table, _ = build_windows(flat, 'TOY', offsets=(3, 6, 9, 12))
    assert not table.empty
    assert (table['outcome'] == 1).all()
    assert np.allclose(table['settle_return'], 0.0)
    assert np.allclose(table['displacement'], 0.0)
    assert np.allclose(table['excursion_up'], 0.0)
    assert np.allclose(table['excursion_down'], 0.0)


def test_one_tick_below_the_strike_resolves_down():
    """The other side of the tie, so the `>=` is pinned from both directions."""
    closes = np.full(N, 100.0)
    closes[29] = 100.0 - 1e-6      # the settling minute of window 2
    table, _ = build_windows(toy_bars(closes), 'TOY', offsets=(3,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    assert row['outcome'] == 0


def test_a_missing_averaging_minute_kills_exactly_the_two_windows_it_feeds():
    """Index 29 is the settling minute of window 2 and the strike minute of
    window 3, so both die and nothing else does."""
    bars = toy_bars().drop(index=[29]).reset_index(drop=True)
    table, report = build_windows(bars, 'TOY', offsets=(3,))
    kept = set(table['window_open'])
    assert START + pd.Timedelta(minutes=15) not in kept
    assert START + pd.Timedelta(minutes=30) not in kept
    assert report.windows_dropped_boundary == 2, report.summary()


def test_a_missing_interior_minute_forward_fills_rather_than_inventing_a_price():
    """A minute with no trades carries no new price; the last trade still holds."""
    bars = toy_bars().drop(index=[20]).reset_index(drop=True)   # window-minute 5
    table, report = build_windows(bars, 'TOY', offsets=(9,))
    row = table.loc[table['window_open'] == START + pd.Timedelta(minutes=15)].iloc[0]
    # Offset 9 reads window-minute 8 = index 23, which is present: 105.75.
    assert float(row['last_price']) == pytest.approx(105.75)
    assert int(row['minutes_missing']) == 1
    assert not bool(row['complete'])
    assert report.windows_dropped_boundary == 0


def test_the_remaining_variance_at_each_offset_is_the_hand_computed_number():
    """`(W - delta - m) + delta/3` with W=15, delta=1: 2.33 minutes at m=12,
    not 3. Asserted as literals, because the existing test recomputes the same
    expression it is checking."""
    config = Config()
    assert config.window_minutes == 15
    assert config.settle_average_minutes == pytest.approx(1.0)
    for offset, expected in ((3, 11 + 1 / 3), (6, 8 + 1 / 3),
                             (9, 5 + 1 / 3), (12, 2 + 1 / 3)):
        assert config.remaining_variance_minutes(offset) == pytest.approx(expected), offset
