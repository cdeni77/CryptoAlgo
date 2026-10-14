"""Tests for the window grid and the venue's settlement conventions.

Both ends are one-minute averages, a tie resolves UP, and a window's strike is the
previous settlement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.config import Config
from core.windows import (
    WindowError, base_rate, build_window_panel, build_windows, floor_to_window,
    minute_grid,
)
from tests.conftest import make_bars


def test_windows_are_aligned_to_the_quarter_hour():
    bars = make_bars(days=2)['BTC-USD']
    table, _ = build_windows(bars, 'BTC-USD')
    opens = pd.DatetimeIndex(table['window_open'].unique())
    assert (opens.minute % 15 == 0).all()
    assert (opens.second == 0).all()
    # Every window settles exactly one window later.
    assert (table['settle_time'] - table['window_open']
            == pd.Timedelta(minutes=15)).all()


def test_both_ends_are_one_minute_averages():
    """The strike is the mean over `[t0 - 1min, t0)` and the settlement the mean over
    `[t1 - 1min, t1)`.
    """
    from core.windows import bar_mean

    bars = make_bars(days=1)['BTC-USD']
    table, _ = build_windows(bars, 'BTC-USD', offsets=(9,))
    means = bar_mean(bars.set_index('event_time'))
    for row in table.head(20).itertuples():
        assert row.strike == pytest.approx(
            means.loc[row.window_open - pd.Timedelta(minutes=1)])
        assert row.settle_price == pytest.approx(
            means.loc[row.settle_time - pd.Timedelta(minutes=1)])
    assert np.allclose(table['settle_return'],
                       table['settle_price'] / table['strike'] - 1)


def test_a_windows_strike_is_the_previous_windows_settlement():
    """A window's strike is the previous window's settlement, so consecutive markets
    chain.

    This dependence is one reason the cross-validation embargo is a day.
    """
    bars = make_bars(days=2)['BTC-USD']
    table, _ = build_windows(bars, 'BTC-USD', offsets=(9,))
    windows = table.drop_duplicates('window_open').sort_values('window_open')
    assert np.allclose(windows['strike'].to_numpy()[1:],
                       windows['settle_price'].to_numpy()[:-1])


def test_the_first_window_is_dropped_for_want_of_a_strike():
    """The first window is dropped, since its strike would come from a window that does
    not exist.
    """
    bars = make_bars(days=1)['BTC-USD']
    table, report = build_windows(bars, 'BTC-USD', offsets=(9,))
    first_grid_window = bars['event_time'].iloc[0].floor('15min')
    assert first_grid_window not in set(table['window_open'])
    with pytest.raises(WindowError, match='under two full windows'):
        build_windows(bars.head(20), 'BTC-USD', offsets=(9,))


def test_a_decision_sees_the_bar_before_it_and_nothing_after():
    """At offset m the last observable price is the close of bar m-1.

    A one-minute leak is 7% of a fifteen-minute window and reads like skill.
    """
    bars = make_bars(days=1)['BTC-USD']
    indexed = bars.set_index('event_time')
    for offset in (1, 3, 9, 14):
        table, _ = build_windows(bars, 'BTC-USD', offsets=(offset,))
        for row in table.head(10).itertuples():
            expected_bar = row.window_open + pd.Timedelta(minutes=offset - 1)
            assert row.last_price == pytest.approx(indexed.loc[expected_bar, 'close'])
            assert row.decision_time == row.window_open + pd.Timedelta(minutes=offset)


def test_a_tie_resolves_up():
    """A tie resolves UP (`strike_type: greater_or_equal`).

    Real ties are extremely rare, so nothing else exercises this path.
    """
    times = pd.date_range('2025-01-01', periods=46, freq='1min', tz='UTC')
    flat = pd.DataFrame({
        'event_time': times, 'open': 100.0, 'high': 100.0, 'low': 100.0,
        'close': 100.0, 'volume': 1.0, 'quote_volume': np.nan, 'trade_count': 1,
    })
    table, _ = build_windows(flat, 'X', offsets=(9,))
    assert not table.empty
    assert (table['outcome'] == 1).all(), 'a tie went to the down side'
    assert np.allclose(table['settle_return'], 0.0)


def test_one_missing_averaging_minute_drops_two_windows():
    """One missing averaging minute drops two windows.

    The minute before a boundary is both the settlement average of the window ending
    there and the strike of the next. Neither is invented. The boundary minute itself is
    used by neither average.
    """
    bars = make_bars(days=1)['BTC-USD']
    # The last minute of a window: index 14 of 0..14.
    averaging = bars.index[bars['event_time'].dt.minute % 15 == 14][20]
    missing = pd.Timestamp(bars.loc[averaging, 'event_time'])
    holed = bars.drop(index=[averaging])
    table, report = build_windows(holed, 'BTC-USD', offsets=(9,))
    assert report.windows_dropped_boundary == 2, report.summary()
    remaining = set(table['window_open'])
    assert missing.floor('15min') not in remaining, 'the settling window survived'
    assert missing.ceil('15min') not in remaining, 'the next window survived'


def test_a_missing_boundary_minute_no_longer_matters():
    """A missing boundary minute drops no window, since neither average uses it."""
    bars = make_bars(days=1)['BTC-USD']
    boundary = bars.index[bars['event_time'].dt.minute % 15 == 0][20]
    table, report = build_windows(bars.drop(index=[boundary]), 'BTC-USD',
                                  offsets=(9,))
    assert report.windows_dropped_boundary == 0
    assert report.windows_with_interior_gaps >= 1


def test_an_interior_gap_is_counted_but_kept():
    """A missing interior minute is counted but kept; the last trade price still holds."""
    bars = make_bars(days=1)['BTC-USD']
    interior = bars.index[bars['event_time'].dt.minute % 15 == 7][10]
    holed = bars.drop(index=[interior])
    table, report = build_windows(holed, 'BTC-USD', offsets=(9,))
    assert report.windows_with_interior_gaps >= 1
    assert report.windows_dropped_boundary == 0
    affected = pd.Timestamp(bars.loc[interior, 'event_time']).floor('15min')
    row = table.loc[table['window_open'] == affected]
    assert not row.empty and np.isfinite(row['last_price'].iloc[0])
    assert row['minutes_missing'].iloc[0] >= 1


def test_offsets_outside_the_window_are_refused():
    bars = make_bars(days=1)['BTC-USD']
    with pytest.raises(WindowError, match='offsets must lie'):
        build_windows(bars, 'BTC-USD', offsets=(0,))
    with pytest.raises(WindowError, match='offsets must lie'):
        build_windows(bars, 'BTC-USD', offsets=(15,))


def test_the_base_rate_is_near_a_half():
    """The base rate is near one half; a large departure is a grid bug.

    Slightly above 0.5 is expected, since ties pay UP.
    """
    bars = make_bars(days=40)
    panel, _ = build_window_panel(bars, Config())
    rate = base_rate(panel)
    assert 0.46 < rate < 0.56, rate


def test_every_offset_produces_the_same_number_of_rows():
    """Every offset has the same row count: four offsets of one window share one label.

    So cross-validation splits on the window, never the row.
    """
    bars = make_bars(days=3)
    panel, _ = build_window_panel(bars, Config())
    counts = panel.groupby('offset').size()
    assert counts.nunique() == 1, counts.to_dict()
    assert set(panel['offset']) == set(Config().decision_offsets)


def test_the_grid_is_gap_free_after_reindexing():
    bars = make_bars(days=1)['BTC-USD'].drop(index=[100, 101, 500])
    grid = minute_grid(bars)
    assert (grid.index.to_series().diff().dropna()
            == pd.Timedelta(minutes=1)).all()
    assert grid['close'].isna().sum() == 3


def test_floor_to_window_is_exact_on_a_boundary():
    times = pd.DatetimeIndex(['2025-01-01 00:15:00', '2025-01-01 00:15:59',
                              '2025-01-01 00:14:59'], tz='UTC')
    floored = floor_to_window(times, 15)
    assert list(floored.strftime('%H:%M')) == ['00:15', '00:15', '00:00']


def test_remaining_variance_is_less_than_wall_clock_time():
    """Remaining variance is `(W - delta - m) + delta/3`, less than wall-clock time.

    The settlement value is a one-minute average, and an average has a third of its
    endpoint's variance. At offset 12 that is 2.33 minutes against 3. Ignoring it would
    overstate sigma by 13%.
    """
    config = Config()
    for offset in config.decision_offsets:
        wall = config.remaining_minutes(offset)
        variance = config.remaining_variance_minutes(offset)
        assert variance < wall, (offset, variance, wall)
        expected = (config.window_minutes - config.settle_average_minutes
                    - offset) + config.settle_average_minutes / 3.0
        assert variance == pytest.approx(expected)

    # Monotone and never negative, including past the last offset.
    values = [config.remaining_variance_minutes(m) for m in range(0, 16)]
    assert all(a >= b for a, b in zip(values, values[1:]))
    assert min(values) >= 0.0


def test_disabling_the_settlement_average_recovers_wall_clock_time():
    """With `settle_average_minutes=0` the correction vanishes."""
    config = Config(settle_average_minutes=0.0)
    for offset in (3, 9, 12):
        assert config.remaining_variance_minutes(offset) == pytest.approx(
            config.remaining_minutes(offset))
