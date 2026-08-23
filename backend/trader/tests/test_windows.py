"""The window grid, and the conventions that make it mean anything.

Two of these tests exist because the previous incarnation of this project got
the same class of thing wrong and it cost every performance number it produced.
The target was anchored on `close(t)` — a last trade that could be twenty
minutes stale on a thin contract — while the simulator entered at the next open,
and 98% of the apparent edge turned out to be that mismatch. So the strike and
the settle price are both *opens*, and a test says so rather than a comment.
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
    # And every window settles exactly one window later.
    assert (table['settle_time'] - table['window_open']
            == pd.Timedelta(minutes=15)).all()


def test_the_strike_and_the_settle_price_are_both_opens():
    """Open-to-open, on both ends.

    This is the correction that invalidated every number this repo produced
    before it. A bar's `close` is its last *trade*; its `open` is the first trade
    at or after the boundary. Mixing the two — a close-anchored strike and an
    open-anchored settlement — books the difference between them as profit, and
    on a thin market that difference is most of the move.
    """
    bars = make_bars(days=1)['BTC-USD']
    table, _ = build_windows(bars, 'BTC-USD', offsets=(9,))
    indexed = bars.set_index('event_time')
    for row in table.head(20).itertuples():
        assert row.strike == pytest.approx(indexed.loc[row.window_open, 'open'])
        assert row.settle_price == pytest.approx(indexed.loc[row.settle_time, 'open'])
    assert np.allclose(table['settle_return'],
                       table['settle_price'] / table['strike'] - 1)


def test_a_decision_sees_the_bar_before_it_and_nothing_after():
    """At offset m the last observable price is the close of bar m-1.

    A one-minute leak in a fifteen-minute window is enormous — it is 7% of the
    whole question — and it reads exactly like skill.
    """
    bars = make_bars(days=1)['BTC-USD']
    indexed = bars.set_index('event_time')
    for offset in (1, 3, 9, 14):
        table, _ = build_windows(bars, 'BTC-USD', offsets=(offset,))
        for row in table.head(10).itertuples():
            expected_bar = row.window_open + pd.Timedelta(minutes=offset - 1)
            assert row.last_price == pytest.approx(indexed.loc[expected_bar, 'close'])
            assert row.decision_time == row.window_open + pd.Timedelta(minutes=offset)


def test_the_outcome_is_strictly_above_the_strike():
    """A flat window is a loss for the up side, which is how the venue resolves."""
    times = pd.date_range('2025-01-01', periods=31, freq='1min', tz='UTC')
    flat = pd.DataFrame({
        'event_time': times, 'open': 100.0, 'high': 100.0, 'low': 100.0,
        'close': 100.0, 'volume': 1.0, 'quote_volume': np.nan, 'trade_count': 1,
    })
    table, _ = build_windows(flat, 'X', offsets=(9,))
    assert (table['outcome'] == 0).all()


def test_one_missing_boundary_minute_drops_two_windows():
    """The window settling on it, and the window opening on it.

    Both are unrecoverable: the strike and the settle price are the numbers the
    venue read, and inventing either is the one repair this module refuses.
    """
    bars = make_bars(days=1)['BTC-USD']
    boundary = bars.index[bars['event_time'].dt.minute % 15 == 0][20]
    holed = bars.drop(index=[boundary])
    table, report = build_windows(holed, 'BTC-USD', offsets=(9,))
    assert report.windows_dropped_boundary == 2
    missing = pd.Timestamp(bars.loc[boundary, 'event_time'])
    remaining = set(table['window_open'])
    assert missing not in remaining
    assert missing - pd.Timedelta(minutes=15) not in remaining


def test_an_interior_gap_is_counted_but_kept():
    """A minute with no trades carries no new price; the last trade still holds."""
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
    """A large departure is a grid bug, not a market fact.

    Slightly *below* 0.5 is expected: an exactly flat window resolves down, and a
    minute grid produces exact ties more often than a continuous model implies.
    """
    bars = make_bars(days=40)
    panel, _ = build_window_panel(bars, Config())
    rate = base_rate(panel)
    assert 0.44 < rate < 0.52, rate


def test_every_offset_produces_the_same_number_of_rows():
    """Four offsets of one window are four rows sharing one label.

    Which is why cross-validation splits on the window and never on the row.
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
