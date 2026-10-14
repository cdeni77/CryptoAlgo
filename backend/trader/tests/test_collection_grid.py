"""Tests for the Polymarket window grid, used instead of paging through all markets.

Building slugs is safe here: a wrong slug returns nothing, and every returned market is
checked against its own end_time.
"""

from __future__ import annotations

import datetime as dt

from research.collect.catalog import batched, window_grid

UTC = dt.timezone.utc


def test_the_grid_is_every_quarter_hour_for_every_asset():
    start = dt.datetime(2026, 1, 8, 0, 0, tzinfo=UTC)
    end = dt.datetime(2026, 1, 8, 1, 0, tzinfo=UTC)
    grid = list(window_grid(start, end, ('btc', 'eth')))
    assert len(grid) == 8  # 4 windows x 2 assets
    assert {slug for slug, _, _ in grid} >= {
        f'btc-updown-15m-{int(start.timestamp())}',
        f'eth-updown-15m-{int(start.timestamp())}'}


def test_the_grid_end_is_exclusive_so_a_live_window_is_not_requested():
    """The end is exclusive, so the window in progress is not requested."""
    start = dt.datetime(2026, 1, 8, 0, 0, tzinfo=UTC)
    grid = list(window_grid(start, start + dt.timedelta(minutes=15), ('btc',)))
    assert len(grid) == 1


def test_the_grid_carries_the_symbol_the_rest_of_the_system_uses():
    start = dt.datetime(2026, 1, 8, tzinfo=UTC)
    _, symbol, opened = next(iter(window_grid(
        start, start + dt.timedelta(minutes=15), ('sol',))))
    assert symbol == 'SOL-USD' and opened == start


def test_the_grid_only_lands_on_quarter_hours():
    start = dt.datetime(2026, 1, 8, tzinfo=UTC)
    end = start + dt.timedelta(days=1)
    assert all(o.minute in (0, 15, 30, 45) and o.second == 0
               for _, _, o in window_grid(start, end, ('btc',)))


def test_batches_never_exceed_the_endpoints_ceiling():
    """Batches never exceed 50 slugs: the endpoint rejects 100 with HTTP 422."""
    items = list(range(137))
    chunks = list(batched(items, 50))
    assert [len(c) for c in chunks] == [50, 50, 37]


def test_batching_preserves_every_item_exactly_once():
    items = list(range(137))
    flat = [x for chunk in batched(items, 50) for x in chunk]
    assert flat == items


def test_an_empty_input_yields_no_batches():
    assert list(batched([], 50)) == []
