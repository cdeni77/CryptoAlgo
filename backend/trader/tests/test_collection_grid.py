"""Cataloguing Polymarket by asking for the windows we want, not by paging.

Pagination walks every 15-minute market on the venue — eight-plus assets — to
extract our three, and it degrades with depth: measured at 1.8 days of history
a minute at the start, 1.3 in the middle and 0.55 four hundred pages in, which
extrapolates past six hours. Asking for specific slugs instead is flat, and 50
of them fit in one request (100 returns HTTP 422).

Constructing an identifier is normally forbidden here, and for a good reason:
a ticker built from a pattern keeps working until the venue renames a series,
and then it silently finds nothing or, worse, the wrong contract. Two things
make it safe in this one place:

  * It was verified, not assumed. Every one of the 10,762 slugs that
    pagination had already discovered is reproduced exactly by the grid, with
    zero off-grid windows. That is a check against real discovered data rather
    than against the pattern's own plausibility.
  * A wrong slug returns nothing rather than something. The failure mode is a
    missing market we then know nothing about — not a real market misread as a
    different one. And every market that IS returned is still cross-checked
    against the venue's own end_time before being written.
"""

from __future__ import annotations

import datetime as dt

from research.collect.catalog import batched, window_grid

UTC = dt.timezone.utc


def test_the_grid_is_every_quarter_hour_for_every_asset():
    start = dt.datetime(2026, 1, 8, 0, 0, tzinfo=UTC)
    end = dt.datetime(2026, 1, 8, 1, 0, tzinfo=UTC)
    grid = list(window_grid(start, end, ('btc', 'eth')))
    assert len(grid) == 8                       # 4 windows x 2 assets
    assert {slug for slug, _, _ in grid} >= {
        f'btc-updown-15m-{int(start.timestamp())}',
        f'eth-updown-15m-{int(start.timestamp())}'}


def test_the_grid_end_is_exclusive_so_a_live_window_is_not_requested():
    """The window in progress has no settled book yet; asking for it wastes a
    request and writes a market that will need re-collecting."""
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
    """50 works; 100 returns HTTP 422. A batch of 51 would fail the whole
    chunk, so the size is a correctness constraint, not a tuning knob."""
    items = list(range(137))
    chunks = list(batched(items, 50))
    assert [len(c) for c in chunks] == [50, 50, 37]


def test_batching_preserves_every_item_exactly_once():
    items = list(range(137))
    flat = [x for chunk in batched(items, 50) for x in chunk]
    assert flat == items


def test_an_empty_input_yields_no_batches():
    assert list(batched([], 50)) == []
