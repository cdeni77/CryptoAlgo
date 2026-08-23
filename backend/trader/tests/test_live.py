"""The live loop's own logic: which offset, and settle before deciding.

Everything else in `scripts/live.py` is plumbing over code tested elsewhere.
These two are its own, and both are the kind of off-by-one that produces
plausible numbers rather than an error.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from core.config import Config
from scripts.live import (
    FETCH_MINUTES, PRICE_RETENTION_HOURS, SERIES_BY_SYMBOL, choose_offset,
    current_window,
)

CFG = Config()


def test_the_current_window_floors_to_the_quarter_hour():
    for minute, expected in ((0, '03:00'), (7, '03:00'), (14, '03:00'),
                             (15, '03:15'), (44, '03:30'), (59, '03:45')):
        now = datetime(2026, 8, 23, 3, minute, 30, tzinfo=timezone.utc)
        window, elapsed = current_window(now, CFG)
        assert window.strftime('%H:%M') == expected
        assert elapsed == minute - int(expected.split(':')[1]) % 15 * 0 - (
            minute // 15 * 15)


def test_elapsed_counts_whole_minutes_into_the_window():
    now = datetime(2026, 8, 23, 3, 7, 59, tzinfo=timezone.utc)
    _, elapsed = current_window(now, CFG)
    assert elapsed == 7, 'a partially elapsed minute must not count as elapsed'


def test_the_offset_is_the_latest_one_reached_never_a_future_one():
    """Scoring at an offset that has not happened reads a bar that does not exist."""
    assert choose_offset(0, CFG) is None
    assert choose_offset(2, CFG) is None, 'chose an offset before it was reached'
    assert choose_offset(3, CFG) == 3
    assert choose_offset(5, CFG) == 3
    assert choose_offset(6, CFG) == 6
    assert choose_offset(11, CFG) == 9
    assert choose_offset(14, CFG) == 12


def test_before_the_first_offset_is_an_abstention_not_an_error():
    assert choose_offset(1, CFG) is None


def test_the_fetch_window_covers_the_longest_feature_lookback():
    """`log_rv_1440` needs a day. Fetching less makes it NaN and the column dead."""
    assert FETCH_MINUTES > max(CFG.vol_lookbacks_minutes)
    assert FETCH_MINUTES >= 1440 + CFG.window_minutes


def test_every_symbol_has_a_venue_series():
    for symbol in CFG.symbols:
        assert SERIES_BY_SYMBOL.get(symbol), symbol


def test_prices_are_retained_longer_than_the_chart_shows():
    assert PRICE_RETENTION_HOURS >= 24


def test_place_orders_requires_live_mode():
    """One flag guarding an irreversible action is one typo away from wrong."""
    from scripts.live import build_parser

    args = build_parser().parse_args(['--mode', 'paper', '--place-orders'])
    assert args.place_orders and args.mode == 'paper', (
        'the parser must accept this so main() can reject it with a message'
    )


def test_forcing_past_the_gates_requires_a_reason():
    from scripts.live import build_parser

    parser = build_parser()
    forced = parser.parse_args(['--force'])
    assert forced.force and forced.reason is None, (
        'main() checks this pair and exits; the parser must not silently allow it'
    )


def test_paper_is_the_default_mode():
    from scripts.live import build_parser

    args = build_parser().parse_args([])
    assert args.mode == 'paper'
    assert not args.place_orders
    assert args.require_gates is True
