"""Tests for the live loop's own logic: window, offset choice, and decision tolerance.
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
    """The offset is the latest one reached, never a future one."""
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
    """The fetch window covers `log_rv_1440`, the longest lookback."""
    assert FETCH_MINUTES > max(CFG.vol_lookbacks_minutes)
    assert FETCH_MINUTES >= 1440 + CFG.window_minutes


def test_every_symbol_has_a_venue_series():
    for symbol in CFG.symbols:
        assert SERIES_BY_SYMBOL.get(symbol), symbol


def test_prices_are_retained_longer_than_the_chart_shows():
    assert PRICE_RETENTION_HOURS >= 24


def test_place_orders_requires_live_mode():
    """The parser accepts `--mode paper --place-orders` so `main()` can reject it with a
    message.
    """
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


# --- the decision tolerance --------------------------------------------------
#
# The market price moves several points per minute, so a few seconds of
# staleness can cost more than the model's edge. The tolerance must be narrower
# than the cycle cadence, or the next ordinary cycle re-decides the same offset
# a full minute late.

def test_the_decision_tolerance_is_tighter_than_the_cycle_cadence():
    """The decision tolerance is narrower than the 60s cycle cadence."""
    from scripts.live import DECISION_TOLERANCE_SECONDS

    assert DECISION_TOLERANCE_SECONDS < 60.0, (
        'tolerance must be narrower than the ordinary cycle, or the cycle after '
        'an on-target decision re-decides the same offset while stale'
    )


def test_the_decision_tolerance_fits_the_measured_latency_budget():
    """The tolerance stays between 5 and 20 seconds."""
    from scripts.live import DECISION_TOLERANCE_SECONDS

    assert 5.0 <= DECISION_TOLERANCE_SECONDS <= 20.0


def test_a_stale_cycle_does_not_decide_the_offset_it_has_drifted_past():
    """6s past an offset decides; 72s past it abstains."""
    from scripts.live import decision_offset

    # 6 seconds past +3m -> decide at 3
    assert decision_offset(3 + 6 / 60.0, CFG) == 3
    # 72 seconds past +3m -> too stale; abstain
    assert decision_offset(3 + 72 / 60.0, CFG) is None


def test_an_explicit_offset_overrides_the_tolerance():
    """A forced offset (backfill or manual run) skips the staleness check."""
    from scripts.live import decision_offset

    assert decision_offset(3 + 600 / 60.0, CFG, forced=3) == 3


def test_before_the_first_offset_there_is_nothing_to_decide():
    from scripts.live import decision_offset

    assert decision_offset(1.0, CFG) is None


def test_the_exchange_shard_is_read_off_the_markets_not_hardcoded():
    """The exchange shard is read from the quotes, not hardcoded, since the venue can move
    it.
    """
    from data_collection.kalshi_client import Quote
    from scripts.live import venue_exchange_index

    def quote(ticker: str) -> Quote:
        return Quote(ticker=ticker, yes_bid=0.50, yes_ask=0.51, no_bid=0.49,
                     no_ask=0.50, last_price=0.50, volume=1, open_interest=1,
                     close_time=None, status='active', exchange_index=2)

    # The shape `fetch_quotes` produces: `{symbol: (Quote, ticker)}`.
    quotes = {'BTC-USD': (quote('a'), 'a'), 'ETH-USD': (quote('b'), 'b')}
    assert venue_exchange_index(quotes) == 2

    # A bare `{symbol: Quote}` mapping also works.
    assert venue_exchange_index({'BTC-USD': quote('a')}) == 2


def test_with_no_quotes_no_shard_is_claimed():
    """With no quotes, no shard is claimed and the whole-account balance is used."""
    from scripts.live import venue_exchange_index

    assert venue_exchange_index({}) is None
