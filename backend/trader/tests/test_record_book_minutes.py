"""The minute sampler that feeds `book_flow`.

It reads the stream cache in process -- no network, no market resolution -- and
stamps each sample with how far past the minute mark it was taken, because that
IS its age: the cache is current to ~34ms, so what makes a row old is the
sampling phase, not the transport.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from livetrade.features import (
    book_history_depth, reset_book_history,
)
from scripts.record_book_minutes import _symbol_for, sample_once

OPEN = dt.datetime(2026, 10, 21, 12, 0, tzinfo=dt.timezone.utc)
# 12:00 UTC window -> its CLOSE is 12:15 UTC, which the ticker names in New York
# local time. Built through the same parser the sampler uses so the test cannot
# encode a different convention.
from scripts.backfill_candles import window_from_ticker


def _ticker_for(series, close_utc):
    ny = close_utc.astimezone(dt.timezone.utc)
    from zoneinfo import ZoneInfo
    local = close_utc.astimezone(ZoneInfo('America/New_York'))
    mon = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
           'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'][local.month - 1]
    return '%s-%02d%s%02d%02d%02d-%02d' % (
        series, local.year % 100, mon, local.day, local.hour, local.minute,
        local.minute)


class _Cache:
    """The parts of `core.stream_book.BookCache` the sampler touches."""

    def __init__(self, books):
        self._books = books

    def tickers(self):
        return sorted(self._books)

    def gapped(self, ticker):
        return False

    def ladder(self, ticker):
        yes, no = self._books[ticker]
        return type('L', (), {'yes': yes, 'no': no, 'stale': False})()


@pytest.fixture(autouse=True)
def _clean():
    reset_book_history()
    yield
    reset_book_history()


def _cache_for(series='KXBTC15M'):
    tk = _ticker_for(series, OPEN + dt.timedelta(minutes=15))
    assert window_from_ticker(tk) is not None, tk
    return tk, _Cache({tk: ([(0.50, 100), (0.49, 200)],
                            [(0.48, 90), (0.47, 180)])})


def test_a_sample_at_the_minute_mark_is_kept():
    tk, cache = _cache_for()
    opened, _ = window_from_ticker(tk)
    assert sample_once(cache, now=opened + dt.timedelta(minutes=3)) == 1
    held = book_history_depth()
    assert len(held) == 1
    assert int(held.iloc[0]['offset_minutes']) == 3
    assert held.iloc[0]['symbol'] == 'BTC-USD'


def test_a_sample_well_past_the_minute_is_refused():
    # `decision_age` would age it negative and `book_flow_features` would drop
    # it; refusing here makes a drifting sampler visible.
    tk, cache = _cache_for()
    opened, _ = window_from_ticker(tk)
    assert sample_once(cache, now=opened + dt.timedelta(minutes=3, seconds=7)) == 0
    assert book_history_depth().empty


def test_a_sample_outside_the_window_is_not_recorded():
    tk, cache = _cache_for()
    opened, _ = window_from_ticker(tk)
    assert sample_once(cache, now=opened - dt.timedelta(minutes=1)) == 0
    assert sample_once(cache, now=opened + dt.timedelta(minutes=16)) == 0


def test_no_cache_is_not_an_error():
    assert sample_once(None) == 0


def test_an_unknown_series_is_skipped_not_guessed():
    tk, cache = _cache_for(series='KXDOGE15M')
    opened, _ = window_from_ticker(tk)
    assert sample_once(cache, now=opened + dt.timedelta(minutes=3)) == 0


def test_the_symbol_comes_from_the_shared_mapping():
    from core.config import SERIES_BY_SYMBOL
    for symbol, series in SERIES_BY_SYMBOL.items():
        assert _symbol_for('%s-26OCT211215-15' % series) == symbol
    assert _symbol_for('NOPE-26OCT211215-15') is None


def test_every_minute_of_a_window_accumulates():
    tk, cache = _cache_for()
    opened, _ = window_from_ticker(tk)
    for m in range(0, 13):
        sample_once(cache, now=opened + dt.timedelta(minutes=m, seconds=0.4))
    held = book_history_depth()
    assert sorted(held['offset_minutes']) == list(range(0, 13))


def test_the_sleep_re_establishes_the_minute_phase():
    """A fixed 60s sleep DRIFTS, and the drift is fatal here.

    `sleep(60)` runs after the work, so each cycle costs 60s plus however long
    sampling took. At ~45ms of work that is 45ms/minute, crossing the
    3-second freshness threshold after ~67 minutes. Measured live 2026-10-21:
    rows were kept for 68 minutes after a restart and then every one was
    refused for six hours, while the heartbeat kept stamping because it marks
    liveness, not success.

    **Calls the production function.** This test used to define its own copy of
    the formula, so reverting the loop to `sleep(PERIOD_SECONDS)` -- the exact
    regression above -- left all nine tests in this file green.
    """
    from scripts.record_book_minutes import PERIOD_SECONDS, wait_seconds

    # Phase is re-established wherever in the minute the work finished.
    for second in (0.0, 0.045, 13.7, 59.2):
        landed = (second + wait_seconds(second)) % PERIOD_SECONDS
        assert landed == pytest.approx(0.0, abs=1e-9), second

    # It is never the bare period: that is the drift this guards.
    assert wait_seconds(0.045) < PERIOD_SECONDS
    # And never zero, or the loop spins when the work overran the boundary.
    assert wait_seconds(59.999) >= 0.5


def test_a_fixed_period_sleep_would_have_drifted_past_the_threshold():
    """The counterfactual, so the regression is a measurement not a belief."""
    from scripts.record_book_minutes import PERIOD_SECONDS
    from livetrade.features import BOOK_SAMPLE_MAX_AGE_SECONDS
    now, work = 960.0, 0.045               # 960 % 60 == 0: start ON the boundary
    assert now % PERIOD_SECONDS == 0
    for i in range(200):
        now += work + PERIOD_SECONDS          # the OLD behaviour
        if (now % PERIOD_SECONDS) > BOOK_SAMPLE_MAX_AGE_SECONDS:
            assert 60 <= i <= 75, 'drifted out at cycle %d' % i
            break
    else:
        raise AssertionError('expected the fixed-period sleep to drift out')
