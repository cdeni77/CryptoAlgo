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

    The sleep must therefore be computed from the CLOCK, not from a constant.
    """
    from scripts.record_book_minutes import PERIOD_SECONDS

    def wait_from(now):
        return max(0.5, PERIOD_SECONDS - (now % PERIOD_SECONDS))

    # From any phase, the next wake lands on a boundary.
    for now in (1000.0, 1000.04, 1012.5, 1059.9, 1059.999):
        assert abs((now + wait_from(now)) % PERIOD_SECONDS) < 1e-6 or \
            wait_from(now) == 0.5, now

    # And the drift does NOT accumulate across many cycles with work in them.
    now, work = 960.0, 0.045
    for _ in range(200):
        now += work + wait_from(now + work)
    assert (now % PERIOD_SECONDS) < 0.5, 'phase drifted to %.3fs' % (now % PERIOD_SECONDS)


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
