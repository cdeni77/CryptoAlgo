"""Tests for timezone and date-slicing invariants on the research store read path.

Reads return UTC whatever the process timezone, since seasonality uses minute-of-day.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from core.datastore import ResearchStore

NEW_YORK = ZoneInfo('America/New_York')


def rows(start: str, periods: int) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq='1min', tz='UTC')
    return pd.DataFrame({
        'venue': 'coinbase_spot', 'symbol': 'BTC-USD', 'event_time': index,
        'available_time': index + pd.Timedelta(minutes=1),
        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.5,
        'volume': 1.0, 'quote_volume': np.nan, 'trade_count': np.nan,
        'quality': 'valid', 'ingested_at': pd.Timestamp('2025-02-01', tz='UTC'),
        'revision': 1,
    })


@pytest.fixture
def store(tmp_path) -> ResearchStore:
    s = ResearchStore(tmp_path)
    s.write('minute_bars', rows('2025-01-01', 600))
    return s


def test_reads_come_back_in_utc(store):
    """Reads come back in UTC, whatever the machine's timezone."""
    frame = store.read('minute_bars')
    assert str(frame['event_time'].dt.tz) == 'UTC'
    assert str(frame['available_time'].dt.tz) == 'UTC'
    assert frame['event_time'].min() == pd.Timestamp('2025-01-01 00:00', tz='UTC')


def test_minute_of_day_is_utc_minute_of_day(store):
    """Minute-of-day, which seasonality is indexed by, is UTC minute-of-day."""
    frame = store.read('minute_bars', start='2025-01-01 02:00', end='2025-01-01 03:00')
    minute_of_day = frame['event_time'].dt.hour * 60 + frame['event_time'].dt.minute
    assert minute_of_day.min() == 120 and minute_of_day.max() == 180


def test_a_date_limited_read_works_at_all(store):
    """A read limited by `start`/`end` strings works."""
    sliced = store.read('minute_bars', start='2025-01-01 02:00', end='2025-01-01 03:00')
    assert len(sliced) == 61
    assert sliced['event_time'].min() == pd.Timestamp('2025-01-01 02:00', tz='UTC')
    assert sliced['event_time'].max() == pd.Timestamp('2025-01-01 03:00', tz='UTC')


def test_as_of_excludes_rows_not_yet_available(store):
    frame = store.read('minute_bars', as_of='2025-01-01 01:00')
    assert (frame['available_time'] <= pd.Timestamp('2025-01-01 01:00', tz='UTC')).all()


class TestEquityHours:
    """`us_equity_hours` marks the US session in both EST and EDT.

    Calls the real `_clock_features` instead of recomputing the comparison.
    """

    def flag(self, when: pd.Timestamp) -> float:
        from core.config import DEFAULT_CONFIG
        from core.features import _clock_features

        frame = pd.DataFrame({
            'decision_time': [when.tz_convert('UTC')],
            'window_open': [when.tz_convert('UTC').floor('15min')],
            'offset': [3],
        })
        return float(_clock_features(frame, DEFAULT_CONFIG)['us_equity_hours'].iloc[0])

    @pytest.mark.parametrize('day', ('2026-07-15', '2026-01-15'))
    def test_the_session_is_flagged_in_both_regimes(self, day):
        """09:30-15:59 New York is the session in summer and in winter."""
        for hhmm in ('09:30', '12:00', '15:59'):
            assert self.flag(pd.Timestamp(f'{day} {hhmm}', tz=NEW_YORK)) == 1.0, \
                f'{day} {hhmm} New York should be the session'
        for hhmm in ('09:29', '16:00', '03:00'):
            assert self.flag(pd.Timestamp(f'{day} {hhmm}', tz=NEW_YORK)) == 0.0, \
                f'{day} {hhmm} New York should not be'

    def test_the_weekend_is_never_the_session(self):
        saturday = pd.Timestamp('2026-07-18 12:00', tz=NEW_YORK)
        assert saturday.dayofweek == 5
        assert self.flag(saturday) == 0.0

    def test_the_old_utc_band_would_have_failed_in_est(self):
        """A fixed 13:30-20:00 UTC band would miss the last hour of trading in EST."""
        january_close = pd.Timestamp('2026-01-15 15:59', tz=NEW_YORK).tz_convert('UTC')
        utc_minute = january_close.hour * 60 + january_close.minute
        assert not (13 * 60 + 30) <= utc_minute < 20 * 60, (
            'the old fixed band happens to include this, so it no longer '
            'demonstrates the problem'
        )
        assert self.flag(january_close) == 1.0


def test_a_date_limited_read_accepts_bounds_that_are_already_tz_aware(store):
    """`start`/`end` bounds that are already tz-aware are accepted."""
    sliced = store.read(
        'minute_bars',
        start=pd.Timestamp('2025-01-01 02:00', tz='UTC'),
        end=pd.Timestamp('2025-01-01 03:00', tz='UTC'),
    )
    assert len(sliced) == 61


def test_a_naive_and_an_aware_bound_select_the_same_rows(store):
    naive = store.read('minute_bars', start='2025-01-01 02:00',
                       end='2025-01-01 03:00')
    aware = store.read('minute_bars',
                       start=pd.Timestamp('2025-01-01 02:00', tz='UTC'),
                       end=pd.Timestamp('2025-01-01 03:00', tz='UTC'))
    assert len(naive) == len(aware)
    assert naive['event_time'].equals(aware['event_time'])


def test_a_bound_in_another_zone_is_converted_not_relabelled(store):
    """A bound in another zone is converted, not relabelled: 21:00 New York is 02:00 UTC
    next day.
    """
    aware = store.read(
        'minute_bars',
        start=pd.Timestamp('2024-12-31 21:00', tz=NEW_YORK),
        end=pd.Timestamp('2024-12-31 22:00', tz=NEW_YORK),
    )
    assert len(aware) == 61
    assert aware['event_time'].min() == pd.Timestamp('2025-01-01 02:00', tz='UTC')
