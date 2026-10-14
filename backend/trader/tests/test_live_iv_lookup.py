"""Tests for `latest_fit`, which falls back to the store when the in-process cache is
empty.

A restart empties `record_implied_vol.CACHE`. The backtest accepts a stored fit up
to `MAX_FIT_AGE_MINUTES` old, and the artifact never saw a NaN there, so live
reads the store too.
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.live import latest_fit, reset_fit_lookup

NOW = pd.Timestamp('2026-08-28 20:40', tz='UTC')


@pytest.fixture(autouse=True)
def _clean():
    reset_fit_lookup()
    yield
    reset_fit_lookup()


def _stored(minutes_ago, sigma=8.0):
    return pd.DataFrame({
        'symbol': ['BTC-USD'],
        'event_time': [NOW - pd.Timedelta(minutes=minutes_ago)],
        'implied_sigma_per_min': [sigma], 'r2': [0.97], 'n_strikes': [12.0],
    })


def test_the_in_process_cache_wins_when_it_is_fresher():
    cache = {'BTC-USD': {'implied_sigma_per_min': 9.0, 'r2': 0.99,
                         'n_strikes': 14.0, 'at': NOW - pd.Timedelta(minutes=1)}}
    fit = latest_fit('BTC-USD', now=NOW, cache=cache, read=lambda: _stored(26))
    assert fit['implied_sigma_per_min'] == 9.0


def test_the_store_is_used_when_the_cache_is_empty():
    """After a restart the store's fit is used."""
    fit = latest_fit('BTC-USD', now=NOW, cache={}, read=lambda: _stored(26))
    assert fit is not None and fit['implied_sigma_per_min'] == 8.0
    assert (NOW - fit['at']).total_seconds() / 60 == pytest.approx(26.0)


def test_a_stored_fit_past_the_age_cap_is_not_used():
    """A stored fit older than MAX_FIT_AGE_MINUTES is not used."""
    assert latest_fit('BTC-USD', now=NOW, cache={},
                      read=lambda: _stored(400)) is None


def test_a_symbol_with_nothing_anywhere_is_none_not_an_error():
    assert latest_fit('SOL-USD', now=NOW, cache={}, read=lambda: _stored(26)) is None


def test_the_store_is_not_read_on_every_call():
    """The store is not read on every call, to keep a synchronous Parquet read off the
    event loop.
    """
    calls = []

    def _read():
        calls.append(1)
        return _stored(26)

    for _ in range(5):
        latest_fit('BTC-USD', now=NOW, cache={}, read=_read)
    assert len(calls) == 1, f'read {len(calls)} times, expected 1'
