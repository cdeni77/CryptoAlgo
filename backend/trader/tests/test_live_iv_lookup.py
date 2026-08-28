"""Live read implied vol only from the in-process cache, which starts empty.

The backtest attaches it with an as-of join over stored history, tolerating a
fit up to `MAX_FIT_AGE_MINUTES` (360) old. Live consulted only
`record_implied_vol.CACHE`, which a restart empties — so after every deploy the
five iv_* features were NaN until the recorder happened to fit again, even
though the store held a usable fit from twenty-six minutes earlier.

That is not a small gap. `--complete-cases` means the artifact was fitted on
66,555 rows that ALL carry an implied-vol fit; it has never seen a NaN there.
Scoring without one is out-of-distribution, on a live account.
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
    """A restart empties the cache; the store still holds the fit."""
    fit = latest_fit('BTC-USD', now=NOW, cache={}, read=lambda: _stored(26))
    assert fit is not None and fit['implied_sigma_per_min'] == 8.0
    assert (NOW - fit['at']).total_seconds() / 60 == pytest.approx(26.0)


def test_a_stored_fit_past_the_age_cap_is_not_used():
    """Beyond MAX_FIT_AGE_MINUTES it describes a different session."""
    assert latest_fit('BTC-USD', now=NOW, cache={},
                      read=lambda: _stored(400)) is None


def test_a_symbol_with_nothing_anywhere_is_none_not_an_error():
    assert latest_fit('SOL-USD', now=NOW, cache={}, read=lambda: _stored(26)) is None


def test_the_store_is_not_read_on_every_call():
    """The hot path is a decision. Reading a Parquet partition per cycle would
    put a synchronous pandas read on the event loop for a value that changes
    every few minutes at most."""
    calls = []

    def _read():
        calls.append(1)
        return _stored(26)

    for _ in range(5):
        latest_fit('BTC-USD', now=NOW, cache={}, read=_read)
    assert len(calls) == 1, f'read {len(calls)} times, expected 1'
