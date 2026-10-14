"""Tests that live refuses a symbol whose bar history is too short or holed.

`volume_z_15` needs 1,455 contiguous bars and `log_rv_1440` needs 1,441. Below
about 721 bars the features are NaN and `decide()` abstains. Between 721 and
1,454 the features are wrong but finite, and the cycle would still trade. The
same history also scales `sigma_remaining`, so it moves the baseline as well.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from scripts.live import MIN_USABLE_BARS, short_symbols


def _frame(n, end=None):
    end = end or pd.Timestamp('2026-09-03 12:00', tz='UTC')
    idx = [end - pd.Timedelta(minutes=i) for i in range(n)][::-1]
    return pd.DataFrame({'event_time': idx, 'close': 1.0, 'volume': 1.0})


def test_a_full_history_passes():
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols}
    assert short_symbols(bars, DEFAULT_CONFIG) == {}


def test_a_symbol_inside_the_silent_corruption_band_is_refused():
    """1,200 bars is fresh and unholed but inside the band where features are wrong."""
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols}
    bars[DEFAULT_CONFIG.symbols[0]] = _frame(1200)
    reasons = short_symbols(bars, DEFAULT_CONFIG)
    assert DEFAULT_CONFIG.symbols[0] in reasons
    assert '1200' in reasons[DEFAULT_CONFIG.symbols[0]]


def test_the_threshold_covers_the_longest_lookback_any_feature_needs():
    """The threshold covers the longest lookback of any feature (1,455 for volume_z_15).
    """
    assert MIN_USABLE_BARS >= 1455


def test_a_holed_history_is_refused_even_when_it_is_long_enough():
    """The guard counts rows, not the span, so a holed 1,500-minute grid is refused."""
    full = _frame(1500)
    holed = pd.concat([full.iloc[:600], full.iloc[900:]], ignore_index=True)
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols}
    bars[DEFAULT_CONFIG.symbols[0]] = holed
    assert DEFAULT_CONFIG.symbols[0] in short_symbols(bars, DEFAULT_CONFIG)


def test_a_missing_symbol_is_reported_rather_than_skipped():
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols[1:]}
    assert DEFAULT_CONFIG.symbols[0] in short_symbols(bars, DEFAULT_CONFIG)
