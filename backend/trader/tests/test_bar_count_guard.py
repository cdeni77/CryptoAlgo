"""Live checked that its bars were FRESH and never that they were COMPLETE.

`stale_symbols` compares only the newest bar's timestamp against
`max_bar_age_seconds`. Nothing looked at how MANY bars came back, and
`get_candles_range` returns a short or holed series after logging an ERROR.

The features need more history than that check implies. `volume_z_15` is a
1,440-minute z-score of a 15-minute mean, so bit-parity needs 1,455 contiguous
bars; `log_rv_1440` needs 1,441. Live fetches 1,500 — a 45-bar margin that
nothing guards.

Measured degradation against the walk-forward value:

    bars   volume_z_15   rv_surprise   rv_slope_long
    1440        105%          16%           3.5%
    1200         46x          67x           6.8x
     800        130x         285x            12x

The shape is the wrong way round. Below ~721 bars `log_rv_1440` fails its
`min_periods` and the NaN propagates to `sigma_remaining`, so `decide()`
abstains — safe. Between 721 and 1,454 the numbers are confidently wrong and the
cycle still trades, and that band is exactly where a partial fetch lands.

It is invisible: a short-but-unholed grid spans what was RETURNED, so coverage
reports 100.0000% and logs at DEBUG.

And it is not only a feature problem — the same `sigma_per_min` scales
`sigma_remaining`, so a short fetch moves the baseline the model is a correction
to.
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
    """1,200 bars is fresh, unholed, reports 100% coverage — and puts
    rv_surprise 67x out."""
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols}
    bars[DEFAULT_CONFIG.symbols[0]] = _frame(1200)
    reasons = short_symbols(bars, DEFAULT_CONFIG)
    assert DEFAULT_CONFIG.symbols[0] in reasons
    assert '1200' in reasons[DEFAULT_CONFIG.symbols[0]]


def test_the_threshold_covers_the_longest_lookback_any_feature_needs():
    """volume_z_15 needs 1,455; log_rv_1440 needs 1,441. A threshold under the
    larger of those readmits the band this guard exists to close."""
    assert MIN_USABLE_BARS >= 1455


def test_a_holed_history_is_refused_even_when_it_is_long_enough():
    """A grid can span 1,500 minutes and contain 400 bars. Counting rows, not
    the span, is the point — coverage reports the span."""
    full = _frame(1500)
    holed = pd.concat([full.iloc[:600], full.iloc[900:]], ignore_index=True)
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols}
    bars[DEFAULT_CONFIG.symbols[0]] = holed
    assert DEFAULT_CONFIG.symbols[0] in short_symbols(bars, DEFAULT_CONFIG)


def test_a_missing_symbol_is_reported_rather_than_skipped():
    bars = {s: _frame(1500) for s in DEFAULT_CONFIG.symbols[1:]}
    assert DEFAULT_CONFIG.symbols[0] in short_symbols(bars, DEFAULT_CONFIG)
