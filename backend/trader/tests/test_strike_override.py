"""Swapping in the venue's strike must move every strike-derived quantity.

`scripts/live.py` records the failure this prevents: an earlier version
overwrote `strike` and recomputed only `z_score`, so `abs_z_score ==
|z_score|` -- an identity that holds in every training row -- was violated at
inference on every cycle, invisibly. A tree ensemble does not degrade
gracefully off its training manifold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from core.windows import build_windows

WINDOW = DEFAULT_CONFIG.window_minutes


def _bars(n_windows: int = 8, start: str = '2026-03-01') -> pd.DataFrame:
    n = n_windows * WINDOW
    idx = pd.date_range(start, periods=n, freq='min', tz='UTC')
    rng = np.random.default_rng(7)
    close = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    return pd.DataFrame({
        'event_time': idx, 'open': close, 'high': close + 0.02,
        'low': close - 0.02, 'close': close, 'volume': 1.0,
    })


def test_without_an_override_nothing_changes():
    table, _ = build_windows(_bars(), 'BTC-USD', DEFAULT_CONFIG, offsets=[3])
    assert (table['strike_source'] == 'proxy').all()


def test_the_override_moves_strike_displacement_and_excursions_together():
    bars = _bars()
    base, _ = build_windows(bars, 'BTC-USD', DEFAULT_CONFIG, offsets=[3])
    opens = pd.DatetimeIndex(base['window_open'].unique())
    shifted = pd.Series(base.groupby('window_open')['strike'].first().to_numpy() + 1.0,
                        index=opens)
    out, _ = build_windows(bars, 'BTC-USD', DEFAULT_CONFIG, offsets=[3],
                           strike_override=shifted)
    assert (out['strike_source'] == 'venue').all()
    assert np.allclose(out['strike'].to_numpy(),
                       base['strike'].to_numpy() + 1.0)
    # every derived quantity recomputed against the NEW strike, not the old one
    assert np.allclose(out['displacement'].to_numpy(),
                       out['last_price'].to_numpy() / out['strike'].to_numpy() - 1.0)
    assert np.allclose(out['excursion_up'].to_numpy(),
                       (base['excursion_up'].to_numpy() + 1.0)
                       * base['strike'].to_numpy() / out['strike'].to_numpy() - 1.0)
    assert not np.allclose(out['displacement'].to_numpy(),
                           base['displacement'].to_numpy())


def test_a_partial_override_marks_only_the_windows_it_covers():
    """The venue does not list every window; those rows keep our strike and
    must say so, or a mixed sample reads as one measurement."""
    bars = _bars()
    base, _ = build_windows(bars, 'BTC-USD', DEFAULT_CONFIG, offsets=[3])
    opens = pd.DatetimeIndex(base['window_open'].unique())
    partial = pd.Series(base.groupby('window_open')['strike'].first().to_numpy() + 1.0,
                        index=opens)
    partial.iloc[1:] = np.nan                     # only the first window covered
    out, _ = build_windows(bars, 'BTC-USD', DEFAULT_CONFIG, offsets=[3],
                           strike_override=partial)
    assert (out['strike_source'] == 'venue').sum() == 1
    assert (out['strike_source'] == 'proxy').sum() == len(out) - 1


class TestTheConfigSwitch:
    def test_the_default_is_the_coinbase_proxy(self):
        from core.config import DEFAULT_CONFIG
        assert DEFAULT_CONFIG.strike_source == 'proxy'

    def test_only_proxy_and_venue_are_accepted(self):
        import dataclasses

        import pytest

        from core.config import DEFAULT_CONFIG
        with pytest.raises(Exception):
            dataclasses.replace(DEFAULT_CONFIG, strike_source='brti').validate()

    def test_a_missing_table_falls_back_rather_than_building_nan_windows(self):
        from core.dataset import venue_strike_series
        assert venue_strike_series(None) is None

    def test_a_table_without_floor_strike_is_refused_not_half_used(self):
        import pandas as pd

        from core.dataset import venue_strike_series
        f = pd.DataFrame({'symbol': ['BTC-USD'],
                          'window_open': pd.to_datetime(['2026-03-01'], utc=True)})
        assert venue_strike_series(f) is None
