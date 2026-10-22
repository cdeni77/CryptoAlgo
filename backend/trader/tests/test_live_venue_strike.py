"""The live path scores on the strike the model was fitted on.

`build_dataset` has honoured `strike_source='venue'` since the venue-native
work; `score_live` did not pass a strike at all, so a model trained on venue
strikes would have been scored live on bar-derived ones. That divergence is
silent: the drift warning fires above 25bp and the Coinbase-vs-BRTI basis is
about 1bp.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from core.dataset import _override_series, score_live
from livetrade import quotes as lq


class Meta:
    def __init__(self, strike):
        self.floor_strike = strike
        self.ticker = 'KXBTC15M-X'


SETTLE = pd.Timestamp('2026-10-22 16:15', tz='UTC')


@pytest.fixture(autouse=True)
def clean_cache():
    lq._MARKETS.clear()                                        # noqa: SLF001
    yield
    lq._MARKETS.clear()                                        # noqa: SLF001


def test_it_reads_the_cache_without_a_network_call():
    lq._MARKETS['BTC-USD'] = (SETTLE, Meta(81665.43))          # noqa: SLF001
    out = lq.cached_floor_strikes(['BTC-USD', 'ETH-USD'], SETTLE)
    assert out == {'BTC-USD': 81665.43}, 'absent, never zero'


def test_a_strike_from_another_window_is_refused():
    """A stale entry would price this window against the wrong barrier."""
    lq._MARKETS['BTC-USD'] = (SETTLE - pd.Timedelta(minutes=15),  # noqa: SLF001
                              Meta(81665.43))
    assert lq.cached_floor_strikes(['BTC-USD'], SETTLE) == {}


@pytest.mark.parametrize('bad', [None, 0.0, -1.0, 'nonsense'])
def test_an_unusable_strike_is_dropped_not_coerced(bad):
    lq._MARKETS['BTC-USD'] = (SETTLE, Meta(bad))               # noqa: SLF001
    assert lq.cached_floor_strikes(['BTC-USD'], SETTLE) == {}


def test_score_live_accepts_and_forwards_the_strike():
    assert 'strikes_by_symbol' in inspect.signature(score_live).parameters
    src = inspect.getsource(score_live)
    assert 'strikes_by_symbol=_override_series' in src, \
        'it must reach build_window_panel, not be accepted and dropped'


def test_the_override_is_shaped_for_build_windows():
    bars = pd.DataFrame({'close': [1.0, 2.0]},
                        index=pd.date_range('2026-10-22', periods=2, freq='min'))
    out = _override_series({'BTC-USD': 81665.43}, {'BTC-USD': bars})
    assert list(out) == ['BTC-USD']
    assert out['BTC-USD'].index.equals(bars.index), 'indexed like its bars'
    assert (out['BTC-USD'] == 81665.43).all()


def test_nothing_overridden_returns_none_so_the_old_path_is_untouched():
    bars = pd.DataFrame({'close': [1.0]},
                        index=pd.date_range('2026-10-22', periods=1, freq='min'))
    assert _override_series(None, {'BTC-USD': bars}) is None
    assert _override_series({}, {'BTC-USD': bars}) is None
    assert _override_series({'BTC-USD': np.nan}, {'BTC-USD': bars}) is None


def test_live_warns_when_only_some_symbols_have_a_venue_strike():
    """A partial fallback is the mixed state C6 forbids; it must not pass
    silently."""
    import scripts.live as live
    src = inspect.getsource(live.run_cycle)
    assert 'cached_floor_strikes' in src
    assert 'fall back to the bar-derived strike' in src
