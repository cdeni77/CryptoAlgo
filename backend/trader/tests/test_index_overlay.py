"""Serving the settlement index live, with spot's activity carried onto it.

C1 measured the venue strike plus the BRTI price beating Coinbase with an
OHLC-mean strike by +0.0033 log loss at t=+5.71. C6 binds the strike and the
price series together, so the live path has to serve the same index the model
was fitted on -- and the index has no volume, verified on both transports, so
the bars become a hybrid that training and live must build identically.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from core.config import DEFAULT_CONFIG
from core.index_bars import ACTIVITY_COLUMNS, carry_activity
from livetrade import bars as live_bars

MIN = pd.Timestamp('2026-10-22 04:00', tz='UTC')


def _spot(n=3):
    return pd.DataFrame({
        'event_time': [MIN + pd.Timedelta(minutes=i) for i in range(n)],
        'open': [100.0] * n, 'high': [101.0] * n,
        'low': [99.0] * n, 'close': [100.5] * n,
        'volume': [7.0 + i for i in range(n)],
        'quote_volume': [700.0 + 100 * i for i in range(n)],
        'trade_count': [70 + 10 * i for i in range(n)],
    })


def _index(n=3):
    return pd.DataFrame({
        'event_time': [MIN + pd.Timedelta(minutes=i) for i in range(n)],
        'open': [200.0] * n, 'high': [201.0] * n,
        'low': [199.0] * n, 'close': [200.5] * n,
        'mean_price': [200.25] * n,
    })


class TestCarryActivity:
    def test_the_index_keeps_its_price_and_takes_spot_activity(self):
        got = carry_activity(_index(), _spot())
        assert got['close'].tolist() == [200.5] * 3, 'index price must survive'
        assert got['volume'].tolist() == [7.0, 8.0, 9.0]
        assert got['mean_price'].tolist() == [200.25] * 3

    def test_an_index_minute_with_no_spot_bar_keeps_a_null_volume(self):
        """Dropping it belongs to --complete-cases, not to this join."""
        got = carry_activity(_index(3), _spot(2))
        assert len(got) == 3
        assert np.isnan(got['volume'].iloc[2])

    def test_it_replaces_rather_than_duplicates_an_existing_column(self):
        idx = _index()
        idx['volume'] = 999.0
        got = carry_activity(idx, _spot())
        assert got['volume'].tolist() == [7.0, 8.0, 9.0]
        assert len([c for c in got.columns if c.startswith('volume')]) == 1

    def test_empty_inputs_pass_through(self):
        assert carry_activity(None, _spot()) is None
        out = carry_activity(_index(), pd.DataFrame())
        assert out['close'].tolist() == [200.5] * 3

    def test_training_and_live_share_this_one_definition(self):
        """Two implementations would give `signed_volume_15` and
        `vwap_gap_15` different meanings in training and serving."""
        import inspect

        from scripts import backfill_index
        assert 'carry_activity' in inspect.getsource(
            backfill_index.attach_coinbase_activity)


class TestOverlayGating:
    @pytest.mark.asyncio
    async def test_the_default_venue_leaves_spot_untouched(self):
        spot = {'BTC-USD': _spot()}
        got = await live_bars._overlay_index(DEFAULT_CONFIG, spot, 1500)
        assert got is spot

    @pytest.mark.asyncio
    async def test_a_failing_overlay_refuses_to_serve_spot_prices(self, monkeypatch):
        """Fitted on the index, served on spot, is the cross-index basis C6
        exists to prevent. Trading nothing beats trading that."""
        cfg = dataclasses.replace(DEFAULT_CONFIG, venue='cfbenchmarks')

        def boom(*a, **k):
            raise RuntimeError('store unreachable')

        monkeypatch.setattr(live_bars, '_index_overlay_sync', boom)
        got = await live_bars._overlay_index(cfg, {'BTC-USD': _spot()}, 1500)
        assert got == {}, 'must refuse, not fall back to spot prices'

    def test_the_activity_columns_are_the_ones_the_index_lacks(self):
        assert ACTIVITY_COLUMNS == ('volume', 'quote_volume', 'trade_count')
