"""Tests for sizing and order type changes that improve the fill rate.

Depth at the touch is usually ample, but a thin tail can shrink within seconds, and the
quote used for sizing is a few seconds old when the order is sent.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.config import Config
from core.decide import Reason, decide


W = pd.Timestamp('2026-08-26 01:15', tz='UTC')


def row(**over):
    base = dict(symbol='SOL-USD', window_open=W,
                settle_time=W + pd.Timedelta(minutes=15), offset=12,
                baseline_probability=0.55, model_probability=0.62,
                ask_up=0.49, ask_down=0.51)
    base.update(over)
    return base


class TestDepthHeadroom:
    """The stake uses only `depth_fraction` of the visible depth, since part of it is
    gone by send time.
    """

    def test_the_default_leaves_headroom(self):
        assert 0.0 < Config().depth_fraction <= 0.75

    def test_the_stake_is_capped_below_the_visible_depth(self):
        config = Config(entry_offsets=(12,))
        # Both sides are set: the cap must hold whichever side `decide()` takes.
        shallow = decide(row(depth_up=2.00, depth_down=2.00), config,
                         bankroll=100.0)
        assert shallow.traded, shallow.reason
        assert shallow.stake <= 2.00 * config.depth_fraction + 0.02, shallow.stake

    def test_ample_depth_does_not_bind(self):
        """With ample depth the cap does not reduce the size."""
        config = Config(entry_offsets=(12,))
        deep = decide(row(depth_up=500.0, depth_down=500.0), config,
                      bankroll=100.0)
        uncapped = decide(row(), config, bankroll=100.0)
        assert deep.contracts == uncapped.contracts

    def test_a_row_with_no_measured_depth_is_unaffected(self):
        """A row without measured depth sizes as before."""
        config = Config(entry_offsets=(12,))
        assert decide(row(), config, bankroll=100.0).contracts > 0


class TestPartialFillsAllowed:
    """Orders use `immediate_or_cancel`, so a partial fill is taken instead of killed."""

    @pytest.mark.asyncio
    async def test_orders_default_to_immediate_or_cancel(self, monkeypatch):
        from data_collection.kalshi_client import KalshiClient

        client = KalshiClient(key_id='k', private_key_pem=None, live=True)
        sent = {}

        async def _request(method, path, *, body=None, **kw):
            sent.update(body or {})
            return {'order': {'status': 'executed'}}

        client._request = _request
        await client.place_order(ticker='KXSOL15M-X', side='down', contracts=9,
                                 limit_price=0.48)
        assert sent['time_in_force'] == 'immediate_or_cancel'


