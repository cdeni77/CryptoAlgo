"""Three changes aimed at actually getting filled, measured from the book.

From 237 windows of real order book at +12m (Predexon tick history):

    ask size at touch      median 241 contracts, p25 27, p10 6
    size retained 45s on   median 1.00x, p25 0.55x, 35% shrink

So depth is usually ample — median 241 against orders of ~9 — and the failures
are concentrated in a thin tail where the touch holds single digits and can halve
within seconds. Meanwhile the quote the stake is sized against is read ~4s before
the order is sent.

Observed live, one window, three separate faults in eight seconds:

    fill_or_kill_insufficient_resting_volume   9 contracts wanted, not enough there
    order_already_exists                       the retry colliding with our own order
    order_already_exists                       and again
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
    """Sizing to 100% of a four-second-old quote is sizing to a number that has
    already moved. p25 retention is 0.55x, so half of what is visible is the
    portion that reliably survives."""

    def test_the_default_leaves_headroom(self):
        assert 0.0 < Config().depth_fraction <= 0.75

    def test_the_stake_is_capped_below_the_visible_depth(self):
        config = Config(entry_offsets=(12,))
        # Set both sides: which one `decide()` takes is its business, and the
        # cap must hold either way.
        shallow = decide(row(depth_up=2.00, depth_down=2.00), config,
                         bankroll=100.0)
        assert shallow.traded, shallow.reason
        assert shallow.stake <= 2.00 * config.depth_fraction + 0.02, shallow.stake

    def test_ample_depth_does_not_bind(self):
        """On the median trade — 241 resting, ~9 wanted — the cap must cost
        nothing, or it would be paid on every trade to save the tail."""
        config = Config(entry_offsets=(12,))
        deep = decide(row(depth_up=500.0, depth_down=500.0), config,
                      bankroll=100.0)
        uncapped = decide(row(), config, bankroll=100.0)
        assert deep.contracts == uncapped.contracts

    def test_a_row_with_no_measured_depth_is_unaffected(self):
        """A backtest row carries no book, and must size exactly as before."""
        config = Config(entry_offsets=(12,))
        assert decide(row(), config, bankroll=100.0).contracts > 0


class TestPartialFillsAllowed:
    """`fill_or_kill` is all-or-nothing: 9 contracts wanted against 5 resting
    returns nothing. `immediate_or_cancel` takes the 5. With a positive edge a
    partial fill strictly beats a kill, and 16 of 323 live fills were already
    partial, so the accounting handles them."""

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


