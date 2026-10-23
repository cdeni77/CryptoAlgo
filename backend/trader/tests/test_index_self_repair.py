"""The index overlay must repair its own history, not sit out the lookback.

The index reaches the store two ways: the live process folds minutes in memory
and flushes them, and `backfill_index` fetches them from CF Benchmarks. A
restart loses the first and the store keeps only what was flushed, so minutes
between the last flush and the restart exist in neither. The overlay then
refuses the whole universe -- correctly, because pricing half of it on spot is
the cross-index basis C6 exists to prevent -- and a hole does not leave a
1,500 minute lookback for twenty-five hours.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from core.config import Config
from livetrade import bars as bars_mod


def _frame(n: int) -> pd.DataFrame:
    end = pd.Timestamp('2026-10-23 01:00', tz='UTC')
    return pd.DataFrame({
        'event_time': pd.date_range(end - pd.Timedelta(minutes=n - 1), end,
                                    freq='min'),
        'close': [100.0] * n,
        'volume': [1.0] * n,
    })


@pytest.fixture(autouse=True)
def _reset():
    bars_mod._INDEX_CACHE.clear()
    bars_mod._INDEX_REPAIR_AT = 0.0
    yield
    bars_mod._INDEX_CACHE.clear()
    bars_mod._INDEX_REPAIR_AT = 0.0


def test_a_short_overlay_triggers_a_repair_and_a_re_read(monkeypatch):
    config = Config(venue='cfbenchmarks')
    spot = {'BTC-USD': _frame(1500)}
    calls = {'overlay': 0, 'repair': 0}

    def fake_overlay(cfg, s, minutes):
        calls['overlay'] += 1
        # Short first, whole after the repair.
        return {'BTC-USD': _frame(1438 if calls['repair'] == 0 else 1500)}

    async def fake_repair(cfg, minutes):
        calls['repair'] += 1
        return 62

    monkeypatch.setattr(bars_mod, '_index_overlay_sync', fake_overlay)
    monkeypatch.setattr(bars_mod, '_repair_index_history', fake_repair)

    out = asyncio.run(bars_mod._overlay_index(config, spot, 1500))
    assert calls['repair'] == 1, 'a short overlay must ask for a repair'
    assert calls['overlay'] == 2, 'the overlay must be re-read after the repair'
    usable = int(pd.to_numeric(out['BTC-USD']['close'], errors='coerce').notna().sum())
    assert usable == 1500


def test_a_whole_overlay_never_calls_the_venue(monkeypatch):
    config = Config(venue='cfbenchmarks')
    calls = {'repair': 0}

    async def fake_repair(cfg, minutes):
        calls['repair'] += 1
        return 0

    monkeypatch.setattr(bars_mod, '_index_overlay_sync',
                        lambda c, s, m: {'BTC-USD': _frame(1500)})
    monkeypatch.setattr(bars_mod, '_repair_index_history', fake_repair)
    asyncio.run(bars_mod._overlay_index(config, {'BTC-USD': _frame(1500)}, 1500))
    assert calls['repair'] == 0


def test_the_repair_is_rate_limited(monkeypatch):
    """A hole the venue genuinely lacks must not be re-asked every cycle."""
    config = Config(venue='cfbenchmarks')
    calls = {'repair': 0}

    async def fake_repair(cfg, minutes):
        calls['repair'] += 1
        return 0

    monkeypatch.setattr(bars_mod, '_index_overlay_sync',
                        lambda c, s, m: {'BTC-USD': _frame(1438)})
    monkeypatch.setattr(bars_mod, '_repair_index_history', fake_repair)
    for _ in range(4):
        asyncio.run(bars_mod._overlay_index(config, {'BTC-USD': _frame(1500)}, 1500))
    assert calls['repair'] == 1, 'the cooldown must hold off the later cycles'


def test_a_failed_repair_does_not_take_the_cycle_down(monkeypatch):
    config = Config(venue='cfbenchmarks')

    async def boom(cfg, minutes):
        raise RuntimeError('venue unreachable')

    monkeypatch.setattr(bars_mod, '_index_overlay_sync',
                        lambda c, s, m: {'BTC-USD': _frame(1438)})
    monkeypatch.setattr(bars_mod, '_repair_index_history', boom)
    out = asyncio.run(bars_mod._overlay_index(config, {'BTC-USD': _frame(1500)}, 1500))
    # Still the short frame: short_symbols refuses it, which is the safe end.
    assert out and len(out['BTC-USD']) == 1438


def test_the_default_venue_is_untouched(monkeypatch):
    """Nothing changes for a run that is not priced on the index."""
    spot = {'BTC-USD': _frame(10)}
    monkeypatch.setattr(bars_mod, '_index_overlay_sync',
                        lambda c, s, m: pytest.fail('overlay ran off-index'))
    out = asyncio.run(bars_mod._overlay_index(Config(), spot, 1500))
    assert out is spot


def test_the_repair_is_rate_limited_against_the_token_budget():
    """`/cfbenchmarks` costs 50 tokens of a 300/s refill.

    The backfill script's own default is 5 requests a second, which spends
    250 -- 83% of the budget -- and the 429s then land on the trading path.
    A repair that throttles the loop it exists to repair is not a repair.
    """
    import inspect
    source = inspect.getsource(bars_mod._repair_index_history)
    assert "'--rate'" in source, 'the repair must set its own rate'
    assert bars_mod.INDEX_REPAIR_RATE * 50 <= 100, (
        'the repair must stay inside a sixth of the 300 token/s refill')
