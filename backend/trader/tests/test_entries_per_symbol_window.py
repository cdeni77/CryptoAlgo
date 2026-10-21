"""Entries per (symbol, window) became a policy on 2026-10-20.

It was a hard-coded invariant: the four decision offsets are one bet observed
four times, so letting each fire puts up to 4x the intended size on a single
15-minute move. It is now `max_entries_per_symbol_window`, defaulting to 1.

What these pin: the default is byte-for-byte the old behaviour, the exposure
arithmetic counts every entry (not every symbol), and the window caps still
bite when the per-symbol gate is relaxed.
"""
from __future__ import annotations

from dataclasses import replace

from core.config import DEFAULT_CONFIG
from core.decide import Decision, Reason, WindowExposure


import pandas as pd

WINDOW = pd.Timestamp('2026-10-20 12:00', tz='UTC')


def _traded(symbol: str, stake: float = 1.0) -> Decision:
    """`traded` is derived — `reason is TRADED and contracts > 0`."""
    return Decision(symbol=symbol, window_open=WINDOW,
                    settle_time=WINDOW + pd.Timedelta(minutes=15), offset=3,
                    reason=Reason.TRADED, contracts=1, stake=stake, price=0.7)


def _refused(symbol: str) -> Decision:
    return Decision(symbol=symbol, window_open=WINDOW,
                    settle_time=WINDOW + pd.Timedelta(minutes=15), offset=3,
                    reason=Reason.EDGE_BELOW_GATE)


def test_the_default_is_still_one_entry_per_symbol_window():
    """The invariant must survive becoming a field."""
    assert DEFAULT_CONFIG.max_entries_per_symbol_window == 1


def test_a_legacy_exposure_without_counts_still_reports_one():
    """`symbols_entered` is how every caller built this before `entry_counts`
    existed. Reporting 0 for a held symbol would re-open a window that is
    already entered — the exact double-entry the gate exists to stop."""
    exposure = WindowExposure(symbols_entered=frozenset({'BTC-USD'}))
    assert exposure.entries_for('BTC-USD') == 1
    assert exposure.entries_for('ETH-USD') == 0


def test_with_increments_the_count_rather_than_only_the_set():
    exposure = WindowExposure()
    exposure = exposure.with_(_traded('BTC-USD'))
    assert exposure.entries_for('BTC-USD') == 1
    exposure = exposure.with_(_traded('BTC-USD'))
    assert exposure.entries_for('BTC-USD') == 2
    # The set still carries the symbol once; the count is what moved.
    assert exposure.symbols_entered == frozenset({'BTC-USD'})
    assert exposure.positions == 2


def test_a_refused_decision_changes_nothing():
    exposure = WindowExposure().with_(_traded('BTC-USD'))
    assert exposure.with_(_refused('BTC-USD')) is exposure
    assert exposure.entries_for('BTC-USD') == 1


def test_stake_accumulates_across_repeat_entries_on_one_symbol():
    """`max_window_exposure_fraction` caps notional, so a second entry on a
    symbol already held must add its stake. Deduplicating by symbol here would
    silently raise the real exposure limit."""
    exposure = WindowExposure()
    exposure = exposure.with_(_traded('BTC-USD', stake=2.0))
    exposure = exposure.with_(_traded('BTC-USD', stake=3.0))
    assert exposure.stake == 5.0
    assert exposure.positions == 2


def test_counts_are_per_symbol_not_shared():
    exposure = WindowExposure()
    exposure = exposure.with_(_traded('BTC-USD'))
    exposure = exposure.with_(_traded('BTC-USD'))
    exposure = exposure.with_(_traded('ETH-USD'))
    assert exposure.entries_for('BTC-USD') == 2
    assert exposure.entries_for('ETH-USD') == 1
    assert exposure.entries_for('SOL-USD') == 0


def test_raising_the_limit_alone_cannot_exceed_the_window_position_cap():
    """The trap worth pinning: `max_positions_per_window` (3) is saturated by
    one entry on each of three symbols. Raising the per-symbol limit without
    raising it too reallocates slots rather than adding trades."""
    config = replace(DEFAULT_CONFIG, max_entries_per_symbol_window=4)
    assert config.max_positions_per_window == 3
    exposure = WindowExposure()
    for _ in range(3):
        exposure = exposure.with_(_traded('BTC-USD'))
    # Four are permitted per symbol, but the window is already full.
    assert exposure.entries_for('BTC-USD') < config.max_entries_per_symbol_window
    assert exposure.positions >= config.max_positions_per_window


# --- the same-side constraint ------------------------------------------------
#
# Multi-entry went live 2026-10-20 10:42 and was reverted 2h later: different
# offsets took OPPOSITE sides of the same market. Three settled with
# revenue $0.00 and -$6.29 -- the entire stake on both legs -- because buying
# the other side nets against the open position rather than hedging it.
# `already_entered` had been preventing this as a side effect.

def _traded_side(symbol: str, side, stake: float = 1.0) -> Decision:
    return Decision(symbol=symbol, window_open=WINDOW,
                    settle_time=WINDOW + pd.Timedelta(minutes=15), offset=3,
                    reason=Reason.TRADED, contracts=1, stake=stake,
                    price=0.7, side=side)


def test_the_first_side_is_recorded():
    from core.decide import Side
    exposure = WindowExposure().with_(_traded_side('BTC-USD', Side.UP))
    assert exposure.side_for('BTC-USD') == 'up'
    assert exposure.side_for('ETH-USD') is None


def test_the_first_side_is_never_overwritten():
    """A later entry must be refused AGAINST the side actually held, so the
    recorded side has to be the first one, not the most recent."""
    from core.decide import Side
    exposure = WindowExposure().with_(_traded_side('BTC-USD', Side.UP))
    exposure = exposure.with_(_traded_side('BTC-USD', Side.UP))
    assert exposure.side_for('BTC-USD') == 'up'


def test_sides_are_tracked_per_symbol():
    from core.decide import Side
    exposure = WindowExposure()
    exposure = exposure.with_(_traded_side('BTC-USD', Side.UP))
    exposure = exposure.with_(_traded_side('ETH-USD', Side.DOWN))
    assert exposure.side_for('BTC-USD') == 'up'
    assert exposure.side_for('ETH-USD') == 'down'


def test_an_exposure_with_no_side_blocks_nothing():
    """Rows that predate the field, or a position stored without a side, must
    not refuse every later entry — that would silently restore first-clear."""
    exposure = WindowExposure(entry_counts={'BTC-USD': 1},
                              symbols_entered=frozenset({'BTC-USD'}))
    assert exposure.side_for('BTC-USD') is None


def test_opposite_side_has_its_own_reason():
    """Distinct from ALREADY_ENTERED: one means 'no entries left', the other
    means 'the model changed its mind'. Conflating them hides the defect."""
    assert Reason.OPPOSITE_SIDE.value == 'opposite_side'
    assert Reason.OPPOSITE_SIDE is not Reason.ALREADY_ENTERED
