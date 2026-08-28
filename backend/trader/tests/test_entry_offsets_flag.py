"""The backtest measured a policy the live loop does not run.

`Config.entry_offsets` restricts which decision offsets may OPEN a position,
and `core/decide.py` honours it — but only `scripts/live.py` had a flag for it,
defaulting to 12. The evaluate/promote side had none, so `entry_offsets` stayed
None, every offset was tradeable, and the backtest ran the first-clear policy.

Those are not close. Measured over 70 days of quotes, per contract after fees
and a 0.5c half-spread:

    first_clear (what the gates measured)   0.040c   t=0.10
    wait_12     (what live trades)          3.304c   t=5.98

So `realised_edge_pp`, `total_return`, `sharpe` and `max_drawdown` described a
strategy nobody deployed — and the one they described is the weakest of the
three.
"""
from __future__ import annotations

import argparse

import pytest

from scripts._common import add_data_arguments, config_from_args


def _config(argv):
    parser = argparse.ArgumentParser()
    add_data_arguments(parser)
    return config_from_args(parser.parse_args(argv))


def test_the_backtest_can_restrict_entries_the_way_live_does():
    config = _config(['--entry-offsets', '12'])
    assert config.entry_offsets == (12,)


def test_several_offsets_are_allowed():
    config = _config(['--entry-offsets', '9', '12'])
    assert config.entry_offsets == (9, 12)


def test_the_default_is_unrestricted_so_existing_runs_are_unchanged():
    """None means every decision offset may enter — the first-clear policy the
    gates have always measured. Changing that default silently would rewrite
    the meaning of every past ledger entry."""
    assert _config([]).entry_offsets is None


def test_an_entry_offset_outside_the_decision_grid_is_refused():
    """Entering at an offset that is never scored is not a policy, it is a
    typo that would silently abstain forever."""
    with pytest.raises(ValueError, match='entry_offsets'):
        _config(['--offsets', '3,6,9,12', '--entry-offsets', '7'])
