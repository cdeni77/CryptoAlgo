"""Tests for `--entry-offsets` on the evaluate/promote side.

Without it the backtest trades the first offset that clears, while live enters only at
+12m. The gates would then measure a policy that is not deployed.
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
    """The default is None (every offset may enter), so past ledger entries keep their meaning."""
    assert _config([]).entry_offsets is None


def test_an_entry_offset_outside_the_decision_grid_is_refused():
    with pytest.raises(ValueError, match='entry_offsets'):
        _config(['--offsets', '3,6,9,12', '--entry-offsets', '7'])
