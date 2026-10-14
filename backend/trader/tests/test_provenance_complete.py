"""Tests that `Config.provenance()` records every field that changes the money numbers."""
from __future__ import annotations

import pytest

from core.config import Config

REQUIRED = ('entry_offsets', 'compound', 'starting_bankroll',
            'max_stake_dollars', 'kelly_fraction', 'min_edge_pp')


@pytest.mark.parametrize('field', REQUIRED)
def test_every_field_that_changes_the_money_is_recorded(field):
    assert field in Config().provenance(), (
        f'{field} changes what the strategy DOES but is not in provenance, so '
        f'the ledger cannot say what produced its numbers')


def test_the_recorded_values_are_the_ones_in_force():
    config = Config(kelly_fraction=0.10, min_edge_pp=3.0,
                    entry_offsets=(12,), compound=False)
    p = config.provenance()
    assert p['kelly_fraction'] == 0.10
    assert p['min_edge_pp'] == 3.0
    assert tuple(p['entry_offsets']) == (12,)
    assert p['compound'] is False


def test_an_unrestricted_entry_policy_records_as_null_not_as_absent():
    """An unrestricted entry policy is recorded as None, not left out."""
    p = Config(entry_offsets=None).provenance()
    assert 'entry_offsets' in p and p['entry_offsets'] is None
