"""The ledger could not say which policy produced its money numbers.

`Config.provenance()` recorded `kelly_fraction` and `min_edge_pp` but not
`entry_offsets`, `compound`, `starting_bankroll` or `max_stake_dollars`. Two
consequences, both silent:

  * The ledger is the record of account and the trial count. Promotion
    20260828T202438Z exists specifically to measure `--entry-offsets 12` —
    it moved realised edge from +2.06 to +4.17pp and drawdown from a failing
    0.402 to a passing 0.232 — and the entry does not record which policy that
    was. A later reader cannot reproduce it.
  * `scripts/live.config_for_artifact` adopts the economics the artifact was
    measured under by reading provenance. It listed `max_stake_dollars`,
    `max_stake_fraction` and `compound`, none of which were ever present, so
    those three were dead code that silently adopted nothing.

`compound` matters most of the four. Sizing off the starting bankroll keeps the
equity curve additive, so its slope IS the per-trade edge; compounding makes it
an exponential of the ESTIMATE of that edge. An artifact measured one way and
traded the other is not the same strategy, and nothing recorded which was which.
"""
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
    """None means "every decision offset may enter" — the first-clear policy.
    Absent and None must not look the same to a reader."""
    p = Config(entry_offsets=None).provenance()
    assert 'entry_offsets' in p and p['entry_offsets'] is None
