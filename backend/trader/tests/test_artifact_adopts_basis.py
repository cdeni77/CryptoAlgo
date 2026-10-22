"""Live follows the price basis the artifact was fitted on.

`scripts.live` has no --venue or --strike-source flag, so without this the
trader cannot follow a migrated model at all: it keeps scoring on
coinbase_spot while the artifact was fitted on cfbenchmarks, and nothing
reports it -- the strike drift warning fires above 25bp and the
Coinbase-vs-BRTI basis is about 1bp.
"""

from __future__ import annotations

import types

from core.config import DEFAULT_CONFIG
from livetrade.artifact import config_for_artifact


def model(**provenance):
    return types.SimpleNamespace(config_provenance=provenance, groups=())


def test_the_basis_is_adopted_from_the_artifact():
    out = config_for_artifact(
        DEFAULT_CONFIG,
        model(venue='cfbenchmarks', strike_source='venue'),
        mode='live')
    assert out.venue == 'cfbenchmarks'
    assert out.strike_source == 'venue'


def test_an_unmigrated_artifact_leaves_the_basis_alone():
    out = config_for_artifact(
        DEFAULT_CONFIG, model(venue='coinbase_spot', strike_source='proxy'),
        mode='live')
    assert out.venue == 'coinbase_spot'
    assert out.strike_source == 'proxy'


def test_a_command_line_setting_still_wins():
    """An explicit operator choice must not be overwritten by the artifact."""
    pinned = DEFAULT_CONFIG.with_overrides(venue='coinbase_spot')
    out = config_for_artifact(
        pinned, model(venue='cfbenchmarks', strike_source='venue'),
        mode='live')
    assert out.venue == 'coinbase_spot', 'cli_overrides must win'
    assert out.strike_source == 'venue', 'but the unpinned field still follows'


def test_a_provenance_without_the_fields_changes_nothing():
    """An older artifact records neither; adopting None would erase config."""
    out = config_for_artifact(DEFAULT_CONFIG, model(), mode='live')
    assert out.venue == DEFAULT_CONFIG.venue
    assert out.strike_source == DEFAULT_CONFIG.strike_source


def test_the_two_move_together_for_the_installed_artifact():
    """C6: the price series and the strike are all-or-nothing."""
    out = config_for_artifact(
        DEFAULT_CONFIG,
        model(venue='cfbenchmarks', strike_source='venue'), mode='live')
    migrated_price = out.venue == 'cfbenchmarks'
    migrated_strike = out.strike_source == 'venue'
    assert migrated_price == migrated_strike, 'a mixed basis is the C6 defect'
