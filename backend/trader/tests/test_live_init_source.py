"""Tests for `config_for_artifact`: live adopts settings from the artifact.

Live adopts the artifact's `init_score_source`, since scoring a market-fitted
residual on the baseline logit is always wrong. It also adopts the economics the
artifact was evaluated under, except fields set explicitly on the command line.
"""
from __future__ import annotations

import types

import pytest

from core.config import Config
from scripts.live import config_for_artifact


def _artifact(source):
    return types.SimpleNamespace(init_score_source=source)


def test_live_adopts_the_artifact_init_source_rather_than_its_own_default():
    config = config_for_artifact(Config(), _artifact('market'), mode='live')
    assert config.init_score_source == 'market'


def test_a_baseline_artifact_is_left_alone():
    config = config_for_artifact(Config(), _artifact('baseline'), mode='paper')
    assert config.init_score_source == 'baseline'


def test_a_market_artifact_in_paper_mode_is_refused_not_silently_scored_as_nan():
    """A market-init artifact in paper mode is refused at startup.

    Paper mode has no Kalshi client and so no quotes, so every window would score NaN.
    """
    with pytest.raises(SystemExit) as excinfo:
        config_for_artifact(Config(), _artifact('market'), mode='paper')
    message = str(excinfo.value)
    assert 'paper' in message and 'market' in message


def _artifact_with(**provenance):
    return types.SimpleNamespace(init_score_source='market',
                                 config_provenance=provenance)


def test_live_adopts_the_economics_the_artifact_was_measured_under():
    """Live adopts the economics (`kelly_fraction`, `min_edge_pp`) the artifact was
    evaluated under.

    `verify` does not check these, because they change what the strategy does, not
    what the model predicts.
    """
    config = config_for_artifact(
        Config(), _artifact_with(kelly_fraction=0.10, min_edge_pp=3.0),
        mode='live')
    assert config.kelly_fraction == 0.10
    assert config.min_edge_pp == 3.0


def test_a_field_the_artifact_did_not_record_keeps_the_running_default():
    """A field recorded as None keeps the running value."""
    config = config_for_artifact(
        Config(), _artifact_with(kelly_fraction=0.10, max_stake_dollars=None),
        mode='live')
    assert config.kelly_fraction == 0.10
    assert config.max_stake_dollars == Config().max_stake_dollars


def test_an_artifact_with_no_provenance_changes_nothing():
    config = config_for_artifact(Config(), _artifact_with(), mode='live')
    assert config.kelly_fraction == Config().kelly_fraction


def test_an_explicit_cli_setting_beats_the_artifact_provenance():
    """A field set on the command line is not replaced by the artifact's value."""
    from dataclasses import replace
    config = replace(Config(compound=True, kelly_fraction=0.25),
                     cli_overrides=frozenset({'compound'}))
    out = config_for_artifact(
        config, _artifact_with(compound=False, kelly_fraction=0.10), mode='live')
    assert out.compound is True, 'the explicit flag was overwritten'
    assert out.kelly_fraction == 0.10, 'the un-overridden field still adopts'


def test_without_an_override_the_artifact_still_wins():
    config = Config(compound=True)
    out = config_for_artifact(config, _artifact_with(compound=False), mode='live')
    assert out.compound is False


def _artifact_with_groups(*groups):
    return types.SimpleNamespace(init_score_source='market',
                                 config_provenance={}, groups=tuple(groups))


def test_an_artifact_that_reads_cross_venue_requires_the_peer_book():
    """An artifact that reads `cross_venue` requires the peer book, as training did."""
    out = config_for_artifact(
        Config(), _artifact_with_groups('cross_venue', 'geometry'), mode='live')
    assert out.require_peer_book is True


def test_an_artifact_without_cross_venue_does_not_require_the_peer_book():
    out = config_for_artifact(
        Config(), _artifact_with_groups('geometry', 'vol_state'), mode='live')
    assert out.require_peer_book is False


def test_an_explicit_no_require_peer_book_beats_the_artifact():
    from dataclasses import replace
    config = replace(Config(require_peer_book=False),
                     cli_overrides=frozenset({'require_peer_book'}))
    out = config_for_artifact(
        config, _artifact_with_groups('cross_venue'), mode='live')
    assert out.require_peer_book is False


def test_live_adopts_the_ensemble_gates_the_artifact_was_measured_under():
    config = config_for_artifact(
        Config(), _artifact_with(edge_lower_bound_z=1.0,
                                 require_member_agreement=True),
        mode='live')
    assert config.edge_lower_bound_z == 1.0
    assert config.require_member_agreement is True
