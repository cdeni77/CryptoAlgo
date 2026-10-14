"""Tests for `--max-stake-dollars`, which overrides the cap the artifact would supply."""

from __future__ import annotations

import pytest

import scripts.live as live


def _config(argv):
    return live.config_from_args(live.build_parser().parse_args(argv))


def test_the_flag_sets_the_cap():
    assert _config(['--max-stake-dollars', '40']).max_stake_dollars == 40.0


def test_omitting_it_leaves_the_default():
    assert _config([]).max_stake_dollars == 25.0


def test_it_is_recorded_as_an_explicit_override():
    cfg = _config(['--max-stake-dollars', '40'])
    assert 'max_stake_dollars' in (cfg.cli_overrides or ())


def test_adoption_does_not_overrule_the_flag():
    class _Model:
        config_provenance = {'max_stake_dollars': 25.0, 'kelly_fraction': 0.05}
        init_score_source = 'baseline'

    cfg = _config(['--max-stake-dollars', '40'])
    adopted = live.config_for_artifact(cfg, _Model(), mode='live')
    assert adopted.max_stake_dollars == 40.0, (
        'the operator set it deliberately; adoption must yield'
    )


def test_adoption_still_wins_when_the_flag_is_absent():
    """Without the flag, the artifact's cap is adopted."""
    class _Model:
        config_provenance = {'max_stake_dollars': 12.0}
        init_score_source = 'baseline'

    adopted = live.config_for_artifact(_config([]), _Model(), mode='live')
    assert adopted.max_stake_dollars == 12.0
