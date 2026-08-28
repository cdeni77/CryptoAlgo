"""The artifact is the authority on which forecaster its correction corrects.

`scripts/live.py` has no `--init-score-source` flag, so `config_from_args`
left `Config.init_score_source` at its 'baseline' default and `model.verify`
refused every market-init artifact at load — a crash loop with a message about
a configuration the operator had no way to set.

Live has no legitimate freedom here: scoring a market-fitted residual on the
baseline logit is always the bug the guard describes. So live adopts the
artifact's value. But adopting it silently would trade one failure for a worse
one — a market-init model needs a live quote to score, and paper mode never
opens a client, so every window would score NaN instead of raising.
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
    """Paper mode opens no Kalshi client, so `fetch_quotes` returns {} and the
    market logit is never attached. Refusing at startup names the problem;
    proceeding would produce a NaN prediction every cycle and look like a model
    that abstains."""
    with pytest.raises(SystemExit) as excinfo:
        config_for_artifact(Config(), _artifact('market'), mode='paper')
    message = str(excinfo.value)
    assert 'paper' in message and 'market' in message


def _artifact_with(**provenance):
    return types.SimpleNamespace(init_score_source='market',
                                 config_provenance=provenance)


def test_live_adopts_the_economics_the_artifact_was_measured_under():
    """`--kelly-fraction 0.10 --min-edge-pp 3.0` promoted the artifact, and
    `scripts/live.py` has neither flag — so `config_from_args` left Config at
    0.25 Kelly and a 1.50pp gate and the live loop traded a policy nothing had
    evaluated.

    `verify`'s material fields do not include these, because they change what
    the strategy DOES rather than what the model SAYS. That distinction is real
    and it is exactly why nothing caught this: the model was scored correctly
    and then acted on wrongly.

    CLAUDE.md measures the cost: 0.25 -> 0.10 moved realised edge per contract
    +0.99pp -> +3.32pp and drawdown 58% -> 21%, because a smaller Kelly also
    floors marginal trades under one contract and refuses them.
    """
    config = config_for_artifact(
        Config(), _artifact_with(kelly_fraction=0.10, min_edge_pp=3.0),
        mode='live')
    assert config.kelly_fraction == 0.10
    assert config.min_edge_pp == 3.0


def test_a_field_the_artifact_did_not_record_keeps_the_running_default():
    """Provenance carries None for anything the promoting run left at default.
    Adopting a None would erase a real setting."""
    config = config_for_artifact(
        Config(), _artifact_with(kelly_fraction=0.10, max_stake_dollars=None),
        mode='live')
    assert config.kelly_fraction == 0.10
    assert config.max_stake_dollars == Config().max_stake_dollars


def test_an_artifact_with_no_provenance_changes_nothing():
    config = config_for_artifact(Config(), _artifact_with(), mode='live')
    assert config.kelly_fraction == Config().kelly_fraction
