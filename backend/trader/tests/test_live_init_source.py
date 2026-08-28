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
