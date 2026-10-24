"""`scripts.live` must accept the price band, and follow an artifact's.

A band set on the retrain and not on the trader grades a policy the account
does not run -- the drift this repo keeps getting caught by. On 2026-10-24 the
flag was added to the retrain alone and the trader crash-looped on
`unrecognized arguments`, because the band reached `Config` only as a default.
"""

from __future__ import annotations

import pytest

from core.config import Config
from livetrade.artifact import config_for_artifact
from scripts import live as live_mod


def _args(argv):
    return live_mod.build_parser().parse_args(argv)


def test_the_parser_accepts_the_band():
    args = _args(['--min-traded-price', '0.10', '--max-traded-price', '0.90'])
    assert args.min_traded_price == pytest.approx(0.10)
    assert args.max_traded_price == pytest.approx(0.90)


def test_an_asymmetric_band_is_refused_by_config():
    """Config owns the rule; the flags must not smuggle past it."""
    with pytest.raises(ValueError, match='symmetric'):
        Config(min_traded_price=0.10, max_traded_price=0.80)


def test_a_symmetric_band_is_accepted():
    config = Config(min_traded_price=0.10, max_traded_price=0.90)
    assert config.min_traded_price == pytest.approx(0.10)
    assert config.max_traded_price == pytest.approx(0.90)


class _Model:
    groups = ()
    config_provenance = {'min_traded_price': 0.10, 'max_traded_price': 0.90}


def test_the_trader_follows_the_artifact_band():
    """Without adoption the trader keeps the Config default while the model was
    graded on something narrower, and nothing says so."""
    out = config_for_artifact(Config(), _Model(), mode='live')
    assert out.min_traded_price == pytest.approx(0.10)
    assert out.max_traded_price == pytest.approx(0.90)


def test_an_explicit_flag_beats_the_artifact():
    base = Config(min_traded_price=0.05, max_traded_price=0.95,
                  cli_overrides=('min_traded_price', 'max_traded_price'))
    out = config_for_artifact(base, _Model(), mode='live')
    assert out.min_traded_price == pytest.approx(0.05)
