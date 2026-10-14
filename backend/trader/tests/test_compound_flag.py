"""Tests for the `--compound` flag.

Compounding is off by default, so the equity curve stays additive and its slope is the
per-trade edge.
"""
from __future__ import annotations

import argparse

from scripts._common import add_data_arguments, config_from_args


def _config(argv):
    p = argparse.ArgumentParser()
    add_data_arguments(p)
    return config_from_args(p.parse_args(argv))


def test_compounding_is_off_unless_asked_for():
    assert _config([]).compound is False


def test_compounding_can_be_turned_on():
    assert _config(['--compound']).compound is True


def test_it_reaches_the_sizing_base():
    """The flag reaches the Config the backtest uses, which `decide()` reads for the
    sizing base.
    """
    from core.decide import kelly_fraction_for  # noqa: F401  (import guard)
    config = _config(['--compound', '--bankroll', '500'])
    assert config.compound is True
    assert config.starting_bankroll == 500.0


# --- the flag must reach the Config, not just the parser -------------------
#
# The flag must also be listed in `cli_overrides`, or `config_for_artifact`
# adopts the artifact's provenance over it.

def test_the_compound_flag_reaches_the_config():
    from scripts.live import build_parser, config_from_args
    args = build_parser().parse_args(
        ['--mode', 'live', '--place-orders', '--compound', '--bankroll', '500'])
    assert args.compound is True, 'parser'
    config = config_from_args(args)
    assert config.compound is True, 'the flag must reach the Config, not just args'
    assert 'compound' in config.cli_overrides, (
        'it must be marked explicit, or config_for_artifact adopts the '
        "artifact's provenance over it")


def test_without_the_flag_compounding_stays_off():
    """Without the flag, compounding stays off and is not marked as an override."""
    from scripts.live import build_parser, config_from_args
    args = build_parser().parse_args(['--mode', 'live', '--place-orders'])
    config = config_from_args(args)
    assert config.compound is False
    assert 'compound' not in config.cli_overrides, (
        'an unset flag must not claim to be an explicit override, or it would '
        "block the artifact's provenance from applying")


def test_the_sizing_base_actually_follows_the_flag():
    """The sizing base is the running bankroll only when `--compound` is set."""
    from scripts.live import build_parser, config_from_args
    running = 550.60
    for argv, expected in (
            (['--mode', 'live', '--place-orders', '--compound', '--bankroll', '500'],
             running),
            (['--mode', 'live', '--place-orders', '--bankroll', '500'], 500.0)):
        config = config_from_args(build_parser().parse_args(argv))
        base = running if config.compound else config.starting_bankroll
        assert base == expected, (argv, base, expected)
