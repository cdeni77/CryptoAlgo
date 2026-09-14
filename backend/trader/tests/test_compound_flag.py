"""`compound` was a Config field with no way to reach it from the command line.

Every backtest ever run was additive — all three ledger entries record
`compound` absent, which is the default False. That is the right DEFAULT: sizing
off the starting bankroll keeps the equity curve additive, so its slope IS the
per-trade edge, while compounding makes it an exponential of the ESTIMATE of
that edge. An earlier run of this repo compounded $100 into $2e17 and reported
it as a return.

But "never measured" and "measured and rejected" are different claims, and only
the second is worth having. The flag makes the alternative runnable without
changing what runs by default.
"""
from __future__ import annotations

import argparse

from scripts._common import add_data_arguments, config_from_args


def _config(argv):
    p = argparse.ArgumentParser()
    add_data_arguments(p)
    return config_from_args(p.parse_args(argv))


def test_compounding_is_off_unless_asked_for():
    """The default decides what every past ledger entry means. Changing it
    silently would rewrite the interpretation of all of them."""
    assert _config([]).compound is False


def test_compounding_can_be_turned_on():
    assert _config(['--compound']).compound is True


def test_it_reaches_the_sizing_base():
    """`decide()` reads `bankroll if config.compound else starting_bankroll`,
    so the flag has to survive into the Config the backtest actually uses."""
    from core.decide import kelly_fraction_for  # noqa: F401  (import guard)
    config = _config(['--compound', '--bankroll', '500'])
    assert config.compound is True
    assert config.starting_bankroll == 500.0


# --- the flag must reach the CONFIG, not just the parser -------------------
#
# `--compound` was declared at the parser on 2026-09-02 and read nowhere else:
# `config_from_args` never copied it, so `config.compound` stayed False and
# `decide()` sized off `starting_bankroll` while the running balance was
# ignored. Measured 2026-09-13 on a $550 account, a trade that should have been
# 9 contracts was 3, because the base in force was the fixed $200.
#
# `6b288fc9` added `compound` to the adoption loop in `config_for_artifact`,
# which skips any field present in `cli_overrides`. Nothing ever put it there,
# so the flag could never win and the artifact's provenance always did. Half
# the fix was built and the half that mattered was not.
#
# Same shape as `--dry-run`, "declared and never read, so `--mode live
# --dry-run --place-orders` parsed cleanly and placed real orders". Testing the
# parser cannot catch it; only the seam can.

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
    """The documented default: sizing additive, so the equity curve's slope IS
    the per-trade edge rather than an exponential of it."""
    from scripts.live import build_parser, config_from_args
    args = build_parser().parse_args(['--mode', 'live', '--place-orders'])
    config = config_from_args(args)
    assert config.compound is False
    assert 'compound' not in config.cli_overrides, (
        'an unset flag must not claim to be an explicit override, or it would '
        "block the artifact's provenance from applying")


def test_the_sizing_base_actually_follows_the_flag():
    """The consequence the flag exists for.

    `decide()`: sizing_base = bankroll if config.compound else starting_bankroll.
    """
    from scripts.live import build_parser, config_from_args
    running = 550.60
    for argv, expected in (
            (['--mode', 'live', '--place-orders', '--compound', '--bankroll', '500'],
             running),
            (['--mode', 'live', '--place-orders', '--bankroll', '500'], 500.0)):
        config = config_from_args(build_parser().parse_args(argv))
        base = running if config.compound else config.starting_bankroll
        assert base == expected, (argv, base, expected)
