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
