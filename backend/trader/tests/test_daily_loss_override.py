"""The daily-loss breaker can be widened, and only by saying so explicitly.

A flag rather than an edit to the dataclass default, so what the loop is running
with shows up in the deploy config and in `ps`, and reverting is deleting a line
rather than remembering to change a number back.
"""
from __future__ import annotations

from scripts.live import build_parser, config_from_args


def _config(argv):
    return config_from_args(build_parser().parse_args(argv))


def test_the_default_is_unchanged_when_the_flag_is_absent():
    assert _config([]).max_daily_loss_fraction == 0.15


def test_the_flag_widens_the_breaker():
    assert _config(['--max-daily-loss-fraction', '0.5']).max_daily_loss_fraction == 0.5


def test_a_value_at_or_above_one_leaves_only_the_ruin_floor():
    config = _config(['--max-daily-loss-fraction', '1.0'])
    limit = -abs(config.max_daily_loss_fraction) * config.starting_bankroll
    assert limit <= -config.starting_bankroll, (
        'the daily rule can no longer fire before the account is gone')
    assert config.ruin_floor_fraction > 0, 'the ruin floor still guards the stake'


def test_the_value_is_not_silently_clamped():
    """A clamp would run the loop with a limit nobody asked for."""
    assert _config(['--max-daily-loss-fraction', '3.0']).max_daily_loss_fraction == 3.0


def test_other_overrides_still_apply_alongside_it():
    config = _config(['--max-daily-loss-fraction', '0.5', '--bankroll', '250'])
    assert config.max_daily_loss_fraction == 0.5
    assert config.starting_bankroll == 250.0
