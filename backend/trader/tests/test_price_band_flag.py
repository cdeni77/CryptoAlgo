"""The traded-price band is a Config field with no CLI flag.

CLAUDE.md is explicit that the backfilled book is quantised to whole cents
outside [0.10, 0.90] — ~0.24c mean rounding error, about half the measured
half-spread — and that economics from backfilled quotes should restrict to that
band or carry the uncertainty. Neither was possible from the command line, so
every cost-stress run silently used [0.05, 0.95] and no measurement of the
quantisation-free band was ever taken.
"""
from scripts._common import add_data_arguments, config_from_args
import argparse


def _config(argv):
    p = argparse.ArgumentParser()
    add_data_arguments(p)
    return config_from_args(p.parse_args(argv))


def test_the_traded_price_band_is_settable_from_the_command_line():
    config = _config(['--min-traded-price', '0.10', '--max-traded-price', '0.90'])
    assert config.min_traded_price == 0.10
    assert config.max_traded_price == 0.90


def test_the_band_defaults_are_left_alone_when_the_flags_are_absent():
    config = _config([])
    assert config.min_traded_price == 0.05
    assert config.max_traded_price == 0.95
