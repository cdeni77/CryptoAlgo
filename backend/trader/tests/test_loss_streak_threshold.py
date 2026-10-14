"""Tests for the `max_consecutive_losses` threshold (25).

A streak of 25 at a 25% win rate is about as rare as 12 on even-money bets.
"""
from __future__ import annotations

from core.config import Config


def _p_streak(win_rate: float, n: int) -> float:
    return (1.0 - win_rate) ** n


def test_the_threshold_is_not_tuned_for_even_money_bets():
    """At a 25% win rate a 12-loss streak is over 3% likely, so the threshold is above 12."""
    assert _p_streak(0.25, 12) > 0.03
    assert Config().max_consecutive_losses > 12


def test_it_restores_the_alarm_rate_that_12_gives_at_even_odds():
    """The threshold at a 25% win rate is about as rare as 12 at even odds."""
    reference = _p_streak(0.50, 12)
    actual = _p_streak(0.25, Config().max_consecutive_losses)
    assert actual < reference * 5, (
        f'{Config().max_consecutive_losses} losses at a 25% win rate is '
        f'{actual:.4%}, against {reference:.4%} for the even-money case')


def test_the_money_breakers_are_untouched():
    """The money breakers (daily loss, ruin floor) are unchanged."""
    config = Config()
    assert config.max_daily_loss_fraction == 0.15
    assert config.ruin_floor_fraction == 0.50
