"""`max_consecutive_losses` was calibrated for near-even bets.

Twelve straight losses is a genuine alarm at 50c contracts and routine at 20c,
because a longshot is SUPPOSED to lose most of the time — that is what a 4:1
payoff means:

    P(12 consecutive losses)
      50c contracts, 50% win rate:  0.024%   <- an alarm
      22c contracts, 30% win rate:  1.384%
      22c contracts, 25% win rate:  3.168%   <- routine
      22c contracts, 22% win rate:  5.072%

The live loop halted on 2026-09-03 after twelve losses at a mean price of 22c.
They cost $13.57 against $96.80 realised, and the P&L-based breaker
(`max_daily_loss_fraction`, ~$43) never came close — so the count breaker fired
on the ODDS while the money breaker, which measures the damage, stayed quiet.

The model had drifted from 51c to 19c contracts over five days as its volatility
estimate rose above the market's, which makes favourites look overpriced and
sends it to the cheap side. That is the strategy's documented other half
working, not a fault.

Raised to 25, which restores roughly the alarm rate a 12 gives at even odds.
This is an interim: the threshold should scale with the price actually being
paid, so it means the same thing at any odds, and a fixed number tuned to
today's prices will be wrong again if the drift reverses.
"""
from __future__ import annotations

from core.config import Config


def _p_streak(win_rate: float, n: int) -> float:
    return (1.0 - win_rate) ** n


def test_the_threshold_is_not_tuned_for_even_money_bets():
    """At the ~22c the strategy actually trades, 12 was a 3% event — which over
    ~180 settlements is a coin flip on whether it fires at all."""
    assert _p_streak(0.25, 12) > 0.03
    assert Config().max_consecutive_losses > 12


def test_it_restores_the_alarm_rate_that_12_gives_at_even_odds():
    """A breaker should mean the same thing wherever it is set. 12 at a 50% win
    rate is a 1-in-4000 event; the new threshold should be comparably rare at
    the win rate the traded prices imply."""
    reference = _p_streak(0.50, 12)
    actual = _p_streak(0.25, Config().max_consecutive_losses)
    assert actual < reference * 5, (
        f'{Config().max_consecutive_losses} losses at a 25% win rate is '
        f'{actual:.4%}, against {reference:.4%} for the even-money case')


def test_the_money_breakers_are_untouched():
    """The count breaker catches a stuck model; these bound real damage. Raising
    one must not loosen the others — they were the ones behaving correctly."""
    config = Config()
    assert config.max_daily_loss_fraction == 0.15
    assert config.ruin_floor_fraction == 0.50
