"""The balance-drift alarm must not fire on the payouts we just booked.

`venue_balance` is sampled at the TOP of the cycle by `reconcile_with_venue`.
`settle_due` then credits our side, and the venue posts the same settlement a
cycle later — so every winning window produced a matched pair of warnings:

    15:15:07  ours $527.98, venue $511.98  (-16.00)   <- we credited
    15:16:09  ours $511.98, venue $527.98  (+16.00)   <- venue caught up

against ETH +$8 and SOL +$8 settling at 19:15 UTC. `adopt_venue_balance`'s own
docstring records this cry-wolf pattern being fixed once before in the other
direction; moving the call site corrected the sign and not the sample.

The cost is not money — entries fire at +12m, three minutes from any settlement
— it is the alarm's credibility. This is the designated detector for an
unrecorded fill, a partial, or a mispriced fee, and a real one was hiding inside
a stream of benign pairs.
"""

from __future__ import annotations

import logging

import pytest

from scripts.live import adopt_venue_balance


class _Account:
    def __init__(self, bankroll):
        self.bankroll = bankroll


class _Writer:
    """Only what `adopt_venue_balance` actually touches. `account` is a METHOD
    on the real PgWriter, not an attribute — getting that wrong made every test
    here fail with "'_Account' object is not callable"."""

    def __init__(self, bankroll):
        self._account = _Account(bankroll)
        self.adopted = None

    def account(self, *a, **k):
        return self._account

    def set_bankroll(self, value, **k):
        self.adopted = value
        self._account.bankroll = value
        return self._account

    def __getattr__(self, name):
        def _noop(*a, **k):
            return self._account
        return _noop


def _levels(caplog):
    return [(r.levelno, r.getMessage()) for r in caplog.records]


def test_a_drift_equal_to_what_we_just_credited_is_not_a_warning(caplog):
    caplog.set_level(logging.INFO, logger='live')
    # We credited $16; the venue has not posted it yet.
    adopt_venue_balance(_Writer(527.98), 511.98, credited_this_cycle=16.0)
    msgs = _levels(caplog)
    assert msgs, 'it must still say something — silence hides the adoption'
    assert not any(lvl >= logging.WARNING for lvl, _ in msgs), (
        'a payout the venue has not posted yet is not a drift')


def test_an_unexplained_drift_still_warns(caplog):
    caplog.set_level(logging.INFO, logger='live')
    # Nothing settled, and the venue disagrees by $16. THAT is the alarm.
    adopt_venue_balance(_Writer(527.98), 511.98, credited_this_cycle=0.0)
    assert any(lvl >= logging.WARNING for lvl, _ in _levels(caplog)), (
        'an unrecorded fill, a partial or a mispriced fee must still fire')


def test_a_drift_larger_than_the_credit_still_warns(caplog):
    """The dangerous case: a real discrepancy hiding behind a settlement."""
    caplog.set_level(logging.INFO, logger='live')
    adopt_venue_balance(_Writer(527.98), 501.98, credited_this_cycle=16.0)
    assert any(lvl >= logging.WARNING for lvl, _ in _levels(caplog))


def test_agreement_says_nothing(caplog):
    caplog.set_level(logging.INFO, logger='live')
    adopt_venue_balance(_Writer(500.0), 500.0, credited_this_cycle=0.0)
    assert not any(lvl >= logging.WARNING for lvl, _ in _levels(caplog))
