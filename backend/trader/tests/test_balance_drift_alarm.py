"""Tests that the balance-drift alarm ignores payouts the venue has not posted yet.

The venue balance lags our settlement credit by one cycle. That lag is not drift; a real
drift must still warn.
"""

from __future__ import annotations

import logging

import pytest

from scripts.live import adopt_venue_balance


class _Account:
    def __init__(self, bankroll):
        self.bankroll = bankroll


class _Writer:
    """Fake writer with only what `adopt_venue_balance` uses. `account` is a method, as
    on PgWriter.
    """

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
    # Nothing settled, and the venue disagrees by $16. That must warn.
    adopt_venue_balance(_Writer(527.98), 511.98, credited_this_cycle=0.0)
    assert any(lvl >= logging.WARNING for lvl, _ in _levels(caplog)), (
        'an unrecorded fill, a partial or a mispriced fee must still fire')


def test_a_drift_larger_than_the_credit_still_warns(caplog):
    """A real discrepancy larger than the pending credit still warns."""
    caplog.set_level(logging.INFO, logger='live')
    adopt_venue_balance(_Writer(527.98), 501.98, credited_this_cycle=16.0)
    assert any(lvl >= logging.WARNING for lvl, _ in _levels(caplog))


def test_agreement_says_nothing(caplog):
    caplog.set_level(logging.INFO, logger='live')
    adopt_venue_balance(_Writer(500.0), 500.0, credited_this_cycle=0.0)
    assert not any(lvl >= logging.WARNING for lvl, _ in _levels(caplog))
