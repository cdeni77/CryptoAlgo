"""Steady-state quiet, money always loud.

The live loop emitted ~14 lines per cycle, twice a minute, around the clock.
Counted over 2,000 lines of real output: ~1,800 were heartbeat — bar counts,
"Coinbase REST client closed", an unchanging settlement count, a coverage report
reading 100.0000% every time, and one `no trade (offset_not_traded)` per symbol
for the two offsets the policy deliberately never enters. The ~20 lines that
mattered — decisions, fills, refusals, balance drift — were buried in it.

A log nobody can read is a log nobody checks, and this one is the only view of an
account that trades unattended. These tests pin the rule rather than the levels:
routine repetition is DEBUG, anything that moves money or departs from the
expected steady state is INFO or louder.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest

from core.decide import Decision, Reason, Side
from core.windows import GridReport

W = pd.Timestamp('2026-08-25 12:00', tz='UTC')


def report(*, minutes_present=1499, minutes_expected=1499, dropped=0):
    return GridReport(
        symbol='BTC-USD', first_minute=W, last_minute=W + pd.Timedelta(minutes=1),
        minutes_expected=minutes_expected, minutes_present=minutes_present,
        windows_total=99, windows_dropped_boundary=dropped,
        windows_with_interior_gaps=0)


def decision(reason: Reason, *, traded=False, contracts=0):
    return Decision(
        symbol='BTC-USD', window_open=W,
        settle_time=W + pd.Timedelta(minutes=15), offset=12,
        reason=reason, side=Side.UP if traded else None,
        price=0.42 if traded else float('nan'),
        effective_cost=0.44 if traded else float('nan'),
        model_probability=0.50, baseline_probability=0.48,
        edge=0.03 if traded else float('nan'), contracts=contracts,
        stake=1.0 if traded else 0.0, price_source='quote')


class TestCoverageReporting:
    """A grid at full coverage says nothing new; a degraded one must shout."""

    def test_a_nominal_grid_is_not_logged_at_info(self):
        from core.windows import coverage_log_level

        assert coverage_log_level(report()) == logging.DEBUG

    def test_missing_minutes_are_logged_at_info(self):
        from core.windows import coverage_log_level

        assert coverage_log_level(
            report(minutes_present=1450)) >= logging.INFO

    def test_dropped_boundaries_are_logged_at_info(self):
        """A dropped boundary is a window that cannot be scored at all."""
        from core.windows import coverage_log_level

        assert coverage_log_level(report(dropped=3)) >= logging.INFO


class TestDecisionReporting:
    """Every decision is recorded to the database regardless; this is only about
    which ones a human reading the log needs to see."""

    def test_a_trade_is_always_logged_at_info(self):
        from scripts.live import decision_log_level

        assert decision_log_level(
            decision(Reason.TRADED, traded=True, contracts=4)) == logging.INFO

    def test_the_designed_steady_state_is_quiet(self):
        """`offset_not_traded` fires on every non-entry offset by design — three
        symbols x two offsets x four windows an hour. It is the single largest
        source of noise and carries no information."""
        from scripts.live import decision_log_level

        assert decision_log_level(
            decision(Reason.OFFSET_NOT_TRADED)) == logging.DEBUG

    @pytest.mark.parametrize('reason', [
        Reason.PRICE_OUT_OF_BAND, Reason.EDGE_BELOW_GATE, Reason.ALREADY_ENTERED,
    ])
    def test_ordinary_refusals_are_quiet(self, reason):
        """Expected outcomes of the gates working. Counted in the cycle summary."""
        from scripts.live import decision_log_level

        assert decision_log_level(decision(reason)) == logging.DEBUG

    @pytest.mark.parametrize('reason', [
        Reason.HALTED, Reason.BANKROLL_FLOOR, Reason.DISAGREEMENT_IMPLAUSIBLE,
    ])
    def test_refusals_that_signal_a_problem_stay_loud(self, reason):
        """A latched breaker, a breached floor, or the model disagreeing with the
        market by an implausible margin are not routine and must not be filtered
        out with the noise."""
        from scripts.live import decision_log_level

        assert decision_log_level(decision(reason)) >= logging.INFO
