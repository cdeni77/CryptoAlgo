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

    def test_the_one_boundary_the_live_grid_always_drops_is_not_news(self):
        """The in-progress window has no settlement yet, so exactly one boundary
        drops on every live cycle, forever. Treating that as an anomaly is how the
        first version of this rule kept the noisiest line at INFO."""
        from core.windows import coverage_log_level

        assert coverage_log_level(report(dropped=1)) == logging.DEBUG

    def test_more_dropped_boundaries_than_the_live_edge_explains_are_news(self):
        """Two or more means windows are being lost for a reason other than the
        clock, and each one is a window that cannot be scored at all."""
        from core.windows import coverage_log_level

        assert coverage_log_level(report(dropped=2)) >= logging.INFO
        assert coverage_log_level(report(dropped=17)) >= logging.INFO


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


class TestHeartbeat:
    """Quiet is the goal; silent is a different failure.

    After the noise was removed the loop emitted nothing at all in steady state,
    which means the log cannot distinguish "running normally" from "hung". The
    container healthcheck only proves the process is alive and can reach the
    database — not that it is still waking on the offsets and deciding.

    One line per 15-minute window is ~96 a day against the ~40,000 it replaced,
    and it is the line an operator actually wants: is it cycling, is it deciding,
    what is the bankroll, how late are the decisions landing.
    """

    def test_a_heartbeat_is_due_when_the_window_rolls(self):
        from scripts.live import heartbeat_due

        assert heartbeat_due(pd.Timestamp('2026-08-25 12:15', tz='UTC'),
                             pd.Timestamp('2026-08-25 12:00', tz='UTC')) is True

    def test_no_heartbeat_inside_the_same_window(self):
        from scripts.live import heartbeat_due

        assert heartbeat_due(pd.Timestamp('2026-08-25 12:00', tz='UTC'),
                             pd.Timestamp('2026-08-25 12:00', tz='UTC')) is False

    def test_the_first_cycle_emits_one(self):
        """With no previous window there is nothing to compare against, and the
        operator most wants a line right after a restart."""
        from scripts.live import heartbeat_due

        assert heartbeat_due(pd.Timestamp('2026-08-25 12:00', tz='UTC'), None) is True

    def test_the_summary_names_what_an_operator_checks(self):
        from scripts.live import heartbeat_summary

        text = heartbeat_summary(
            window_open=pd.Timestamp('2026-08-25 12:00', tz='UTC'),
            cycles=14, decisions=6, traded=2, bankroll=108.83, lag_seconds=6.4)
        for token in ('12:00', '14', '6', '2', '108.83', '6.4'):
            assert token in text, f'{token!r} missing from {text!r}'
