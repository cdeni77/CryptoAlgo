"""Tests for live log levels: routine repetition is DEBUG, anything about money is INFO
or louder.
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
    """A grid at full coverage logs at DEBUG; a degraded one logs louder."""

    def test_a_nominal_grid_is_not_logged_at_info(self):
        from core.windows import coverage_log_level

        assert coverage_log_level(report()) == logging.DEBUG

    def test_missing_minutes_are_logged_at_info(self):
        from core.windows import coverage_log_level

        assert coverage_log_level(
            report(minutes_present=1450)) >= logging.INFO

    def test_the_one_boundary_the_live_grid_always_drops_is_not_news(self):
        """One dropped boundary (the window in progress) is expected every cycle, so it
        is DEBUG.
        """
        from core.windows import coverage_log_level

        assert coverage_log_level(report(dropped=1)) == logging.DEBUG

    def test_more_dropped_boundaries_than_the_live_edge_explains_are_news(self):
        """Two or more dropped boundaries mean lost windows, so they log at INFO or louder."""
        from core.windows import coverage_log_level

        assert coverage_log_level(report(dropped=2)) >= logging.INFO
        assert coverage_log_level(report(dropped=17)) >= logging.INFO


class TestDecisionReporting:
    """Every decision is recorded in the database; this decides only which ones appear
    in the log.
    """

    def test_a_trade_is_always_logged_at_info(self):
        from scripts.live import decision_log_level

        assert decision_log_level(
            decision(Reason.TRADED, traded=True, contracts=4)) == logging.INFO

    def test_the_designed_steady_state_is_quiet(self):
        """`offset_not_traded` fires on every non-entry offset by design, so it is DEBUG."""
        from scripts.live import decision_log_level

        assert decision_log_level(
            decision(Reason.OFFSET_NOT_TRADED)) == logging.DEBUG

    @pytest.mark.parametrize('reason', [
        Reason.PRICE_OUT_OF_BAND, Reason.EDGE_BELOW_GATE, Reason.ALREADY_ENTERED,
    ])
    def test_ordinary_refusals_are_quiet(self, reason):
        """Ordinary gate refusals are DEBUG; the cycle summary counts them."""
        from scripts.live import decision_log_level

        assert decision_log_level(decision(reason)) == logging.DEBUG

    @pytest.mark.parametrize('reason', [
        Reason.HALTED, Reason.BANKROLL_FLOOR, Reason.DISAGREEMENT_IMPLAUSIBLE,
    ])
    def test_refusals_that_signal_a_problem_stay_loud(self, reason):
        """A halt, a breached floor, or an implausible disagreement logs at INFO or louder."""
        from scripts.live import decision_log_level

        assert decision_log_level(decision(reason)) >= logging.INFO


class TestHeartbeat:
    """One heartbeat line per 15-minute window shows the loop is still cycling and
    deciding.

    Without it a quiet log cannot tell "running" from "hung".
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
        """The first cycle after a restart emits a heartbeat."""
        from scripts.live import heartbeat_due

        assert heartbeat_due(pd.Timestamp('2026-08-25 12:00', tz='UTC'), None) is True

    def test_the_summary_names_what_an_operator_checks(self):
        from scripts.live import heartbeat_summary

        text = heartbeat_summary(
            window_open=pd.Timestamp('2026-08-25 12:00', tz='UTC'),
            cycles=14, decisions=6, traded=2, bankroll=108.83, lag_seconds=6.4)
        for token in ('12:00', '14', '6', '2', '108.83', '6.4'):
            assert token in text, f'{token!r} missing from {text!r}'
