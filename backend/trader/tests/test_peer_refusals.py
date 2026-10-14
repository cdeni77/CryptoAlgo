"""Peer refusals are counted by reason per window and reported once per window."""
from __future__ import annotations

import logging

import pandas as pd

from livetrade import features
from livetrade.features import (
    PEER_USED, cross_venue_row, peer_refusal, take_peer_counts,
)

NOW = pd.Timestamp('2026-10-14 12:12:02', tz='UTC')
WINDOW = pd.Timestamp('2026-10-14 12:00', tz='UTC')


class _Quote:
    yes_bid, yes_ask = 0.50, 0.52


def _pm(age=3.0, window=WINDOW, bid=40.0, ask=42.0):
    return {'best_bid': bid, 'best_ask': ask, 'window': window,
            'at': NOW - pd.Timedelta(seconds=age)}


def test_each_reason_is_named():
    assert peer_refusal(None, now=NOW, window_open=WINDOW) == 'no_book'
    assert peer_refusal(_pm(window=WINDOW - pd.Timedelta(minutes=15)),
                        now=NOW, window_open=WINDOW) == 'wrong_window'
    assert peer_refusal(_pm(age=600), now=NOW, window_open=WINDOW) == 'too_old'
    assert peer_refusal(_pm(age=-2), now=NOW, window_open=WINDOW) == 'future'
    assert peer_refusal(_pm(ask=float('nan')), now=NOW,
                        window_open=WINDOW) == 'one_sided'
    assert peer_refusal(_pm(), now=NOW, window_open=WINDOW) is None


def test_the_window_is_checked_before_the_age():
    pm = _pm(age=600, window=WINDOW - pd.Timedelta(minutes=15))
    assert peer_refusal(pm, now=NOW, window_open=WINDOW) == 'wrong_window'


def test_verdicts_are_counted_per_window_and_taken_once():
    features._PEER_COUNTS.clear()
    for pm in (_pm(), _pm(), _pm(age=600), None):
        cross_venue_row(pm, _Quote(), now=NOW, window_open=WINDOW, symbol='BTC-USD')
    assert take_peer_counts(WINDOW) == {PEER_USED: 2, 'too_old': 1, 'no_book': 1}
    assert take_peer_counts(WINDOW) == {}


def test_taking_a_window_drops_older_ones():
    features._PEER_COUNTS.clear()
    older = WINDOW - pd.Timedelta(minutes=15)
    cross_venue_row(None, _Quote(), now=NOW, window_open=older)
    cross_venue_row(None, _Quote(), now=NOW, window_open=WINDOW)
    take_peer_counts(WINDOW)
    assert not features._PEER_COUNTS


def test_the_per_row_line_stays_at_debug(caplog):
    with caplog.at_level(logging.DEBUG, logger='live'):
        cross_venue_row(_pm(age=600), _Quote(), now=NOW, window_open=WINDOW,
                        symbol='ETH-USD')
    lines = [r for r in caplog.records if 'refused' in r.message]
    assert lines and all(r.levelno == logging.DEBUG for r in lines)
    assert 'too_old' in lines[0].message
    features._PEER_COUNTS.clear()


def test_the_heartbeat_line_carries_the_counts():
    from scripts.live import heartbeat_summary

    text = heartbeat_summary(
        window_open=WINDOW, cycles=15, decisions=3, traded=0, bankroll=500.0,
        lag_seconds=2.1, peer={PEER_USED: 1, 'too_old': 1, 'wrong_window': 1})
    assert 'peer used 1/3 refused too_old=1 wrong_window=1' in text


def test_the_heartbeat_line_without_peer_reads_is_unchanged():
    from scripts.live import heartbeat_summary

    text = heartbeat_summary(
        window_open=WINDOW, cycles=15, decisions=3, traded=0, bankroll=500.0,
        lag_seconds=2.1, peer={})
    assert 'peer' not in text
