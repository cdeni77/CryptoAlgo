"""Tests for `require_peer_book`, which refuses a row without the peer venue's book.

Training requires the peer (`--complete-cases`), so live refusing it keeps the two
consistent.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.config import Config
from core.decide import Reason, decide


def _row(**over):
    row = {
        'symbol': 'BTC-USD', 'offset': 12,
        'baseline_probability': 0.40, 'model_probability': 0.52,
        'ask_up': 0.44, 'ask_down': 0.58,
        'venue_prob_gap': 0.01,
        'sigma_remaining': 0.0008,
    }
    row.update(over)
    return row


def test_a_row_without_a_peer_gap_is_refused():
    config = Config(require_peer_book=True)
    out = decide(_row(venue_prob_gap=float('nan')), config, bankroll=200.0)
    assert not out.traded
    assert out.reason is Reason.NO_PEER_BOOK


def test_a_row_with_a_peer_gap_is_unaffected():
    config = Config(require_peer_book=True)
    assert decide(_row(), config, bankroll=200.0).reason is not Reason.NO_PEER_BOOK


def test_the_guard_is_off_by_default_so_past_entries_keep_their_meaning():
    """The guard is off by default, so past ledger entries keep their meaning."""
    assert Config().require_peer_book is False
    out = decide(_row(venue_prob_gap=float('nan')), Config(), bankroll=200.0)
    assert out.reason is not Reason.NO_PEER_BOOK


def test_a_row_that_never_carried_the_column_is_not_refused():
    """A row with no `venue_prob_gap` column (a config without cross_venue) is not refused."""
    config = Config(require_peer_book=True)
    row = _row()
    row.pop('venue_prob_gap')
    assert decide(row, config, bankroll=200.0).reason is not Reason.NO_PEER_BOOK
