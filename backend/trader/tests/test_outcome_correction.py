"""Tests that `predictions.outcome` is corrected to the venue's settlement.

The live branch of `model_minus_market` reads this column, while the backtest
uses `venue_outcome`. Grading live rows on our Coinbase label would grade the
model on our own proxy.
"""

from __future__ import annotations

import inspect

import pandas as pd

from scripts.correct_outcomes import venue_labels

OPEN = pd.Timestamp('2026-09-16T15:00:00Z')


class _Store:
    def __init__(self, frame):
        self._frame = frame

    def read(self, name):
        assert name == 'venue_settlements'
        return self._frame


def _frame(rows):
    return pd.DataFrame(rows)


def test_the_newest_revision_of_a_window_wins():
    """The newest revision of a window's settlement wins."""
    store = _Store(_frame([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'settled_up': False, 'available_time': pd.Timestamp('2026-09-16T15:20Z')},
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'settled_up': True, 'available_time': pd.Timestamp('2026-09-16T15:40Z')},
    ]))
    labels = venue_labels(store)
    assert labels == [('BTC-USD', OPEN, True)]


def test_only_the_venue_asked_for_is_returned():
    """Only the requested venue is returned; Polymarket settles on a different oracle.
    """
    store = _Store(_frame([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'settled_up': True, 'available_time': OPEN},
        {'venue': 'polymarket', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'settled_up': False, 'available_time': OPEN},
    ]))
    assert venue_labels(store, venue='kalshi') == [('BTC-USD', OPEN, True)]


def test_a_missing_label_is_dropped_not_guessed():
    store = _Store(_frame([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'settled_up': None, 'available_time': OPEN},
    ]))
    assert venue_labels(store) == []


def test_an_empty_store_is_not_an_error():
    assert venue_labels(_Store(pd.DataFrame())) == []


def test_the_correction_only_touches_rows_that_disagree():
    """Only rows that disagree are updated, so the reported count is the disagreement and a
    re-run changes nothing.
    """
    from core.pg_writer import PgWriter

    src = inspect.getsource(PgWriter.correct_outcomes_from_venue)
    assert 'Prediction.outcome != value' in src
    assert 'Prediction.outcome.isnot(None)' in src, (
        'an unsettled row must be left to settle_predictions, not forced')


def test_the_step_runs_after_the_settlements_it_reads():
    import scripts.run_live as rl

    src = inspect.getsource(rl)
    assert 'scripts.correct_outcomes' in src, 'the step is not wired in'
    assert (src.index("'scripts.collect_settlements'")
            < src.index("'scripts.correct_outcomes'")), (
        'it reads what collect_settlements writes, so it must run after it')
