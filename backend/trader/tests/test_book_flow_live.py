"""`book_flow` on the live path, and that it equals the backtest's.

All six `book_flow` features were NaN on every real-money decision while the
deployed artifact was fitted on them: `book_flow_features` is called from
`core/dataset.py` alone, and `livetrade/features.py` never built it. The loop
warned every cycle and nothing escalated it.

Two things have to hold for the repair to be worth anything:

  * the live values must EQUAL the backtest's on the same book, bit for bit --
    so the live path calls `book_flow_features` itself rather than restating
    its arithmetic;
  * a decision at offset `m` must not read minute `m+1`. A one-minute leak in a
    fifteen-minute window is 7% of the question and reads exactly like skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.book_flow import BOOK_FLOW, book_flow_features
from livetrade.features import (
    BOOK_SAMPLE_MAX_AGE_SECONDS, book_flow_row, book_history_depth,
    record_book_minute, reset_book_history,
)

OPEN = pd.Timestamp('2026-10-21T12:00Z')
SYMBOL = 'BTC-USD'


def _levels(bid, ask, bid_size, ask_size):
    """(yes_levels, no_levels). NO levels are NO-denominated: a YES ask of
    0.52 is a NO bid of 0.48 — the denomination trap this repo already has a
    name for."""
    return ([(bid, bid_size), (round(bid - 0.01, 4), bid_size * 2)],
            [(round(1.0 - ask, 4), ask_size), (round(1.0 - ask - 0.01, 4), ask_size * 2)])


def _record(upto, *, age=0.0):
    reset_book_history()
    for off in range(0, upto + 1):
        yes, no = _levels(0.50 + 0.001 * off, 0.52 + 0.001 * off,
                          100 + off, 90 + off)
        record_book_minute(SYMBOL, window_open=OPEN, offset=off,
                           yes_levels=yes, no_levels=no, age_seconds=age)


def _scored(offset):
    return pd.DataFrame([{'symbol': SYMBOL, 'window_open': OPEN,
                          'offset': offset}])


@pytest.fixture(autouse=True)
def _clean():
    reset_book_history()
    yield
    reset_book_history()


def test_the_live_values_equal_the_backtest_on_the_same_book():
    _record(12)
    live = book_flow_row(_scored(12))
    research = book_flow_features(
        pd.DataFrame([{'symbol': SYMBOL, 'window_open': OPEN, 'offset': 12}]),
        book_history_depth())
    for col in BOOK_FLOW:
        assert live[col].iloc[0] == research[col].iloc[0], col


def test_every_feature_is_produced():
    _record(12)
    row = book_flow_row(_scored(12)).iloc[0]
    missing = [c for c in BOOK_FLOW if not np.isfinite(row[c])]
    assert not missing, 'still NaN live: %s' % missing


def test_a_decision_cannot_read_a_later_minute():
    # The guard the module docstring calls the whole risk. Plant a wild swing
    # AFTER the decision and assert the answer is bit-identical.
    _record(12)
    clean = book_flow_row(_scored(12)).iloc[0]
    for off in (13, 14):
        yes, no = _levels(0.90, 0.95, 9999, 1)
        record_book_minute(SYMBOL, window_open=OPEN, offset=off,
                           yes_levels=yes, no_levels=no, age_seconds=0.0)
    after = book_flow_row(_scored(12)).iloc[0]
    for col in BOOK_FLOW:
        assert clean[col] == after[col], col


def test_a_stale_sample_is_refused_rather_than_stored():
    # The trap this was built around: the live decision cycle starts ~7s into
    # the minute, and `decision_age` ages a minute-grid source FORWARD, so a
    # 7s-old row is dropped by `book_flow_features` and the features come back
    # NaN anyway. Refusing at record time makes the sampler's drift visible
    # instead of hiding it behind absent features.
    assert record_book_minute(SYMBOL, window_open=OPEN, offset=3,
                              yes_levels=[(0.5, 10)], no_levels=[(0.48, 10)],
                              age_seconds=0.0)
    assert not record_book_minute(
        SYMBOL, window_open=OPEN, offset=4,
        yes_levels=[(0.5, 10)], no_levels=[(0.48, 10)],
        age_seconds=BOOK_SAMPLE_MAX_AGE_SECONDS + 0.1)
    assert sorted(book_history_depth()['offset_minutes']) == [3]


def test_an_empty_book_is_not_a_reading():
    assert not record_book_minute(SYMBOL, window_open=OPEN, offset=3,
                                  yes_levels=[], no_levels=[])
    assert book_history_depth().empty


def test_history_is_evicted_by_window_not_by_arrival():
    # Three symbols are sampled in the same minute of the same window and must
    # not evict each other; a NEW window clears the old one.
    for sym in ('BTC-USD', 'ETH-USD', 'SOL-USD'):
        record_book_minute(sym, window_open=OPEN, offset=1,
                           yes_levels=[(0.5, 10)], no_levels=[(0.48, 10)])
    assert len(book_history_depth()) == 3
    later = OPEN + pd.Timedelta(minutes=15)
    record_book_minute('BTC-USD', window_open=later, offset=1,
                       yes_levels=[(0.5, 10)], no_levels=[(0.48, 10)])
    held = book_history_depth()
    assert set(held['window_open']) == {later}


def test_no_history_gives_nan_not_zero():
    # Zero says "the book did not move", which is a claim. The model is fitted
    # with these missing on rows that have no history.
    row = book_flow_row(_scored(12)).iloc[0]
    assert all(np.isnan(row[c]) for c in BOOK_FLOW)


def test_the_touch_size_is_the_size_at_the_best_price():
    # `build_depth` defines `yes_bid_size` as the size at the best price and
    # nothing behind it; a second convention under the same column name would
    # make live and the backtest describe different books.
    reset_book_history()
    record_book_minute(SYMBOL, window_open=OPEN, offset=1,
                       yes_levels=[(0.50, 10), (0.49, 999)],
                       no_levels=[(0.48, 7), (0.47, 888)])
    row = book_history_depth().iloc[0]
    assert row['yes_bid'] == pytest.approx(0.50)
    assert row['yes_bid_size'] == pytest.approx(10)
    assert row['yes_ask'] == pytest.approx(0.52)
    assert row['yes_ask_size'] == pytest.approx(7)


def test_only_the_rows_being_scored_come_back():
    _record(12)
    scored = pd.DataFrame([{'symbol': SYMBOL, 'window_open': OPEN, 'offset': 3},
                           {'symbol': SYMBOL, 'window_open': OPEN, 'offset': 12}])
    got = book_flow_row(scored)
    assert len(got) == 2
    assert list(got.columns) == list(BOOK_FLOW)
