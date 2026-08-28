"""The backtest must settle on the venue's own result, not on our proxy.

**The leak this closes, measured.** With the backtest settling on our
Coinbase-derived label while trading against the venue's BRTI-priced quotes,
and `market_prob` live as a feature, the model learned to bet the disagreement
between the two indices — and won it by construction:

    labels agree on 96.51% of traded windows
      win rate where they AGREE  : 56.17%
      win rate where they DIFFER : 72.77%   (n=448)

Rescoring the same 12,821 trades on the venue's settlement took the win rate
from 56.75% to 55.16% and the edge from 8.68% to 4.99% of stake. About 43% of
the apparent edge was the label.

Our label stays as the fallback: the venue's settlements do not reach the whole
span, and dropping those windows would discard most of the sample. What must
never happen is preferring ours where theirs exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.backtest import settlement_outcomes


def _frame(outcome, venue=None):
    f = pd.DataFrame({
        'symbol': ['BTC-USD', 'BTC-USD'],
        'window_open': pd.to_datetime(['2026-07-01T12:00Z', '2026-07-01T12:15Z']),
        'outcome': outcome,
    })
    if venue is not None:
        f['venue_outcome'] = venue
    return f


def test_the_venue_result_wins_where_we_hold_it():
    got = settlement_outcomes(_frame([1, 0], venue=[0.0, 1.0]))
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:00Z'))] is False
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:15Z'))] is True


def test_our_label_fills_in_where_the_venue_has_none():
    """Kalshi purges older markets, so their settlements do not reach the whole
    span. Dropping those windows would discard most of the sample."""
    got = settlement_outcomes(_frame([1, 0], venue=[np.nan, 1.0]))
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:00Z'))] is True
    assert got[('BTC-USD', pd.Timestamp('2026-07-01T12:15Z'))] is True


def test_a_frame_without_the_venue_column_still_settles():
    got = settlement_outcomes(_frame([1, 0]))
    assert set(got.values()) == {True, False}


def test_a_window_with_neither_label_is_absent_rather_than_guessed():
    """An unsettled window must not be booked as a loss: `book.settle` skips
    what it has no outcome for, and inventing one would pay or charge for a
    result nobody has."""
    got = settlement_outcomes(_frame([np.nan, 0], venue=[np.nan, np.nan]))
    assert ('BTC-USD', pd.Timestamp('2026-07-01T12:00Z')) not in got
    assert len(got) == 1
