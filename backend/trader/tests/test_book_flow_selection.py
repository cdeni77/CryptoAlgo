"""Tests that `book_flow` chooses its observer and drops unusable books.

It applies the same source ranking and the same crossed/stale filters as
`attach_quotes`, so `market_state` and `book_flow` read the same book on a row.
Unusable rows are dropped, not demoted, because a flow feature has no NaN
fallback and would average a crossed book into a trend.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.book_flow import book_flow_features

OPEN = pd.Timestamp('2026-07-01T12:00Z')


def _windows(offset=12):
    return pd.DataFrame([{'symbol': 'BTC-USD', 'window_open': OPEN, 'offset': offset}])


def _book(rows):
    return pd.DataFrame([
        {'venue': 'kalshi', 'symbol': 'BTC-USD', 'window_open': OPEN,
         'offset_minutes': o, 'yes_bid': b, 'yes_ask': a,
         'yes_bid_size': bs, 'yes_ask_size': asz,
         'quote_age_seconds': age, 'source': src}
        for o, b, a, bs, asz, age, src in rows])


def _imbalance(frame):
    """Reads `imbalance_persistence`: the other columns are changes and are 0 for a
    constant book.
    """
    return frame['imbalance_persistence'].iloc[0]


def test_a_crossed_book_is_refused():
    """A crossed book (ask below bid) produces no feature."""
    crossed = _book([(o, 0.80, 0.20, 900.0, 10.0, 1.0, 'live_touch') for o in range(13)])
    out = book_flow_features(_windows(), crossed)
    assert not np.isfinite(_imbalance(out)), 'a crossed book must not produce a feature'


def test_a_stale_book_is_refused():
    stale = _book([(o, 0.40, 0.42, 900.0, 10.0, 4000.0, 'backfill') for o in range(13)])
    out = book_flow_features(_windows(), stale)
    assert not np.isfinite(_imbalance(out))


def test_the_decision_instant_observer_wins_a_contested_minute():
    """When two observers report the same minute, source preference picks the winner."""
    rows = []
    for o in range(13):
        rows.append((o, 0.40, 0.42, 10.0, 990.0, 0.5, 'backfill'))  # ask-heavy
        rows.append((o, 0.40, 0.42, 990.0, 10.0, np.nan, 'live_touch'))  # bid-heavy
    out = book_flow_features(_windows(), _book(rows))
    assert _imbalance(out) > 0, 'live_touch is ranked first and is bid-heavy'


def test_an_unusable_row_does_not_displace_a_usable_one():
    rows = []
    for o in range(13):
        rows.append((o, 0.90, 0.10, 990.0, 10.0, np.nan, 'live_touch'))  # crossed
        rows.append((o, 0.40, 0.42, 10.0, 990.0, 0.5, 'backfill'))  # fine
    out = book_flow_features(_windows(), _book(rows))
    assert np.isfinite(_imbalance(out)), 'the usable backfill row must survive'
    assert _imbalance(out) < 0, 'and it is ask-heavy'


def test_the_age_bar_is_the_one_everything_else_uses():
    from core.book_flow import MAX_QUOTE_AGE_SECONDS as flow_bar
    from core.metrics import MAX_QUOTE_AGE_SECONDS as gate_bar
    from core.quotes import DEFAULT_MAX_AGE as price_bar

    assert flow_bar == gate_bar == price_bar, (
        'three filters on one concept must move together, or the money and the '
        'features are measured on different samples')
