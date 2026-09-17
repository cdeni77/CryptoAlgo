"""`book_flow` must pick its observer, and refuse a book that is not one.

`attach_quotes` refuses a crossed or stale quote and ranks the observers.
`core/book_flow.py` did neither: it deduplicated with
`sort_values('offset_minutes')` — part of the dedup key itself, so it imposed
no order among duplicates and the winner was whatever the store returned last.

Measured 2026-09-16 on 724,594 selected Kalshi rows:

    26.0% (188,280) carried quotes older than 30s
    37,036 were CROSSED books
    on 48,350 rows it read a different observer than `attach_quotes` did

So `market_state` and `book_flow` could describe two different samples of the
same book on the same row. Unusable rows are DROPPED rather than demoted: a
flow feature has no NaN column to fall back to, so a crossed book would be
silently averaged into a trend.
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
    """`imbalance_persistence`, deliberately — the other three are CHANGES and
    a book held constant across the window makes all of them 0 regardless of
    which observer won. The level is what distinguishes them."""
    return frame['imbalance_persistence'].iloc[0]


def test_a_crossed_book_is_refused():
    """Ask below bid is not a book. Kept, it feeds a guaranteed-profit shape
    straight into a flow feature."""
    crossed = _book([(o, 0.80, 0.20, 900.0, 10.0, 1.0, 'live_touch') for o in range(13)])
    out = book_flow_features(_windows(), crossed)
    assert not np.isfinite(_imbalance(out)), 'a crossed book must not produce a feature'


def test_a_stale_book_is_refused():
    stale = _book([(o, 0.40, 0.42, 900.0, 10.0, 4000.0, 'backfill') for o in range(13)])
    out = book_flow_features(_windows(), stale)
    assert not np.isfinite(_imbalance(out))


def test_the_decision_instant_observer_wins_a_contested_minute():
    """Same minute from two observers. The winner must be chosen by source
    preference, not by whichever row the store returned last."""
    rows = []
    for o in range(13):
        rows.append((o, 0.40, 0.42, 10.0, 990.0, 0.5, 'backfill'))    # ask-heavy
        rows.append((o, 0.40, 0.42, 990.0, 10.0, np.nan, 'live_touch'))  # bid-heavy
    out = book_flow_features(_windows(), _book(rows))
    assert _imbalance(out) > 0, 'live_touch is ranked first and is bid-heavy'


def test_an_unusable_row_does_not_displace_a_usable_one():
    rows = []
    for o in range(13):
        rows.append((o, 0.90, 0.10, 990.0, 10.0, np.nan, 'live_touch'))  # crossed
        rows.append((o, 0.40, 0.42, 10.0, 990.0, 0.5, 'backfill'))       # fine
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
