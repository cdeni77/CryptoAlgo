"""Book DYNAMICS — how the ladder is moving, not how it stands.

`core/book_features.py` reads the book as a snapshot at the decision instant:
spread, imbalance, depth ratios, convexity. Nothing reads the DIRECTION it is
moving, and `venue_depth` carries a row at every minute 0..15 — 16,345 of 18,321
windows have ten or more — so the history is there and unused.

**The mechanism.** A resting book is a claim about where the price should be;
a book whose bid side is building while the ask thins is a claim being revised.
The barrier framing says the only forecastable quantity is `sigma_remaining`,
and a book draining fast is a market that expects to move. That is a different
statement from the level of imbalance, which `market_state` already carries.

**Why this and not the trade tape.** The tape is REJECTED and stays rejected:
sixteen cells across four projections and four offsets, orthogonalised against
price and price-squared, clustered on the UTC day — not one significant, largest
t=-1.90 on 1,800 windows. Book imbalance is the one thing that survived as
open rather than closed:

    "Daily corr went +0.0074 (t=+0.30) to +0.0253 (t=+1.84) over 67 days as the
     sample grew. Does not clear the bar the trade tape failed at, but unlike
     the tape it is positive and strengthening. Re-test when the backfill
     completes rather than filing as rejected."

**The lookahead guard is the whole risk.** A decision at offset `m` may read
minutes 0..m and nothing after. A one-minute leak in a fifteen-minute window is
7% of the question and reads exactly like skill, which is why the first test in
`tests/test_book_flow.py` plants a wild swing in minutes 13-14 and asserts a
+12m decision is bit-identical without them.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BOOK_FLOW = (
    'imbalance_change_3',    # touch imbalance now minus three minutes ago
    'depth_trend_3',         # log ratio of total resting size over three minutes
    'spread_change_3',       # spread now minus three minutes ago, in cents
    'imbalance_persistence',  # mean imbalance so far — standing pressure, not a blip
)
LOOKBACK = 3


def _imbalance(bid, ask):
    total = bid + ask
    return np.where(total > 0, (bid - ask) / np.where(total > 0, total, 1.0), np.nan)


def book_flow_features(windows: pd.DataFrame, depth: pd.DataFrame, *,
                       venue: str = 'kalshi',
                       lookback: int = LOOKBACK) -> pd.DataFrame:
    """One row per window row, from the minutes at or before its decision.

    NaN, never zero, where the history is absent: zero says "the book did not
    move", which is a claim, and the model is fitted with these missing on the
    rows that have no history.
    """
    out = pd.DataFrame(index=windows.index)
    for col in BOOK_FLOW:
        out[col] = np.nan
    if depth is None or not len(depth) or not len(windows):
        return out

    book = depth[depth['venue'] == venue].copy()
    if not len(book):
        return out
    for col in ('yes_bid_size', 'yes_ask_size', 'yes_bid', 'yes_ask'):
        book[col] = pd.to_numeric(book.get(col), errors='coerce')
    book['offset_minutes'] = pd.to_numeric(book['offset_minutes'], errors='coerce')
    book['window_open'] = pd.to_datetime(book['window_open'], utc=True)
    book['_imb'] = _imbalance(book['yes_bid_size'].to_numpy(),
                              book['yes_ask_size'].to_numpy())
    book['_total'] = book['yes_bid_size'] + book['yes_ask_size']
    book['_spread'] = (book['yes_ask'] - book['yes_bid']) * 100.0
    # One row per minute: the same minute can arrive from more than one observer.
    book = (book.sort_values('offset_minutes')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='last'))
    keyed = {k: g.set_index('offset_minutes')
             for k, g in book.groupby(['symbol', 'window_open'])}

    offsets = pd.to_numeric(windows.get('offset', windows.get('offset_minutes')),
                            errors='coerce')
    opens = pd.to_datetime(windows['window_open'], utc=True)
    for i, (sym, open_at, off) in enumerate(zip(windows['symbol'], opens, offsets)):
        g = keyed.get((sym, open_at))
        if g is None or not np.isfinite(off):
            continue
        # AT OR BEFORE the decision. Never after — see the module docstring.
        past = g[g.index <= int(off)]
        if not len(past):
            continue
        now = past.iloc[-1]
        then_rows = past[past.index <= int(off) - lookback]
        pos = out.index[i]
        if len(past) >= 2:
            out.at[pos, 'imbalance_persistence'] = float(
                np.nanmean(past['_imb'].to_numpy()))
        if not len(then_rows):
            continue
        then = then_rows.iloc[-1]
        out.at[pos, 'imbalance_change_3'] = float(now['_imb'] - then['_imb'])
        out.at[pos, 'spread_change_3'] = float(now['_spread'] - then['_spread'])
        if now['_total'] > 0 and then['_total'] > 0:
            out.at[pos, 'depth_trend_3'] = float(
                np.log(now['_total'] / then['_total']))
    return out
