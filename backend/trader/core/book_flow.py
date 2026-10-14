"""Book dynamics features: how the ladder moves, not how it stands.

Point-in-time: a decision at offset `m` reads book minutes 0..m and nothing after.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.metrics import MAX_QUOTE_AGE_SECONDS
from core.quotes import QUOTE_SOURCE_PRIORITY

logger = logging.getLogger(__name__)

BOOK_FLOW = (
    'imbalance_change_3',    # touch imbalance now minus three minutes ago
    'depth_trend_3',         # log ratio of total resting size over three minutes
    'spread_change_3',       # spread now minus three minutes ago, in cents
    'imbalance_persistence',  # mean imbalance so far: standing pressure, not a blip
)
LOOKBACK = 3


def _imbalance(bid, ask):
    total = bid + ask
    return np.where(total > 0, (bid - ask) / np.where(total > 0, total, 1.0), np.nan)


def book_flow_features(windows: pd.DataFrame, depth: pd.DataFrame, *,
                       venue: str = 'kalshi',
                       lookback: int = LOOKBACK,
                       decision_latency_seconds: float = 3.0) -> pd.DataFrame:
    """One row per window row, from the book minutes at or before its decision.

    NaN, never zero, where history is absent: zero claims "the book did not move".
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
    # Drop crossed and stale books, then pick one observer per minute.
    # The ranking matches `attach_quotes`: source preference, then age within a source.
    # Unusable rows are dropped, not demoted: a flow feature has no NaN fallback.
    # Age is decision-relative, so a row seen after its minute's decision is stale.
    from core.quotes import decision_age
    age = decision_age(book, 'fresh', decision_latency_seconds)
    bid, ask = book['yes_bid'], book['yes_ask']
    sane = ((bid.isna() | ((bid >= 0.0) & (bid <= 1.0)))
            & (ask.isna() | ((ask >= 0.0) & (ask <= 1.0)))
            & (bid.isna() | ask.isna() | (ask >= bid)))
    fresh = age.isna() | ((age >= 0.0) & (age <= MAX_QUOTE_AGE_SECONDS))
    book = book[sane & fresh]
    if not len(book):
        return out
    if 'source' in book.columns:
        rank = {name: i for i, name in enumerate(QUOTE_SOURCE_PRIORITY)}
        book = book.assign(_rank=book['source'].map(rank)
                           .fillna(len(rank)).astype(int),
                           _age=age.reindex(book.index))
    else:
        book = book.assign(_rank=0, _age=age.reindex(book.index))
    book = (book.sort_values(['_rank', '_age'], na_position='last')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='first')
                .drop(columns=['_rank', '_age']))
    keyed = {k: g.set_index('offset_minutes')
             for k, g in book.groupby(['symbol', 'window_open'])}

    offsets = pd.to_numeric(windows.get('offset', windows.get('offset_minutes')),
                            errors='coerce')
    opens = pd.to_datetime(windows['window_open'], utc=True)
    for i, (sym, open_at, off) in enumerate(zip(windows['symbol'], opens, offsets)):
        g = keyed.get((sym, open_at))
        if g is None or not np.isfinite(off):
            continue
        # At or before the decision, never after.
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
