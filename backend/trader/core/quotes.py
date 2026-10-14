"""Price a window row against the venue's recorded book.

Without a quote, `core/decide.py` prices against the calibrated baseline.
This module supplies the real book, so the backtest trades against the market.

Denomination: the UP side costs `yes_ask`; the DOWN side costs `1 - yes_bid`.
Never the mid. The two costs sum to `1 + spread`; a sum below 1 is a crossed book.

A row with no quote keeps NaN, and `price_source` stays `baseline`.
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# The store's depth columns, renamed to what `core.book_features` expects.
# Declared here so a rename cannot leave FEATURE_GROUPS pointing at a missing column.
DEPTH_MAP = {
    'yes_bid_size': 'bid_at_touch', 'yes_ask_size': 'ask_at_touch',
    'depth_bid_1c': 'bid_1c', 'depth_ask_1c': 'ask_1c',
    'depth_bid_5c': 'bid_5c', 'depth_ask_5c': 'ask_5c',
    'depth_bid_total': 'bid_vol', 'depth_ask_total': 'ask_vol',
}
QUOTE_COLUMNS = (('ask_up', 'ask_down', 'market_probability',
                  'quote_age_seconds', 'quote_source', 'quote_observer',
                  # Not features: `decide()` reads these to cap the stake.
                  'depth_up', 'depth_down')
                 + tuple(DEPTH_MAP.values()))
# Which observer to believe when several recorded the same minute.
# `quote_age_seconds` means a different thing in each producer, so it cannot
# rank across sources; it breaks ties only within one source.
# `live_touch` ranks first: it is the quote the live loop actually decided on,
# and the model must be fitted against the same object it trades against.
QUOTE_SOURCE_PRIORITY = tuple(
    s.strip() for s in os.getenv(
        'QUOTE_SOURCE_PRIORITY', 'live_touch,live_ws,live,backfill').split(',')
    if s.strip())

# A stale book is not a price this decision could have taken. Stale quotes are
# where fake edge comes from. Shares `Config.max_quote_age_seconds` with
# `core.metrics` and the live order check.
DEFAULT_MAX_AGE = float(DEFAULT_CONFIG.max_quote_age_seconds)

# The one observer whose `available_time` bounds its quote's age. The trading
# cycle reads the book at or after `event_time` (the decision instant) and
# stamps `available_time` when it writes the row. Other sources lack this:
# the backfill's `available_time` is when the backfill ran.
AGE_BOUNDED_SOURCES = ('live_touch',)

# Buckets for the source/age coverage report, in seconds.
AGE_BUCKETS = ((2.0, '<=2s'), (5.0, '2-5s'), (30.0, '5-30s'),
               (float('inf'), '>30s'))


def derived_quote_age(book: pd.DataFrame) -> pd.Series:
    """`available_time - event_time` in seconds on age-bounded rows, else NaN.

    An upper bound on the priced book's distance from the decision instant, for
    rows with no recorded age. It includes post-decision latency, so it errs
    toward stale.
    """
    out = pd.Series(np.nan, index=book.index, dtype=float)
    if not {'source', 'event_time', 'available_time'} <= set(book.columns):
        return out
    bounded = book['source'].isin(AGE_BOUNDED_SOURCES)
    if not bounded.any():
        return out
    event = pd.to_datetime(book.loc[bounded, 'event_time'], utc=True, errors='coerce')
    available = pd.to_datetime(book.loc[bounded, 'available_time'], utc=True,
                               errors='coerce')
    seconds = (available - event).dt.total_seconds()
    out.loc[bounded] = seconds.clip(lower=0.0).to_numpy(dtype=float)
    return out


def filter_sources(book: pd.DataFrame, sources: Sequence[str]) -> pd.DataFrame:
    if not sources:
        return book
    if 'source' not in book.columns:
        logger.warning('quote_sources=%s asked for, but the depth table records '
                       'no observer; nothing can be shown to qualify',
                       list(sources))
        return book.iloc[0:0]
    return book[book['source'].isin(tuple(sources))]


# How each producer measures `quote_age_seconds`. The backfill counts back from
# the row's minute. `live_touch` counts back from the decision instant. The
# minute-grid recorders count forward: they sample after the minute.
MINUTE_BACKWARD_SOURCES = ('backfill',)


def decision_age(book: pd.DataFrame, null_age: str,
                 latency_seconds: float) -> pd.Series:
    """Seconds from when each row's book was seen to the decision instant.

    The decision instant is the row's minute plus `latency_seconds`. A negative
    value is a book seen after the decision, which a decision cannot use.
    """
    raw = _effective_age(book, null_age)
    source = (book['source'] if 'source' in book.columns
              else pd.Series(None, index=book.index, dtype=object))
    from_minute = source.isin(MINUTE_BACKWARD_SOURCES)
    from_decision = source.isin(AGE_BOUNDED_SOURCES)
    forward = ~(from_minute | from_decision)
    age = raw.copy()
    age[from_minute] = raw[from_minute] + latency_seconds
    age[forward] = latency_seconds - raw[forward]
    return age


def _effective_age(book: pd.DataFrame, null_age: str) -> pd.Series:
    """The age a row is judged on.

    Under 'stale', a missing age is bounded from the row's own timestamps where
    possible, and stays NaN (dropped) otherwise."""
    age = (pd.to_numeric(book['quote_age_seconds'], errors='coerce')
           if 'quote_age_seconds' in book.columns
           else pd.Series(np.nan, index=book.index, dtype=float))
    if null_age == 'stale':
        bound = derived_quote_age(book)
        age = age.where(age.notna(), bound)
    return age


def attach_quotes(windows: pd.DataFrame, depth: pd.DataFrame, *,
                  venue: str = 'kalshi',
                  other_venue: Optional[str] = 'polymarket',
                  max_age_seconds: float = DEFAULT_MAX_AGE,
                  sources: Sequence[str] = (),
                  null_age: str = 'fresh',
                  peer_max_age_seconds: Optional[float] = None,
                  peer_null_age: str = 'fresh',
                  decision_latency_seconds: float = 3.0) -> pd.DataFrame:
    """`ask_up` / `ask_down` on each window row, from `venue_depth`.

    Joined on (symbol, window_open, offset) exactly, never as-of across offsets:
    a later offset's book is lookahead.

    Only `venue` prices a trade. The other venue is a cross-venue feature,
    never an execution price.

    Args:
        sources: when non-empty, only these observers count on both venues.
        null_age: 'stale' drops a quote with no age (after `derived_quote_age`);
            'fresh' keeps it.
        peer_max_age_seconds: the other venue's age cap; None follows `max_age_seconds`.
        peer_null_age: the other venue's null-age rule.
        decision_latency_seconds: the decision instant is the minute plus this.

    Both venues are point-in-time: a row is usable only if its book was seen at
    or before the decision instant. The output `quote_age_seconds` is that
    decision-relative age.
    """
    out = windows.copy()
    for column in QUOTE_COLUMNS:
        out[column] = np.nan
    out['quote_source'] = None
    out['quote_observer'] = None
    if depth is None or not len(depth) or not len(out):
        return out

    book = filter_sources(depth[depth['venue'] == venue], sources).copy()
    if not len(book):
        return out

    bid = pd.to_numeric(book['yes_bid'], errors='coerce')
    ask = pd.to_numeric(book['yes_ask'], errors='coerce')
    age = decision_age(book, null_age, decision_latency_seconds)

    # A price outside [0, 1] is not a probability, and an ask below the bid is
    # not a book: it would show a guaranteed profit.
    sane = ((bid.isna() | ((bid >= 0.0) & (bid <= 1.0)))
            & (ask.isna() | ((ask >= 0.0) & (ask <= 1.0)))
            & (bid.isna() | ask.isna() | (ask >= bid)))
    fresh = (age >= 0.0) & (age <= max_age_seconds)
    if null_age == 'fresh':
        fresh = fresh | age.isna()
    keep = sane & fresh

    book['ask_up'] = ask.where(keep)
    book['ask_down'] = (1.0 - bid).where(keep)
    # The market's forecast is the mid, not the ask. Scoring the market at its
    # ask would credit the model with half the spread.
    book['market_probability'] = ((bid + ask) / 2.0).where(
        keep & bid.notna() & ask.notna())
    # The stake cap `decide()` reads, in dollars on the side crossed. Live
    # supplies it from the ladder; the backtest must size the same way.
    # NaN, never zero, when no size is reported: zero would refuse every trade.
    # Raw column names: the DEPTH_MAP renames have not run yet here.
    def _size(name):
        return (pd.to_numeric(book[name], errors='coerce')
                if name in book.columns
                else pd.Series(np.nan, index=book.index))

    bid_size, ask_size = _size('yes_bid_size'), _size('yes_ask_size')
    book['depth_up'] = (ask_size * book['ask_up']).where(keep)
    book['depth_down'] = (bid_size * book['ask_down']).where(keep)
    book['quote_age_seconds'] = age.where(keep)
    book['quote_source'] = np.where(keep, venue, None)
    # Which observer priced the row. `quote_source` holds the venue.
    book['quote_observer'] = (np.where(keep, book['source'], None)
                              if 'source' in book.columns else None)

    # One row per (symbol, window, offset): collapse duplicate observers before
    # joining. Ranked by:
    #   1. `_unusable`: a row that failed `sane`/`fresh` never displaces one that passed.
    #   2. `_source_rank`: see QUOTE_SOURCE_PRIORITY.
    #   3. age, as a tie-break within one source.
    book['_unusable'] = (~keep).astype(int)
    if 'source' in book.columns:
        rank = {name: i for i, name in enumerate(QUOTE_SOURCE_PRIORITY)}
        book['_source_rank'] = (book['source'].map(rank)
                                .fillna(len(rank)).astype(int))
    else:
        book['_source_rank'] = 0
    book = (book.sort_values(['_unusable', '_source_rank', 'quote_age_seconds'],
                             na_position='last')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='first')
                .drop(columns=['_unusable', '_source_rank']))

    for src, dst in DEPTH_MAP.items():
        book[dst] = pd.to_numeric(book.get(src), errors='coerce').where(keep) \
            if src in book.columns else np.nan

    take = ['symbol', 'window_open', 'offset_minutes', 'ask_up', 'ask_down',
            'market_probability', 'quote_age_seconds', 'quote_source',
            'quote_observer',
            # The stake cap. Listed explicitly because QUOTE_COLUMNS is also
            # the drop list above.
            'depth_up', 'depth_down']
    take += list(DEPTH_MAP.values())
    merged = out.drop(columns=list(QUOTE_COLUMNS)).merge(
        book[take],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'],
        how='left')
    merged = merged.drop(columns=['offset_minutes'])

    # The other venue, under its own prefix. The two books must never share a column.
    if other_venue:
        peer = filter_sources(depth[depth['venue'] == other_venue],
                              sources).copy()
        merged['pm_market_probability'] = np.nan
        merged['pm_spread'] = np.nan
        if len(peer):
            pbid = pd.to_numeric(peer['yes_bid'], errors='coerce')
            pask = pd.to_numeric(peer['yes_ask'], errors='coerce')
            # The same staleness bar as the traded side. `venue_prob_gap` is a
            # difference of the two venues, and a timing asymmetry there reads as skill.
            # A missing stamp is kept: absence is not evidence of staleness.
            peer_cap = (max_age_seconds if peer_max_age_seconds is None
                        else float(peer_max_age_seconds))
            page = decision_age(peer, peer_null_age, decision_latency_seconds)
            peer['quote_age_seconds'] = page
            pfresh = (page >= 0.0) & (page <= peer_cap)
            if peer_null_age == 'fresh':
                pfresh = pfresh | page.isna()
            two_sided = pbid.notna() & pask.notna() & (pask >= pbid) & pfresh
            peer['pm_market_probability'] = ((pbid + pask) / 2.0).where(two_sided)
            peer['pm_spread'] = (pask - pbid).where(two_sided)
            # Rank duplicates instead of taking store order. An arbitrary winner
            # that fails `two_sided` would void `venue_prob_gap` while a usable
            # duplicate exists. Same rule as the traded side.
            peer['_unusable'] = (~two_sided).astype(int)
            peer = (peer.sort_values(['_unusable', 'quote_age_seconds'],
                                     na_position='last')
                        .drop_duplicates(['symbol', 'window_open',
                                          'offset_minutes'], keep='first')
                        .drop(columns=['_unusable']))
            merged = merged.drop(columns=['pm_market_probability', 'pm_spread']).merge(
                peer[['symbol', 'window_open', 'offset_minutes',
                      'pm_market_probability', 'pm_spread']],
                left_on=['symbol', 'window_open', 'offset'],
                right_on=['symbol', 'window_open', 'offset_minutes'],
                how='left').drop(columns=['offset_minutes'])
    # decide() reads rows in order; a lost or reordered row changes which
    # windows trade.
    if len(merged) != len(out):
        raise ValueError(f'quote join changed the row count: '
                         f'{len(out)} -> {len(merged)}')
    return merged


def age_bucket(age: pd.Series) -> pd.Series:
    """Label each age with its AGE_BUCKETS bucket; a missing age is 'null'."""
    age = pd.to_numeric(age, errors='coerce').abs()
    out = pd.Series('null', index=age.index, dtype=object)
    lower = -np.inf
    for upper, name in AGE_BUCKETS:
        out[(age > lower) & (age <= upper)] = name
        lower = upper
    return out


def source_age_table(frame: pd.DataFrame, *, source_column: str = 'source',
                     age_column: str = 'quote_age_seconds') -> pd.DataFrame:
    """Row counts by observer x age bucket, one row per observer.

    Run on the raw depth table it shows what a source/age filter is about to
    discard; on the joined window table, what actually priced the rows.
    """
    if frame is None or not len(frame):
        return pd.DataFrame()
    source = (frame[source_column].fillna('none').astype(str)
              if source_column in frame.columns
              else pd.Series('none', index=frame.index))
    age = (frame[age_column] if age_column in frame.columns
           else pd.Series(np.nan, index=frame.index))
    table = pd.crosstab(source.rename('source'), age_bucket(age).rename('age'))
    order = [name for _, name in AGE_BUCKETS] + ['null']
    table = table.reindex(columns=[c for c in order if c in table.columns],
                          fill_value=0)
    table['total'] = table.sum(axis=1)
    return table.reset_index()


def quote_coverage(table: pd.DataFrame) -> dict:
    """How many rows a real price reached. `market_windows` gates on this."""
    if not len(table) or 'ask_up' not in table.columns:
        return {'rows': 0, 'quoted': 0, 'share': 0.0, 'by_source_age': []}
    quoted = table['ask_up'].notna()
    # Which observer priced the quoted rows, and how old its quote was.
    by_source = source_age_table(table.loc[quoted], source_column='quote_observer')
    return {'rows': int(len(table)), 'quoted': int(quoted.sum()),
            'share': int(quoted.sum()) / max(len(table), 1),
            'by_source_age': by_source.to_dict('records')}
