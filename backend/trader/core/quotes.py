"""Price a window row against the venue's recorded book.

**This closes the circularity the audit named.** `core/decide.py` falls back to
`p_market = baseline_probability` when a row carries no `ask_up`/`ask_down` —
which is every backtest row — so the simulated counterparty is the model's own
training target. Two of the eighteen gates, `market_windows` and
`model_minus_market`, read NaN and fail for want of a price rather than for want
of skill. `AUDIT_REPORT.md` calls that the sharpest complaint against the whole
result, and eight months of collected book is the answer to it.

**Denomination is the thing that kills you quietly.** A Kalshi YES book quotes
what YES costs. The price of the DOWN side is `1 - yes_bid`, because buying NO
is taking the other side of what YES bidders will pay — not `yes_bid`, and not
the mid, which is a price nobody will sell you. Getting it wrong is an error of
the spread with the sign inverted on one side, and nothing raises. It is the
same class of mistake as the `no_levels` trap that put a 0.51 YES ask in the
column holding a 0.51 NO bid.

A useful property falls out and is asserted in the tests: the two costs must sum
to `1 + spread`. Summing to exactly 1 means the mid was used; summing to less
than 1 means the book is crossed and the row is not a book at all.

**A row with no quote keeps NaN on purpose.** Most of five years predates the
venue, and those rows must go on pricing against the calibrated baseline exactly
as before. `price_source` then still reads `baseline`, which is the distinction
the schema has always insisted on: a backtest and a fill are different claims.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

# The store's depth columns, renamed to what `core.book_features` expects.
# Declared here so a rename cannot leave FEATURE_GROUPS pointing at a column
# nothing joins — which is exactly how a run reported "46 features (12 empty),
# 2 trees": a quarter of the matrix was all-NaN and only a warning said so.
DEPTH_MAP = {
    'yes_bid_size': 'bid_at_touch', 'yes_ask_size': 'ask_at_touch',
    'depth_bid_1c': 'bid_1c', 'depth_ask_1c': 'ask_1c',
    'depth_bid_5c': 'bid_5c', 'depth_ask_5c': 'ask_5c',
    'depth_bid_total': 'bid_vol', 'depth_ask_total': 'ask_vol',
}
QUOTE_COLUMNS = (('ask_up', 'ask_down', 'market_probability',
                  'quote_age_seconds', 'quote_source',
                  # Not features — `decide()` reads these to cap the stake.
                  'depth_up', 'depth_down')
                 + tuple(DEPTH_MAP.values()))
# A book quoted twenty minutes ago is not a price this decision could have
# taken. Measured on the real store, 88.1% of backfilled minute rows are fresher
# than 30s and 70.8% fresher than 5s, so tightening this costs little coverage.
#
# It matters that this MATCHES `core.metrics.MAX_QUOTE_AGE_SECONDS`: trades were
# priced against quotes up to 900s old while `model_minus_market` counted only
# those under 30s, so the money and the forecast comparison were measured on
# different samples — and stale quotes are exactly where fake edge comes from
# (model_minus_market ran 9x higher at 900s than at 5s).
DEFAULT_MAX_AGE = float(os.getenv('QUOTE_MAX_AGE_SECONDS', '30'))


def attach_quotes(windows: pd.DataFrame, depth: pd.DataFrame, *,
                  venue: str = 'kalshi',
                  other_venue: Optional[str] = 'polymarket',
                  max_age_seconds: float = DEFAULT_MAX_AGE) -> pd.DataFrame:
    """`ask_up` / `ask_down` on each window row, from `venue_depth`.

    Joined on (symbol, window_open, offset) EXACTLY. Not as-of across offsets:
    a decision at +3m priced with the book at +12m would be nine minutes of a
    fifteen-minute window ahead of itself, which is most of the question and
    reads exactly like skill. The at-or-before rule already lives one layer
    down, in how `venue_depth`'s minute rows are built.

    Only `venue` is used. Polymarket is a cross-venue FEATURE, never the book we
    execute against — pricing a Kalshi trade off a Polymarket quote would book a
    fill at a price the venue never showed.
    """
    out = windows.copy()
    for column in QUOTE_COLUMNS:
        out[column] = np.nan
    out['quote_source'] = None
    if depth is None or not len(depth) or not len(out):
        return out

    book = depth[depth['venue'] == venue].copy()
    if not len(book):
        return out

    bid = pd.to_numeric(book['yes_bid'], errors='coerce')
    ask = pd.to_numeric(book['yes_ask'], errors='coerce')
    age = pd.to_numeric(book.get('quote_age_seconds'), errors='coerce')

    # A price outside [0, 1] is not a probability, and an ask below the bid is
    # not a book — kept, it would show a guaranteed profit and the backtest
    # would take it every time.
    sane = ((bid.isna() | ((bid >= 0.0) & (bid <= 1.0)))
            & (ask.isna() | ((ask >= 0.0) & (ask <= 1.0)))
            & (bid.isna() | ask.isna() | (ask >= bid)))
    fresh = age.isna() | (age.abs() <= max_age_seconds)
    keep = sane & fresh

    book['ask_up'] = ask.where(keep)
    book['ask_down'] = (1.0 - bid).where(keep)
    # The market's FORECAST, which is the mid — not the ask, which is what a
    # trade costs. `model_minus_market` compares log losses, so scoring the
    # market at its ask would credit the model with half the spread as skill on
    # every row, in its own favour.
    book['market_probability'] = ((bid + ask) / 2.0).where(
        keep & bid.notna() & ask.notna())
    # The stake cap `decide()` reads, in DOLLARS on the side actually crossed.
    # `scripts/live.py` supplies these from the live ladder; without them here
    # the backtest would size against `max_stake_dollars` alone while live sized
    # against the book, and "one decide()" is the invariant that stops the two
    # from drifting. NaN, never zero, when the venue reported no size: zero
    # refuses every trade where the honest answer is "unmeasured".
    # The RAW column names: `bid_at_touch`/`ask_at_touch` are the DEPTH_MAP
    # renames and that loop has not run yet at this point in the function.
    def _size(name):
        return (pd.to_numeric(book[name], errors='coerce')
                if name in book.columns
                else pd.Series(np.nan, index=book.index))

    bid_size, ask_size = _size('yes_bid_size'), _size('yes_ask_size')
    book['depth_up'] = (ask_size * book['ask_up']).where(keep)
    book['depth_down'] = (bid_size * book['ask_down']).where(keep)
    book['quote_age_seconds'] = age.where(keep)
    book['quote_source'] = np.where(keep, venue, None)

    # One row per (symbol, window, offset). `venue_depth` can hold the same
    # minute from more than one observer — that is what `source` is for — so
    # collapse before joining rather than fanning the window table out.
    book = (book.sort_values('quote_age_seconds', na_position='last')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='first'))

    for src, dst in DEPTH_MAP.items():
        book[dst] = pd.to_numeric(book.get(src), errors='coerce').where(keep) \
            if src in book.columns else np.nan

    take = ['symbol', 'window_open', 'offset_minutes', 'ask_up', 'ask_down',
            'market_probability', 'quote_age_seconds', 'quote_source',
            # The stake cap. Listed here and not derived from QUOTE_COLUMNS
            # because that tuple is also the drop list above; keeping them in
            # step is the reason this line exists rather than a comprehension.
            'depth_up', 'depth_down']
    take += list(DEPTH_MAP.values())
    merged = out.drop(columns=list(QUOTE_COLUMNS)).merge(
        book[take],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'],
        how='left')
    merged = merged.drop(columns=['offset_minutes'])

    # The other venue, under its own prefix. `cross_venue` needs both books on
    # one row and they must never share a column: a Polymarket bid sitting in a
    # column holding a Kalshi one is the `no_levels` denomination trap wearing a
    # different name.
    if other_venue:
        peer = depth[depth['venue'] == other_venue].copy()
        merged['pm_market_probability'] = np.nan
        merged['pm_spread'] = np.nan
        if len(peer):
            pbid = pd.to_numeric(peer['yes_bid'], errors='coerce')
            pask = pd.to_numeric(peer['yes_ask'], errors='coerce')
            two_sided = pbid.notna() & pask.notna() & (pask >= pbid)
            peer['pm_market_probability'] = ((pbid + pask) / 2.0).where(two_sided)
            peer['pm_spread'] = (pask - pbid).where(two_sided)
            peer = peer.drop_duplicates(
                ['symbol', 'window_open', 'offset_minutes'], keep='first')
            merged = merged.drop(columns=['pm_market_probability', 'pm_spread']).merge(
                peer[['symbol', 'window_open', 'offset_minutes',
                      'pm_market_probability', 'pm_spread']],
                left_on=['symbol', 'window_open', 'offset'],
                right_on=['symbol', 'window_open', 'offset_minutes'],
                how='left').drop(columns=['offset_minutes'])
    # `how='left'` preserves order and count, but the table is fed to decide()
    # row by row and losing or reordering rows silently changes which windows
    # were traded. Cheap to assert, expensive to discover.
    if len(merged) != len(out):
        raise ValueError(f'quote join changed the row count: '
                         f'{len(out)} -> {len(merged)}')
    return merged


def quote_coverage(table: pd.DataFrame) -> dict:
    """How many rows a real price actually reached.

    `market_windows` gates on exactly this, so it is worth reporting rather than
    inferring from a gate that failed.
    """
    if not len(table) or 'ask_up' not in table.columns:
        return {'rows': 0, 'quoted': 0, 'share': 0.0}
    quoted = int(table['ask_up'].notna().sum())
    return {'rows': int(len(table)), 'quoted': quoted,
            'share': quoted / max(len(table), 1)}
