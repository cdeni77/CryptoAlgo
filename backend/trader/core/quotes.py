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

from typing import Optional

import numpy as np
import pandas as pd

QUOTE_COLUMNS = ('ask_up', 'ask_down', 'quote_age_seconds', 'quote_source')
# A book quoted twenty minutes ago is not a price this decision could have
# taken. Generous by default because `quote_age_seconds` is carried alongside,
# so a study can tighten it without re-joining.
DEFAULT_MAX_AGE = 900.0


def attach_quotes(windows: pd.DataFrame, depth: pd.DataFrame, *,
                  venue: str = 'kalshi',
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
    book['quote_age_seconds'] = age.where(keep)
    book['quote_source'] = np.where(keep, venue, None)

    # One row per (symbol, window, offset). `venue_depth` can hold the same
    # minute from more than one observer — that is what `source` is for — so
    # collapse before joining rather than fanning the window table out.
    book = (book.sort_values('quote_age_seconds', na_position='last')
                .drop_duplicates(['symbol', 'window_open', 'offset_minutes'],
                                 keep='first'))

    merged = out.drop(columns=list(QUOTE_COLUMNS)).merge(
        book[['symbol', 'window_open', 'offset_minutes',
              'ask_up', 'ask_down', 'quote_age_seconds', 'quote_source']],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'],
        how='left')
    merged = merged.drop(columns=['offset_minutes'])
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
