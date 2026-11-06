"""Whether a resting order would have filled, from the venue's own tape.

The backtest books every order it decides to place. A resting order is booked
only when somebody crosses to it, and whether they do is not independent of
whether the order was right: measured on 5,489 orders this account actually
rested, fill probability rises from 48% to 99.8% with run-through depth while
the edge on those same orders falls from +18pp to -28pp per contract. Booking
all of them keeps the profitable half of a trade the venue never hands over.

`margin` is how far the most aggressive print ran BEYOND our limit, in cents.
It is a proxy for fill certainty, not fill certainty itself -- a deep
run-through means the level was consumed, so the queue cleared; a shallow touch
may never reach us. Queue position is not observable from the tape, which is
why this returns a probability and not a decision.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

# Fitted on orders this account rested from 2026-10-18, excluding tickets that
# never reached the book (API rejections, stale-quote skips): those are
# operational outcomes and judging a market model against them is a category
# error. Upper edge of each bucket in cents, then P(fill).
DEFAULT_FILL_CURVE: tuple[tuple[float, float], ...] = (
    (0.0, 0.483),
    (1.0, 0.706),
    (2.0, 0.870),
    (3.0, 0.964),
    (6.0, 0.969),
    (12.0, 0.992),
    (25.0, 0.998),
    (float('inf'), 0.998),
)

# Below this the order was never reached at all, so no queue argument applies.
NEVER_REACHED_MARGIN_CENTS = 0.0


@dataclass(frozen=True)
class FillModel:
    """P(fill) as a step function of run-through margin in cents."""

    curve: tuple[tuple[float, float], ...] = DEFAULT_FILL_CURVE

    def probability(self, margin_cents) -> np.ndarray:
        """P(fill) for each margin. NaN margin returns NaN, never a default.

        A missing margin means the tape did not cover that market, which is a
        coverage problem. Substituting a rate would hide it behind a number.
        """
        m = np.asarray(margin_cents, dtype=float)
        out = np.full(m.shape, np.nan, dtype=float)
        lo = -np.inf
        for hi, p in self.curve:
            band = np.isfinite(m) & (m > lo) & (m <= hi)
            out[band] = p
            lo = hi
        return out


def run_through_margin(
    side: Sequence[str],
    limit_price: Sequence[float],
    sell_low: Sequence[float],
    buy_high: Sequence[float],
) -> np.ndarray:
    """Cents the aggressor ran beyond our limit. Positive means it reached us.

    **`taker_book_side='ask'` is the SELLING aggressor and prints at the bid**,
    so its `min_price` is what fills a resting YES bid -- the opposite of the
    natural reading of the label. Buying NO at q rests a YES OFFER at 1-q,
    which a BUYING aggressor (`taker_book_side='bid'`, printing at the ask)
    lifts. Swapping these inverts every result and looks plausible doing it.
    """
    s = pd.Series(side).astype(str).str.lower().to_numpy()
    p = np.asarray(limit_price, dtype=float)
    low = np.asarray(sell_low, dtype=float)
    high = np.asarray(buy_high, dtype=float)
    up = s == 'up'
    return 100.0 * np.where(up, p - low, high - (1.0 - p))


def expected_fill(
    frame: pd.DataFrame,
    model: Optional[FillModel] = None,
    *,
    side: str = 'side',
    limit: str = 'limit_price',
    sell_low: str = 'sell_low',
    buy_high: str = 'buy_high',
) -> pd.Series:
    """P(fill) per row, for weighting a backtest's bookings.

    Returns NaN where the tape does not cover the row. The caller decides what
    an uncovered row means; this does not quietly call it a fill.
    """
    model = model or FillModel()
    margin = run_through_margin(frame[side], frame[limit],
                                frame[sell_low], frame[buy_high])
    return pd.Series(model.probability(margin), index=frame.index, name='p_fill')


# How long a resting order really lives, in whole minutes of tape. The config
# asks for 60s but the sweep has been observed cancelling at ~180s, and
# `backfill_trades._rows` keeps only min/max per (minute, side) -- per-print
# timestamps are discarded, so a 60s rest cannot be separated from its minute.
# Widening this cannot be corrected for later; it is why the model stays
# optimistic by ~0.75pp/ct out of sample.
TAPE_REST_MINUTES = 3

_MAX_WINDOW_OFFSET = 15


def attach_tape_extremes(
    windows: pd.DataFrame,
    tape: pd.DataFrame,
    *,
    rest_minutes: int = TAPE_REST_MINUTES,
) -> pd.DataFrame:
    """Add `sell_low` and `buy_high`: the aggressor extremes while an order rests.

    For a row deciding at offset `o`, these cover offsets `o .. o+rest_minutes`
    of the SAME (symbol, window). Nothing before `o` is included: a print that
    happened before the order existed cannot have filled it, and folding it in
    is lookahead that reads as a fill.

    Rows the tape does not cover keep NaN, so `expected_fill` returns NaN and
    the caller has to decide what an uncovered row means.
    """
    need = {'symbol', 'window_open', 'offset_minutes', 'taker_book_side'}
    missing = need - set(tape.columns)
    if missing or not len(tape):
        logger_missing = ', '.join(sorted(missing)) or 'no rows'
        import logging
        logging.getLogger(__name__).warning(
            'venue_trades cannot be joined (%s); fills stay unsimulated',
            logger_missing)
        out = windows.copy()
        out['sell_low'] = np.nan
        out['buy_high'] = np.nan
        return out

    t = tape.copy()
    t['window_open'] = pd.to_datetime(t['window_open'], utc=True)
    t['offset_minutes'] = pd.to_numeric(t['offset_minutes'], errors='coerce')
    t = t[t['offset_minutes'].between(0, _MAX_WINDOW_OFFSET)]

    sell = (t[t['taker_book_side'] == 'ask']
            .groupby(['symbol', 'window_open', 'offset_minutes'])['min_price'].min())
    buy = (t[t['taker_book_side'] == 'bid']
           .groupby(['symbol', 'window_open', 'offset_minutes'])['max_price'].max())
    grid = pd.DataFrame({'sell_low': sell, 'buy_high': buy})
    if grid.empty:
        out = windows.copy()
        out['sell_low'] = np.nan
        out['buy_high'] = np.nan
        return out

    # Reindex every (symbol, window) onto the full minute grid so the forward
    # look is over a fixed span rather than over however many minutes happened
    # to trade -- otherwise a quiet market silently gets a longer horizon.
    keys = grid.index.droplevel('offset_minutes').unique()
    full = pd.MultiIndex.from_tuples(
        [(s, w, float(o)) for s, w in keys for o in range(_MAX_WINDOW_OFFSET + 1)],
        names=['symbol', 'window_open', 'offset_minutes'])
    grid = grid.reindex(full)

    span = int(rest_minutes) + 1
    g = grid.groupby(level=['symbol', 'window_open'], sort=False)
    # Reverse the minute order so a trailing window becomes a forward one.
    fwd_low = (g['sell_low'].apply(lambda s: s[::-1].rolling(span, min_periods=1).min()[::-1])
               .reset_index(level=0, drop=True) if False else None)
    rev = grid[::-1]
    rolled = (rev.groupby(level=['symbol', 'window_open'], sort=False)
              .rolling(span, min_periods=1).agg({'sell_low': 'min', 'buy_high': 'max'}))
    rolled.index = rolled.index.droplevel([0, 1])
    rolled = rolled[::-1]

    out = windows.copy()
    out['window_open'] = pd.to_datetime(out['window_open'], utc=True)
    key = pd.MultiIndex.from_arrays(
        [out['symbol'], out['window_open'],
         pd.to_numeric(out['offset_minutes'], errors='coerce').astype(float)])
    out['sell_low'] = rolled['sell_low'].reindex(key).to_numpy()
    out['buy_high'] = rolled['buy_high'].reindex(key).to_numpy()
    return out


def _uniform_for(symbol: str, window_open, offset) -> float:
    """A stable uniform in [0,1) for one decision.

    Keyed on the decision's own identity, NOT on a running RNG: a global
    generator makes the result depend on how many rows came before, so the same
    backtest scores differently when an unrelated fold changes. Deterministic
    also means a rerun reproduces the same fills, which a money figure needs.
    """
    import hashlib
    key = f'{symbol}|{pd.Timestamp(window_open).isoformat()}|{int(offset)}'
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], 'big') / float(1 << 64)


def would_fill(
    symbol: str,
    window_open,
    offset,
    side: str,
    limit_price: float,
    sell_low: float,
    buy_high: float,
    model: Optional[FillModel] = None,
) -> Optional[bool]:
    """Whether this resting order fills. None when the tape does not cover it.

    None is not False: an uncovered row is a gap in the data, and silently
    calling it a miss would quietly delete trades from the backtest.
    """
    margin = run_through_margin([side], [limit_price], [sell_low], [buy_high])
    p = (model or FillModel()).probability(margin)[0]
    if not np.isfinite(p):
        return None
    return _uniform_for(symbol, window_open, offset) < float(p)
