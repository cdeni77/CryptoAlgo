"""The 15-minute window grid, and the barrier problem stated as a table.

A Kalshi crypto up/down market is not a direction bet. It opens on a
quarter-hour boundary, records the price there as its strike, and settles on
whether the price at the next boundary is strictly above it. By the time a
decision is being made, some of the window has already happened — so the
question is never "which way will it go" but:

    given that price has already moved `x` from the strike, and `n` minutes of
    movement remain, what is the chance it finishes above?

That is a barrier crossing, and its answer is `F(x / sigma_n)` for some
distribution `F`. The displacement `x` is *known exactly*. The only forecast
required is `sigma_n`. This module produces the table that makes that
statement testable: one row per (symbol, window, decision offset), carrying
the displacement, the excursion so far, the settle price and the outcome.

**Conventions, stated once because getting them wrong is silent.**

* A bar's `event_time` is the minute it *opens*. The bar covering
  `[10:03, 10:04)` has `event_time` 10:03 and is knowable at 10:04.
* The strike is `open` of the bar at `t0`, and the settle price is `open` of
  the bar at `t0 + 15`. Both are "the first trade at or after the boundary",
  which makes the window's return an open-to-open return — the only kind this
  project trusts. The previous incarnation of this repo anchored a target on
  `close(t)`, a last trade that could be twenty minutes stale, and 98% of the
  apparent edge turned out to be that staleness.
* A decision at offset `m` sees the close of the bar covering `[m-1, m)` and
  nothing after it. `decision_time = t0 + m` is therefore both the timestamp
  of the decision and the row's `available_time`.
* The outcome is `settle_price > strike`, strictly. A dead-flat window is a
  loss for the "up" side, which is how the venue resolves it.

**A window is dropped, never repaired, when its strike or settle minute has no
bar.** Interior gaps are forward-filled for the displacement — a minute with no
trades genuinely carries no new information, so the last trade is the correct
point-in-time price — but a missing boundary would have to be invented, and the
whole point of the strike is that it is the number the venue read.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Columns every window table carries. Ordered: identity, geometry, outcome,
# quality. `decision_time` is the point-in-time key — every feature joined onto
# this table must be computable from bars strictly before it.
WINDOW_COLUMNS = (
    'symbol', 'window_open', 'settle_time', 'offset', 'decision_time',
    'strike', 'last_price', 'displacement',
    'excursion_up', 'excursion_down',
    'settle_price', 'settle_return', 'outcome',
    'minutes_missing', 'complete',
)


class WindowError(ValueError):
    """The bar frame cannot be laid on a window grid."""


def floor_to_window(times: pd.DatetimeIndex, window_minutes: int) -> pd.DatetimeIndex:
    """The open time of the window each timestamp falls in."""
    return times.floor(f'{window_minutes}min')


def minute_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex a bar frame onto a gap-free minute grid.

    Missing minutes arrive as all-NaN rows rather than being absent, so the
    reshape below can assume a rectangular grid and gaps become a countable
    quality measure instead of a silent misalignment. A frame with a gap and a
    frame without one used to produce differently-shaped windows from the same
    slice of wall-clock time.
    """
    if bars.empty:
        raise WindowError('no bars')
    frame = bars.set_index('event_time').sort_index()
    if frame.index.has_duplicates:
        frame = frame[~frame.index.duplicated(keep='last')]
    grid = pd.date_range(frame.index[0], frame.index[-1], freq='1min', tz=frame.index.tz)
    return frame.reindex(grid)


@dataclass(frozen=True)
class GridReport:
    """What the grid looked like, so a caller can refuse a bad one."""

    symbol: str
    first_minute: pd.Timestamp
    last_minute: pd.Timestamp
    minutes_expected: int
    minutes_present: int
    windows_total: int
    windows_dropped_boundary: int
    windows_with_interior_gaps: int

    @property
    def minute_coverage(self) -> float:
        return self.minutes_present / self.minutes_expected if self.minutes_expected else float('nan')

    @property
    def boundary_drop_rate(self) -> float:
        return (self.windows_dropped_boundary / self.windows_total
                if self.windows_total else float('nan'))

    def summary(self) -> str:
        return (
            f"{self.symbol}: {self.minutes_present:,}/{self.minutes_expected:,} minutes "
            f"({self.minute_coverage:.4%}), {self.windows_total:,} windows, "
            f"{self.windows_dropped_boundary:,} dropped for a missing boundary "
            f"({self.boundary_drop_rate:.3%}), "
            f"{self.windows_with_interior_gaps:,} with interior gaps"
        )


def build_windows(
    bars: pd.DataFrame,
    symbol: str,
    config: Config = DEFAULT_CONFIG,
    *,
    offsets: Optional[Sequence[int]] = None,
) -> tuple[pd.DataFrame, GridReport]:
    """One row per (window, decision offset) for a single symbol's minute bars.

    Vectorised through a reshape rather than a loop: the minute grid is trimmed
    to whole windows, reshaped to `(n_windows, window_minutes)`, and every
    offset is then a column slice. On five years of one-minute bars that is
    seconds rather than an afternoon, and — more usefully — it makes the
    alignment a property of the array shape instead of an off-by-one hiding in
    an index expression.
    """
    window = int(config.window_minutes)
    offsets = tuple(int(o) for o in (offsets if offsets is not None else config.decision_offsets))
    if not offsets:
        raise WindowError('no decision offsets')
    bad = [o for o in offsets if not 1 <= o < window]
    if bad:
        raise WindowError(
            f"offsets must lie in [1, {window}) — a decision at 0 has seen no "
            f"bar and a decision at {window} is the settlement itself; got {bad}"
        )

    grid = minute_grid(bars)

    # Trim to a whole number of windows, starting on a boundary. One extra
    # minute is needed beyond the last window because the settle price is the
    # *next* window's opening bar.
    start = floor_to_window(grid.index[:1], window)[0]
    if start < grid.index[0]:
        start = start + pd.Timedelta(minutes=window)
    offset_into = int((start - grid.index[0]) / pd.Timedelta(minutes=1))
    usable = len(grid) - offset_into - 1
    n_windows = usable // window
    if n_windows < 1:
        raise WindowError(f"{symbol}: {len(grid)} minutes is under one full window")

    body = grid.iloc[offset_into: offset_into + n_windows * window]
    settle_rows = grid.iloc[offset_into + window: offset_into + n_windows * window + 1: window]

    def reshaped(column: str) -> np.ndarray:
        return body[column].to_numpy(dtype=float).reshape(n_windows, window)

    opens, highs, lows, closes = (reshaped(c) for c in ('open', 'high', 'low', 'close'))

    window_open = body.index[::window]
    strike = opens[:, 0]
    settle_price = settle_rows['open'].to_numpy(dtype=float)
    if settle_price.shape[0] != n_windows:  # defensive: the slice above must line up
        raise WindowError(
            f"{symbol}: {settle_price.shape[0]} settle prices for {n_windows} windows"
        )

    minutes_missing = np.isnan(closes).sum(axis=1)
    boundary_ok = np.isfinite(strike) & np.isfinite(settle_price)

    # Forward-fill closes *within* the window for the displacement. A minute
    # with no trades carries no new price, so the last trade is the correct
    # point-in-time value; inventing an open for a missing boundary is not the
    # same thing, which is why those windows are dropped instead.
    filled = pd.DataFrame(closes).ffill(axis=1).to_numpy()
    # A leading gap has nothing to fill from — fall back to the strike, which
    # is by construction the price at the window's first instant.
    filled = np.where(np.isnan(filled), strike[:, None], filled)

    settle_return = settle_price / strike - 1.0
    outcome = (settle_price > strike).astype(np.int8)

    frames = []
    for offset in offsets:
        last_price = filled[:, offset - 1]
        with np.errstate(invalid='ignore'):
            high_so_far = np.nanmax(highs[:, :offset], axis=1)
            low_so_far = np.nanmin(lows[:, :offset], axis=1)
        high_so_far = np.where(np.isnan(high_so_far), np.maximum(strike, last_price), high_so_far)
        low_so_far = np.where(np.isnan(low_so_far), np.minimum(strike, last_price), low_so_far)
        frames.append(pd.DataFrame({
            'symbol': symbol,
            'window_open': window_open,
            'settle_time': window_open + pd.Timedelta(minutes=window),
            'offset': np.int16(offset),
            'decision_time': window_open + pd.Timedelta(minutes=offset),
            'strike': strike,
            'last_price': last_price,
            'displacement': last_price / strike - 1.0,
            'excursion_up': high_so_far / strike - 1.0,
            'excursion_down': low_so_far / strike - 1.0,
            'settle_price': settle_price,
            'settle_return': settle_return,
            'outcome': outcome,
            'minutes_missing': minutes_missing.astype(np.int16),
            'complete': minutes_missing == 0,
        }))

    table = pd.concat(frames, ignore_index=True)
    # `boundary_ok` is exactly `strike` and `settle_price` both present, so the
    # filter is written on the columns rather than by mapping the mask back
    # through the window index — one expression, no searchsorted to get wrong.
    table = table.loc[table['strike'].notna() & table['settle_price'].notna()]
    table = table.sort_values(['window_open', 'symbol', 'offset'], ignore_index=True)

    report = GridReport(
        symbol=symbol,
        first_minute=grid.index[0],
        last_minute=grid.index[-1],
        minutes_expected=len(grid),
        minutes_present=int(grid['close'].notna().sum()),
        windows_total=n_windows,
        windows_dropped_boundary=int((~boundary_ok).sum()),
        windows_with_interior_gaps=int((minutes_missing[boundary_ok] > 0).sum()),
    )
    return table[list(WINDOW_COLUMNS)], report


def build_window_panel(
    bars_by_symbol: dict[str, pd.DataFrame],
    config: Config = DEFAULT_CONFIG,
    *,
    offsets: Optional[Sequence[int]] = None,
) -> tuple[pd.DataFrame, dict[str, GridReport]]:
    """`build_windows` across the universe, concatenated and sorted by decision time."""
    tables, reports = [], {}
    for symbol in sorted(bars_by_symbol):
        bars = bars_by_symbol[symbol]
        if bars is None or bars.empty:
            logger.warning('%s: no bars, skipped', symbol)
            continue
        table, report = build_windows(bars, symbol, config, offsets=offsets)
        tables.append(table)
        reports[symbol] = report
        logger.info(report.summary())
    if not tables:
        raise WindowError('no symbol produced any windows')
    panel = pd.concat(tables, ignore_index=True)
    panel = panel.sort_values(['decision_time', 'symbol', 'offset'], ignore_index=True)
    return panel, reports


def base_rate(panel: pd.DataFrame) -> float:
    """Fraction of windows that settle up.

    Expected to sit slightly below 0.5: a window that does not move is a loss
    for the up side, and a flat minute-to-minute grid produces exact ties more
    often than a continuous model would suggest. Any large departure from 0.5
    is a bug in the grid, not a market fact — check it before reading anything
    else in a report.
    """
    return float(panel.drop_duplicates(['symbol', 'window_open'])['outcome'].mean())
