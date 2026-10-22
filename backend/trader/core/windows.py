"""The 15-minute window grid: one row per (symbol, window, decision offset).

Each row carries the displacement from the strike, the excursion so far, the
settle price and the outcome. The barrier question is: given displacement `x`
and `n` minutes left, what is P(finish at or above the strike)? The answer is
`F(x / sigma_n)`; `x` is known exactly, so only `sigma_n` needs a forecast.

Venue conventions (from the market's own rules):

* Both ends are one-minute averages. The strike is the mean over `[t0 - 1min, t0)`
  and the settlement value is the mean over `[t1 - 1min, t1)`.
* A window's strike is the previous window's settlement value, so markets chain.
* A tie resolves up: `strike_type` is `greater_or_equal`, so the test is `>=`.
* The settlement index is CF Benchmarks BRTI. Coinbase bars are a close proxy,
  not the same number; disagreement concentrates in near-ties.

Our conventions:

* A bar's `event_time` is the minute it opens. The bar `[10:03, 10:04)` is knowable at 10:04.
* A decision at offset `m` sees the close of bar `[m-1, m)` and nothing after.
  `decision_time = t0 + m` is also the row's `available_time`.

A window whose strike or settlement minute has no bar is dropped, never repaired.
Interior gaps are forward-filled: a minute with no trades carries no new price.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core.config import Config, DEFAULT_CONFIG

logger = logging.getLogger(__name__)

# Columns every window table carries: identity, geometry, outcome, quality.
# `decision_time` is the point-in-time key: every joined feature must be
# computable from bars strictly before it.
WINDOW_COLUMNS = (
    'symbol', 'window_open', 'settle_time', 'offset', 'decision_time',
    # `strike_source` says whether this row was built from the venue's own
    # `floor_strike` or from our Coinbase OHLC mean. A mixed sample that
    # cannot distinguish them reads as one measurement, which is the defect
    # `price_source` exists to prevent on the quote side.
    'strike', 'strike_source', 'last_price', 'displacement',
    'excursion_up', 'excursion_down',
    'settle_price', 'settle_return', 'outcome',
    'minutes_missing', 'complete',
)


def bar_mean(frame: pd.DataFrame) -> pd.Series:
    """The average price within each bar.

    **`mean_price` where the source has it, `(O+H+L+C)/4` otherwise.**

    The venue settles on "the simple average of the sixty seconds" of its
    index. `(O+H+L+C)/4` is a proxy for that, tolerable because both ends of
    the target use it and most of the bias cancels. But the CF Benchmarks
    feed publishes the real quantity -- `avg_60s_data`, the venue computing
    its own 60-second mean -- and `scripts/record_index_stream` and
    `scripts/backfill_index` both store it as `mean_price`.

    Using the proxy when the actual number is in the next column would throw
    away the whole reason that feed was built, and it would do it silently:
    the strike and the settlement value are both this function, so a 1bp
    error enters the target at both ends.

    A source without the column falls back row by row, so Coinbase bars are
    unchanged and a partially covered span is not dropped.
    """
    ohlc = (frame['open'] + frame['high'] + frame['low'] + frame['close']) / 4.0
    if 'mean_price' not in frame.columns:
        return ohlc
    exact = pd.to_numeric(frame['mean_price'], errors='coerce')
    return exact.where(np.isfinite(exact), ohlc)


class WindowError(ValueError):
    """The bar frame cannot be laid on a window grid."""


def floor_to_window(times: pd.DatetimeIndex, window_minutes: int) -> pd.DatetimeIndex:
    """The open time of the window each timestamp falls in."""
    return times.floor(f'{window_minutes}min')


def minute_grid(bars: pd.DataFrame) -> pd.DataFrame:
    """Reindex a bar frame onto a gap-free minute grid.

    Missing minutes become all-NaN rows, so the reshape sees a rectangle and
    gaps are countable.
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


def coverage_log_level(report: 'GridReport') -> int:
    """How loudly to report a grid: DEBUG when it is exactly as expected.

    Missing minutes, interior gaps or more than one dropped boundary stay at INFO.
    """
    import logging

    if report.minutes_present < report.minutes_expected:
        return logging.INFO
    # One dropped boundary is the normal live state: the window in progress
    # has no settlement minute yet.
    if report.windows_dropped_boundary > 1:
        return logging.INFO
    if report.windows_with_interior_gaps > 0:
        return logging.INFO
    return logging.DEBUG


def build_windows(
    bars: pd.DataFrame,
    symbol: str,
    config: Config = DEFAULT_CONFIG,
    *,
    offsets: Optional[Sequence[int]] = None,
    include_unsettled: bool = False,
    strike_override: Optional[pd.Series] = None,
) -> tuple[pd.DataFrame, GridReport]:
    """One row per (window, decision offset) for a single symbol's minute bars.

    The minute grid is trimmed to whole windows and reshaped to
    `(n_windows, window_minutes)`, so every offset is a column slice.

    `include_unsettled=True` pads the grid so the window in progress survives.
    Its strike is real; `settle_price`, `settle_return` and `outcome` are NaN.
    Live and backtest share this code path, so their features are identical.

    A row is withheld when the feed does not reach `decision_time`: a
    forward-fill there would invent a `last_price`.
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

    # Trim to a whole number of windows, starting on a boundary. The settlement
    # average is over the window's own last minute, so no minute past the end is needed.
    start = floor_to_window(grid.index[:1], window)[0]
    if start < grid.index[0]:
        start = start + pd.Timedelta(minutes=window)
    offset_into = int((start - grid.index[0]) / pd.Timedelta(minutes=1))

    # Minutes of the trailing window that have not happened yet. Padded with
    # all-NaN rows so the reshape sees a rectangle.
    real_minutes = len(grid)
    pad_minutes = 0
    if include_unsettled:
        remainder = (len(grid) - offset_into) % window
        if remainder:
            pad_minutes = window - remainder
            extra = pd.date_range(grid.index[-1] + pd.Timedelta(minutes=1),
                                  periods=pad_minutes, freq='1min', tz=grid.index.tz)
            grid = grid.reindex(grid.index.append(extra))

    n_windows = (len(grid) - offset_into) // window
    if n_windows < 2:
        raise WindowError(
            f"{symbol}: {len(grid)} minutes is under two full windows, and the "
            f"first window has no strike — its strike is the previous window's "
            f"settlement average, which does not exist yet"
        )

    body = grid.iloc[offset_into: offset_into + n_windows * window]
    means = bar_mean(body).to_numpy(dtype=float).reshape(n_windows, window)

    def reshaped(column: str) -> np.ndarray:
        return body[column].to_numpy(dtype=float).reshape(n_windows, window)

    highs, lows, closes = (reshaped(c) for c in ('high', 'low', 'close'))

    # The settlement value is the mean over the window's last minute, and the
    # strike is the same quantity for the window before: one array, shifted.
    # The first window has no predecessor and is dropped.
    settle_all = means[:, window - 1]
    window_open = body.index[::window][1:]
    strike = settle_all[:-1]
    settle_price = settle_all[1:]

    # **The venue's own strike replaces ours HERE, before anything derives
    # from it.** Every strike-derived quantity -- the displacement, the
    # z-score, both excursions, the path statistics and the outcome -- is
    # computed below from `strike`, so overriding it at this one point moves
    # all of them together. `scripts/live.py` records what happens otherwise:
    # an earlier version overwrote the strike and recomputed only `z_score`,
    # so `abs_z_score == |z_score|` -- an identity that holds in every
    # training row -- was violated at inference on every cycle, invisibly.
    strike_is_venue = np.zeros(len(strike), dtype=bool)
    if strike_override is not None and len(strike_override):
        want = pd.Series(np.asarray(strike, dtype=float), index=window_open)
        have = pd.to_numeric(strike_override, errors='coerce')
        have = have[~have.index.duplicated(keep='last')].reindex(want.index)
        use = have.notna().to_numpy()
        strike = np.where(use, have.to_numpy(dtype=float), want.to_numpy())
        strike_is_venue = use
    highs, lows, closes = highs[1:], lows[1:], closes[1:]
    means = means[1:]
    n_windows -= 1

    minutes_missing = np.isnan(closes).sum(axis=1)
    boundary_ok = np.isfinite(strike) & np.isfinite(settle_price)

    # Forward-fill closes within the window for the displacement: a minute with
    # no trades carries no new price. Missing boundaries are dropped instead.
    filled = pd.DataFrame(closes).ffill(axis=1).to_numpy()
    # A leading gap has nothing to fill from: use the strike, which is the
    # price at the window's first instant.
    filled = np.where(np.isnan(filled), strike[:, None], filled)

    settle_return = settle_price / strike - 1.0
    # `>=`, not `>`: the venue's `strike_type` is `greater_or_equal`.
    settled = np.isfinite(settle_price)
    if include_unsettled:
        # Float, so an undecided window stays NaN. `(nan >= x)` is False,
        # which would file every in-progress window as a loss.
        outcome = np.where(settled, settle_price >= strike, np.nan)
    else:
        outcome = (settle_price >= strike).astype(np.int8)

    frames = []
    for offset in offsets:
        last_price = filled[:, offset - 1]
        # `errstate` does not silence numpy's all-NaN-slice RuntimeWarning,
        # so `warnings` suppresses it too.
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            high_so_far = np.nanmax(highs[:, :offset], axis=1)
            low_so_far = np.nanmin(lows[:, :offset], axis=1)
        high_so_far = np.where(np.isnan(high_so_far), np.maximum(strike, last_price), high_so_far)
        low_so_far = np.where(np.isnan(low_so_far), np.minimum(strike, last_price), low_so_far)

        missing = minutes_missing.copy()
        if include_unsettled and pad_minutes and n_windows:
            # The trailing window's later minutes have not happened. Count only
            # gaps the decision could have seen.
            missing[-1] = int(np.isnan(closes[-1, :offset]).sum())
        frames.append(pd.DataFrame({
            'symbol': symbol,
            'window_open': window_open,
            'settle_time': window_open + pd.Timedelta(minutes=window),
            'offset': np.int16(offset),
            'decision_time': window_open + pd.Timedelta(minutes=offset),
            'strike': strike,
            # Which strike this row used. A row that cannot say whether it was
            # graded against the venue's number or ours is the same defect as
            # a quote that cannot say whether it was a fill or a backtest.
            'strike_source': np.where(strike_is_venue, 'venue', 'proxy'),
            'last_price': last_price,
            'displacement': last_price / strike - 1.0,
            'excursion_up': high_so_far / strike - 1.0,
            'excursion_down': low_so_far / strike - 1.0,
            'settle_price': settle_price,
            'settle_return': settle_return,
            'outcome': outcome,
            'minutes_missing': missing.astype(np.int16),
            'complete': missing == 0,
        }))

    table = pd.concat(frames, ignore_index=True)
    # `strike` is always required: it is the previous window's settlement average.
    # `settle_price` is required only for settled windows; a live window has none yet.
    keep = table['strike'].notna()
    if not include_unsettled:
        keep = keep & table['settle_price'].notna()
    else:
        # The trailing window's real bars stop at `window - pad_minutes`. A
        # decision past that would read a forward-filled price from a feed that
        # has not caught up, so those rows are withheld.
        if pad_minutes:
            last_real = window - pad_minutes
            stale = (
                (table['window_open'] == window_open[-1])
                & (table['offset'] > last_real)
            )
            if stale.any():
                logger.warning(
                    '%s: withholding %d row(s) for the window opening %s — the '
                    'feed reaches minute %d of %d, so offsets past +%dm would be '
                    'decided on a forward-filled price rather than a traded one',
                    symbol, int(stale.sum()), window_open[-1], last_real, window,
                    last_real)
            keep = keep & ~stale
    table = table.loc[keep]
    table = table.sort_values(['window_open', 'symbol', 'offset'], ignore_index=True)

    report = GridReport(
        symbol=symbol,
        first_minute=grid.index[0],
        last_minute=grid.index[-1],
        minutes_expected=real_minutes,
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
    include_unsettled: bool = False,
    strikes_by_symbol: Optional[dict[str, pd.Series]] = None,
) -> tuple[pd.DataFrame, dict[str, GridReport]]:
    """`build_windows` across the universe, concatenated and sorted by decision time.

    `include_unsettled` passes through. Live needs it; the backtest must never
    use it, because an unsettled row has a NaN label.
    """
    tables, reports = [], {}
    for symbol in sorted(bars_by_symbol):
        bars = bars_by_symbol[symbol]
        if bars is None or bars.empty:
            logger.warning('%s: no bars, skipped', symbol)
            continue
        table, report = build_windows(bars, symbol, config, offsets=offsets,
                                      strike_override=(strikes_by_symbol or {}).get(symbol),
                                      include_unsettled=include_unsettled)
        tables.append(table)
        reports[symbol] = report
        logger.log(coverage_log_level(report), report.summary())
    if not tables:
        raise WindowError('no symbol produced any windows')
    panel = pd.concat(tables, ignore_index=True)
    panel = panel.sort_values(['decision_time', 'symbol', 'offset'], ignore_index=True)
    return panel, reports


def base_rate(panel: pd.DataFrame) -> float:
    """Fraction of windows that settle up.

    Expect slightly above 0.5, because a tie pays the up side. A large departure
    from 0.5 is a bug in the grid.
    """
    return float(panel.drop_duplicates(['symbol', 'window_open'])['outcome'].mean())
