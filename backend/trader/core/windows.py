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

**The conventions are the venue's, read off a live market's own rules.** They
are not a modelling choice, and three of them are not what a reasonable person
would guess:

> If the simple average of the sixty seconds of CF Benchmarks' BRTI before
> 12:45 AM EDT on Aug 23, 2026 is **at least** the simple average of the sixty
> seconds of CF Benchmarks' BRTI before 12:30 AM EDT, then the market resolves
> to Yes.

* **Both ends are one-minute averages, not point prices.** The strike is the mean
  over `[t0 - 1min, t0)` and the settlement value is the mean over
  `[t1 - 1min, t1)`. An earlier version of this module used `open(t0)` and
  `open(t1)`, which is a different quantity and a noisier one.
* **The strike of a window is the settlement value of the window before it.** Both
  are the mean over the same minute — the one ending at `t0`. Consecutive markets
  therefore chain, which is a real structural dependence and one more reason the
  cross-validation embargo is a day rather than a window.
* **A tie resolves UP.** `strike_type` is `greater_or_equal`, so the comparison is
  `>=`. A strict `>` hands every dead-flat window to the down side — and a tie
  is not the common case a "minute grid" framing suggests: both ends are
  one-minute OHLC means of a liquid asset, so an exact tie is measure-zero,
  1 in 173,937 real BTC windows. `>=` is right because it matches the venue's
  `strike_type`, not because ties are frequent enough to matter on their own.
* The settlement index is **CF Benchmarks BRTI**, not Coinbase spot. This module
  builds the target from Coinbase bars because that is the history available;
  Coinbase is a large BRTI constituent, so it is a close proxy and not the same
  number. That basis is now MEASURED rather than an open risk: `venue_settlements`
  holds the venue's own settlement on 56,385 Kalshi markets, and the Coinbase
  label agrees with it on 96.98% of shared windows, with essentially all of the
  disagreement concentrated in near-ties (see `CLAUDE.md`).

Two conventions that are ours and unchanged:

* A bar's `event_time` is the minute it *opens*. The bar covering
  `[10:03, 10:04)` has `event_time` 10:03 and is knowable at 10:04.
* A decision at offset `m` sees the close of the bar covering `[m-1, m)` and
  nothing after it. `decision_time = t0 + m` is both the timestamp of the
  decision and the row's `available_time`.

**A window is dropped, never repaired, when the minute either average is taken
over has no bar.** Interior gaps are forward-filled for the displacement — a
minute with no trades genuinely carries no new information, so the last trade is
the correct point-in-time price — but a missing average would have to be
invented, and the whole point of the strike is that it is the number the venue
read.
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


def bar_mean(frame: pd.DataFrame) -> pd.Series:
    """The average price within each bar: `(O + H + L + C) / 4`.

    Standing in for "the simple average of the sixty seconds", which one-minute
    bars cannot reproduce exactly. The OHLC mean is the usual proxy and is
    materially better than the close alone, which is one of the sixty
    observations rather than a summary of them.

    Two alternatives were considered and are worth naming: `(H + L) / 2` ignores
    where the bar spent its time, and `close` is what an earlier version of this
    module effectively used and is the noisiest of the three. Any of them biases
    the strike and the settlement value *identically* — both ends use this same
    function — so the difference largely cancels in the comparison, which is why
    the approximation is tolerable at all.
    """
    return (frame['open'] + frame['high'] + frame['low'] + frame['close']) / 4.0


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


def coverage_log_level(report: 'GridReport') -> int:
    """How loudly to report a grid: DEBUG when it is exactly as expected.

    The live loop rebuilds this grid twice a minute and the line read
    "1,499/1,499 minutes (100.0000%), 99 windows, 0 dropped" every single time.
    Repeating an unchanging fact around the clock buries the lines that matter,
    and this log is the only view of an account trading unattended.

    A shortfall in minutes or a dropped boundary is different: a dropped boundary
    is a window that cannot be scored at all, and missing minutes are how a stale
    feed announces itself. Those stay visible.
    """
    import logging

    if report.minutes_present < report.minutes_expected:
        return logging.INFO
    # **One dropped boundary is the permanent live steady state**, not an
    # anomaly. The window currently in progress has no settlement minute yet, so
    # it is dropped on every single cycle, forever. The first version of this
    # rule tested `> 0` and therefore kept the noisiest line in the log at INFO —
    # which was the whole thing it was written to stop.
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
) -> tuple[pd.DataFrame, GridReport]:
    """One row per (window, decision offset) for a single symbol's minute bars.

    Vectorised through a reshape rather than a loop: the minute grid is trimmed
    to whole windows, reshaped to `(n_windows, window_minutes)`, and every
    offset is then a column slice. On five years of one-minute bars that is
    seconds rather than an afternoon, and — more usefully — it makes the
    alignment a property of the array shape instead of an off-by-one hiding in
    an index expression.

    **`include_unsettled` is what makes live scoring possible at all.** The trim
    to whole windows (`// window`) and the settlement read at `means[:, window-1]`
    together mean the window *currently in progress* is absent twice over: it has
    neither a full complement of minutes nor a settlement minute. So
    `core/dataset.py:score_live` asked for the window it was deciding, got an
    empty slice, and raised `DatasetError` on every cycle — at offsets 3, 6, 9
    and 12 alike. The window only became scoreable once it had already settled.
    `scripts/live.py` did not catch it and its loop catches only
    `KeyboardInterrupt`, so paper and live trading crash-looped and had never
    completed a single window.

    With `include_unsettled=True` the grid is padded to a whole window so the
    trailing partial window survives the reshape. Its `strike` is real — that is
    the *previous* window's minute-14 mean, which has happened — while
    `settle_price`, `settle_return` and `outcome` come back NaN, because they have
    not. Everything the decision needs (`last_price`, `displacement`, the
    excursions) reads bars strictly before `decision_time` exactly as before, so
    the arithmetic is shared with the backtest rather than reimplemented. That
    sharing is why the two paths were measured bit-identical to 16 decimal
    places, and it is the reason this is a flag on `build_windows` rather than a
    second window builder.

    A row is withheld when the feed does not actually reach `decision_time`. The
    interior forward-fill is the right answer for a minute in which nothing
    traded, but it cannot distinguish that from a minute that has not been
    fetched yet — and ffilling a stale feed would invent a `last_price` and hand
    it to the barrier as fact.
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

    # Trim to a whole number of windows, starting on a boundary. No extra minute
    # is needed past the end: the settlement average is taken over the window's
    # *own* last minute, so it is inside the body.
    start = floor_to_window(grid.index[:1], window)[0]
    if start < grid.index[0]:
        start = start + pd.Timedelta(minutes=window)
    offset_into = int((start - grid.index[0]) / pd.Timedelta(minutes=1))

    # How many minutes of the trailing window are missing because it has not
    # happened yet. Padded with all-NaN rows so the reshape below still sees a
    # rectangle; `pad_minutes` is then how far back the real bars stop.
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

    # The settlement value is the mean over the last minute of the window, and
    # the strike is that same quantity for the window before — so one array,
    # shifted. The first window is dropped for want of a predecessor.
    settle_all = means[:, window - 1]
    window_open = body.index[::window][1:]
    strike = settle_all[:-1]
    settle_price = settle_all[1:]
    highs, lows, closes = highs[1:], lows[1:], closes[1:]
    means = means[1:]
    n_windows -= 1

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
    # `>=`, not `>`. The venue's `strike_type` is `greater_or_equal`, so a window
    # that ends exactly where it started pays the up side. On a minute grid exact
    # ties are not rare, and a strict comparison hands every one of them to the
    # wrong side.
    settled = np.isfinite(settle_price)
    if include_unsettled:
        # Float, so an undecided window can say so. `(nan >= x)` is False, which
        # would file every in-progress window as a loss.
        outcome = np.where(settled, settle_price >= strike, np.nan)
    else:
        outcome = (settle_price >= strike).astype(np.int8)

    frames = []
    for offset in offsets:
        last_price = filled[:, offset - 1]
        # `errstate` does not silence numpy's all-NaN-slice RuntimeWarning, which
        # is a different mechanism (`warnings`, not the floating-point error
        # state). Both are suppressed here so the intent in the original guard
        # actually holds.
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            high_so_far = np.nanmax(highs[:, :offset], axis=1)
            low_so_far = np.nanmin(lows[:, :offset], axis=1)
        high_so_far = np.where(np.isnan(high_so_far), np.maximum(strike, last_price), high_so_far)
        low_so_far = np.where(np.isnan(low_so_far), np.minimum(strike, last_price), low_so_far)

        missing = minutes_missing.copy()
        if include_unsettled and pad_minutes and n_windows:
            # The trailing window's later minutes have not happened; counting them
            # as gaps would say the data is bad rather than young. Count only what
            # the decision could have seen.
            missing[-1] = int(np.isnan(closes[-1, :offset]).sum())
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
            'minutes_missing': missing.astype(np.int16),
            'complete': missing == 0,
        }))

    table = pd.concat(frames, ignore_index=True)
    # `boundary_ok` is exactly `strike` and `settle_price` both present, so the
    # filter is written on the columns rather than by mapping the mask back
    # through the window index — one expression, no searchsorted to get wrong.
    # `strike` is always required: it is the previous window's settlement average
    # and inventing one would defeat the point of a barrier. `settle_price` is
    # required only when the caller wants settled windows — a window being
    # decided has none yet, and that is the normal case live.
    keep = table['strike'].notna()
    if not include_unsettled:
        keep = keep & table['settle_price'].notna()
    else:
        # The trailing window's real bars stop at index `window - pad_minutes`. A
        # decision at offset `m` reads index `m - 1`, so anything at or past the
        # pad boundary is being forward-filled out of a feed that simply has not
        # caught up. `filled` would hand the barrier a fabricated `last_price`
        # and nothing downstream could tell.
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
) -> tuple[pd.DataFrame, dict[str, GridReport]]:
    """`build_windows` across the universe, concatenated and sorted by decision time.

    `include_unsettled` is passed straight through; see `build_windows`. The live
    path needs it and the backtest must never have it, because an unsettled row
    carries a NaN label.
    """
    tables, reports = [], {}
    for symbol in sorted(bars_by_symbol):
        bars = bars_by_symbol[symbol]
        if bars is None or bars.empty:
            logger.warning('%s: no bars, skipped', symbol)
            continue
        table, report = build_windows(bars, symbol, config, offsets=offsets,
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

    Expected to sit slightly ABOVE 0.5, not below: `strike_type` is
    `greater_or_equal`, so a tie pays the up side rather than losing it. This
    docstring used to claim the opposite — "expected to sit slightly below
    0.5... a window that does not move is a loss for the up side" — which
    described the superseded strict-`>` behaviour and would have flagged
    correct output as a bug. Measured on real data: 0.5009 (BTC), 0.5031
    (ETH), both above 0.5, exactly as `>=` implies. Ties are also not the
    common case a "minute grid" framing suggests — both ends of the target are
    one-minute OHLC means of a liquid asset, so an exact tie is measure-zero:
    1 in 173,937 real BTC windows. Any large departure from 0.5 is still a bug
    in the grid, not a market fact — check it before reading anything else in
    a report.
    """
    return float(panel.drop_duplicates(['symbol', 'window_open'])['outcome'].mean())
