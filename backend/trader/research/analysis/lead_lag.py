"""Does the Kalshi price lag spot, or Polymarket, at the seconds scale?

The live audit found the model beats `F(x/sigma)` at every offset and does not
beat the Kalshi price at any. One explanation is that the price already
contains whatever the spot features see, within seconds. This measures that
directly, with no model in the way.

Method, per symbol and per market (one 15-minute window):

* **Kalshi** — the raw `venue_book_events` frames replayed through the same
  `BookCache` the live path folds, sampled on a 1-second grid. The mid is turned
  into an implied barrier z, `Phi^-1(mid)`.
* **Spot** — `spot_trades` (symbol, trade_id, time, price, size), last price per
  second. Its z change over one second is the log return divided by the
  remaining sigma, `sigma_1s * sqrt(remaining variance seconds)`, with `sigma_1s`
  a trailing 15-minute realised volatility. That is `dx / sigma_n`, the change
  in the barrier's own argument. Without a trades table it falls back to
  one-minute bars, which move once a minute and CANNOT resolve a seconds-scale
  lead; the report says so.
* **Polymarket** — `pm_ladder` mids, as Kalshi's. The archive samples that venue
  about once a minute (the stream is not archived), so the same caveat applies
  and the report states the median sampling interval.

Then cross-correlate the 1-second z changes inside each market at lags
-60..+60s, pooled across markets and bucketed by minutes remaining.

**Sign convention: a POSITIVE lag means the other series LEADS Kalshi** —
corr(dz_kalshi(t), dz_other(t - lag)). A peak at +5 says Kalshi moves five
seconds after spot.

`lead_share` is an information-share-style summary: the squared correlation at
positive lags over that at all non-zero lags. 0.5 is symmetric; near 1 the other
series leads. `delayed_share` is the part of Kalshi's response to the other
series that arrives AFTER the same second, from the positive correlations at
lags >= 0 — the fraction a decision taken on the stale price would miss.

    python research/analysis/lead_lag.py --start 2026-10-12 --end 2026-10-13
    python research/analysis/lead_lag.py --symbols BTC-USD --max-lag 30
"""

from __future__ import annotations

# `core` is rooted at backend/trader/, which Python does not add to sys.path for
# a script run from a subdirectory.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import norm

from core.config import DEFAULT_CONFIG

logger = logging.getLogger('lead_lag')

WINDOW_SECONDS = DEFAULT_CONFIG.window_minutes * 60
SIGMA_LOOKBACK_SECONDS = 900
MID_CLIP = 0.02
PROFILE_LAGS = (-60, -30, -10, -5, -3, -2, -1, 0, 1, 2, 3, 5, 10, 30, 60)


# ---- series ---------------------------------------------------------------

def _seconds(values) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.floor('s') \
        if isinstance(values, pd.Series) else pd.to_datetime(values, utc=True).floor('s')


def kalshi_mid_series(events: pd.DataFrame) -> pd.DataFrame:
    """Replay `venue_book_events` rows into a mid per market per second.

    Frames are regrouped on (market_ticker, event_time, seq) — `core/spool`
    explains why seq alone is not unique — and folded through `BookCache`, the
    one applier. Sequence checking is skipped (seq=None): a replay of a subset
    of markets is not a connection, and every market would read as gapped.
    Returns symbol, market_ticker, time (whole seconds), mid; one-sided seconds
    are left out rather than guessed.
    """
    from core.stream_book import BookCache
    from data_collection.stream.base import BookEvent

    if events is None or not len(events):
        return pd.DataFrame(columns=['symbol', 'market_ticker', 'time', 'mid'])
    frame = events.copy()
    frame['event_time'] = pd.to_datetime(frame['event_time'], utc=True)
    frame['seq'] = pd.to_numeric(frame.get('seq'), errors='coerce')
    frame = frame.sort_values(['market_ticker', 'event_time', 'seq'], kind='stable')
    out = []
    for ticker, rows in frame.groupby('market_ticker', sort=True):
        cache = BookCache(max_age_seconds=float('inf'))
        symbol = rows['symbol'].iloc[0]
        times = rows['event_time'].to_numpy()
        seqs = rows['seq'].to_numpy()
        kinds = rows['kind'].to_numpy()
        sides = rows['side'].to_numpy()
        prices = pd.to_numeric(rows['price'], errors='coerce').to_numpy(dtype=float)
        sizes = pd.to_numeric(rows['size'], errors='coerce').to_numpy(dtype=float)
        absolute = (rows['absolute'].astype(bool).to_numpy()
                    if 'absolute' in rows.columns else np.ones(len(rows), bool))
        stamps = pd.DatetimeIndex(times)
        seconds = stamps.floor('s')
        samples: dict = {}
        start = 0
        n = len(rows)
        while start < n:
            end = start + 1
            while (end < n and times[end] == times[start]
                   and (seqs[end] == seqs[start]
                        or (np.isnan(seqs[end]) and np.isnan(seqs[start])))):
                end += 1
            yes = [(prices[i], sizes[i]) for i in range(start, end) if sides[i] == 'yes']
            no = [(prices[i], sizes[i]) for i in range(start, end) if sides[i] == 'no']
            cache.apply(BookEvent(
                venue='kalshi', market_ticker=ticker, kind=str(kinds[start]),
                received=stamps[start].timestamp(), seq=None, yes=yes, no=no,
                absolute=bool(absolute[start])))
            # Sampled once per second, after its LAST frame: the state at the
            # end of the second, the same convention as the spot series.
            if end >= n or seconds[end] != seconds[start]:
                ladder = cache.ladder(ticker)
                if ladder is not None and ladder.yes and ladder.no:
                    bid = max(p for p, _ in ladder.yes)
                    ask = 1.0 - max(p for p, _ in ladder.no)
                    if ask >= bid:
                        samples[seconds[start]] = (bid + ask) / 2.0
            start = end
        if samples:
            out.append(pd.DataFrame({'symbol': symbol, 'market_ticker': ticker,
                                     'time': list(samples), 'mid': list(samples.values())}))
    if not out:
        return pd.DataFrame(columns=['symbol', 'market_ticker', 'time', 'mid'])
    return pd.concat(out, ignore_index=True)


def spot_second_prices(trades: Optional[pd.DataFrame] = None,
                       bars: Optional[pd.DataFrame] = None) -> tuple[pd.DataFrame, str]:
    """Last trade price per symbol per second, and the resolution it came from.

    Falls back to one-minute bar closes stamped at the bar's END — the first
    moment the close was knowable — when no trade table is given.
    """
    if trades is not None and len(trades):
        t = trades.copy()
        when = 'time' if 'time' in t.columns else 'event_time'
        t['time'] = _seconds(t[when])
        t['price'] = pd.to_numeric(t['price'], errors='coerce')
        order = [when] + (['trade_id'] if 'trade_id' in t.columns else [])
        t = t.sort_values(order, kind='stable')
        last = t.groupby(['symbol', 'time'], sort=True)['price'].last().reset_index()
        return last, 'trades'
    if bars is not None and len(bars):
        b = bars.copy()
        b['time'] = (pd.to_datetime(b['event_time'], utc=True)
                     + pd.Timedelta(minutes=1)).dt.floor('s')
        b['price'] = pd.to_numeric(b['close'], errors='coerce')
        return b[['symbol', 'time', 'price']].dropna(), 'minute_bars'
    return pd.DataFrame(columns=['symbol', 'time', 'price']), 'none'


def pm_mid_series(ladder: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Polymarket mids from `pm_ladder`, stamped when they were observed.

    `no_levels` is NO-denominated there too, so the YES ask is 1 - best NO bid.
    """
    import json

    if ladder is None or not len(ladder):
        return pd.DataFrame(columns=['symbol', 'window_open', 'time', 'mid'])
    rows = []
    for record in ladder.to_dict('records'):
        try:
            yes = json.loads(record.get('yes_levels') or '[]')
            no = json.loads(record.get('no_levels') or '[]')
        except (TypeError, ValueError):
            continue
        if not yes or not no:
            continue
        bid = max(float(p) for p, _ in yes)
        ask = 1.0 - max(float(p) for p, _ in no)
        if ask < bid:
            continue
        observed = record.get('available_time') or record.get('event_time')
        rows.append({'symbol': record.get('symbol'),
                     'window_open': pd.Timestamp(record['window_open']),
                     'time': pd.Timestamp(observed).floor('s'),
                     'mid': (bid + ask) / 2.0})
    frame = pd.DataFrame(rows, columns=['symbol', 'window_open', 'time', 'mid'])
    if len(frame):
        frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
        frame['time'] = pd.to_datetime(frame['time'], utc=True)
    return frame


def market_windows(mids: pd.DataFrame, ticker_windows: Optional[pd.DataFrame] = None,
                   window_minutes: int = DEFAULT_CONFIG.window_minutes) -> dict:
    """market_ticker -> window_open.

    From a (market_ticker, window_open) table where the store has one —
    `venue_depth` does — because a ticker's shape is a guess. Otherwise the
    window the market's own frames mostly fall in.
    """
    known = {}
    if ticker_windows is not None and len(ticker_windows):
        pairs = ticker_windows.dropna(subset=['market_ticker', 'window_open'])
        opens = pd.to_datetime(pairs['window_open'], utc=True)
        known = dict(zip(pairs['market_ticker'], opens))
    out = {}
    for ticker, rows in mids.groupby('market_ticker'):
        if ticker in known:
            out[ticker] = known[ticker]
        else:
            median = pd.to_datetime(rows['time'], utc=True).median()
            out[ticker] = median.floor(f'{window_minutes}min')
    return out


# ---- alignment --------------------------------------------------------------

def _implied_z(mid: np.ndarray) -> np.ndarray:
    return norm.ppf(np.clip(mid, MID_CLIP, 1.0 - MID_CLIP))


def _grid(window_open: pd.Timestamp, seconds: int = WINDOW_SECONDS) -> pd.DatetimeIndex:
    return pd.date_range(window_open, periods=seconds, freq='s')


def _on_grid(times, values, grid: pd.DatetimeIndex, *, limit: Optional[int] = None
             ) -> np.ndarray:
    """Last value at or before each grid second (as-of), NaN before the first."""
    series = pd.Series(np.asarray(values, dtype=float),
                       index=pd.DatetimeIndex(pd.to_datetime(times, utc=True)))
    series = series[~series.index.duplicated(keep='last')].sort_index()
    return series.reindex(grid, method='ffill', limit=limit).to_numpy(dtype=float)


@lru_cache(maxsize=8)
def remaining_variance_seconds(seconds: int = WINDOW_SECONDS) -> np.ndarray:
    """Variance-seconds left at each second of a window — the settlement value
    is a one-minute mean, so this is `Config.remaining_variance_minutes`."""
    return np.array([DEFAULT_CONFIG.remaining_variance_minutes(s / 60.0)
                     for s in range(seconds)]) * 60.0


def spot_dz(prices: pd.DataFrame, symbol: str, window_open: pd.Timestamp,
            seconds: int = WINDOW_SECONDS) -> np.ndarray:
    """1-second change in the spot-implied barrier z across one window."""
    part = prices[prices['symbol'] == symbol]
    lookback = window_open - pd.Timedelta(seconds=SIGMA_LOOKBACK_SECONDS)
    part = part[(part['time'] >= lookback - pd.Timedelta(minutes=2))
                & (part['time'] < window_open + pd.Timedelta(seconds=seconds))]
    if not len(part):
        return np.full(seconds, np.nan)
    wide = pd.date_range(lookback, window_open + pd.Timedelta(seconds=seconds - 1),
                         freq='s')
    price = _on_grid(part['time'], part['price'], wide)
    r = np.diff(np.log(price), prepend=np.nan)
    sigma = (pd.Series(r).rolling(SIGMA_LOOKBACK_SECONDS, min_periods=60).std()
             .shift(1).to_numpy())
    r, sigma = r[-seconds:], sigma[-seconds:]
    remaining = remaining_variance_seconds(seconds)
    with np.errstate(divide='ignore', invalid='ignore'):
        dz = r / (sigma * np.sqrt(np.maximum(remaining, 1.0)))
    return np.where(np.isfinite(dz), dz, np.nan)


def mid_dz(times, mids, window_open: pd.Timestamp,
           seconds: int = WINDOW_SECONDS) -> np.ndarray:
    """1-second change in the price-implied z. NaN where the mid sits at the
    clip on either side of the change: a pinned price cannot move with spot,
    and counting it as "did not move" would read as a lag."""
    mid = _on_grid(times, mids, _grid(window_open, seconds))
    z = _implied_z(mid)
    dz = np.diff(z, prepend=np.nan)
    pinned = (mid <= MID_CLIP) | (mid >= 1.0 - MID_CLIP)
    pinned = pinned | np.concatenate([[True], pinned[:-1]])
    return np.where(pinned, np.nan, dz)


# ---- correlation ------------------------------------------------------------

@dataclass
class LagAccumulator:
    """Pooled Pearson sums per (bucket, lag), so markets never pair across."""

    max_lag: int
    buckets: Sequence[str]
    sums: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = (len(self.buckets), 2 * self.max_lag + 1)
        self.sums = {k: np.zeros(shape) for k in ('n', 'x', 'y', 'xx', 'yy', 'xy')}

    def add(self, kalshi: np.ndarray, other: np.ndarray, bucket: np.ndarray) -> None:
        """`bucket` indexes `self.buckets` per second of the KALSHI series."""
        T = len(kalshi)
        nb = len(self.buckets)
        for j, lag in enumerate(range(-self.max_lag, self.max_lag + 1)):
            if lag >= 0:
                x, y, b = kalshi[lag:], other[:T - lag], bucket[lag:]
            else:
                x, y, b = kalshi[:T + lag], other[-lag:], bucket[:T + lag]
            ok = np.isfinite(x) & np.isfinite(y) & (b >= 0)
            if not ok.any():
                continue
            x, y, b = x[ok], y[ok], b[ok]
            for key, w in (('n', None), ('x', x), ('y', y), ('xx', x * x),
                           ('yy', y * y), ('xy', x * y)):
                self.sums[key][:, j] += np.bincount(b, weights=w, minlength=nb)

    def correlation(self, rows: Optional[Sequence[int]] = None) -> tuple[np.ndarray, np.ndarray]:
        """Correlation per lag over the chosen bucket rows (all by default)."""
        pick = slice(None) if rows is None else list(rows)
        s = {k: v[pick].sum(axis=0) for k, v in self.sums.items()}
        n = s['n']
        with np.errstate(divide='ignore', invalid='ignore'):
            cov = s['xy'] / n - (s['x'] / n) * (s['y'] / n)
            vx = s['xx'] / n - (s['x'] / n) ** 2
            vy = s['yy'] / n - (s['y'] / n) ** 2
            rho = cov / np.sqrt(vx * vy)
        return np.where(n > 2, rho, np.nan), n

    @property
    def lags(self) -> np.ndarray:
        return np.arange(-self.max_lag, self.max_lag + 1)


def summarise_profile(lags: np.ndarray, rho: np.ndarray, n: np.ndarray) -> dict:
    finite = np.isfinite(rho)
    if not finite.any():
        return {'n': 0, 'peak_lag': np.nan, 'peak_corr': np.nan,
                'corr_at_0': np.nan, 'lead_share': np.nan, 'delayed_share': np.nan}
    peak = int(np.nanargmax(np.where(finite, rho, -np.inf)))
    pos, neg = (lags > 0) & finite, (lags < 0) & finite
    sq_pos, sq_neg = float(np.sum(rho[pos] ** 2)), float(np.sum(rho[neg] ** 2))
    after = (lags >= 0) & finite
    gain = np.clip(rho[after], 0.0, None)
    later = np.clip(rho[(lags > 0) & finite], 0.0, None)
    zero = rho[lags == 0]
    return {
        'n': int(n[lags == 0][0]) if (lags == 0).any() else int(n.max()),
        'peak_lag': int(lags[peak]), 'peak_corr': float(rho[peak]),
        'corr_at_0': float(zero[0]) if len(zero) else np.nan,
        'lead_share': sq_pos / (sq_pos + sq_neg) if sq_pos + sq_neg > 0 else np.nan,
        'delayed_share': float(later.sum() / gain.sum()) if gain.sum() > 0 else np.nan,
    }


def bucket_labels(bucket_minutes: int) -> list[str]:
    edges = list(range(0, DEFAULT_CONFIG.window_minutes, bucket_minutes))
    return [f'{lo}-{min(lo + bucket_minutes, DEFAULT_CONFIG.window_minutes)}m left'
            for lo in edges]


def bucket_index(seconds: int, bucket_minutes: int) -> np.ndarray:
    """Minutes-remaining bucket per second; -1 (excluded) inside the settlement
    averaging minute, where the price is converging on a mean already partly
    printed and `x / sigma_n` no longer describes it."""
    remaining = (seconds - np.arange(seconds)) / 60.0
    idx = np.floor((remaining - 1e-9) / bucket_minutes).astype(int)
    idx = np.clip(idx, 0, len(bucket_labels(bucket_minutes)) - 1)
    averaging = remaining <= float(DEFAULT_CONFIG.settle_average_minutes) + 1e-9
    return np.where(averaging, -1, idx)


# ---- the study --------------------------------------------------------------

@dataclass
class StudyResult:
    pair: str
    resolution: str
    lags: np.ndarray
    profiles: dict             # symbol -> (rho, n) pooled over buckets
    summary: pd.DataFrame      # symbol x bucket rows
    markets: int


def run_pair(kalshi: pd.DataFrame, windows: dict, other_dz, *, pair: str,
             resolution: str, max_lag: int = 60, bucket_minutes: int = 3,
             seconds: int = WINDOW_SECONDS) -> StudyResult:
    """Cross-correlate Kalshi's z changes with `other_dz(symbol, window_open)`."""
    labels = bucket_labels(bucket_minutes)
    buckets = bucket_index(seconds, bucket_minutes)
    accumulators: dict[str, LagAccumulator] = {}
    markets = 0
    for ticker, rows in kalshi.groupby('market_ticker'):
        window_open = windows.get(ticker)
        if window_open is None:
            continue
        symbol = rows['symbol'].iloc[0]
        k = mid_dz(rows['time'], rows['mid'], window_open, seconds)
        o = other_dz(symbol, window_open)
        if o is None or not np.isfinite(o).any() or not np.isfinite(k).any():
            continue
        acc = accumulators.setdefault(symbol, LagAccumulator(max_lag, labels))
        acc.add(k, o, buckets)
        markets += 1
    rows, profiles = [], {}
    lags = np.arange(-max_lag, max_lag + 1)
    for symbol, acc in sorted(accumulators.items()):
        rho, n = acc.correlation()
        profiles[symbol] = (rho, n)
        rows.append({'symbol': symbol, 'bucket': 'all', **summarise_profile(lags, rho, n)})
        for i, label in enumerate(labels):
            rho_b, n_b = acc.correlation([i])
            rows.append({'symbol': symbol, 'bucket': label,
                         **summarise_profile(lags, rho_b, n_b)})
    return StudyResult(pair=pair, resolution=resolution, lags=lags,
                       profiles=profiles, summary=pd.DataFrame(rows), markets=markets)


def kalshi_vs_spot(kalshi: pd.DataFrame, windows: dict, prices: pd.DataFrame,
                   resolution: str, **kwargs) -> StudyResult:
    def other(symbol, window_open):
        return spot_dz(prices, symbol, window_open, kwargs.get('seconds', WINDOW_SECONDS))
    return run_pair(kalshi, windows, other, pair='kalshi vs spot',
                    resolution=resolution, **kwargs)


def kalshi_vs_polymarket(kalshi: pd.DataFrame, windows: dict, pm: pd.DataFrame,
                         **kwargs) -> StudyResult:
    seconds = kwargs.get('seconds', WINDOW_SECONDS)
    groups = {key: part for key, part in pm.groupby(['symbol', 'window_open'])}

    def other(symbol, window_open):
        part = groups.get((symbol, window_open))
        if part is None or not len(part):
            return None
        return mid_dz(part['time'], part['mid'], window_open, seconds)

    gaps = (pm.sort_values('time').groupby(['symbol', 'window_open'])['time']
            .diff().dt.total_seconds().dropna())
    resolution = (f'polymarket sampled every {gaps.median():.0f}s (median)'
                  if len(gaps) else 'polymarket sampling unknown')
    return run_pair(kalshi, windows, other, pair='kalshi vs polymarket',
                    resolution=resolution, **kwargs)


def format_result(result: StudyResult) -> str:
    lines = [f'== {result.pair} == ({result.markets:,} markets; {result.resolution})']
    if not result.profiles:
        lines.append('  no overlapping data')
        return '\n'.join(lines)
    summary = result.summary.copy()
    for col in ('peak_corr', 'corr_at_0', 'lead_share', 'delayed_share'):
        summary[col] = summary[col].map(lambda v: f'{v:+.3f}' if np.isfinite(v) else 'nan')
    lines.append('  positive lag = the other series LEADS Kalshi by that many seconds')
    lines.append('  ' + summary.to_string(index=False).replace('\n', '\n  '))
    profile = {'lag_s': [lag for lag in PROFILE_LAGS if abs(lag) <= result.lags.max()]}
    for symbol, (rho, _) in result.profiles.items():
        profile[symbol] = [f'{rho[result.lags == lag][0]:+.3f}' for lag in profile['lag_s']]
    lines.append('\n  correlation profile, all buckets pooled:')
    lines.append('  ' + pd.DataFrame(profile).to_string(index=False).replace('\n', '\n  '))
    return '\n'.join(lines)


# ---- CLI ----------------------------------------------------------------------

def _utc(value):
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def load_inputs(store, symbols, start, end) -> dict:
    """Everything the study reads, each optional, with what was missing said."""
    from core.datastore import DataStoreError

    def _read(name, **kw):
        try:
            return store.read(name, symbols=symbols, start=start, end=end,
                              min_quality=None, **kw)
        except DataStoreError as exc:
            logger.warning('%s unavailable (%s)', name, exc)
            return pd.DataFrame()

    events = _read('venue_book_events', venue='kalshi')
    trades = _read('spot_trades')
    bars = pd.DataFrame()
    if not len(trades):
        logger.warning('no spot_trades rows: falling back to one-minute bars, which '
                       'move once a minute and CANNOT resolve a seconds-scale lead')
        bars = _read('minute_bars', venue=DEFAULT_CONFIG.venue,
                     start=None if start is None else start - pd.Timedelta(minutes=20))
    pm = _read('pm_ladder', venue='polymarket')
    depth = _read('venue_depth', venue='kalshi')
    return {'events': events, 'trades': trades, 'bars': bars, 'pm': pm,
            'depth': depth}


def run_study(inputs: dict, *, max_lag: int = 60, bucket_minutes: int = 3) -> list[StudyResult]:
    kalshi = kalshi_mid_series(inputs.get('events'))
    if not len(kalshi):
        logger.warning('no Kalshi book frames in range; nothing to study')
        return []
    depth = inputs.get('depth')
    ticker_windows = (depth[['market_ticker', 'window_open']]
                      if depth is not None and len(depth)
                      and {'market_ticker', 'window_open'} <= set(depth.columns)
                      else None)
    windows = market_windows(kalshi, ticker_windows)
    prices, resolution = spot_second_prices(inputs.get('trades'), inputs.get('bars'))
    if resolution == 'minute_bars':
        resolution = 'spot from ONE-MINUTE bars: seconds-scale lags are not resolved'
    out = [kalshi_vs_spot(kalshi, windows, prices, resolution,
                          max_lag=max_lag, bucket_minutes=bucket_minutes)]
    pm = pm_mid_series(inputs.get('pm'))
    if len(pm):
        out.append(kalshi_vs_polymarket(kalshi, windows, pm, max_lag=max_lag,
                                        bucket_minutes=bucket_minutes))
    else:
        logger.warning('no Polymarket ladder rows in range; Kalshi vs Polymarket '
                       'skipped')
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=None, help='defaults to $RESEARCH_STORE')
    parser.add_argument('--symbols', default=','.join(DEFAULT_CONFIG.symbols))
    parser.add_argument('--start', default=None, help='ISO time, inclusive '
                        '(default: 24h before --end)')
    parser.add_argument('--end', default=None, help='ISO time, exclusive (default now)')
    parser.add_argument('--max-lag', type=int, default=60)
    parser.add_argument('--bucket-minutes', type=int, default=3,
                        help='minutes-remaining bucket width')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)-7s %(message)s')

    from core.datastore import ResearchStore

    end = _utc(args.end) or pd.Timestamp.now(tz='UTC').floor('min')
    start = _utc(args.start) or end - pd.Timedelta(days=1)
    store = ResearchStore(args.store or os.getenv('RESEARCH_STORE'))
    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    inputs = load_inputs(store, symbols, start, end)
    results = run_study(inputs, max_lag=args.max_lag,
                        bucket_minutes=args.bucket_minutes)
    print(f'lead-lag, {start} .. {end}, symbols {",".join(symbols)}\n')
    for result in results:
        print(format_result(result) + '\n')
    return 0 if results else 1


if __name__ == '__main__':
    raise SystemExit(main())
