"""The `index_proxy` feature group: displacement measured on the settlement index.

Strike and settlement are one-minute means of the index (BRTI and siblings),
not Coinbase. These columns use the per-second constituent median from
`scripts/record_index.py`:

* `proxy_displacement_z`: displacement from the proxy strike, in units of `sigma_remaining`.
* `proxy_minus_spot_bps`: how far the index sits from Coinbase now.
* `proxy_n_exchanges`: how many venues the newest proxy row was built from.

Point-in-time: the strike is the mean of rows stamped in `[window_open - 60s, window_open)`.
The current value is the newest row stamped strictly before `decision_time`,
within `max_age_seconds`. A row stamped `t` was computed before `t + 1`.

The backtest and the live path both go through `minute_summary` and
`attach_index_proxy`. Decisions fall on minute boundaries, so the summary is lossless.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

INDEX_PROXY = ('proxy_displacement_z', 'proxy_minus_spot_bps', 'proxy_n_exchanges')
# Raw columns `attach_index_proxy` adds. The z-score needs the fitted
# `sigma_remaining`, so `build_features` finishes it inside the fold.
RAW_COLUMNS = ('proxy_strike', 'proxy_last', 'proxy_displacement',
               'proxy_minus_spot_bps', 'proxy_n_exchanges')
# The strike is a 60-second mean. With fewer valid seconds it is not the
# quantity the venue averages.
MIN_STRIKE_SAMPLES = 30
SUMMARY_COLUMNS = ('symbol', 'minute', 'mean', 'samples', 'last', 'last_time',
                   'last_n')
MINUTE = pd.Timedelta(minutes=1)


def _valid(frame: pd.DataFrame) -> pd.DataFrame:
    proxy = pd.to_numeric(frame['proxy'], errors='coerce')
    return frame.assign(proxy=proxy)[np.isfinite(proxy.to_numpy(dtype=float))]


def minute_summary(proxy_1s: pd.DataFrame) -> pd.DataFrame:
    """Per (symbol, minute): mean, valid-sample count, and the newest valid row.

    `minute` is the minute's start, so the summary for `[t - 60s, t)` has
    `minute == t - 1min`.
    """
    if proxy_1s is None or not len(proxy_1s):
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    frame = _valid(proxy_1s[['symbol', 'event_time', 'proxy', 'n_exchanges']].copy())
    if not len(frame):
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    frame['event_time'] = pd.to_datetime(frame['event_time'], utc=True)
    frame = frame.sort_values(['symbol', 'event_time']).drop_duplicates(
        ['symbol', 'event_time'], keep='last')
    frame['minute'] = frame['event_time'].dt.floor('min')
    grouped = frame.groupby(['symbol', 'minute'], sort=True)
    out = grouped.agg(mean=('proxy', 'mean'), samples=('proxy', 'size'),
                      last=('proxy', 'last'), last_time=('event_time', 'last'),
                      last_n=('n_exchanges', 'last')).reset_index()
    return out[list(SUMMARY_COLUMNS)]


def load_proxy_minutes(store_root: Path | str, *, symbols: Optional[Sequence[str]] = None,
                       start=None, end=None) -> pd.DataFrame:
    """`minute_summary` over the stored table, aggregated in DuckDB.

    Duplicate stamps collapse to the newest revision, as `ResearchStore.read` does.
    """
    import duckdb

    root = Path(store_root) / 'index_proxy_1s'
    if not any(root.rglob('*.parquet')):
        return pd.DataFrame(columns=list(SUMMARY_COLUMNS))
    clauses, params = ["quality = 'valid'", 'proxy IS NOT NULL', 'NOT isnan(proxy)'], []
    if symbols:
        clauses.append(f"symbol IN ({', '.join('?' for _ in symbols)})")
        params.extend(s.upper() for s in symbols)
    if start is not None:
        clauses.append('epoch(event_time) >= ?')
        params.append(pd.Timestamp(start).timestamp() - 60.0)
    if end is not None:
        clauses.append('epoch(event_time) <= ?')
        params.append(pd.Timestamp(end).timestamp() + 60.0)
    sql = (
        'SELECT symbol, CAST(floor(epoch(event_time) / 60) * 60 AS BIGINT) AS minute, '
        'avg(proxy) AS mean, count(*) AS samples, '
        'arg_max(proxy, event_time) AS last, max(epoch(event_time)) AS last_time, '
        'arg_max(n_exchanges, event_time) AS last_n '
        'FROM (SELECT * FROM read_parquet(?, hive_partitioning = false, '
        'union_by_name = true) WHERE ' + ' AND '.join(clauses) +
        ' QUALIFY row_number() OVER (PARTITION BY symbol, event_time '
        'ORDER BY available_time DESC) = 1) '
        'GROUP BY 1, 2 ORDER BY 1, 2')
    with duckdb.connect() as con:
        out = con.execute(sql, [str(root / '**' / '*.parquet'), *params]).df()
    out['minute'] = pd.to_datetime(out['minute'], unit='s', utc=True)
    out['last_time'] = pd.to_datetime(out['last_time'], unit='s', utc=True)
    return out[list(SUMMARY_COLUMNS)]


def attach_index_proxy(windows: pd.DataFrame, minutes: pd.DataFrame, *,
                       max_age_seconds: float = 5.0) -> pd.DataFrame:
    """The raw proxy columns on a window table (one row per symbol/window/offset).

    Needs `symbol`, `window_open`, `decision_time` and `last_price`. Rows with
    no usable proxy carry NaN; the row count never changes.
    """
    out = windows.copy()
    if minutes is None or not len(minutes):
        for column in RAW_COLUMNS:
            out[column] = np.nan
        return out
    summary = minutes.copy()
    summary['minute'] = pd.to_datetime(summary['minute'], utc=True)
    summary['last_time'] = pd.to_datetime(summary['last_time'], utc=True)
    keys = pd.DataFrame({
        'symbol': out['symbol'].to_numpy(),
        'strike_minute': pd.to_datetime(out['window_open'], utc=True) - MINUTE,
        'last_minute': pd.to_datetime(out['decision_time'], utc=True) - MINUTE,
        'decision_time': pd.to_datetime(out['decision_time'], utc=True),
    })
    strike = keys.merge(
        summary[['symbol', 'minute', 'mean', 'samples']].rename(
            columns={'minute': 'strike_minute'}),
        on=['symbol', 'strike_minute'], how='left')
    last = keys.merge(
        summary[['symbol', 'minute', 'last', 'last_time', 'last_n']].rename(
            columns={'minute': 'last_minute'}),
        on=['symbol', 'last_minute'], how='left')
    assert len(strike) == len(out) and len(last) == len(out), 'proxy join fanned out'

    strike_value = np.where(
        pd.to_numeric(strike['samples'], errors='coerce').fillna(0).to_numpy()
        >= MIN_STRIKE_SAMPLES,
        pd.to_numeric(strike['mean'], errors='coerce').to_numpy(dtype=float), np.nan)
    age = (last['decision_time'] - last['last_time']).dt.total_seconds().to_numpy()
    # `last_time < decision_time` holds by construction (the row is from the
    # minute before the decision). The age bound excludes a stalled recorder.
    usable = np.isfinite(age) & (age > 0) & (age <= float(max_age_seconds))
    last_value = np.where(usable, pd.to_numeric(last['last'], errors='coerce')
                          .to_numpy(dtype=float), np.nan)
    last_n = np.where(usable, pd.to_numeric(last['last_n'], errors='coerce')
                      .to_numpy(dtype=float), np.nan)

    spot = (pd.to_numeric(out['last_price'], errors='coerce').to_numpy(dtype=float)
            if 'last_price' in out.columns else np.full(len(out), np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        out['proxy_strike'] = strike_value
        out['proxy_last'] = last_value
        out['proxy_displacement'] = last_value / strike_value - 1.0
        out['proxy_minus_spot_bps'] = 1e4 * (last_value / spot - 1.0)
    out['proxy_n_exchanges'] = last_n
    return out


def proxy_displacement_z(proxy_displacement, sigma_remaining) -> np.ndarray:
    """Displacement over the remaining sigma: `z_score`, measured on the index."""
    displacement = pd.to_numeric(pd.Series(proxy_displacement), errors='coerce')
    sigma = pd.to_numeric(pd.Series(sigma_remaining), errors='coerce')
    sigma = sigma.where(sigma > 0)
    return (displacement.to_numpy(dtype=float) / sigma.to_numpy(dtype=float))


def index_features_row(cache_snapshot: pd.DataFrame, window_open, decision_time,
                       sigma_remaining: float, *, last_price: float = float('nan'),
                       symbol: Optional[str] = None,
                       max_age_seconds: float = 5.0) -> dict:
    """The `index_proxy` group for one live decision, from the in-process cache.

    `cache_snapshot` is `record_index.CACHE.snapshot(symbol)`, in the stored
    table's shape. Rows at or after `decision_time` are dropped, so a late snapshot cannot leak.
    """
    window_open = pd.Timestamp(window_open)
    decision_time = pd.Timestamp(decision_time)
    window_open = window_open.tz_localize('UTC') if window_open.tzinfo is None \
        else window_open.tz_convert('UTC')
    decision_time = decision_time.tz_localize('UTC') if decision_time.tzinfo is None \
        else decision_time.tz_convert('UTC')
    snap = cache_snapshot if cache_snapshot is not None else pd.DataFrame(
        columns=['symbol', 'event_time', 'proxy', 'n_exchanges'])
    if symbol is None:
        symbol = (str(snap['symbol'].iloc[0]) if len(snap) and 'symbol' in snap
                  else 'UNKNOWN')
    snap = snap.copy()
    snap['symbol'] = symbol
    if len(snap):
        times = pd.to_datetime(snap['event_time'], utc=True)
        snap = snap[(times >= window_open - MINUTE) & (times < decision_time)]
    row = pd.DataFrame({'symbol': [symbol], 'window_open': [window_open],
                        'decision_time': [decision_time], 'last_price': [last_price]})
    attached = attach_index_proxy(row, minute_summary(snap),
                                  max_age_seconds=max_age_seconds)
    z = proxy_displacement_z(attached['proxy_displacement'], [sigma_remaining])
    return {'proxy_displacement_z': float(z[0]),
            'proxy_minus_spot_bps': float(attached['proxy_minus_spot_bps'].iloc[0]),
            'proxy_n_exchanges': float(attached['proxy_n_exchanges'].iloc[0])}
