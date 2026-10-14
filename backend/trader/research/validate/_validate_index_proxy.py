"""Does the index proxy call Kalshi's settlement better than our Coinbase label?

Our label (`core/windows.py`) is a Coinbase one-minute OHLC mean at each end of
the window. Kalshi settles on CF Benchmarks' index, and the two disagree on
~35% of windows within 1bp of the strike — the near-ties this system trades
most. `index_proxy_1s` is a per-second median of constituent-exchange mids
(`core/index_proxy.py`). This asks the only question that justifies keeping it:

    on the windows Kalshi has settled, how often does each estimate agree with
    the venue, by how far the window finished from its strike?

Both labels use the venue's convention: the strike is the mean over
`[t0 - 60s, t0)`, the settlement value the mean over `[t1 - 60s, t1)`, and
`>=` resolves up. For the proxy the means are over its one-second rows; for
Coinbase over the one-minute bar's `(O+H+L+C)/4`. Only windows where BOTH
labels exist are compared, so the two columns are scored on identical rows.

The table is bucketed by the Coinbase |move| (the axis of the existing label
audit) and again by the proxy's own |move|. A proxy that helps should close the
gap in the narrow buckets and match Coinbase everywhere else.
"""
from __future__ import annotations

# See `_validate_label.py`: `core`/`scripts` are rooted at backend/trader/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd

BUCKETS = (0.0, 1.0, 2.0, 5.0, 10.0, np.inf)
LABELS = ('<1bp', '1-2bp', '2-5bp', '5-10bp', '>10bp')
MINUTE = pd.Timedelta(minutes=1)


def coinbase_ends(bars_by_symbol: dict, window_minutes: int = 15) -> pd.DataFrame:
    """`(symbol, window_open, cb_strike, cb_settle)` from one-minute bars."""
    from core.windows import bar_mean

    frames = []
    for symbol, bars in bars_by_symbol.items():
        if bars is None or not len(bars):
            continue
        means = pd.Series(bar_mean(bars).to_numpy(dtype=float),
                          index=pd.to_datetime(bars['event_time'], utc=True))
        means = means[~means.index.duplicated(keep='last')]
        opens = means.index.floor(f'{window_minutes}min').unique()
        strike = means.reindex(opens - MINUTE).to_numpy()
        settle = means.reindex(opens + pd.Timedelta(minutes=window_minutes)
                               - MINUTE).to_numpy()
        frames.append(pd.DataFrame({'symbol': symbol, 'window_open': opens,
                                    'cb_strike': strike, 'cb_settle': settle}))
    if not frames:
        return pd.DataFrame(columns=['symbol', 'window_open', 'cb_strike', 'cb_settle'])
    return pd.concat(frames, ignore_index=True)


def proxy_ends(minutes: pd.DataFrame, window_minutes: int = 15,
               min_samples: Optional[int] = None) -> pd.DataFrame:
    """`(symbol, window_open, px_strike, px_settle)` from proxy minute summaries."""
    from core.index_features import MIN_STRIKE_SAMPLES

    need = MIN_STRIKE_SAMPLES if min_samples is None else min_samples
    if minutes is None or not len(minutes):
        return pd.DataFrame(columns=['symbol', 'window_open', 'px_strike', 'px_settle'])
    m = minutes.copy()
    m['minute'] = pd.to_datetime(m['minute'], utc=True)
    m['value'] = np.where(m['samples'] >= need, m['mean'], np.nan)
    lookup = m.set_index(['symbol', 'minute'])['value']
    # Every window whose strike OR settlement minute the proxy covers.
    span = pd.Timedelta(minutes=window_minutes)
    candidates = pd.concat([
        pd.DataFrame({'symbol': m['symbol'],
                      'window_open': (m['minute'] + MINUTE).dt.floor(f'{window_minutes}min')}),
        pd.DataFrame({'symbol': m['symbol'],
                      'window_open': (m['minute'] + MINUTE - span).dt.floor(f'{window_minutes}min')}),
    ]).drop_duplicates(ignore_index=True)

    def at(times):
        index = pd.MultiIndex.from_arrays([candidates['symbol'], times])
        return lookup.reindex(index).to_numpy(dtype=float)
    candidates['px_strike'] = at(candidates['window_open'] - MINUTE)
    candidates['px_settle'] = at(candidates['window_open'] + span - MINUTE)
    return candidates


def compare(settlements: pd.DataFrame, coinbase: pd.DataFrame,
            proxy: pd.DataFrame) -> pd.DataFrame:
    """One row per Kalshi-settled window both labels can score."""
    venue = settlements
    if 'venue' in venue.columns:
        venue = venue[venue['venue'] == 'kalshi']
    venue = venue.dropna(subset=['symbol', 'window_open', 'settled_up']).copy()
    venue['window_open'] = pd.to_datetime(venue['window_open'], utc=True)
    if 'available_time' in venue.columns:
        venue = venue.sort_values('available_time')
    venue = venue.drop_duplicates(['symbol', 'window_open'], keep='last')
    venue = venue[['symbol', 'window_open', 'settled_up']]
    coinbase, proxy = coinbase.copy(), proxy.copy()
    for frame in (coinbase, proxy):
        frame['window_open'] = pd.to_datetime(frame['window_open'], utc=True)
    joined = (venue.merge(coinbase, on=['symbol', 'window_open'], how='inner')
              .merge(proxy, on=['symbol', 'window_open'], how='inner'))
    joined = joined.dropna(subset=['cb_strike', 'cb_settle', 'px_strike', 'px_settle'])
    joined['kalshi_up'] = joined['settled_up'].astype(float) > 0.5
    joined['cb_up'] = joined['cb_settle'] >= joined['cb_strike']
    joined['px_up'] = joined['px_settle'] >= joined['px_strike']
    joined['cb_move_bps'] = (1e4 * (joined['cb_settle'] / joined['cb_strike'] - 1)).abs()
    joined['px_move_bps'] = (1e4 * (joined['px_settle'] / joined['px_strike'] - 1)).abs()
    return joined.reset_index(drop=True)


def agreement_table(joined: pd.DataFrame, by: str = 'cb_move_bps') -> pd.DataFrame:
    """Agreement with Kalshi per |move| bucket, plus an `all` row."""
    rows = []
    bands = pd.cut(joined[by], list(BUCKETS), labels=list(LABELS), right=False)
    groups = [(label, joined[bands == label]) for label in LABELS]
    groups.append(('all', joined))
    for label, part in groups:
        n = len(part)
        cb = float((part['cb_up'] == part['kalshi_up']).mean()) if n else np.nan
        px = float((part['px_up'] == part['kalshi_up']).mean()) if n else np.nan
        rows.append({'move': label, 'windows': n, 'coinbase_agree': cb,
                     'proxy_agree': px, 'proxy_minus_coinbase': px - cb})
    return pd.DataFrame(rows)


def render(table: pd.DataFrame, title: str) -> str:
    lines = [title, f"{'move':>8} {'windows':>8} {'coinbase':>9} {'proxy':>9} {'gain':>8}"]
    for row in table.itertuples(index=False):
        def pct(v):
            return f'{100 * v:8.2f}%' if np.isfinite(v) else f"{'-':>9}"
        gain = (f'{100 * row.proxy_minus_coinbase:+7.2f}pp'
                if np.isfinite(row.proxy_minus_coinbase) else f"{'-':>9}")
        lines.append(f'{row.move:>8} {row.windows:>8,} {pct(row.coinbase_agree)}'
                     f' {pct(row.proxy_agree)} {gain}')
    return '\n'.join(lines)


def main(argv=None) -> int:
    from core.config import DEFAULT_CONFIG
    from core.dataset import load_minute_bars
    from core.datastore import ResearchStore
    from core.index_features import load_proxy_minutes

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--store', default=os.getenv('RESEARCH_STORE'))
    args = parser.parse_args(argv)

    config = DEFAULT_CONFIG
    store = ResearchStore(args.store)
    minutes = load_proxy_minutes(store.root)
    if not len(minutes):
        print('index_proxy_1s is empty — run the `index` component first')
        return 1
    settled = store.read('venue_settlements')
    if not len(settled):
        print('venue_settlements is empty — run scripts.collect_settlements')
        return 1
    lo = pd.to_datetime(minutes['minute'], utc=True).min() - pd.Timedelta(minutes=20)
    hi = pd.to_datetime(minutes['minute'], utc=True).max() + pd.Timedelta(minutes=20)
    bars = load_minute_bars(config, store=store, start=lo, end=hi)
    joined = compare(settled, coinbase_ends(bars, config.window_minutes),
                     proxy_ends(minutes, config.window_minutes))
    if not len(joined):
        print('no Kalshi-settled window is covered by both labels yet')
        return 1
    print(f'{len(joined):,} Kalshi-settled windows scored by both labels, '
          f'{joined["window_open"].min()} .. {joined["window_open"].max()}\n')
    print(render(agreement_table(joined, 'cb_move_bps'),
                 'Agreement with Kalshi, by |Coinbase move from strike|'))
    print()
    print(render(agreement_table(joined, 'px_move_bps'),
                 'Agreement with Kalshi, by |proxy move from strike|'))
    for symbol, part in joined.groupby('symbol'):
        print()
        print(render(agreement_table(part, 'cb_move_bps'), f'{symbol} only'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
