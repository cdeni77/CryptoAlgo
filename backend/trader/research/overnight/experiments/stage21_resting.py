"""Would a RESTING bid have filled, and was the fill adversely selected?

The trade: market 65, model says 67. Lifting the ask at 66 costs ~0.7c of fee
and leaves ~+0.3c. Resting at the bid, 65, costs NOTHING -- the venue charges
makers nothing, confirmed on its own order ticket -- and leaves +2c. The only
question is whether it fills, and what you own when it does.

**Answerable from data already held.** `venue_depth` carries the Kalshi book at
every minute 0..15 of every window, live-recorded since 2026-08-25, so for a
decision at offset m we walk the REMAINING minutes of the same window and ask
whether the ask ever came down to our bid. Nothing is placed, nothing
extrapolated.

Two assumptions, stated rather than buried:

* A resting buy at P fills when someone sells INTO it, so we require the best
  ask at a later minute to be <= P. Merely touching is not enough.
* It assumes QUEUE PRIORITY, that we are first in line at P. We are not. So the
  fill RATE is an UPPER BOUND -- if it does not fill often enough on paper it
  certainly will not live.

The control is the point: `taker pnl on the same rows` is what crossing the
spread would have made on exactly the windows that filled. If resting fills
only on windows that go on to lose, the maker column falls while that one does
not, and that is adverse selection made visible rather than argued about.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from core.metrics import LIVE_OBSERVERS                       # noqa: E402

SCORED = 'research/overnight/deadband/full/scored.parquet'


def load():
    s = pd.read_parquet(SCORED, columns=[
        'symbol', 'window_open', 'offset', 'venue_outcome', 'outcome',
        'quote_observer', 'quote_age_seconds', 'market_probability',
        'model_probability', 'ask_up', 'ask_down'])
    s = s[s.quote_observer.isin(LIVE_OBSERVERS)]
    s = s[(s.quote_age_seconds.isna()) | (s.quote_age_seconds.abs() <= 30)]
    s['y'] = pd.to_numeric(s.venue_outcome, errors='coerce').fillna(
        pd.to_numeric(s.outcome, errors='coerce'))
    s = s.dropna(subset=['y', 'market_probability', 'model_probability',
                         'ask_up', 'ask_down'])
    s['window_open'] = pd.to_datetime(s.window_open, utc=True)
    import duckdb
    d = duckdb.connect().execute("""
        select symbol, window_open, offset_minutes, yes_bid, yes_ask
        from read_parquet('data/research/venue_depth/**/*.parquet', union_by_name=true)
        where venue = 'kalshi'
    """).fetchdf()
    d['window_open'] = pd.to_datetime(d.window_open, utc=True)
    for c in ('yes_bid', 'yes_ask'):
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d['offset_minutes'] = pd.to_numeric(d.offset_minutes, errors='coerce')
    d = d.dropna(subset=['yes_ask', 'offset_minutes'])
    d = d[(d.yes_ask > 0) & (d.yes_ask < 1)
          & (d.yes_bid.isna() | (d.yes_ask >= d.yes_bid))]
    return s, d


def simulate(s, fut, below_cents: float, gate_pp: float = 0.0):
    rows = []
    for r in s.itertuples():
        g = fut.get((r.symbol, r.window_open))
        if g is None:
            continue
        off, ask_arr, bid_arr = g
        at = off == r.offset
        if not at.any():
            continue
        i = np.flatnonzero(at)[-1]
        later = off > r.offset
        if not later.any():
            continue
        up = r.model_probability >= r.market_probability
        # **The edge gate, which the first version of this script omitted.**
        # Without it the simulation rests an order on EVERY scored row, i.e.
        # buys something 50,140 times with no view, and both columns land near
        # -10pp -- that is selection, not economics. The gate is applied to the
        # edge against the RESTING price, which is the trade actually taken.
        if up:
            bid = bid_arr[i]
            if not np.isfinite(bid):
                continue
            limit = bid - below_cents / 100.0
            filled = bool(np.nanmin(ask_arr[later]) <= limit + 1e-9)
            taker_px, won = float(r.ask_up), (r.y == 1)
        else:
            ask = ask_arr[i]
            if not np.isfinite(ask):
                continue
            limit = (1.0 - ask) - below_cents / 100.0
            no_ask = 1.0 - bid_arr[later]
            filled = bool(np.nanmin(no_ask) <= limit + 1e-9) if np.isfinite(
                np.nanmin(no_ask)) else False
            taker_px, won = float(r.ask_down), (r.y == 0)
        if not (0.01 < limit < 0.99) or not np.isfinite(taker_px):
            continue
        q_side = (r.model_probability if up else 1.0 - r.model_probability)
        if (q_side - limit) * 100.0 < gate_pp:
            continue
        fee = np.ceil(0.07 * taker_px * (1 - taker_px) * 10_000) / 10_000
        rows.append((r.window_open, bool(filled), bool(won), limit, taker_px,
                     (1.0 if won else 0.0) - limit,
                     (1.0 if won else 0.0) - taker_px - fee))
    return pd.DataFrame(rows, columns=['window_open', 'filled', 'won', 'limit',
                                       'taker_px', 'maker_pnl', 'taker_pnl'])


def main():
    s, d = load()
    print(f"scored live rows {len(s):,}   depth rows {len(d):,}   "
          f"{str(s.window_open.min())[:10]} -> {str(s.window_open.max())[:10]}\n")
    fut = {}
    for k, g in d.sort_values('offset_minutes').groupby(['symbol', 'window_open']):
        fut[k] = (g.offset_minutes.to_numpy(), g.yes_ask.to_numpy(),
                  g.yes_bid.to_numpy())
    print(f"  {'rest at':<14}{'attempts':>9}{'fills':>8}{'rate':>8}"
          f"{'MAKER pp/ct':>13}{'t':>7}{'TAKER same rows':>17}")
    import os
    gate = float(os.getenv('GATE_PP', '2.0'))
    print(f"  edge gate: model's probability must beat the RESTING price by "
          f"{gate:.1f}pp\n")
    for k in (0.0, 1.0, 2.0):
        t = simulate(s, fut, k, gate)
        if not len(t):
            print(f"  bid -{k:.0f}c: none"); continue
        f = t[t.filled]
        if len(f) < 30:
            print(f"  {'the bid' if k==0 else f'bid -{k:.0f}c':<14}{len(t):>9,}"
                  f"{len(f):>8,}{len(f)/len(t)*100:>7.1f}%   only {len(f)} fills")
            continue
        w = f.groupby('window_open').maker_pnl.mean()
        tt = w.mean() / (w.std(ddof=1) / np.sqrt(len(w)))
        print(f"  {'the bid' if k==0 else f'bid -{k:.0f}c':<14}{len(t):>9,}"
              f"{len(f):>8,}{len(f)/len(t)*100:>7.1f}%"
              f"{f.maker_pnl.mean()*100:>+12.3f}{tt:>+7.2f}"
              f"{f.taker_pnl.mean()*100:>+16.3f}")
    print("\n  TAKER same rows = crossing the spread on exactly the windows that")
    print("  filled. Adverse selection shows up as that column holding while the")
    print("  maker column falls. Fill rate is an UPPER bound (assumes queue priority).")


if __name__ == '__main__':
    main()
