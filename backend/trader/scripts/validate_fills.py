"""Refit and score `core.fills` against orders the account really rested.

The backtest books every order it places. The venue books the ones somebody
crossed to. This measures the difference, out of sample, and prints the curve
`core.fills.DEFAULT_FILL_CURVE` should carry.

    python -m scripts.validate_fills [--start 2026-10-18] [--rest-minutes 3]

Ground truth excludes tickets that never reached the book -- API rejections and
stale-quote skips are operational outcomes, and scoring a market model against
them measures the wrong thing. What remains is orders that rested and either
filled or did not.
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from core.datastore import ResearchStore
from core.fills import FillModel, run_through_margin

# The sweep has been observed cancelling at ~180s against a 60s setting, so an
# order's real life spans the placement minute and the two after it. Per-minute
# tape buckets cannot resolve it finer: `backfill_trades._rows` keeps min/max
# per (minute, side) and discards per-print timestamps.
REST_MINUTES = 3
BUCKET_EDGES = (-np.inf, 0.0, 1.0, 2.0, 3.0, 6.0, 12.0, 25.0, np.inf)

GROUND_TRUTH = text("""
    SELECT o.market_ticker, o.window_open, o.offset_minutes, o.side,
           o.limit_price, o.contracts, o.status,
           (CASE WHEN o.side='up' THEN p.outcome ELSE 1-p.outcome END) AS won
    FROM order_tickets o
    JOIN predictions p ON p.symbol=o.symbol AND p.window_open=o.window_open
                      AND p.offset_minutes=o.offset_minutes
    WHERE o.window_open >= :start AND o.market_ticker IS NOT NULL
      AND p.outcome IS NOT NULL
      AND (o.status='filled' OR o.note LIKE :lost OR o.note LIKE :cancelled)
""")


def load(start: str, rest_minutes: int) -> pd.DataFrame:
    eng = create_engine(os.environ['DATABASE_URL'])
    tix = pd.read_sql(GROUND_TRUTH, eng, params={
        'start': start, 'lost': '%no such order%',
        'cancelled': '%cancelled after%'})
    tix['window_open'] = pd.to_datetime(tix['window_open'], utc=True)
    tix['filled'] = tix['status'].eq('filled')

    tape = ResearchStore(os.environ.get('RESEARCH_STORE', '/app/data/research')
                         ).read('venue_trades')
    tape['window_open'] = pd.to_datetime(tape['window_open'], utc=True)
    tape = tape[tape['window_open'] >= start]

    parts = []
    for off in sorted(tix['offset_minutes'].dropna().unique()):
        g = tape[tape['offset_minutes'].between(int(off), int(off) + rest_minutes)]
        sell = g[g['taker_book_side'] == 'ask'].groupby('market_ticker')['min_price'].min()
        buy = g[g['taker_book_side'] == 'bid'].groupby('market_ticker')['max_price'].max()
        t = tix[tix['offset_minutes'] == off].copy()
        t['sell_low'] = t['market_ticker'].map(sell)
        t['buy_high'] = t['market_ticker'].map(buy)
        parts.append(t)
    t = pd.concat(parts, ignore_index=True)
    t['margin_c'] = run_through_margin(t['side'], t['limit_price'],
                                       t['sell_low'], t['buy_high'])
    t['pnl_c'] = 100 * (t['won'] - t['limit_price']) * t['contracts']
    return t[t['margin_c'].notna()]


def fit_curve(frame: pd.DataFrame) -> tuple[tuple[float, float], ...]:
    """Bucketed fill rates, forced non-decreasing.

    A deeper run-through consumed more of the level, so it can only make the
    fill more likely. A dip is sampling noise, and shipping one would let the
    backtest book a shallow touch more readily than a deep one -- exactly the
    error this model exists to remove.
    """
    b = buckets(frame)
    return tuple((float(u), float(p)) for u, p in
                 zip(b['upper_c'], np.maximum.accumulate(b['fill_rate'])))


def buckets(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lo, hi in zip(BUCKET_EDGES[:-1], BUCKET_EDGES[1:]):
        g = frame[(frame['margin_c'] > lo) & (frame['margin_c'] <= hi)]
        if not len(g):
            continue
        rows.append({
            'upper_c': hi, 'orders': len(g), 'fill_rate': g['filled'].mean(),
            'pp_per_ct': g['pnl_c'].sum() / g['contracts'].sum()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--start', default='2026-10-18')
    ap.add_argument('--rest-minutes', type=int, default=REST_MINUTES)
    args = ap.parse_args()

    t = load(args.start, args.rest_minutes)
    print(f'ground truth: {len(t):,} rested orders, {t["filled"].mean():.1%} filled '
          f'({int((~t["filled"]).sum())} not)')

    print('\n  fill rate and edge by run-through margin')
    print(f'    {"upper":>9s}{"orders":>8s}{"filled":>9s}{"pp/ct":>9s}')
    for _, r in buckets(t).iterrows():
        print(f'    {r.upper_c:9.0f}{int(r.orders):8d}{r.fill_rate:9.1%}{r.pp_per_ct:+9.2f}')

    # Fit on the earlier half of WINDOWS, score on the later. Splitting on rows
    # would let the same window teach and grade.
    cut = t['window_open'].quantile(0.5)
    tr, te = t[t['window_open'] <= cut], t[t['window_open'] > cut].copy()
    te['p'] = FillModel(fit_curve(tr)).probability(te['margin_c'])
    te = te[te['p'].notna()]

    naive = te['pnl_c'].sum() / te['contracts'].sum()
    weighted = (te['pnl_c'] * te['p']).sum() / (te['contracts'] * te['p']).sum()
    actual = te[te['filled']]
    paid = actual['pnl_c'].sum() / actual['contracts'].sum()

    print(f'\n  held out: {len(te):,} orders after {cut:%Y-%m-%d %H:%M}')
    print(f'    naive backtest (100% fill)   {naive:+.2f} pp/ct  '
          f'error {naive - paid:+.2f}')
    print(f'    fill-weighted                {weighted:+.2f} pp/ct  '
          f'error {weighted - paid:+.2f}')
    print(f'    what the venue paid          {paid:+.2f} pp/ct')

    # Shipped curve is refitted on EVERYTHING: the split above validates the
    # procedure, and throwing half the data away afterwards buys nothing.
    print('\n  DEFAULT_FILL_CURVE for core/fills.py (refitted on all rows):')
    for u, p in fit_curve(t):
        print(f"    ({'float(\"inf\")' if np.isinf(u) else f'{u:.1f}'}, {p:.3f}),")


if __name__ == '__main__':
    main()
