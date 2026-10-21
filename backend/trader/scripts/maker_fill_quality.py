"""Did a resting order fill, was it a MAKER fill, and how far from the touch?

**The one number the strategy now turns on.** Measured 2026-10-17 on 54,100
live-recorded rows graded on the venue's own settlement, charging the fee the
way the venue charges it:

    scenario                          tradeable   signal    cost      NET      t
    TAKER: cross the spread + fee        33.9%     1.525   1.620   -0.095   -1.44
    MAKER: rest at the touch, $0 fee    100.0%     0.916   0.000   +0.916   +3.41
    MAKER: fill 1c worse than touch      51.4%     1.209   1.000   +0.209   +1.34
    MAKER: fill 2c worse than touch      24.8%     1.246   2.000   -0.754   -0.43

**Break-even is about 1.2 cents of adverse selection.** Inside it the model
pays; beyond it it does not. Nothing in this repo answers it: all 328 live fills
came back `is_taker: true`, because `KalshiClient.place_order` defaults to
`immediate_or_cancel`, which cannot rest by construction. The maker rate is
already modelled as zero (`Config.maker_fee_multiplier = 0.0`) and the venue's
own order ticket agrees -- a 12-contract limit at 63c against a 65/66 book
quotes "$7.56 ($0 fee)".

**This script places nothing.** Resting orders are placed by hand, in the app or
by whatever the operator prefers; `scripts/sync_venue.py` then pulls them into
`venue_fills`, which already carries `is_taker`, `price`, `contracts` and
`created_time`. This reads those back, reconstructs the touch that stood at the
moment of each fill from `venue_depth`, and reports the gap. That division is
deliberate: the measurement needs no new order path, and an order path built to
answer a question tends to be kept and traded.

    python -m scripts.sync_venue          # pull the ledger first
    python -m scripts.maker_fill_quality  # then read it back
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger('maker_fill_quality')
BREAK_EVEN_CENTS = 1.2


def _args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--since', default=None, help='ISO date; default all')
    p.add_argument('--database-url', default=os.getenv('DATABASE_URL'))
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    return p.parse_args()


def _fills(database_url, since) -> pd.DataFrame:
    from sqlalchemy import create_engine, select
    from core.pg_writer import VenueFill
    q = select(VenueFill.trade_id, VenueFill.ticker, VenueFill.side,
               VenueFill.contracts, VenueFill.price, VenueFill.is_taker,
               VenueFill.created_time)
    if since:
        q = q.where(VenueFill.created_time >= pd.Timestamp(since, tz='UTC'))
    engine = create_engine(database_url, future=True)
    try:
        with engine.connect() as c:
            rows = c.execute(q).all()
    finally:
        engine.dispose()
    return pd.DataFrame(rows, columns=['trade_id', 'ticker', 'side', 'contracts',
                                       'price', 'is_taker', 'created_time'])


def _touch_at(store, fills: pd.DataFrame) -> pd.Series:
    """The best price standing on the fill's own side, at the minute it filled.

    From `venue_depth`, which records every minute. NaN where no row covers the
    fill -- never a default, because a defaulted touch would report a fill as
    perfectly priced when in fact nothing is known about it.
    """
    out = pd.Series(np.nan, index=fills.index, dtype=float)
    if not len(fills):
        return out
    try:
        depth = store.read('venue_depth')
    except Exception as exc:                                   # noqa: BLE001
        logger.warning('no venue_depth (%s); cannot reconstruct the touch',
                       str(exc)[:60])
        return out
    if depth is None or not len(depth):
        return out
    depth = depth.copy()
    depth['event_time'] = pd.to_datetime(depth['event_time'], utc=True)
    for col in ('yes_bid', 'yes_ask'):
        if col in depth:
            depth[col] = pd.to_numeric(depth[col], errors='coerce')
    for i, row in fills.iterrows():
        t = pd.to_datetime(row['created_time'], utc=True, errors='coerce')
        if pd.isna(t):
            continue
        g = depth[depth.get('market_ticker', depth.get('ticker')) == row['ticker']]
        if not len(g):
            continue
        g = g[(g.event_time >= t - pd.Timedelta(minutes=2)) & (g.event_time <= t)]
        if not len(g):
            continue
        last = g.sort_values('event_time').iloc[-1]
        # Buying UP lifts the YES ask; buying DOWN lifts the NO ask, which is
        # 1 - best yes bid. Same convention as `core/decide.price_and_edge`.
        px = last.get('yes_ask') if row['side'] == 'up' else (
            1.0 - last['yes_bid'] if pd.notna(last.get('yes_bid')) else np.nan)
        out.at[i] = float(px) if pd.notna(px) else np.nan
    return out


def main() -> int:
    a = _args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s')
    if not a.database_url:
        raise SystemExit('set DATABASE_URL (or pass --database-url)')
    fills = _fills(a.database_url, a.since)
    if not len(fills):
        print('no fills in venue_fills — run `python -m scripts.sync_venue` first')
        return 0
    n_taker = int((fills.is_taker == True).sum())            # noqa: E712
    n_maker = int((fills.is_taker == False).sum())           # noqa: E712
    print(f'{len(fills):,} fills   taker {n_taker:,}   maker {n_maker:,}   '
          f'unknown {len(fills) - n_taker - n_maker:,}\n')
    if not n_maker:
        print('NO MAKER FILLS YET. That is the finding, not an error: until a')
        print('resting order fills, the +0.916pp / t=+3.41 maker row in the')
        print('table above is a model, not a measurement. Rest some orders and')
        print('re-run.')
        return 0
    from core.datastore import ResearchStore
    store = ResearchStore(a.research_store) if a.research_store else ResearchStore()
    maker = fills[fills.is_taker == False].copy()             # noqa: E712
    maker['touch'] = _touch_at(store, maker)
    known = maker.dropna(subset=['touch', 'price'])
    print(f'maker fills with a reconstructable touch: {len(known):,} of {len(maker):,}')
    if not len(known):
        print('  none — venue_depth does not cover these minutes.')
        return 0
    slip = (known.price - known.touch) * 100.0
    print(f'\n  fill price minus the touch, in CENTS '
          f'(negative = better than crossing):')
    for q in (10, 25, 50, 75, 90):
        print(f'    p{q:<3}{np.percentile(slip, q):>+8.3f}')
    print(f'    mean {slip.mean():+.3f}   n={len(slip):,}')
    print(f'\n  BREAK-EVEN IS {BREAK_EVEN_CENTS}c. '
          f'{"PASSES" if slip.mean() < BREAK_EVEN_CENTS else "FAILS"}: '
          f'mean adverse selection {slip.mean():+.3f}c against a '
          f'{BREAK_EVEN_CENTS}c budget.')
    print('\n  Caveat the number cannot show: this measures fills that HAPPENED.')
    print('  Orders that never filled are the other half of the cost and are')
    print('  not in venue_fills at all — a high fill rate on adverse moves and')
    print('  a low one otherwise is exactly what adverse selection looks like.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
