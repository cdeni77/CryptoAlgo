"""What the order book says about fills, adverse selection, and a new feature.

Three questions, none of which could be asked before this data existed. The live
fill estimates rest on 338 orders; the adverse-selection estimate rests on
nothing at all.

**1. Would the order have filled?** For each window the model would have traded,
compare the contracts it wanted against the size actually resting at or inside
its limit. `decide()` already caps the stake at the touch when a book is present,
so the question is what that cap costs in practice.

**2. Adverse selection.** The one that matters. If the windows where size was
available perform WORSE than the windows where it was not, then being filled is
itself bad news — someone was willing to sell precisely when we were wrong — and
every economics number in this project is optimistic. If there is no difference,
the fill assumption is sound and the measured edge stands.

**3. Book imbalance as a feature.** `CLAUDE.md` lists imbalance, ladder slope,
depth asymmetry and level counts as "plausible carriers that appear in no feature
group". Tested here against the market's own residual, controlling for price,
clustered by day — the same bar the trade tape failed at t = -1.90 over sixteen
cells. One projection, pre-registered, not a sweep over the ladder.

Question 2 is the one to read first. It is the only one that could invalidate
rather than merely extend.
"""

from __future__ import annotations

# This file moved into research/analysis/ during a repo cleanup; `core`/`scripts`
# are packages rooted at backend/trader/, which Python does not add to
# sys.path automatically for a script run from a subdirectory (only the
# script's OWN directory is added). Without this, every `from core...`
# import below raises ModuleNotFoundError the moment the file is not sitting
# directly in backend/trader/.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))

import math
import os

import numpy as np
import pandas as pd

from core.config import DEFAULT_CONFIG
from core.datastore import ResearchStore
from core.promotion import load_live
from scripts.retro_forecast_test import score_artifact

pd.set_option('display.width', 220)
TAKER = 0.07
HALF = 0.005
OFFSET = int(os.getenv('BOOK_OFFSET', '12'))


def fee(p):
    return np.ceil(TAKER * p * (1.0 - p) * 10_000.0) / 10_000.0


def main() -> int:
    cfg = DEFAULT_CONFIG
    store = ResearchStore(os.getenv('RESEARCH_STORE'))

    # **Read the unified table, not a JSONL side-archive.** This used to load
    # `data/book_at_decision.jsonl` — a truncated, +12m-only pull that has since
    # been retired. `venue_depth` carries every source at every minute with a
    # `source` column, so the offset is a parameter now and the same analysis
    # runs against live recording or Predexon backfill without a second reader.
    depth = store.read('venue_depth')
    depth = depth[(depth['venue'] == 'kalshi')
                  & (depth['offset_minutes'] == OFFSET)]
    b = depth.drop_duplicates(['symbol', 'window_open'], keep='last').copy()
    b['window_open'] = pd.to_datetime(b['window_open'], utc=True)
    if not len(b):
        print(f'no venue_depth rows at +{OFFSET}m'); return 1
    print(f'book rows {len(b):,} at +{OFFSET}m  '
          f'{b["window_open"].min().date()} .. {b["window_open"].max().date()}  '
          f'sources: {b["source"].value_counts().to_dict()}')
    quotes = store.read('venue_quotes')
    quotes['window_open'] = pd.to_datetime(quotes['window_open'], utc=True)
    usable = quotes.loc[quotes['usable'].astype(bool)].copy()
    model = load_live(config=cfg)
    scored = score_artifact(model, cfg, usable)
    j = scored.merge(
        usable[['symbol', 'window_open', 'offset_minutes', 'market_probability',
                'spread']],
        left_on=['symbol', 'window_open', 'offset'],
        right_on=['symbol', 'window_open', 'offset_minutes'], how='inner')
    j = j[(j['offset'] == 12) & j['outcome'].isin([0, 1, True, False])].copy()
    j['outcome'] = j['outcome'].astype(float)
    j['spread'] = j['spread'].fillna(0.01)
    j = j.merge(b, on=['symbol', 'window_open'], how='inner',
                suffixes=('', '_book'))
    print(f'joined with scores: {len(j):,} windows\n')
    if len(j) < 100:
        print('too few joined rows to conclude anything')
        return 1

    mid = j['market_probability'].to_numpy(float)
    half = np.maximum(j['spread'].to_numpy(float) / 2.0, HALF)
    q = j['model_probability'].to_numpy(float)
    y = j['outcome'].to_numpy(float)
    ay = np.clip(mid + half, 1e-4, 1 - 1e-4)
    an = np.clip((1 - mid) + half, 1e-4, 1 - 1e-4)
    ey, en = q - ay - fee(ay), (1 - q) - an - fee(an)
    yes = ey >= en
    j['side_yes'] = yes
    j['price'] = np.where(yes, ay, an)
    j['edge'] = np.where(yes, ey, en)
    j['pnl_per'] = np.where(yes, y - ay - fee(ay), (1 - y) - an - fee(an))

    cost = j['price'] + fee(j['price'])
    kelly = np.clip((np.where(yes, q, 1 - q) - cost) / np.maximum(1 - cost, 1e-6), 0, 1)
    stake = np.minimum(cfg.kelly_fraction * kelly * cfg.starting_bankroll,
                       cfg.max_stake_fraction * cfg.starting_bankroll)
    stake = np.minimum(stake, cfg.max_stake_dollars or 25.0)
    j['wanted'] = np.floor(stake / cost).astype(int)

    # A YES buy lifts the resting ASK side; a NO buy lifts the resting BID side
    # (selling YES). `ask_at_touch` / `bid_at_touch` are the sizes at the touch,
    # and the 1c figures are what a limit one tick through would reach.
    j['avail_touch'] = np.where(yes, j['yes_ask_size'],
                                j['yes_bid_size']).astype(float)
    j['avail_1c'] = np.where(yes, j['depth_ask_1c'],
                             j['depth_bid_1c']).astype(float)

    trade = (j['edge'] > cfg.min_edge_pp / 100.0) & (j['wanted'] >= cfg.min_contracts) \
        & (j['price'] >= cfg.min_traded_price) & (j['price'] <= cfg.max_traded_price)
    t = j[trade].copy()
    print('=' * 78)
    print(f'1. WOULD IT HAVE FILLED?  ({len(t):,} windows the model would trade)')
    print('=' * 78)
    for label, col in (('at the touch', 'avail_touch'), ('within 1c', 'avail_1c')):
        full = (t[col] >= t['wanted']).mean()
        none = (t[col] <= 0).mean()
        ratio = np.clip(t[col] / t['wanted'].replace(0, np.nan), 0, 1)
        print(f'  {label:<14} full fill {full:6.1%}   nothing resting {none:6.1%}   '
              f'mean fillable share {ratio.mean():6.1%}')
    print(f'  median wanted {t["wanted"].median():.0f} contracts, '
          f'median available at touch {t["avail_touch"].median():.0f}')

    print('\n' + '=' * 78)
    print('2. ADVERSE SELECTION — do the fillable windows perform worse?')
    print('=' * 78)
    t['fillable'] = t['avail_touch'] >= t['wanted']
    for name, part in t.groupby('fillable'):
        lbl = 'fillable' if name else 'NOT fillable'
        print(f'  {lbl:<13} n={len(part):>5}  mean edge claimed '
              f'{100*part["edge"].mean():6.2f}pp  realised '
              f'{100*part["pnl_per"].mean():+7.3f}c  win rate '
              f'{(part["pnl_per"] > 0).mean():5.1%}')
    a = t[t['fillable']]['pnl_per']
    c = t[~t['fillable']]['pnl_per']
    if len(a) > 5 and len(c) > 5:
        diff = a.mean() - c.mean()
        se = math.sqrt(a.var(ddof=1) / len(a) + c.var(ddof=1) / len(c))
        print(f'\n  difference (fillable - not): {100*diff:+.3f}c  '
              f't {diff/se if se else float("nan"):+.2f}')
        print('  Negative and significant would mean being filled is itself bad news')
        print('  and every economics number here is optimistic.')

    print('\n' + '=' * 78)
    print('3. BOOK IMBALANCE as a feature — does it beat the market residual?')
    print('=' * 78)
    # Imbalance is derived here rather than stored: (bid - ask) / (bid + ask)
    # over the resting totals. Storing a ratio would fix one projection of the
    # ladder and foreclose the others, which is the whole reason the levels are
    # kept.
    bid_tot = j['depth_bid_total'].astype(float)
    ask_tot = j['depth_ask_total'].astype(float)
    total = bid_tot + ask_tot
    imb = ((bid_tot - ask_tot) / total.where(total > 0)).astype(float)
    ok = np.isfinite(imb) & np.isfinite(mid)
    resid = y[ok] - mid[ok]
    x = imb[ok].to_numpy()
    design = np.column_stack([np.ones(ok.sum()), mid[ok], mid[ok] ** 2])
    beta, *_ = np.linalg.lstsq(design, x, rcond=None)
    x_orth = x - design @ beta
    day = j.loc[ok, 'window_open'].dt.floor('D').to_numpy()
    per_day = []
    for d in np.unique(day):
        m = day == d
        if m.sum() >= 8 and x_orth[m].std() > 0 and resid[m].std() > 0:
            per_day.append(np.corrcoef(x_orth[m], resid[m])[0, 1])
    arr = np.array(per_day)
    pooled = np.corrcoef(x_orth, resid)[0, 1]
    tstat = (arr.mean() / (arr.std(ddof=1) / math.sqrt(len(arr)))
             if len(arr) > 2 else float('nan'))
    print(f'  pooled corr(imbalance | price, outcome - mid) = {pooled:+.4f}')
    print(f'  mean daily corr {arr.mean():+.4f}  t {tstat:+.3f}  days {len(arr)}')
    print('  (the trade tape failed exactly this test at t = -1.90 over 16 cells)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
