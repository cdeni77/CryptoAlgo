"""D12 — age-threshold sensitivity as DATA QUALITY, not as a trading hyperparameter.

No strategy metric appears here. The quantities reported are properties of the
dataset and of the market price itself:

  * retained rows / windows / symbol-windows, and coverage by month, symbol, offset
  * the MARKET quote's own log loss and calibration against the venue's settlement

The last is a statement about the price, not about any model: a quote that is a
genuinely resting order should still be a decent forecast of its own market's
outcome. A quote that is a missing update should not be, because it describes a book
that no longer exists.

**The threshold is applied to BACKFILL ONLY.** D10 established that
`quote_age_seconds` measures different things per source -- recency of the last
recorded change for `backfill`, the sampler's own polling lag (a constant ~32s) for
the live recorders. Applying one cut to both deletes the independent recorder for a
reason that has nothing to do with staleness.
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H

THRESHOLDS = [1, 2, 5, 10, 15, 30, 60, 120, None]

px = H.market_prices(max_book_age_s=1e9)      # unfiltered
lab = H.venue_labels()
d = px.merge(lab, on=['symbol', 'window_open'], how='inner')
d['y'] = d.settled_up.astype(int)
d['is_backfill'] = d.quote_source == 'backfill'
print(f'unfiltered: {len(d):,} rows, {d.window_open.nunique():,} windows, '
      f'{d.groupby(["symbol","window_open"]).ngroups:,} symbol-windows')
print(f'  backfill {d.is_backfill.sum():,}  other {(~d.is_backfill).sum():,}')

def keep(th):
    if th is None:
        return np.ones(len(d), bool)
    # live rows are unaffected: their age is sampler lag, already handled by the
    # as-of join on available_time
    return (~d.is_backfill) | (d.book_age_s.notna() & (d.book_age_s <= th))

print(f"\n{'max backfill':>13} {'rows':>9} {'%kept':>7} {'windows':>8} {'sym-win':>8} "
      f"{'market logloss':>15} {'market calib':>13} {'market brier':>13}")
rows = []
for th in THRESHOLDS:
    k = keep(th); g = d[k]
    if not len(g): continue
    ll = float(np.mean(H._ll(g.p_mkt, g.y)))
    cal = H.calibration(g.p_mkt, g.y)
    br = float(np.mean((g.p_mkt - g.y) ** 2))
    lbl = 'none' if th is None else f'{th}s'
    print(f"{lbl:>13} {len(g):>9,} {k.mean()*100:>6.1f}% {g.window_open.nunique():>8,} "
          f"{g.groupby(['symbol','window_open']).ngroups:>8,} {ll:>15.6f} "
          f"{cal['error']:>13.5f} {br:>13.6f}")
    rows.append({'threshold_s': th, 'rows': int(len(g)), 'kept_pct': float(k.mean()*100),
                 'windows': int(g.window_open.nunique()),
                 'symbol_windows': int(g.groupby(['symbol','window_open']).ngroups),
                 'market_log_loss': ll, 'market_calibration_error': cal['error'],
                 'market_brier': br})

print("\n--- the market quote's OWN forecast quality, by its age band (backfill only) ---")
b = d[d.is_backfill].copy()
b['band'] = pd.cut(b.book_age_s, [-.01,1,2,5,10,15,30,60,120,1e9],
                   labels=['<1s','1-2s','2-5s','5-10s','10-15s','15-30s','30-60s',
                           '60-120s','>120s'])
print(f"{'age band':>10} {'n':>9} {'market logloss':>15} {'market calib':>13} {'mean |p-0.5|':>13}")
for bd, g in b.groupby('band', observed=True):
    if len(g) < 200: continue
    print(f"{str(bd):>10} {len(g):>9,} {np.mean(H._ll(g.p_mkt,g.y)):>15.6f} "
          f"{H.calibration(g.p_mkt,g.y)['error']:>13.5f} {(g.p_mkt-0.5).abs().mean():>13.4f}")

print("\n--- coverage change by month, at each candidate threshold ---")
d['month'] = d.window_open.dt.to_period('M').astype(str)
cov = {}
for th in [1, 5, 15, 30, None]:
    k = keep(th)
    cov['none' if th is None else f'{th}s'] = (d[k].groupby('month')
                                               .apply(lambda g: g.groupby(['symbol','window_open']).ngroups,
                                                      include_groups=False))
print(pd.DataFrame(cov).to_string())

pd.DataFrame(rows).to_json(HERE / 'd12_threshold_sensitivity.json',
                           orient='records', indent=2)
print('\nwrote d12_threshold_sensitivity.json')
