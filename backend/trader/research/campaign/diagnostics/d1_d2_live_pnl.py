"""D1/D2: is the live P&L distinguishable from noise, and do >90c favourites carry it?

Reads the venue's OWN settlement ledger (trades_db.venue_settlements) exported to TSV.
Every number here is the venue's, not our bookkeeping.
"""
import sys, numpy as np, pandas as pd

TSV = sys.argv[1]
cols = ['ticker','market_result','yes_contracts','no_contracts','yes_cost','no_cost',
        'revenue','fee_cost','pnl','settled_time']
df = pd.read_csv(TSV, sep='\t', names=cols, header=None)
df['settled_time'] = pd.to_datetime(df['settled_time'], utc=True, format='mixed')
for c in ['yes_contracts','no_contracts','yes_cost','no_cost','revenue','fee_cost','pnl']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['contracts'] = df.yes_contracts + df.no_contracts
df['cost'] = df.yes_cost.fillna(0) + df.no_cost.fillna(0)
df = df[df.contracts > 0].copy()
df['side'] = np.where(df.yes_contracts > 0, 'yes', 'no')
df['price'] = df.cost / df.contracts                      # avg entry price per contract
df['won'] = (df.market_result == df.side)
df['symbol'] = df.ticker.str.extract(r'KX(BTC|ETH|SOL)15M')[0]
# the 15-minute window is the correlation unit: three symbols settle together
df['window'] = df.settled_time.dt.floor('15min')
# per-contract P&L in probability points (cents)
df['pnl_pc'] = df.pnl / df.contracts
df['fee_pc'] = df.fee_cost / df.contracts
df['month'] = df.settled_time.dt.to_period('M').astype(str)

def cluster_t(g_pnl, g_n):
    """t-stat on TOTAL P&L, clustered on the window. Sums within cluster."""
    s = pd.DataFrame({'p': g_pnl, 'w': g_n}).groupby('w')['p'].sum()
    n = len(s)
    return s.sum(), s.mean(), s.std(ddof=1)/np.sqrt(n)*n, s.mean()/(s.std(ddof=1)/np.sqrt(n)), n

def block_boot(series_by_window, B=20000, seed=0):
    rng = np.random.default_rng(seed)
    v = series_by_window.values; n = len(v)
    idx = rng.integers(0, n, size=(B, n))
    return np.quantile(v[idx].sum(axis=1), [0.025, 0.5, 0.975])

print("="*78)
print("D1 — IS THE LIVE P&L DISTINGUISHABLE FROM NOISE?")
print("="*78)
tot, mu, se_tot, t, nclust = cluster_t(df.pnl, df.window)
print(f"settlements          {len(df):>10,}")
print(f"contracts            {df.contracts.sum():>10,.0f}")
print(f"gross cost           ${df.cost.sum():>9,.2f}")
print(f"fees paid            ${df.fee_cost.sum():>9,.2f}")
print(f"total venue P&L      ${tot:>9,.2f}")
print(f"P&L per contract      {df.pnl.sum()/df.contracts.sum()*100:>9.3f} cents = pp of edge")
print(f"independent windows  {nclust:>10,}")
print(f"cluster-robust t      {t:>10.3f}   (clustered on the 15-min window)")
w = pd.DataFrame({'p': df.pnl, 'w': df.window}).groupby('w')['p'].sum()
lo, med, hi = block_boot(w)
print(f"block bootstrap 95%  [${lo:,.2f}, ${hi:,.2f}]   median ${med:,.2f}")
print(f"  -> {'DISTINGUISHABLE' if lo>0 else 'NOT distinguishable from zero'}")

print(f"\nby month:")
for m, g in df.groupby('month'):
    gw = pd.DataFrame({'p': g.pnl, 'w': g.window}).groupby('w')['p'].sum()
    tt = gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw)))
    print(f"  {m}  n={len(g):>5}  contracts={g.contracts.sum():>6,.0f}  "
          f"pnl=${g.pnl.sum():>8.2f}  pp/contract={g.pnl.sum()/g.contracts.sum()*100:>7.3f}  t={tt:>6.2f}")

print("\n" + "="*78)
print("D2 — DO >90c FAVOURITES CARRY IT, AND DO THEY BEAT BREAK-EVEN?")
print("="*78)
bins = [0, .10, .30, .50, .70, .80, .90, .95, 1.01]
df['bucket'] = pd.cut(df.price, bins, right=False)
print(f"{'price bucket':>14} {'n':>5} {'contracts':>10} {'cost$':>9} {'pnl$':>8} "
      f"{'pp/ct':>7} {'win%':>6} {'breakeven%':>10} {'excess_pp':>10} {'t':>6}")
rows = []
for b, g in df.groupby('bucket', observed=True):
    if len(g) == 0: continue
    # break-even probability: must recover price + fee per contract
    be = (g.cost.sum() + g.fee_cost.sum()) / g.contracts.sum()
    win = g.won.sum()/len(g)
    # contract-weighted win rate is the right comparison against a cost basis
    win_w = (g.won * g.contracts).sum()/g.contracts.sum()
    excess = (win_w - be) * 100
    gw = pd.DataFrame({'p': g.pnl, 'w': g.window}).groupby('w')['p'].sum()
    tt = gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw))) if len(gw) > 1 else np.nan
    print(f"{str(b):>14} {len(g):>5} {g.contracts.sum():>10,.0f} {g.cost.sum():>9.2f} "
          f"{g.pnl.sum():>8.2f} {g.pnl.sum()/g.contracts.sum()*100:>7.3f} {win_w*100:>6.2f} "
          f"{be*100:>10.2f} {excess:>10.3f} {tt:>6.2f}")
    rows.append((str(b), len(g), g.contracts.sum(), g.pnl.sum(), excess, tt))

print("\nconcentration:")
hi90 = df[df.price >= 0.90]; lo90 = df[df.price < 0.90]
for nm, g in [('>=90c', hi90), ('<90c', lo90)]:
    gw = pd.DataFrame({'p': g.pnl, 'w': g.window}).groupby('w')['p'].sum()
    lo_, _, hi_ = block_boot(gw)
    tt = gw.mean()/(gw.std(ddof=1)/np.sqrt(len(gw)))
    print(f"  {nm:>6}  n={len(g):>5}  contracts={g.contracts.sum():>6,.0f}  "
          f"pnl=${g.pnl.sum():>8.2f} ({g.pnl.sum()/tot*100:>6.1f}% of total)  "
          f"pp/ct={g.pnl.sum()/g.contracts.sum()*100:>7.3f}  t={tt:>6.2f}  "
          f"boot95=[{lo_:,.2f},{hi_:,.2f}]")

print("\nsingle-settlement concentration (is one trade the whole result?):")
top = df.reindex(df.pnl.abs().sort_values(ascending=False).index).head(5)
for _, r in top.iterrows():
    print(f"  {r.ticker:<28} {r.side:>3} {r.contracts:>4.0f}ct @ {r.price:.3f}  pnl=${r.pnl:>8.2f}")
print(f"  total P&L excluding single largest winner: ${tot - top.pnl.iloc[0]:.2f}")
print(f"  sum of top-5 |pnl|: ${top.pnl.sum():.2f} against total ${tot:.2f}")
