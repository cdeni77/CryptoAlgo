"""D4b: is the backfill-vs-live gap a real bias, or regression to the mean?

Two controls:
 1. SYMMETRY. Condition on the BACKFILL mid instead of the live mid. If the sign
    of the gap flips, the pattern in D4 was conditioning bias from independent
    noise, not a bias in the backfill.
 2. WHICH PRICE IS TRUE. Score both against the venue's own settlement. The real
    market price must be the better-calibrated forecast. This cannot be argued
    around: a price that predicts settlement worse is not the price that traded.
"""
import duckdb, numpy as np, pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

pair = pd.read_parquet('research/campaign/diagnostics/d4_pairs.parquet')
pair['mid_bf'] = (pair.ask_bf + pair.bid_bf)/2

print("="*78); print("CONTROL 1 — SYMMETRY")
print("="*78)
print(f"{'band':>14} {'n':>7} {'cond on LIVE mid':>18} {'n':>7} {'cond on BACKFILL mid':>22}")
for lo, hi in [(0,.10),(.10,.30),(.30,.50),(.50,.70),(.70,.90),(.90,1.01)]:
    a = pair[(pair.mid_lv>=lo)&(pair.mid_lv<hi)]
    b = pair[(pair.mid_bf>=lo)&(pair.mid_bf<hi)]
    if len(a)<30 or len(b)<30: continue
    print(f"[{lo:.2f},{hi:.2f}) {len(a):>7,} {a.d_ask.mean()*100:>+17.3f}pp {len(b):>7,} {b.d_ask.mean()*100:>+21.3f}pp")
print("\nIf the two columns have OPPOSITE signs the D4 pattern is regression to the")
print("mean from independent noise. If they AGREE the backfill is genuinely biased.")

con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
sett = con.execute("""
with s as (select *, row_number() over (partition by venue,symbol,event_time order by available_time desc) rn
  from read_parquet('data/research/venue_settlements/**/*.parquet', union_by_name=true))
select symbol, window_open, settled_up from s
where rn=1 and venue='kalshi' and settled_up is not null""").df()
sett['window_open'] = pd.to_datetime(sett.window_open, utc=True)
p = pair.copy()
p['window_open'] = pd.to_datetime(p.window_open, utc=True)
m = p.merge(sett, on=['symbol','window_open'], how='inner')
m['y'] = m.settled_up.astype(int)
print(f"\n{'='*78}"); print("CONTROL 2 — WHICH PRICE PREDICTS THE SETTLEMENT?")
print("="*78)
print(f"paired rows with a venue label: {len(m):,}")

def score(prob, y, name):
    prob = np.clip(prob, 1e-6, 1-1e-6)
    ll = log_loss(y, prob); br = brier_score_loss(y, prob)
    # calibration error: |mean(prob) - mean(y)| in 10 bins, population-weighted
    bins = pd.qcut(prob, 10, duplicates='drop')
    ce = pd.DataFrame({'p':prob,'y':y,'b':bins}).groupby('b', observed=True).apply(
        lambda g: abs(g.p.mean()-g.y.mean())*len(g), include_groups=False).sum()/len(y)
    print(f"  {name:>26}  log_loss {ll:.6f}   brier {br:.6f}   calib_err {ce:.5f}")
    return ll

ll_lv = score(m.mid_lv, m.y, 'live mid (stream)')
ll_bf = score(m.mid_bf, m.y, 'backfill mid (Predexon)')
print(f"\n  live - backfill log loss = {ll_lv-ll_bf:+.6f}  "
      f"({'LIVE is the better forecast' if ll_lv<ll_bf else 'BACKFILL is the better forecast'})")

# paired bootstrap on the window, for a CI on the difference
m['w'] = m.window_open
def pll(prob, y): 
    prob=np.clip(prob,1e-6,1-1e-6); return -(y*np.log(prob)+(1-y)*np.log(1-prob))
m['d'] = pll(m.mid_lv.values, m.y.values) - pll(m.mid_bf.values, m.y.values)
g = m.groupby('w')['d'].mean()
rng = np.random.default_rng(0); v=g.values; n=len(v)
bs = v[rng.integers(0,n,size=(10000,n))].mean(axis=1)
print(f"  paired bootstrap 95% CI on (live - backfill) log loss: "
      f"[{np.quantile(bs,.025):+.6f}, {np.quantile(bs,.975):+.6f}]  over {n:,} windows")

print(f"\n{'='*78}"); print("CONSEQUENCE — apparent edge if a model prices off the backfill")
print("="*78)
# a model that believes the LIVE mid is fair but trades at the BACKFILL ask
m['apparent_edge_pp'] = (m.mid_lv - m.ask_bf)*100
m['true_edge_pp']     = (m.mid_lv - m.ask_lv)*100
for nm in ['apparent_edge_pp','true_edge_pp']:
    sel = m[m[nm] >= 1.5]
    if not len(sel): print(f"  {nm}: no rows clear 1.5pp"); continue
    real = (sel.y - sel.ask_lv)*100     # what the trade actually pays, per contract
    print(f"  gate>=1.5pp on {nm:>18}: {len(sel):>6,} rows ({len(sel)/len(m)*100:>5.2f}%)  "
          f"realised pp/contract at the TRUE ask {real.mean():>+7.3f}")
