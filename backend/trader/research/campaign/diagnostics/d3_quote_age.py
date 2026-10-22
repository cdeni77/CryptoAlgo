"""D3: how does quote age affect the market benchmark, and is any backfill subset usable?

D4b showed the backfill mid is 0.0425 nats worse than the live mid at predicting
settlement. The question that decides the campaign: is that concentrated in STALE
backfill rows, leaving a fresh subset that is as good as the live book? If yes,
Tier B survives with an age cut. If no, the only trustworthy market price in this
repository is the ~5,000 symbol-windows of live recording.
"""
import numpy as np, pandas as pd, duckdb
from sklearn.metrics import log_loss

pair = pd.read_parquet('research/campaign/diagnostics/d4_pairs.parquet')
pair['mid_bf'] = (pair.ask_bf + pair.bid_bf)/2
pair['mid_lv'] = (pair.ask_lv + pair.bid_lv)/2
con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
sett = con.execute("""
with s as (select *, row_number() over (partition by venue,symbol,event_time order by available_time desc) rn
  from read_parquet('data/research/venue_settlements/**/*.parquet', union_by_name=true))
select symbol, window_open, settled_up from s
where rn=1 and venue='kalshi' and settled_up is not null""").df()
sett['window_open']=pd.to_datetime(sett.window_open,utc=True)
pair['window_open']=pd.to_datetime(pair.window_open,utc=True)
m = pair.merge(sett, on=['symbol','window_open'], how='inner')
m['y']=m.settled_up.astype(int)

def ll(p,y):
    p=np.clip(p,1e-6,1-1e-6); return float(log_loss(y,p))

print("="*86)
print("LOG LOSS vs SETTLEMENT, BY BACKFILL QUOTE AGE  (live mid on the same rows = reference)")
print("="*86)
print(f"{'backfill age':>16} {'n':>8} {'ll_backfill':>12} {'ll_live':>10} {'gap':>10} {'mean|bf-lv|pp':>14}")
edges=[(0,5),(5,15),(15,30),(30,60),(60,120),(120,300),(300,1e9)]
for lo,hi in edges:
    g=m[(m.age_bf>=lo)&(m.age_bf<hi)]
    if len(g)<200: continue
    a,b=ll(g.mid_bf,g.y), ll(g.mid_lv,g.y)
    print(f"[{lo:>5.0f},{hi if hi<1e8 else np.inf:>6.0f}) {len(g):>8,} {a:>12.6f} {b:>10.6f} "
          f"{a-b:>+10.6f} {(g.mid_bf-g.mid_lv).abs().mean()*100:>14.3f}")
g=m[m.age_bf.isna()]
if len(g)>200:
    a,b=ll(g.mid_bf,g.y), ll(g.mid_lv,g.y)
    print(f"{'null age':>16} {len(g):>8,} {a:>12.6f} {b:>10.6f} {a-b:>+10.6f} "
          f"{(g.mid_bf-g.mid_lv).abs().mean()*100:>14.3f}")

print("\nEVEN THE FRESHEST backfill rows: is the gap gone?")
fresh=m[m.age_bf<=5]
if len(fresh)>200:
    rng=np.random.default_rng(0)
    d=(-(fresh.y*np.log(np.clip(fresh.mid_bf,1e-6,1-1e-6))+(1-fresh.y)*np.log(np.clip(1-fresh.mid_bf,1e-6,1-1e-6)))
       -(-(fresh.y*np.log(np.clip(fresh.mid_lv,1e-6,1-1e-6))+(1-fresh.y)*np.log(np.clip(1-fresh.mid_lv,1e-6,1-1e-6)))))
    gw=pd.DataFrame({'d':d,'w':fresh.window_open}).groupby('w')['d'].mean()
    v=gw.values;n=len(v);bs=v[rng.integers(0,n,size=(10000,n))].mean(axis=1)
    print(f"  n={len(fresh):,} over {n:,} windows; backfill-minus-live log loss "
          f"{d.mean():+.6f}  95% CI [{np.quantile(bs,.025):+.6f}, {np.quantile(bs,.975):+.6f}]")

print("\n" + "="*86)
print("WHY: is the gap staleness, or is the backfill a DIFFERENT book?")
print("="*86)
# if it were staleness, the backfill should match the live book at an EARLIER minute
best=[]
for lag in range(0,6):
    sub=m[['symbol','window_open','offset_minutes','mid_bf']].copy()
    sub['offset_minutes']=sub.offset_minutes-lag
    j=sub.merge(m[['symbol','window_open','offset_minutes','mid_lv']],
                on=['symbol','window_open','offset_minutes'],how='inner')
    best.append((lag,len(j),(j.mid_bf-j.mid_lv).abs().mean()*100))
print(f"  {'lag(min)':>9} {'n':>8} {'mean |bf - live@t-lag|pp':>26}")
for lag,n,e in best: print(f"  {lag:>9} {n:>8,} {e:>26.3f}")
print("  (a staleness story predicts the error MINIMISES at some lag>0)")
