"""D5b: is `live_touch` a trustworthy market price?

It matters enormously. The full live ladder covers only 2026-08-25..09-16 (6,488
symbol-windows, 23 days). `live_touch` — the trading loop's own REST fetch at the
decision instant — covers 2026-08-27..10-17 (14,178 symbol-windows) at exactly the
four decision offsets. If it agrees with the stream ladder it more than doubles the
trustworthy sample and extends it through October. If it does not, the campaign has
23 days of usable market price.

Known hazard from CLAUDE.md: live_touch used to re-fetch the ladder AFTER the order
was placed, so its depth described a book our own fill had eaten into. Fixed
2026-09-16 via `transport`. Both eras are tested separately.
"""
import duckdb, numpy as np, pandas as pd
from sklearn.metrics import log_loss

con=duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
con.execute("""create view d as
with x as (select *, row_number() over (
   partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('data/research/venue_depth/**/*.parquet',union_by_name=true))
select * from x where rn=1 and venue='kalshi'""")

pair = con.execute("""
select a.symbol, a.window_open, a.offset_minutes, a.transport,
       a.yes_ask ask_t, a.yes_bid bid_t,
       b.yes_ask ask_w, b.yes_bid bid_w
from d a join d b on a.symbol=b.symbol and a.window_open=b.window_open
 and a.offset_minutes=b.offset_minutes
where a.source='live_touch' and b.source in ('live_ws','live')
 and a.yes_ask is not null and b.yes_ask is not null
 and a.yes_bid is not null and b.yes_bid is not null""").df()
print(f"paired live_touch vs stream ladder: {len(pair):,}")
if not len(pair):
    raise SystemExit("no overlap")
pair['mid_t']=(pair.ask_t+pair.bid_t)/2; pair['mid_w']=(pair.ask_w+pair.bid_w)/2
pair['d_mid']=(pair.mid_t-pair.mid_w)*100
pair['window_open']=pd.to_datetime(pair.window_open,utc=True)
print(f"  span {pair.window_open.min()} -> {pair.window_open.max()}")
print(f"  mean  (touch-stream) mid gap {pair.d_mid.mean():+.4f}pp   "
      f"mean |gap| {pair.d_mid.abs().mean():.4f}pp   median |gap| {pair.d_mid.abs().median():.4f}pp")
print(f"  share identical to 0.1c: {(pair.d_mid.abs()<0.1).mean()*100:.2f}%")
print(f"  share within 1pp:        {(pair.d_mid.abs()<=1.0).mean()*100:.2f}%")
print("\n  (compare: backfill vs stream was mean |gap| 7.498pp, median 4.000pp)")

print("\n--- alignment: does live_touch match the stream at its OWN minute? ---")
piv = con.execute("""select symbol,window_open,offset_minutes,(yes_ask+yes_bid)/2 mid
 from d where source in ('live_ws','live') and yes_ask is not null and yes_bid is not null""").df()
piv['window_open']=pd.to_datetime(piv.window_open,utc=True)
w = piv.pivot_table(index=['symbol','window_open'],columns='offset_minutes',values='mid',aggfunc='last')
t = pair.merge(w, left_on=['symbol','window_open'], right_index=True, how='inner')
offs=[c for c in w.columns]
for lag in (-1,0,1):
    vals=[]
    for o in offs:
        sel=t.offset_minutes==(o+lag)
        if sel.sum()==0: continue
        vals.append(np.abs(t.loc[sel,o].to_numpy(float)-t.loc[sel,'mid_t'].to_numpy(float)))
    if not vals: continue
    v=np.concatenate(vals); v=v[np.isfinite(v)]
    print(f"  touch@t vs stream@t{-lag:+d}: n={len(v):>7,} mean |err| {np.nanmean(v)*100:>7.3f}pp "
          f"median {np.nanmedian(v)*100:>7.3f}pp")

print("\n--- by transport era (the post-fill-contamination fix landed 2026-09-16) ---")
for tr, g in pair.groupby(pair.transport.fillna('(null, pre-2026-09-16)')):
    print(f"  {str(tr):>26}: n={len(g):>6,}  mean gap {g.d_mid.mean():+7.4f}pp  "
          f"mean |gap| {g.d_mid.abs().mean():6.4f}pp")

print("\n--- does live_touch predict settlement as well as the stream? ---")
sett=con.execute("""with s as (select *, row_number() over (partition by venue,symbol,event_time
  order by available_time desc) rn from read_parquet('data/research/venue_settlements/**/*.parquet',union_by_name=true))
 select symbol,window_open,settled_up from s where rn=1 and venue='kalshi' and settled_up is not null""").df()
sett['window_open']=pd.to_datetime(sett.window_open,utc=True)
m=pair.merge(sett,on=['symbol','window_open'],how='inner'); m['y']=m.settled_up.astype(int)
f=lambda p: float(log_loss(m.y, np.clip(p,1e-6,1-1e-6)))
print(f"  n={len(m):,}   live_touch mid log loss {f(m.mid_t):.6f}   stream mid {f(m.mid_w):.6f}"
      f"   gap {f(m.mid_t)-f(m.mid_w):+.6f}")
