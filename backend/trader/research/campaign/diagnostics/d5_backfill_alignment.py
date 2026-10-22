"""D5: are backfilled snapshots legitimate point-in-time features / executable prices?

D3's lag sweep said the backfill's error against the live book MINIMISES at a lag
of one minute. That is an alignment claim and it needs a per-row test, not a mean.
For every backfill observation, find which live MINUTE it actually matches best.
If the mode is offset-1 rather than offset, the backfill's `offset_minutes` is
mis-stamped by a minute and every backtest priced off it compared a fresh forecast
against a one-minute-old price.
"""
import numpy as np, pandas as pd, duckdb

con=duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
con.execute("""create view d as
with x as (select *, row_number() over (
   partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('data/research/venue_depth/**/*.parquet',union_by_name=true))
select * from x where rn=1 and venue='kalshi'""")

bf = con.execute("""select symbol,window_open,offset_minutes,
   (yes_ask+yes_bid)/2 mid, quote_age_seconds age
 from d where source='backfill' and yes_ask is not null and yes_bid is not null""").df()
lv = con.execute("""select symbol,window_open,offset_minutes,
   (yes_ask+yes_bid)/2 mid from d
 where source in ('live_ws','live') and yes_ask is not null and yes_bid is not null""").df()
print(f"backfill {len(bf):,}   live {len(lv):,}")

# wide table of the live book across the whole window, per (symbol, window)
piv = lv.pivot_table(index=['symbol','window_open'], columns='offset_minutes',
                     values='mid', aggfunc='last')
bf = bf.merge(piv, left_on=['symbol','window_open'], right_index=True, how='inner')
offs = [c for c in piv.columns]
print(f"backfill rows with a live book in the same window: {len(bf):,}")

M = bf[offs].to_numpy(dtype=float)
own = bf.mid.to_numpy(dtype=float)
diff = np.abs(M - own[:, None])
ok = ~np.isnan(diff)
diff_f = np.where(ok, diff, np.inf)
argmin = np.array(offs, dtype=float)[np.argmin(diff_f, axis=1)]
rel = argmin - bf.offset_minutes.to_numpy(dtype=float)
rel = rel[np.isfinite(rel)]

print("\n--- which live minute does a backfill row ACTUALLY match? (relative to its stamp) ---")
vc = pd.Series(rel).value_counts().sort_index()
for k, v in vc.items():
    if abs(k) <= 5:
        print(f"  offset{int(k):+d}: {v:>8,}  ({v/len(rel)*100:>5.2f}%)")
print(f"  median relative offset: {np.median(rel):+.1f}   mean {rel.mean():+.3f}")

print("\n--- mean |error| of the backfill against each live minute ---")
for lag in range(-2, 4):
    cols = [o for o in offs]
    vals = []
    for o in cols:
        sel = bf.offset_minutes == (o + lag)
        if sel.sum() == 0: continue
        vals.append(np.abs(bf.loc[sel, o].to_numpy(dtype=float) - bf.loc[sel, 'mid'].to_numpy(dtype=float)))
    if not vals: continue
    v = np.concatenate(vals); v = v[np.isfinite(v)]
    star = '  <-- BEST' if lag == 1 else ''
    print(f"  backfill@t vs live@t{-lag:+d}: n={len(v):>8,}  mean |err| {np.nanmean(v)*100:>7.3f}pp"
          f"   median {np.nanmedian(v)*100:>7.3f}pp{star}")

print("\n--- restricted to rows the backfill CALLS fresh (age <= 5s) ---")
b2 = bf[bf.age <= 5]
for lag in range(-1, 3):
    vals=[]
    for o in offs:
        sel = b2.offset_minutes == (o + lag)
        if sel.sum()==0: continue
        vals.append(np.abs(b2.loc[sel,o].to_numpy(dtype=float)-b2.loc[sel,'mid'].to_numpy(dtype=float)))
    if not vals: continue
    v=np.concatenate(vals); v=v[np.isfinite(v)]
    print(f"  backfill@t vs live@t{-lag:+d}: n={len(v):>8,}  mean |err| {np.nanmean(v)*100:>7.3f}pp"
          f"   median {np.nanmedian(v)*100:>7.3f}pp")
