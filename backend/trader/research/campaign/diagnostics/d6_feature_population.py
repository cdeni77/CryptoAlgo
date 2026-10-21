"""D6: are the deployed model's 49 features populated consistently across the
historical (backfill) and live-recorded eras?

The deployed artifact declares 49 features across 8 groups, including `cross_venue`,
`market_state` and `implied_vol` -- all three of which exist only where the venue's
book was collected. A group that is populated in one era and NaN in another makes a
fitted model two models blended, and the evaluation measures the blend.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
STORE = HERE.parents[2] / 'data' / 'research'

prov = json.loads((HERE.parents[2] / 'models' / 'forecast.provenance.json').read_text())
feats = prov['features']; groups = prov['groups']
print(f"deployed artifact declares {len(feats)} features in {len(groups)} groups:")
print(f"  {groups}\n")

from core.features import FEATURE_GROUPS
by_group = {g: [f for f in FEATURE_GROUPS.get(g, ()) if f in feats] for g in groups}
for g, fs in by_group.items():
    print(f"  {g:<16} {len(fs):>2} of the artifact's features")
unmapped = [f for f in feats if not any(f in v for v in by_group.values())]
if unmapped:
    print(f"  UNMAPPED to any declared group: {unmapped}")

con = duckdb.connect()
print("\n--- availability of each book-dependent source, by month ---")
q = con.execute(f"""
with d as (select *, row_number() over (
    partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
  from read_parquet('{STORE}/venue_depth/**/*.parquet', union_by_name=true))
select strftime(window_open,'%Y-%m') m,
  count(distinct case when venue='kalshi' and source in ('live','live_ws')
        then symbol||window_open::varchar end) kalshi_stream,
  count(distinct case when venue='kalshi' and source='backfill'
        then symbol||window_open::varchar end) kalshi_backfill,
  count(distinct case when venue='kalshi' and source='live_touch'
        then symbol||window_open::varchar end) kalshi_touch,
  count(distinct case when venue='polymarket'
        then symbol||window_open::varchar end) polymarket
from d where rn=1 and window_open >= '2026-06-01' group by 1 order by 1""").df()
print(q.to_string(index=False))

print("\n--- implied vol fits by month and symbol (feeds the `implied_vol` group) ---")
iv = con.execute(f"""select strftime(event_time,'%Y-%m') m, symbol, count(*) n,
  avg(case when r2 >= 0.90 then 1.0 else 0.0 end) usable_share
 from read_parquet('{STORE}/venue_implied_vol/**/*.parquet', union_by_name=true)
 where event_time >= '2026-06-01' group by 1,2 order by 1,2""").df()
print(iv.pivot_table(index='m', columns='symbol', values='n', fill_value=0).to_string())
print("\nshare of fits with r2 >= 0.90 (the live gate):")
print(iv.pivot_table(index='m', columns='symbol', values='usable_share').round(3).to_string())

print("\n--- VERDICT ---")
tot = q.set_index('m')
for m in tot.index:
    r = tot.loc[m]
    flags = []
    if r.kalshi_stream == 0: flags.append('NO trustworthy Kalshi book')
    if r.polymarket == 0:    flags.append('NO cross_venue')
    print(f"  {m}: stream {r.kalshi_stream:>6,}  backfill {r.kalshi_backfill:>6,}  "
          f"touch {r.kalshi_touch:>6,}  pm {r.polymarket:>6,}"
          + ('   <-- ' + '; '.join(flags) if flags else ''))
