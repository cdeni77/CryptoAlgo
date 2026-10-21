"""D9 — is a large backfill `quote_age_seconds` a genuinely QUIET book, or a HOLE
in the Predexon feed?

This decides whether the apparent baseline edge is fictitious or merely fragile.

* If the book truly did not change for 90 seconds, the stale price is a real price
  and buying at it is a real (if adversely-selected) opportunity.
* If Predexon simply has no record for that span while the venue's book DID move,
  the price is fiction and every row carrying it is unusable.

The live stream recorder is the independent witness: where backfill claims "no change
for N seconds", did the live book move?
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H

con = duckdb.connect()
con.execute(f"""create view d as
with x as (select *, row_number() over (
  partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('{H.STORE}/venue_depth/**/*.parquet', union_by_name=true))
select * from x where rn=1 and venue='kalshi'""")

pair = con.execute("""
select a.symbol, a.window_open, a.offset_minutes,
       a.quote_age_seconds book_age,
       (a.yes_bid+a.yes_ask)/2 mid_bf,
       (b.yes_bid+b.yes_ask)/2 mid_lv,
       (c.yes_bid+c.yes_ask)/2 mid_lv_prev
from d a
join d b on a.symbol=b.symbol and a.window_open=b.window_open
        and b.offset_minutes=a.offset_minutes and b.source in ('live_ws','live')
left join d c on a.symbol=c.symbol and a.window_open=c.window_open
        and c.offset_minutes=a.offset_minutes-1 and c.source in ('live_ws','live')
where a.source='backfill'
  and a.yes_ask is not null and a.yes_bid is not null
  and b.yes_ask is not null and b.yes_bid is not null""").df()
print(f"paired backfill vs live: {len(pair):,}")

pair['band'] = pd.cut(pair.book_age, [-.01,1,5,15,30,60,120,1e9],
                      labels=['<1s','1-5s','5-15s','15-30s','30-60s','60-120s','>120s'])
pair['lv_moved'] = (pair.mid_lv - pair.mid_lv_prev).abs()
pair['bf_vs_lv'] = (pair.mid_bf - pair.mid_lv).abs()

print("\n--- when the BACKFILL says the book has not changed, did the LIVE book move? ---")
print(f"{'backfill book age':>18} {'n':>8} {'mean|bf-live|pp':>17} {'mean live move/min pp':>23}")
for b, g in pair.groupby('band', observed=True):
    if len(g) < 100: continue
    print(f"{str(b):>18} {len(g):>8,} {g.bf_vs_lv.mean()*100:>17.3f} "
          f"{g.lv_moved.mean()*100:>23.3f}")

print("""
Read: if a large backfill book age meant a genuinely quiet book, the LIVE book would
also be nearly unchanged (right column small) and the two would agree (left column
small). If instead the live book kept moving while the backfill froze, the backfill
price is a feed hole, not a quiet market.""")

print("\n--- share of high-age rows where the live book moved more than 2pp in a minute ---")
for b, g in pair.groupby('band', observed=True):
    if len(g) < 100: continue
    moved = (g.lv_moved > 0.02).mean()
    print(f"  {str(b):>8}: {moved*100:>6.2f}%   (n={len(g):,})")
