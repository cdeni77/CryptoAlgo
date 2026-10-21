"""D7 — CORRECTION to D3/D4/D5. Is the backfill stale, or is the LIVE row late?

Measured: every source stamps `event_time` at window_open+offset exactly, but
`available_time` differs sharply:

    source       available_time - stamp     own quote_age
    backfill              +0s                  0.33s
    live / live_ws       +32s                 32.6s
    live_touch            +3s                  NaN

So the live ladder row labelled offset m describes the book at **m + 32 seconds**.
That single fact reinterprets the earlier results:

* the ~7.5pp gap between backfill@m and live@m is a ~32-second time gap, not
  necessarily backfill error;
* the live mid's better settlement prediction (0.4541 vs 0.4966) is partly just 32
  more seconds of information inside a 15-minute window;
* and for a decision taken at offset m, the live row at m **was not available yet** —
  using it as the executable price is lookahead of 32 seconds.

The fair comparison puts both at the same instant. For a decision at m + latency:
  - backfill@m        is available (available_time = m)
  - live@(m-1)        is available (available_time = m - 28s)
  - live@m            is NOT available (available_time = m + 32s)

So the test is: at the decision instant, which AVAILABLE price predicts settlement
better -- the backfill at m, or the live row from m-1?
"""
import duckdb, numpy as np, pandas as pd
from sklearn.metrics import log_loss

con = duckdb.connect()
con.execute("""create view d as
with x as (select *, row_number() over (
  partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('data/research/venue_depth/**/*.parquet', union_by_name=true))
select * from x where rn=1 and venue='kalshi'""")

pair = con.execute("""
select a.symbol, a.window_open, a.offset_minutes,
       (a.yes_bid+a.yes_ask)/2 mid_bf,            -- backfill AT m   (avail m+0s)
       (b.yes_bid+b.yes_ask)/2 mid_lv_prev,       -- live   AT m-1   (avail m-28s)
       (c.yes_bid+c.yes_ask)/2 mid_lv_now         -- live   AT m     (avail m+32s) = LOOKAHEAD
from d a
join d b on a.symbol=b.symbol and a.window_open=b.window_open
        and b.offset_minutes = a.offset_minutes - 1
join d c on a.symbol=c.symbol and a.window_open=c.window_open
        and c.offset_minutes = a.offset_minutes
where a.source='backfill' and b.source in ('live_ws','live') and c.source in ('live_ws','live')
  and a.yes_ask is not null and a.yes_bid is not null
  and b.yes_ask is not null and b.yes_bid is not null
  and c.yes_ask is not null and c.yes_bid is not null
  and a.offset_minutes between 2 and 14""").df()
pair['window_open'] = pd.to_datetime(pair.window_open, utc=True)

sett = con.execute("""with s as (select *, row_number() over (
   partition by venue,symbol,event_time order by available_time desc) rn
  from read_parquet('data/research/venue_settlements/**/*.parquet', union_by_name=true))
 select symbol, window_open, settled_up from s
 where rn=1 and venue='kalshi' and settled_up is not null""").df()
sett['window_open'] = pd.to_datetime(sett.window_open, utc=True)
m = pair.merge(sett, on=['symbol','window_open'], how='inner')
m['y'] = m.settled_up.astype(int)
print(f"three-way aligned rows with a venue label: {len(m):,} "
      f"over {m.window_open.nunique():,} windows")

def ll(p): return float(log_loss(m.y, np.clip(p, 1e-6, 1-1e-6)))
r = {'backfill @ m  (available at m)':      ll(m.mid_bf),
     'live     @ m-1 (available at m-28s)': ll(m.mid_lv_prev),
     'live     @ m  (available at m+32s)':  ll(m.mid_lv_now)}
print("\n--- log loss against the venue's settlement ---")
for k, v in r.items(): print(f"  {k:<38} {v:.6f}")

def boot(a, b, label):
    pa = np.clip(a,1e-6,1-1e-6); pb = np.clip(b,1e-6,1-1e-6)
    la = -(m.y*np.log(pa)+(1-m.y)*np.log(1-pa))
    lb = -(m.y*np.log(pb)+(1-m.y)*np.log(1-pb))
    g = pd.DataFrame({'d': (la-lb).to_numpy(), 'w': m.window_open}).groupby('w')['d'].mean()
    v = g.to_numpy(); n = len(v)
    rng = np.random.default_rng(0)
    bs = v[rng.integers(0,n,size=(10000,n))].mean(axis=1)
    print(f"  {label}: {v.mean():+.6f}  95% CI [{np.quantile(bs,.025):+.6f}, "
          f"{np.quantile(bs,.975):+.6f}]  over {n:,} windows")

print("\n--- paired differences (negative = the FIRST named is better) ---")
boot(m.mid_bf,      m.mid_lv_prev, 'backfill@m  minus  live@m-1   (both AVAILABLE at the decision)')
boot(m.mid_bf,      m.mid_lv_now,  'backfill@m  minus  live@m     (live is 32s of LOOKAHEAD)')
boot(m.mid_lv_prev, m.mid_lv_now,  'live@m-1    minus  live@m     (pure 60s of freshness)')

print("\n--- how much is 32 seconds worth? the freshness gradient ---")
print("  if `live@m-1 minus live@m` is close to `backfill@m minus live@m`, then the")
print("  backfill's whole apparent handicap is TIMING, and it is a legitimate")
print("  point-in-time price. If the backfill is much worse than live@m-1, it carries")
print("  real error on top of the timing.")
