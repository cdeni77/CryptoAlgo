"""D4: do cent-quantised backfill prices create false apparent edge at the gate?

The test is a PAIRED comparison. Where the live recorder (deci-cent precise) and
the Predexon backfill (integer cents) describe the same (symbol, window, offset),
the difference between them is pure measurement error. If that error is unbiased
the backfill is usable as an executable price; if it is signed, every backtest
priced off the backfill buys at a price the venue never showed.
"""
import duckdb, numpy as np, pandas as pd

con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
con.execute("""create view d as
with x as (select *, row_number() over (
    partition by venue,symbol,event_time,source,offset_minutes
    order by available_time desc) rn
  from read_parquet('data/research/venue_depth/**/*.parquet', union_by_name=true))
select * from x where rn=1 and venue='kalshi'""")

q = lambda s: con.execute(s).df()

# live_ws is the stream sampler: deci-cent precise, no carry-forward.
pair = q("""
select a.symbol, a.window_open, a.offset_minutes,
       a.yes_ask as ask_bf,  a.yes_bid as bid_bf,  a.quote_age_seconds as age_bf,
       b.yes_ask as ask_lv,  b.yes_bid as bid_lv,  b.quote_age_seconds as age_lv
from d a join d b
  on a.symbol=b.symbol and a.window_open=b.window_open
 and a.offset_minutes=b.offset_minutes
where a.source='backfill' and b.source in ('live_ws','live')
  and a.yes_ask is not null and b.yes_ask is not null
  and a.yes_bid is not null and b.yes_bid is not null
""")
print(f"paired backfill vs live observations: {len(pair):,}")
print(f"  span {pair.window_open.min()} -> {pair.window_open.max()}")

pair['d_ask'] = pair.ask_bf - pair.ask_lv
pair['d_bid'] = pair.bid_bf - pair.bid_lv
pair['mid_lv'] = (pair.ask_lv + pair.bid_lv) / 2

print("\n--- is the backfill price BIASED against the live price? (cents) ---")
print(f"{'band':>14} {'n':>7} {'mean d_ask':>11} {'t':>8} {'mean d_bid':>11} {'t':>8} {'|d_ask|p90':>11}")
for lo, hi in [(0,.10),(.10,.30),(.30,.50),(.50,.70),(.70,.90),(.90,1.01)]:
    g = pair[(pair.mid_lv>=lo)&(pair.mid_lv<hi)]
    if len(g) < 30: continue
    ta = g.d_ask.mean()/(g.d_ask.std(ddof=1)/np.sqrt(len(g)))
    tb = g.d_bid.mean()/(g.d_bid.std(ddof=1)/np.sqrt(len(g)))
    print(f"[{lo:.2f},{hi:.2f}) {len(g):>7,} {g.d_ask.mean()*100:>11.4f} {ta:>8.2f} "
          f"{g.d_bid.mean()*100:>11.4f} {tb:>8.2f} {g.d_ask.abs().quantile(.9)*100:>11.3f}")

print("\n--- off-cent share (does the backfill resolve sub-cent prices?) ---")
for nm, col in [('backfill ask','ask_bf'), ('live ask','ask_lv')]:
    v = pair[col].round(6)
    off = ((v*100 - (v*100).round()).abs() > 1e-6).mean()
    print(f"  {nm:>13}: {off*100:>6.2f}% off the whole-cent grid")

print("\n--- THE EDGE TEST: apparent edge from pricing off the backfill ---")
# A buyer of YES pays the ask. If the backfill ask is LOWER than the live ask,
# a backtest books a cheaper entry than the venue would have given: false edge.
pair['false_edge_pp'] = (pair.ask_lv - pair.ask_bf) * 100   # >0 == backtest flattered
for lo, hi in [(0,.10),(.10,.30),(.30,.50),(.50,.70),(.70,.90),(.90,1.01)]:
    g = pair[(pair.mid_lv>=lo)&(pair.mid_lv<hi)]
    if len(g) < 30: continue
    t = g.false_edge_pp.mean()/(g.false_edge_pp.std(ddof=1)/np.sqrt(len(g)))
    print(f"  [{lo:.2f},{hi:.2f}) n={len(g):>7,}  mean false edge {g.false_edge_pp.mean():>+8.4f}pp "
          f"(t={t:>6.2f})  sd {g.false_edge_pp.std():>6.3f}pp")
t = pair.false_edge_pp.mean()/(pair.false_edge_pp.std(ddof=1)/np.sqrt(len(pair)))
print(f"  OVERALL  n={len(pair):,}  mean false edge {pair.false_edge_pp.mean():+.4f}pp (t={t:.2f})")
print(f"  share where backfill ask is cheaper by >=1.5pp (the gate): "
      f"{(pair.false_edge_pp>=1.5).mean()*100:.3f}%")
pair.to_parquet('research/campaign/diagnostics/d4_pairs.parquet', index=False)
