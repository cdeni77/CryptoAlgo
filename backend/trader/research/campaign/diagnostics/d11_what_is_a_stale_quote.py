"""D11 — does a large backfill `quote_age_seconds` mean an UNCHANGED book or a
MISSING SOURCE UPDATE?

This is settled against ground truth rather than against another sampler.
`venue_book_events` is the raw Kalshi WebSocket tape -- every book frame the venue
sent, 1.18 billion rows, 2026-09-30 to 2026-10-17. If the backfill reports "no change
for N seconds" and the tape shows the venue sending frames throughout that span, the
backfill is missing updates. If the tape is also silent, the book really was quiet
and the quote is a valid resting price.

Deliberately NOT conditioned on any model metric. The question is about the data.
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H

DAY = sys.argv[1] if len(sys.argv) > 1 else '2026-10-08'
print(f'overlap day under test: {DAY}  (backfill ends 2026-10-14, tape starts 2026-09-30)')

con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
con.execute("PRAGMA threads=4")

# ---- the backfill rows for that day, with their claimed age ---------------
bf = con.execute(f"""
with x as (select *, row_number() over (
  partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('{H.STORE}/venue_depth/**/*.parquet', union_by_name=true))
select market_ticker, symbol, event_time, offset_minutes,
       quote_age_seconds as claimed_age, yes_bid, yes_ask
from x
where rn=1 and venue='kalshi' and source='backfill'
  and event_time >= TIMESTAMPTZ '{DAY} 00:00:00+00'
  and event_time <  TIMESTAMPTZ '{DAY} 23:59:00+00'
  and quote_age_seconds is not null
  and yes_bid is not null and yes_ask is not null""").df()
print(f'backfill rows that day: {len(bf):,}  tickers {bf.market_ticker.nunique():,}')
if not len(bf):
    raise SystemExit('no backfill rows on that day')

# ---- the tape, aggregated to one row per (ticker, second) -----------------
files = f"{H.STORE}/venue_book_events/**/{DAY.replace('-','')}T*.parquet"
tape = con.execute(f"""
select market_ticker, date_trunc('second', event_time) sec, count(*) frames
from read_parquet('{files}', union_by_name=true)
where market_ticker in (select distinct market_ticker from bf_t)
group by 1,2""".replace('bf_t', 'bf')) if False else None

con.register('bf', bf)
tape = con.execute(f"""
select e.market_ticker, date_trunc('second', e.event_time) sec, count(*) frames
from read_parquet('{files}', union_by_name=true) e
semi join bf on bf.market_ticker = e.market_ticker
group by 1,2""").df()
print(f'tape seconds with at least one frame: {len(tape):,}')
con.register('tape', tape)

# ---- for each backfill row, how many frames did the venue send during the
#      span the backfill claims was quiet?
res = con.execute("""
select b.market_ticker, b.symbol, b.event_time, b.offset_minutes, b.claimed_age,
       coalesce(sum(t.frames), 0) frames_in_span,
       count(t.sec) seconds_with_frames
from bf b
left join tape t
  on t.market_ticker = b.market_ticker
 and t.sec >  b.event_time - to_seconds(cast(b.claimed_age as integer))
 and t.sec <= b.event_time
group by 1,2,3,4,5""").df()
print(f'joined: {len(res):,}')

res['band'] = pd.cut(res.claimed_age, [-.01, 1, 2, 5, 10, 15, 30, 60, 120, 1e9],
                     labels=['<1s','1-2s','2-5s','5-10s','10-15s','15-30s',
                             '30-60s','60-120s','>120s'])
print("\n=== did the VENUE send frames during the span the backfill calls quiet? ===")
print(f"{'claimed age':>12} {'n':>8} {'mean frames':>12} {'median':>8} "
      f"{'% with ZERO frames':>20} {'frames/sec':>11}")
for b, g in res.groupby('band', observed=True):
    if len(g) < 50: continue
    zero = (g.frames_in_span == 0).mean() * 100
    span = g.claimed_age.clip(lower=1)
    print(f"{str(b):>12} {len(g):>8,} {g.frames_in_span.mean():>12.1f} "
          f"{g.frames_in_span.median():>8.0f} {zero:>19.2f}% "
          f"{(g.frames_in_span / span).mean():>11.2f}")

print("""
Read:
  * '% with ZERO frames' high  -> the book really was quiet; the quote is VALID.
  * '% with ZERO frames' low   -> the venue was sending updates the backfill missed;
                                  the quote is a MISSING UPDATE and is not a real price.
""")
res.to_parquet(HERE / f'd11_tape_vs_backfill_{DAY}.parquet', index=False)
print(f'wrote d11_tape_vs_backfill_{DAY}.parquet')
