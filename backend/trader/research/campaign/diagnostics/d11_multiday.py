"""D11 (multi-day) — backfill 'quiet book' claims vs the raw venue tape.

Ground truth: `venue_book_events`, every Kalshi WebSocket frame. For each backfill
row, count the frames the venue actually sent during the span the backfill reports as
unchanged. Run over the whole tape/backfill overlap, 2026-09-30 .. 2026-10-14.
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H

days = pd.date_range('2026-09-30', '2026-10-14', freq='D').strftime('%Y-%m-%d')
con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk; con.execute('PRAGMA threads=4')
all_res = []
for DAY in days:
    try:
        bf = con.execute(f"""
        with x as (select *, row_number() over (
          partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
         from read_parquet('{H.STORE}/venue_depth/**/*.parquet', union_by_name=true))
        select market_ticker, symbol, event_time, offset_minutes,
               quote_age_seconds as claimed_age
        from x
        where rn=1 and venue='kalshi' and source='backfill'
          and event_time >= TIMESTAMPTZ '{DAY} 00:00:00+00'
          and event_time <  TIMESTAMPTZ '{DAY} 23:59:59+00'
          and quote_age_seconds is not null""").df()
        if not len(bf):
            print(f'{DAY}: no backfill rows', flush=True); continue
        con.register('bf', bf)
        files = f"{H.STORE}/venue_book_events/**/{DAY.replace('-','')}T*.parquet"
        tape = con.execute(f"""
        select e.market_ticker, date_trunc('second', e.event_time) sec, count(*) frames
        from read_parquet('{files}', union_by_name=true) e
        semi join bf on bf.market_ticker = e.market_ticker
        group by 1,2""").df()
        con.register('tape', tape)
        r = con.execute("""
        select b.market_ticker, b.symbol, b.event_time, b.offset_minutes, b.claimed_age,
               coalesce(sum(t.frames), 0) frames_in_span
        from bf b left join tape t
          on t.market_ticker = b.market_ticker
         and t.sec >  b.event_time - to_seconds(cast(greatest(b.claimed_age,1) as integer))
         and t.sec <= b.event_time
        group by 1,2,3,4,5""").df()
        all_res.append(r)
        print(f'{DAY}: backfill {len(bf):,}  tape-seconds {len(tape):,}  joined {len(r):,}',
              flush=True)
    except Exception as e:
        print(f'{DAY}: SKIP ({str(e)[:90]})', flush=True)

if not all_res:
    raise SystemExit('no overlap produced')
res = pd.concat(all_res, ignore_index=True)
res.to_parquet(HERE / 'd11_tape_vs_backfill.parquet', index=False)
print(f'\ntotal rows: {len(res):,}')

res['band'] = pd.cut(res.claimed_age, [-.01,1,2,5,10,15,30,60,120,1e9],
                     labels=['<1s','1-2s','2-5s','5-10s','10-15s','15-30s',
                             '30-60s','60-120s','>120s'])
print("\n=== did the VENUE send frames during the span the backfill calls quiet? ===")
print(f"{'claimed age':>12} {'n':>8} {'mean frames':>12} {'median':>9} "
      f"{'% ZERO frames':>15} {'frames/sec':>11}")
for b, g in res.groupby('band', observed=True):
    if len(g) < 20: continue
    print(f"{str(b):>12} {len(g):>8,} {g.frames_in_span.mean():>12.1f} "
          f"{g.frames_in_span.median():>9.0f} "
          f"{(g.frames_in_span==0).mean()*100:>14.2f}% "
          f"{(g.frames_in_span/g.claimed_age.clip(lower=1)).mean():>11.2f}")
print("""
  high '% ZERO frames' -> genuinely quiet book, the quote is a VALID resting price
  low  '% ZERO frames' -> the venue was updating and the backfill missed it;
                          the quote is a MISSING UPDATE, not a real price""")
