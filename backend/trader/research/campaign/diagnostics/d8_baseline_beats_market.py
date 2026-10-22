"""D8 — AUDIT: the barrier baseline alone appears to beat the Kalshi price.

E3 scored model_minus_market +0.007963 (CI [+0.00623, +0.00966], p=0.0000) over
7,932 windows, using NO machine learning -- just F(x/sigma) with a per-fold fitted
scale and tail. That is a large claim and RESEARCH_PROTOCOL.md section 5 requires the
pipeline to be audited before it is believed.

**CLAUDE.md records this exact failure mode:** "comparing a fresh forecast against a
stale price manufactures edge. Measured: a first pass fetched only to +13.5m, and the
resulting minute-14 rows showed the baseline 'beating' the market by 0.246 nats at
t = 8.4, which was entirely the carry-forward."

The mechanism: Predexon serves book CHANGES, so the quote at minute m is the last
change at or before m. A quiet book is carried forward. `available_time` still reads
m, so the as-of join accepts it -- but the PRICE may be a minute old while the
baseline's spot input is fresh to the second.

The test: does the apparent edge concentrate in rows where the source's own
`quote_age_seconds` (time since the last book change) is large?
"""
import sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import harness as H

OOF = HERE.parent / 'experiments' / 'oof_E3_baseline_only.parquet'
oof = pd.read_parquet(OOF)
oof['window_open'] = pd.to_datetime(oof.window_open, utc=True)
print(f'E3 OOF: {len(oof):,} rows, {oof.window_open.nunique():,} windows')

# the SOURCE's own staleness: time since the last book change
con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
age = con.execute(f"""
with x as (select *, row_number() over (
   partition by venue,symbol,event_time,source,offset_minutes order by available_time desc) rn
 from read_parquet('{H.STORE}/venue_depth/**/*.parquet', union_by_name=true))
select symbol, window_open, offset_minutes, source,
       quote_age_seconds as book_age_s
from x where rn=1 and venue='kalshi'""").df()
age['window_open'] = pd.to_datetime(age.window_open, utc=True)

px = H.market_prices()[['symbol','window_open','offset_minutes','quote_source',
                        'quote_age_s','quote_event_time']]
d = (oof.merge(px, on=['symbol','window_open','offset_minutes'], how='left')
        .merge(age, left_on=['symbol','window_open','offset_minutes','quote_source'],
               right_on=['symbol','window_open','offset_minutes','source'], how='left'))
print(f'joined: {len(d):,}   book_age present on {d.book_age_s.notna().mean()*100:.1f}%')

d['skill'] = H._ll(d.p_mkt, d.y) - H._ll(d.p_model, d.y)

def seg(by, label):
    print(f"\n--- apparent baseline edge by {label} ---")
    print(f"{label:>20} {'n':>8} {'win':>6} {'mmm':>11} {'CI low':>11} {'CI high':>11}")
    for k, g in d.groupby(by, observed=True):
        if len(g) < 500: continue
        mean, lo, hi, p, nw = H.model_minus_market(g.p_model, g.p_mkt, g.y, g.window_open)
        print(f"{str(k):>20} {len(g):>8,} {nw:>6,} {mean:>+11.6f} {lo:>+11.6f} {hi:>+11.6f}")

seg('quote_source', 'quote source')
d['book_age_band'] = pd.cut(d.book_age_s, [-.01, 1, 5, 15, 30, 60, 120, 1e9],
                            labels=['<1s','1-5s','5-15s','15-30s','30-60s','60-120s','>120s'])
seg('book_age_band', 'book age (last change)')
seg('offset_minutes', 'offset')

print("\n--- THE TEST ---")
fresh = d[d.book_age_s <= 5]
stale = d[d.book_age_s > 30]
for nm, g in (('book age <= 5s', fresh), ('book age > 30s', stale)):
    if len(g) < 500: continue
    mean, lo, hi, p, nw = H.model_minus_market(g.p_model, g.p_mkt, g.y, g.window_open)
    print(f"  {nm:>16}: n={len(g):>8,} windows={nw:>5,} mmm={mean:+.6f} "
          f"CI [{lo:+.6f},{hi:+.6f}]")
print("\n  If the edge survives on FRESH rows it is not carry-forward.")
print("  If it lives only in the stale rows, it is the 2026 defect recurring.")
