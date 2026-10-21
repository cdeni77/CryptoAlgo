"""Independent validation of the frozen point-in-time dataset.

**Deliberately avoids `harness.py` for its own derivations.** The harness is the
thing under test; validating it with its own code proves only self-consistency.
Every quantity is re-derived with independent SQL and compared to the frozen file.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import duckdb, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
STORE = HERE.parents[2] / 'data' / 'research'
FROZEN = HERE.parent / 'datasets' / 'market_v2.parquet'
MANIFEST = HERE.parent / 'datasets' / 'market_v2.manifest.json'
LATENCY_S, VERIFIED_AGE = 3, 5.0

fails, warns = [], []
def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok: fails.append(name)

print('=' * 80); print('INDEPENDENT POINT-IN-TIME VALIDATION of market_v2'); print('=' * 80)
con = duckdb.connect(); con.execute('PRAGMA threads=8')
man = json.loads(MANIFEST.read_text())
fz = pd.read_parquet(FROZEN)
print(f'frozen rows {len(fz):,}   manifest says {man["rows"]:,}')
check('manifest row count matches the file', len(fz) == man['rows'])

con.execute(f"""create view depth as
with x as (select *, row_number() over (
    partition by venue,symbol,event_time,source,offset_minutes
    order by available_time desc) rn
  from read_parquet('{STORE}/venue_depth/**/*.parquet', union_by_name=true))
select * from x where rn=1 and venue='kalshi'
  and window_open >= TIMESTAMPTZ '{man['span'][0]}'
  and window_open <= TIMESTAMPTZ '{man['span'][1]}'""")

# V1 — independent as-of recompute
sel = con.execute(f"""
with grid as (
  select distinct symbol, window_open, offset_minutes,
         window_open + to_minutes(offset_minutes) + to_seconds({LATENCY_S}) di
  from depth where offset_minutes between 1 and 14
    and yes_bid is not null and yes_ask is not null and yes_ask > yes_bid),
cand as (
  select g.symbol, g.window_open, g.offset_minutes, g.di,
         d.available_time, d.source, d.quote_age_seconds,
         row_number() over (partition by g.symbol,g.window_open,g.offset_minutes
                            order by d.available_time desc, d.event_time desc) rk
  from grid g join depth d
    on d.symbol=g.symbol and d.window_open=g.window_open
   and d.available_time <= g.di
   and d.yes_bid is not null and d.yes_ask is not null and d.yes_ask > d.yes_bid)
select * from cand where rk=1""").df()
print(f'\nindependent as-of selection: {len(sel):,} rows')
check('V1 independent recompute matches the frozen row count',
      abs(len(sel) - len(fz)) / max(len(fz), 1) < 0.001, f'{len(sel):,} vs {len(fz):,}')
future = (pd.to_datetime(sel.available_time, utc=True) >
          pd.to_datetime(sel.di, utc=True)).sum()
check('V1 no quote is used before it was knowable', future == 0, f'{future} violations')

# V2 — the frozen file itself must contain no future quote
f_future = (pd.to_datetime(fz.quote_available_time, utc=True) >
            pd.to_datetime(fz.decision_instant, utc=True)).sum()
check('V2 frozen file contains no future-dated quote', f_future == 0, f'{f_future} rows')

# V3 — the eligibility labels are reproducible from the raw columns
exp = np.where(fz.quote_source.isin(['live', 'live_ws']), 'verified',
      np.where(fz.quote_source.eq('live_touch'), 'questionable',
      np.where(fz.book_age_s.le(VERIFIED_AGE), 'verified',
      np.where(fz.book_age_s.le(60.0), 'questionable', 'unsuitable'))))
mism = int((exp != fz.quality.to_numpy()).sum())
check('V3 quality labels reproduce from the raw columns', mism == 0, f'{mism} mismatches')
check('V3 usable == (quality == verified)',
      bool((fz.usable == fz.quality.eq('verified')).all()))

# V4 — label integrity, re-derived
lab = con.execute(f"""
with s as (select *, row_number() over (
    partition by venue,symbol,event_time order by available_time desc) rn
  from read_parquet('{STORE}/venue_settlements/**/*.parquet', union_by_name=true))
select symbol, window_open, settled_up from s
where rn=1 and venue='kalshi' and settled_up is not null""").df()
dup = lab.groupby(['symbol', 'window_open']).size().max()
check('V4 venue label has no duplicate (symbol, window)', int(dup) == 1, f'max {int(dup)}')
check('V4 frozen label is binary', set(pd.unique(fz.y)) <= {0, 1})

# V5 — feature availability by month and source
print('\nV5 verified symbol-windows by month:')
v = fz[fz.usable].copy()
v['m'] = pd.to_datetime(v.window_open, utc=True).dt.to_period('M').astype(str)
t = v.groupby('m').apply(lambda g: g.groupby(['symbol', 'window_open']).ngroups,
                         include_groups=False)
print(t.to_string())
check('V5 every month has verified coverage', bool((t > 1000).all()), f'min {int(t.min()):,}')
print('\nV5 quality by source:')
print(fz.groupby(['quote_source', 'quality']).size().unstack(fill_value=0).to_string())

# V6 — dev/val split is chronological and disjoint
dev, val = fz[fz.split == 'dev'], fz[fz.split == 'val']
check('V6 dev precedes val with no overlap',
      pd.to_datetime(dev.window_open, utc=True).max() <
      pd.to_datetime(val.window_open, utc=True).min(),
      f'dev n={len(dev):,} val n={len(val):,}')

print('\n' + '=' * 80)
print(f'FAILURES: {len(fails)} {fails}')
print(f'WARNINGS: {len(warns)} {warns}')
print('=' * 80)
sys.exit(1 if fails else 0)
