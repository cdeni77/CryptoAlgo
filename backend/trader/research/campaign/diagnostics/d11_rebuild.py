"""D11 (rebuilt) — reconstruct the historical Kalshi top of book from raw WebSocket
frames and compare it to the backfilled quote AT THE DECISION INSTANT.

The first version counted frames and inferred staleness from the count. That was
not valid: a frame is any event at any level, and 1,707 frames is not 1,707
economically meaningful book moves. This version reconstructs the actual book.

Method
------
`venue_book_events` carries one row per level per frame: `kind` in {snapshot, delta},
`absolute` True only for snapshots, `side` in {yes, no} (Kalshi's book is TWO BID
stacks, so the YES ask is 1 - best NO bid), signed `size` for deltas.

For a target instant t and ticker k:
  1. find the newest snapshot (k, seq_s) with event_time <= t
  2. level size = snapshot size + sum of all deltas for that price with
     seq > seq_s and event_time <= t
  3. drop levels at or below MIN_SIZE (1e-3) -- `delta` is a signed change, so a
     level driven to zero accumulates float residue and would otherwise survive
     above the true touch
  4. best_yes_bid = max yes price surviving; best_no_bid = max no price surviving;
     yes_ask = 1 - best_no_bid

Validity controls
-----------------
* **Sequence continuity is checked GLOBALLY, not per ticker.** `seq` runs 1..N across
  every market on one subscription, so within a ticker it legitimately jumps. A
  per-ticker gap check would flag every delta.
* Rows with no snapshot at or before t are reported as UNMATCHED, never folded --
  folding a delta into nothing invents a book.
* Event time vs receive time is measured, not assumed.
* Results are broken out by symbol, date, offset and price band.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
STORE = HERE.parents[2] / 'data' / 'research'
MIN_SIZE = 1e-3
DAYS = sys.argv[1].split(',') if len(sys.argv) > 1 else [
    '2026-10-02', '2026-10-05', '2026-10-08', '2026-10-11', '2026-10-13']

con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk; con.execute('PRAGMA threads=16')
t0 = time.time()
print('=' * 86); print('D11 REBUILD — book reconstruction vs backfill at the decision instant')
print('=' * 86)
print(f'days: {DAYS}')

all_cmp, continuity, unmatched_rows = [], [], 0
for DAY in DAYS:
    tag = DAY.replace('-', '')
    files = f"{STORE}/venue_book_events/**/{tag}T*.parquet"

    # ---- validity control 1: event vs receive time, and event types -------
    meta = con.execute(f"""
    select count(*) frames,
           count(*) filter (where kind='snapshot') snaps,
           count(*) filter (where kind='delta') deltas,
           count(*) filter (where absolute) absolute_true,
           count(distinct market_ticker) tickers,
           count(distinct seq) seqs, min(seq) seq_lo, max(seq) seq_hi,
           avg(date_diff('millisecond', event_time, available_time)) recv_lag_ms,
           count(*) filter (where quality <> 'valid') bad_quality
    from read_parquet('{files}', union_by_name=true)""").df().iloc[0]
    # ---- validity control 2: GLOBAL sequence continuity ------------------
    gaps = con.execute(f"""
    with s as (select distinct seq from read_parquet('{files}', union_by_name=true)),
         o as (select seq, lag(seq) over (order by seq) prev from s)
    select count(*) filter (where seq - prev > 1) gap_count,
           coalesce(sum(seq - prev - 1) filter (where seq - prev > 1), 0) missing_seqs
    from o where prev is not null""").df().iloc[0]
    continuity.append({
        'day': DAY, 'frames': int(meta.frames), 'snapshots': int(meta.snaps),
        'deltas': int(meta.deltas), 'tickers': int(meta.tickers),
        'seq_span': int(meta.seq_hi - meta.seq_lo + 1), 'distinct_seqs': int(meta.seqs),
        'seq_gaps': int(gaps.gap_count), 'missing_seqs': int(gaps.missing_seqs),
        'recv_lag_ms': float(meta.recv_lag_ms), 'bad_quality': int(meta.bad_quality)})
    print(f"\n{DAY}: {int(meta.frames):,} frames  {int(meta.snaps):,} snapshots  "
          f"{int(meta.tickers):,} tickers  seq gaps {int(gaps.gap_count):,} "
          f"(missing {int(gaps.missing_seqs):,})  recv lag {float(meta.recv_lag_ms):.1f}ms "
          f"[{time.time()-t0:.0f}s]", flush=True)

    # ---- the decision instants we want, from the BACKFILL rows -----------
    targets = con.execute(f"""
    with x as (select *, row_number() over (
        partition by venue,symbol,event_time,source,offset_minutes
        order by available_time desc) rn
      from read_parquet('{STORE}/venue_depth/**/*.parquet', union_by_name=true))
    select market_ticker, symbol, window_open, offset_minutes,
           event_time as decision_ts, quote_age_seconds as backfill_age,
           yes_bid as bf_bid, yes_ask as bf_ask
    from x
    where rn=1 and venue='kalshi' and source='backfill'
      and yes_bid is not null and yes_ask is not null
      and offset_minutes between 1 and 14
      and event_time >= TIMESTAMPTZ '{DAY} 00:00:00+00'
      and event_time <  TIMESTAMPTZ '{DAY} 23:59:59+00'""").df()
    if not len(targets):
        print(f'  {DAY}: no backfill targets'); continue
    con.register('tg', targets)
    print(f"  targets: {len(targets):,} over {targets.market_ticker.nunique():,} tickers",
          flush=True)

    # ---- fold: snapshot + signed deltas, per target ----------------------
    rec = con.execute(f"""
    with ev as (
      select e.market_ticker, e.seq, e.kind, e.side, e.price, e.size, e.event_time
      from read_parquet('{files}', union_by_name=true) e
      semi join tg on tg.market_ticker = e.market_ticker
      where e.quality = 'valid'),
    -- newest snapshot at or before each target
    snap as (
      select t.market_ticker, t.decision_ts, max(e.seq) snap_seq
      from tg t join ev e
        on e.market_ticker = t.market_ticker
       and e.kind = 'snapshot' and e.event_time <= t.decision_ts
      group by 1,2),
    base as (
      select s.market_ticker, s.decision_ts, e.side, e.price, sum(e.size) sz
      from snap s join ev e
        on e.market_ticker = s.market_ticker and e.seq = s.snap_seq
       and e.kind = 'snapshot'
      group by 1,2,3,4),
    delt as (
      select s.market_ticker, s.decision_ts, e.side, e.price, sum(e.size) sz
      from snap s join ev e
        on e.market_ticker = s.market_ticker
       and e.kind = 'delta' and e.seq > s.snap_seq
       and e.event_time <= s.decision_ts
      group by 1,2,3,4),
    lvl as (
      select coalesce(b.market_ticker, d.market_ticker) market_ticker,
             coalesce(b.decision_ts,  d.decision_ts)  decision_ts,
             coalesce(b.side, d.side) side,
             coalesce(b.price, d.price) price,
             coalesce(b.sz, 0) + coalesce(d.sz, 0) sz
      from base b full outer join delt d
        on b.market_ticker=d.market_ticker and b.decision_ts=d.decision_ts
       and b.side=d.side and b.price=d.price)
    select market_ticker, decision_ts,
           max(price) filter (where side='yes' and sz > {MIN_SIZE}) rec_yes_bid,
           max(price) filter (where side='no'  and sz > {MIN_SIZE}) rec_no_bid,
           count(*)   filter (where side='yes' and sz > {MIN_SIZE}) n_yes_lvl,
           count(*)   filter (where side='no'  and sz > {MIN_SIZE}) n_no_lvl
    from lvl group by 1,2""").df()
    print(f"  reconstructed: {len(rec):,}  [{time.time()-t0:.0f}s]", flush=True)

    m = targets.merge(rec, on=['market_ticker', 'decision_ts'], how='left')
    m['day'] = DAY
    unmatched_rows += int(m.rec_yes_bid.isna().sum() + m.rec_no_bid.isna().sum())
    all_cmp.append(m)

cmp = pd.concat(all_cmp, ignore_index=True)
cont = pd.DataFrame(continuity)
cmp['rec_ask'] = 1.0 - cmp.rec_no_bid
cmp['matched'] = cmp.rec_yes_bid.notna() & cmp.rec_no_bid.notna()
cmp['rec_mid'] = (cmp.rec_yes_bid + cmp.rec_ask) / 2
cmp['bf_mid'] = (cmp.bf_bid + cmp.bf_ask) / 2
cmp['d_mid_pp'] = (cmp.bf_mid - cmp.rec_mid) * 100
cmp.to_parquet(HERE / 'd11_rebuild.parquet', index=False)

print('\n' + '=' * 86); print('VALIDITY CONTROLS'); print('=' * 86)
print(cont.to_string(index=False))
print(f"\ntotal targets {len(cmp):,}   MATCHED {cmp.matched.sum():,} "
      f"({cmp.matched.mean()*100:.2f}%)   UNMATCHED {(~cmp.matched).sum():,} "
      f"({(~cmp.matched).mean()*100:.2f}%)")
print("  unmatched = no snapshot at or before the instant, or one side empty; "
      "these are reported, never folded")

ok = cmp[cmp.matched].copy()
print('\n' + '=' * 86)
print('RECONSTRUCTED BOOK vs BACKFILL QUOTE, by the backfill\'s claimed age')
print('=' * 86)
ok['band'] = pd.cut(ok.backfill_age, [-.01,1,2,5,10,15,30,60,120,1e9],
                    labels=['<1s','1-2s','2-5s','5-10s','10-15s','15-30s','30-60s','60-120s','>120s'])
print(f"{'claimed age':>12} {'n':>7} {'mean|d_mid|':>12} {'median':>9} {'p90':>8} "
      f"{'%exact':>8} {'%<=1pp':>8}")
for b, g in ok.groupby('band', observed=True):
    if len(g) < 20: continue
    print(f"{str(b):>12} {len(g):>7,} {g.d_mid_pp.abs().mean():>12.3f} "
          f"{g.d_mid_pp.abs().median():>9.3f} {g.d_mid_pp.abs().quantile(.9):>8.3f} "
          f"{(g.d_mid_pp.abs()<0.05).mean()*100:>7.2f}% {(g.d_mid_pp.abs()<=1).mean()*100:>7.2f}%")

for by, lbl in (('symbol','symbol'), ('day','date'), ('offset_minutes','offset')):
    print(f"\n--- mean |backfill - reconstructed| by {lbl} (pp) ---")
    t = ok.groupby(by).agg(n=('d_mid_pp','size'), mean_abs=('d_mid_pp', lambda s: s.abs().mean()),
                           bias=('d_mid_pp','mean'))
    print(t.round(3).to_string())

ok['price_band'] = pd.cut(ok.rec_mid, [0,.1,.3,.5,.7,.9,1.01])
print("\n--- by reconstructed price band ---")
print(ok.groupby('price_band', observed=True).agg(
    n=('d_mid_pp','size'), mean_abs=('d_mid_pp', lambda s: s.abs().mean()),
    bias=('d_mid_pp','mean')).round(3).to_string())
print(f"\ntotal runtime {time.time()-t0:.0f}s")
