"""Build order-flow features from the raw Kalshi tape. ONE pass per day, cached.

Writes `research/overnight/features/flow_<DAY>.parquet` — one row per
(market_ticker, minute_of_window). Resumable: a day whose file exists is skipped.

**What the tape is.** `venue_book_events` is `orderbook_delta`: `kind` in
{snapshot, delta}, `side` in {yes, no} (Kalshi's book is TWO BID stacks), signed
`size` for deltas. **There is no trade channel stored**, so a negative delta is
*size removed* -- a cancellation OR a fill, and the two are NOT distinguishable.
Every removal-derived column is named `*_removed_ambig` so no downstream reader can
mistake it for executed volume.

**Point-in-time discipline.** Flow attributed to minute m covers the half-open
interval [m-1, m), which is strictly before the decision instant at m. The
reconstructed book is folded only from events with `event_time <= minute mark`.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import duckdb, numpy as np, pandas as pd

HERE = Path(__file__).resolve().parent
STORE = HERE.parents[2] / 'data' / 'research'
MIN_SIZE = 1e-3


def build_day(con, day: str, out: Path) -> int:
    tag = day.replace('-', '')
    files = f"{STORE}/venue_book_events/**/{tag}T*.parquet"
    # every (ticker, window_open) the tape covers, from the ticker name itself
    con.execute(f"""create or replace view ev as
    select market_ticker, symbol, seq, kind, side, price, size, event_time
    from read_parquet('{files}', union_by_name=true)
    where quality = 'valid'""")
    tickers = con.execute("""
    select market_ticker, symbol from ev group by 1,2""").df()
    if not len(tickers):
        return 0
    # **The ticker is NOT parsed.** `KXBTC15M-26SEP301000-00` encodes the CLOSE in
    # Eastern time, not the open, and not UTC -- the real window_open for that
    # ticker is 2026-09-30 09:45 ET. Rather than hard-code a venue convention that
    # has already moved once (CLAUDE.md records the suffix change), the mapping is
    # taken from `venue_depth`, which carries both columns and is authoritative.
    wmap = con.execute(f"""
    select market_ticker, min(window_open) window_open
    from read_parquet('{STORE}/venue_depth/**/*.parquet', union_by_name=true)
    where venue='kalshi' and market_ticker is not null
    group by 1""").df()
    wmap['window_open'] = pd.to_datetime(wmap.window_open, utc=True)
    tickers = tickers.merge(wmap, on='market_ticker', how='inner')
    if not len(tickers):
        return 0
    grid = tickers.loc[tickers.index.repeat(16)].copy()
    grid['minute'] = np.tile(np.arange(16), len(tickers))
    grid['mark'] = grid.window_open + pd.to_timedelta(grid.minute, unit='m')
    grid = grid[['market_ticker', 'symbol', 'window_open', 'minute', 'mark']]
    con.register('grid', grid)

    # ---- reconstructed top of book at each mark -------------------------
    book = con.execute(f"""
    with snap as (
      select g.market_ticker, g.mark, max(e.seq) snap_seq
      from grid g join ev e on e.market_ticker=g.market_ticker
       and e.kind='snapshot' and e.event_time <= g.mark
      group by 1,2),
    base as (
      select s.market_ticker, s.mark, e.side, e.price, sum(e.size) sz
      from snap s join ev e on e.market_ticker=s.market_ticker
       and e.seq=s.snap_seq and e.kind='snapshot' group by 1,2,3,4),
    delt as (
      select s.market_ticker, s.mark, e.side, e.price, sum(e.size) sz
      from snap s join ev e on e.market_ticker=s.market_ticker
       and e.kind='delta' and e.seq > s.snap_seq and e.event_time <= s.mark
      group by 1,2,3,4),
    lvl as (
      select coalesce(b.market_ticker,d.market_ticker) market_ticker,
             coalesce(b.mark,d.mark) mark, coalesce(b.side,d.side) side,
             coalesce(b.price,d.price) price, coalesce(b.sz,0)+coalesce(d.sz,0) sz
      from base b full outer join delt d
        on b.market_ticker=d.market_ticker and b.mark=d.mark
       and b.side=d.side and b.price=d.price)
    select market_ticker, mark,
      max(price) filter (where side='yes' and sz>{MIN_SIZE}) yes_best,
      max(price) filter (where side='no'  and sz>{MIN_SIZE}) no_best,
      sum(sz)    filter (where side='yes' and sz>{MIN_SIZE}) yes_total,
      sum(sz)    filter (where side='no'  and sz>{MIN_SIZE}) no_total,
      count(*)   filter (where side='yes' and sz>{MIN_SIZE}) yes_levels,
      count(*)   filter (where side='no'  and sz>{MIN_SIZE}) no_levels
    from lvl group by 1,2""").df()
    con.register('book', book)

    # size resting AT the touch, which is what depletion is measured against
    touch = con.execute(f"""
    with snap as (
      select g.market_ticker, g.mark, max(e.seq) snap_seq
      from grid g join ev e on e.market_ticker=g.market_ticker
       and e.kind='snapshot' and e.event_time <= g.mark group by 1,2),
    lvl as (
      select s.market_ticker, s.mark, e.side, e.price, sum(e.size) sz
      from snap s join ev e on e.market_ticker=s.market_ticker
       and ((e.seq=s.snap_seq and e.kind='snapshot')
         or (e.kind='delta' and e.seq>s.snap_seq and e.event_time<=s.mark))
      group by 1,2,3,4)
    select l.market_ticker, l.mark,
      sum(l.sz) filter (where l.side='yes' and l.price=b.yes_best) yes_touch_size,
      sum(l.sz) filter (where l.side='no'  and l.price=b.no_best)  no_touch_size
    from lvl l join book b on b.market_ticker=l.market_ticker and b.mark=l.mark
    group by 1,2""").df()
    con.register('touch', touch)

    # ---- FLOW in the minute BEFORE each mark: [mark-1min, mark) ---------
    flow = con.execute("""
    select g.market_ticker, g.mark,
      sum(case when e.side='yes' and e.size>0 then e.size else 0 end) yes_added,
      sum(case when e.side='yes' and e.size<0 then -e.size else 0 end) yes_removed_ambig,
      sum(case when e.side='no'  and e.size>0 then e.size else 0 end) no_added,
      sum(case when e.side='no'  and e.size<0 then -e.size else 0 end) no_removed_ambig,
      count(*) filter (where e.side='yes') yes_events,
      count(*) filter (where e.side='no')  no_events,
      count(*) filter (where e.size>0) n_adds,
      count(*) filter (where e.size<0) n_removes_ambig
    from grid g left join ev e
      on e.market_ticker=g.market_ticker and e.kind='delta'
     and e.event_time >= g.mark - INTERVAL 1 MINUTE and e.event_time < g.mark
    group by 1,2""").df()

    out_df = (grid.merge(book, on=['market_ticker', 'mark'], how='left')
                  .merge(touch, on=['market_ticker', 'mark'], how='left')
                  .merge(flow,  on=['market_ticker', 'mark'], how='left'))
    out_df['day'] = day
    out_df.to_parquet(out, index=False)
    return len(out_df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2026-09-30')
    ap.add_argument('--end', default='2026-10-14')
    a = ap.parse_args()
    con = duckdb.connect(); con.execute('PRAGMA threads=12')
    days = pd.date_range(a.start, a.end, freq='D').strftime('%Y-%m-%d')
    t0 = time.time()
    for d in days:
        out = HERE / f'flow_{d}.parquet'
        if out.exists():
            print(f'{d}: cached ({out.stat().st_size//1024}KB)', flush=True); continue
        try:
            n = build_day(con, d, out)
            print(f'{d}: {n:,} rows  [{time.time()-t0:.0f}s]', flush=True)
        except Exception as e:
            print(f'{d}: FAILED {str(e)[:150]}', flush=True)
    print(f'done in {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
