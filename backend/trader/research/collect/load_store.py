"""Load the consolidated collection into the research store.

Summarises each window's snapshots to one `venue_depth` row per minute 0..15,
in DuckDB so memory stays bounded. Collection prices are cents; store prices
are dollars. Run: `python -m research.collect.load_store [--what depth] [--month YYYY-MM]`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA = Path(os.getenv('COLLECT_DATA')
            or Path(__file__).resolve().parents[2] / 'data' / 'collection')
DERIVED = DATA / 'derived'
WINDOW_MINUTES = 16                      # 0..15 inclusive
CENTS = 100.0

# `source` is part of `venue_depth`'s event key, so a reconstructed book and a
# recorded one coexist. Must equal the value `scripts/build_depth.py` writes;
# `_validate_depth` filters on it.
BACKFILL_SOURCE = 'backfill'

# collection column -> store column. Prices are converted; sizes are not.
_PRICES = {'best_bid': 'yes_bid', 'best_ask': 'yes_ask'}
_SIZES = {
    'bid_at_touch': 'yes_bid_size', 'ask_at_touch': 'yes_ask_size',
    'bid_1c': 'depth_bid_1c', 'ask_1c': 'depth_ask_1c',
    'bid_5c': 'depth_bid_5c', 'ask_5c': 'depth_ask_5c',
    'bid_vol': 'depth_bid_total', 'ask_vol': 'depth_ask_total',
    'bid_levels': 'levels_bid', 'ask_levels': 'levels_ask',
}


def to_depth_rows(snaps: pd.DataFrame) -> pd.DataFrame:
    """Rename and rescale one already-summarised frame into store columns."""
    if snaps.empty:
        return pd.DataFrame(columns=['venue', 'symbol', 'event_time'])
    out = pd.DataFrame(index=snaps.index)
    out['venue'] = snaps['venue']
    out['symbol'] = snaps['symbol']
    out['market_ticker'] = snaps['market_id']
    out['window_open'] = snaps['window_open']
    out['offset_minutes'] = snaps['offset_minutes'].astype(int)
    # The minute mark, not the tick's timestamp: the row is the book at minute m.
    out['event_time'] = (pd.to_datetime(snaps['window_open'], utc=True)
                         + pd.to_timedelta(snaps['offset_minutes'], unit='m'))
    # Reconstructed, so the minute mark is also when it became knowable.
    out['available_time'] = out['event_time']
    out['quality'] = 'valid'
    out['source'] = BACKFILL_SOURCE
    for src, dst in _PRICES.items():
        out[dst] = pd.to_numeric(snaps[src], errors='coerce') / CENTS
    for src, dst in _SIZES.items():
        out[dst] = pd.to_numeric(snaps[src], errors='coerce')
    # How stale the book was at the mark. Predexon serves book changes, so a
    # quiet market carries forward.
    #
    # Subtract as timestamps, never as integers: DuckDB returns datetime64[us],
    # and `ts` is in milliseconds.
    stamp = next((c for c in ('snap_ts', 'ts') if c in snaps.columns), None)
    if stamp is not None:
        taken = pd.to_datetime(pd.to_numeric(snaps[stamp], errors='coerce'),
                               unit='ms', utc=True)
        out['quote_age_seconds'] = (
            out['event_time'].values - taken.values) / pd.Timedelta(seconds=1)
    return out


def summarise_window(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per minute 0..15: the last snapshot at or before each mark.

    Pandas, for tests and single windows. `summarise_partition` does the same
    in DuckDB for a whole partition.
    """
    if snaps.empty:
        return pd.DataFrame(columns=['offset_minutes'])
    frame = snaps.copy()
    frame['offset_minutes'] = (
        pd.to_numeric(frame['offset_seconds'], errors='coerce') // 60)
    # A snapshot past the close belongs to the next window's book.
    frame = frame[(frame['offset_minutes'] >= 0)
                  & (frame['offset_minutes'] < WINDOW_MINUTES)]
    if frame.empty:
        return pd.DataFrame(columns=['offset_minutes'])
    frame = frame.sort_values('offset_seconds')
    # Point-in-time: take the last tick at or before the mark, never the last
    # tick inside minute m, which is up to 59 seconds after it. Must match the
    # ASOF JOIN in `_SQL`.
    marks = pd.DataFrame({'offset_minutes': range(int(WINDOW_MINUTES))})
    marks['mark_seconds'] = (marks['offset_minutes'] * 60).astype(float)
    keys = [k for k in ('market_id', 'window_open') if k in frame.columns]
    out = []
    for key, group in (frame.groupby(keys, as_index=False, sort=False)
                       if keys else [((), frame)]):
        group = group.copy()
        group['offset_seconds'] = pd.to_numeric(
            group['offset_seconds'], errors='coerce').astype(float)
        group = group.dropna(subset=['offset_seconds']).sort_values('offset_seconds')
        if group.empty:
            continue
        taken = pd.merge_asof(
            marks.sort_values('mark_seconds'), group,
            left_on='mark_seconds', right_on='offset_seconds',
            direction='backward', suffixes=('', '_snap'))
        # Omit a minute with nothing at or before it; never date a later book
        # to an earlier minute.
        taken = taken[taken['offset_seconds'].notna()]
        if keys:
            for name, value in zip(keys, key if isinstance(key, tuple) else (key,)):
                taken[name] = value
        out.append(taken)
    last = (pd.concat(out, ignore_index=True) if out
            else pd.DataFrame(columns=['offset_minutes']))
    if last.empty:
        return pd.DataFrame(columns=['offset_minutes'])
    return to_depth_rows(last)


_SQL = """
-- One row per (window, minute 0..15): the last snapshot AT OR BEFORE the mark.
--
-- An ASOF JOIN, not a GROUP BY the minute. Grouping takes the last tick that
-- FELL IN minute m, which is the book at the END of that minute -- up to 59
-- seconds AFTER the decision instant the row claims to describe. That is a
-- lookahead leak, and it surfaced only because it made `quote_age_seconds`
-- negative. A snapshot at t+3m10s serves minute 4 onward, never minute 3.
WITH windows AS (
    SELECT DISTINCT venue, symbol, market_id, window_open
    FROM read_parquet(?)
),
marks AS (
    SELECT w.venue, w.symbol, w.market_id, w.window_open,
           m.generate_series AS offset_minutes,
           epoch_ms(w.window_open) + m.generate_series * 60000 AS mark_ms
    FROM windows w
    CROSS JOIN generate_series(0, ?) AS m
),
snaps AS (
    SELECT market_id, window_open, ts, best_bid, best_ask,
           bid_at_touch, ask_at_touch, bid_1c, ask_1c, bid_5c, ask_5c,
           bid_vol, ask_vol, bid_levels, ask_levels
    FROM read_parquet(?)
)
SELECT k.venue, k.symbol, k.market_id, k.window_open, k.offset_minutes,
       s.ts AS snap_ts, s.best_bid, s.best_ask,
       s.bid_at_touch, s.ask_at_touch, s.bid_1c, s.ask_1c, s.bid_5c, s.ask_5c,
       s.bid_vol, s.ask_vol, s.bid_levels, s.ask_levels
FROM marks k
ASOF JOIN snaps s
  ON k.market_id = s.market_id
 AND k.window_open = s.window_open
 AND s.ts <= k.mark_ms
"""


def summarise_partition(path) -> pd.DataFrame:
    """The same summary as `summarise_window`, for a whole partition, in DuckDB.

    An ASOF JOIN takes the last snapshot at or before each minute mark. DuckDB
    spills to disk, so memory stays bounded.
    """
    import duckdb
    con = duckdb.connect(config={'max_temp_directory_size': '32GB'})  # bounded: a killed query must fail, not fill the disk
    try:
        frame = con.execute(
            _SQL, [str(path), WINDOW_MINUTES - 1, str(path)]).fetch_df()
    finally:
        con.close()
    return to_depth_rows(frame) if len(frame) else pd.DataFrame()


def load_depth(store, *, month=None, log=print) -> int:
    total = 0
    parts = sorted(DERIVED.glob('venue=*/symbol=*/month=*/data.parquet'))
    for path in parts:
        tags = {p.split('=')[0]: p.split('=')[1] for p in path.parts if '=' in p}
        if month and tags.get('month') != month:
            continue
        rows = summarise_partition(path)
        n = store.write('venue_depth', rows) if len(rows) else 0
        total += n
        log(f'  {tags["venue"]:11} {tags["symbol"]:8} {tags["month"]}  '
            f'{n:>8,} minute rows')
    return total


def settlement_rows(records) -> pd.DataFrame:
    """Kalshi settlements in the store's own column names.

    The store names the column `market_ticker`; the collection says
    `market_id`. A wrong name does not raise.

    Drops `expiration_value`: `venue_settlements` has no price column. The
    proxy-bias check reads the JSONL directly.
    """
    rows = []
    for r in records:
        opened = pd.Timestamp(r['window_open'])
        result = str(r.get('result') or '').strip().lower()
        rows.append({
            'venue': 'kalshi', 'symbol': r.get('symbol'),
            'event_time': opened, 'available_time': opened, 'quality': 'valid',
            'market_ticker': r.get('market_id'), 'window_open': opened,
            'close_time': pd.Timestamp(r['close_time']) if r.get('close_time') else pd.NaT,
            'result': result or None,
            'settled_up': True if result == 'yes' else (False if result == 'no' else None),
        })
    return pd.DataFrame(rows)


def implied_vol_rows(records) -> pd.DataFrame:
    """Ladder fits in the store's column names."""
    rows = []
    for r in records:
        at = pd.Timestamp(r['event_time'])
        rows.append({
            'venue': 'kalshi', 'symbol': r.get('symbol'),
            'event_time': at, 'available_time': at, 'quality': 'valid',
            'event_ticker': r.get('event_ticker'),
            'close_time': pd.Timestamp(r['close_time']) if r.get('close_time') else pd.NaT,
            'minutes_to_close': r.get('minutes_to_close'),
            'implied_sigma_per_min': r.get('implied_sigma_per_min'),
            'implied_spot': r.get('implied_spot'),
            'atm_strike': r.get('atm_strike'),
            'n_strikes': r.get('n_strikes'), 'r2': r.get('r2'),
        })
    return pd.DataFrame(rows)


def catalog_settlement_rows(records) -> pd.DataFrame:
    """Predexon catalog rows that carry a `result`, as settlements.

    Kalshi's own API purges markets after a few months; the catalog reaches
    back over the whole book era. A row without a result is skipped, not
    stored blank.
    """
    rows = []
    for r in records:
        if str(r.get('venue') or 'kalshi').lower() != 'kalshi':
            continue
        result = str(r.get('result') or '').strip().lower()
        if result not in ('yes', 'no'):
            continue
        opened = pd.Timestamp(r['window_open'])
        rows.append({
            'venue': 'kalshi', 'symbol': r.get('symbol'),
            'event_time': opened, 'available_time': opened, 'quality': 'valid',
            'market_ticker': r.get('market_id'), 'window_open': opened,
            'close_time': pd.NaT, 'result': result,
            'settled_up': result == 'yes',
        })
    return pd.DataFrame(rows)


def load_settlements(store, *, log=print) -> int:
    path = DATA / 'kalshi_settlements.jsonl'
    if not path.exists():
        log('  no kalshi_settlements.jsonl'); return 0
    records = []
    with open(path) as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    frame = settlement_rows(records)
    n = store.write('venue_settlements', frame) if len(frame) else 0
    log(f'  venue_settlements  {n:>8,} rows (Kalshi API, with prices)')

    # Then the catalog, which reaches back further. The two agree on
    # `settled_up` where they overlap.
    cat = DATA / 'kalshi_catalog.jsonl'
    if cat.exists():
        recs = []
        with open(cat) as handle:
            for line in handle:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    continue
        extra = catalog_settlement_rows(recs)
        if len(extra):
            m = store.write('venue_settlements', extra)
            log(f'  venue_settlements  {m:>8,} rows (Predexon catalog, 2026-01..08)')
            n += m
    return n


def load_implied_vol(store, *, log=print) -> int:
    path = DATA / 'implied_vol.jsonl'
    if not path.exists():
        log('  no implied_vol.jsonl'); return 0
    records = []
    with open(path) as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    frame = implied_vol_rows(records)
    n = store.write('venue_implied_vol', frame) if len(frame) else 0
    log(f'  venue_implied_vol  {n:>8,} rows')
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--what', default='all',
                        choices=('all', 'depth', 'settlements', 'implied_vol'))
    parser.add_argument('--month', default=None, help='only this YYYY-MM')
    args = parser.parse_args()

    from core.datastore import ResearchStore
    store = ResearchStore(os.getenv('RESEARCH_STORE'))

    if args.what in ('all', 'depth'):
        print('venue_depth (summarised to one row per minute 0..15)')
        print(f'  {load_depth(store, month=args.month):,} rows total\n')
    if args.what in ('all', 'settlements'):
        load_settlements(store)
    if args.what in ('all', 'implied_vol'):
        load_implied_vol(store)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
