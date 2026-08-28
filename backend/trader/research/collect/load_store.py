"""Load the consolidated collection into the research store.

Run:
    python -m research.collect.load_store              # everything
    python -m research.collect.load_store --what depth
    python -m research.collect.load_store --month 2026-01

**Why this exists.** 255 million snapshots were collected and nothing reads
them. Every consumer — the four validators, `core/dataset.py`,
`core/features.py` — reads the research store; the collection writes to
`data/collection/`. Until this runs, `venue_depth` holds 885 rows and
`venue_settlements` holds none, which is why every validator reported "not
enough overlap" against a complete corpus.

Two properties are load-bearing, and both were nearly got wrong.

**It SUMMARISES rather than copies.** `venue_depth` holds one row per minute
0..15 of a window, not one per snapshot. That is deliberate: the offset grid is
itself under test, and a table sampled only where the model currently scores
would foreclose the question. 116,242 windows x 16 minutes is ~1.9M rows
against 255M snapshots — a 130x reduction that loses nothing the schema was
ever meant to hold.

**It STREAMS.** `ResearchStore.write` reads an existing partition whole in
order to merge revisions, and one derived partition holds 44 million rows.
Handing it those directly is precisely the shape that froze this machine when
`consolidate` held a partition in memory. The summarisation therefore happens
in DuckDB, which spills rather than allocating, and the store only ever sees
the ~40k rows a single (venue, symbol, month) partition summarises to.

**Prices are cents in the collection and DOLLARS in the store.** The existing
`venue_depth` rows read `yes_bid 0.63`; the collection holds `63`. Loading one
into the other unconverted is a hundredfold error, in the same direction every
time, and nothing downstream would raise.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

DATA = Path(os.getenv('COLLECT_DATA', 'data/collection'))
DERIVED = DATA / 'derived'
WINDOW_MINUTES = 16                      # 0..15 inclusive
CENTS = 100.0

# `venue_depth`'s event key includes `source` precisely so a reconstructed book
# and a recorded one coexist as independent observations rather than one
# superseding the other. Measured before that existed: 58 overlapping
# (symbol, window) pairs and the comparison still read zero rows, because the
# live row's later `available_time` silently won.
# The canonical value, matching `scripts/build_depth.py:241`. Inventing a
# new one ('predexon_backfill') did not raise: `_validate_depth` filters on
# == 'backfill' and silently reported zero rows against 1.49M.
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
    # The minute mark itself, not the tick's own timestamp: the row asserts
    # "this was the book at minute m", and the tick that proves it may be
    # anywhere in the preceding sixty seconds.
    out['event_time'] = (pd.to_datetime(snaps['window_open'], utc=True)
                         + pd.to_timedelta(snaps['offset_minutes'], unit='m'))
    # Reconstructed after the fact, so the minute mark is also when it became
    # knowable. A live row's available_time is its poll instant, which is why
    # `source` and not this column keeps the two apart.
    out['available_time'] = out['event_time']
    out['quality'] = 'valid'
    out['source'] = BACKFILL_SOURCE
    for src, dst in _PRICES.items():
        out[dst] = pd.to_numeric(snaps[src], errors='coerce') / CENTS
    for src, dst in _SIZES.items():
        out[dst] = pd.to_numeric(snaps[src], errors='coerce')
    # How stale the book was at this minute mark. Predexon serves book CHANGES,
    # so a quiet market carries forward, and a forward fill indistinguishable
    # from an observation lets a fresh forecast "beat" a stale price.
    #
    # Subtracted as timestamps, never as integers. `event_time` comes back from
    # DuckDB as datetime64[**us**], not [ns], so `.astype('int64')` yields
    # MICROseconds -- against a `ts` in milliseconds that is a 1000x error, and
    # it produced ages of -1.78 billion seconds rather than raising.
    stamp = next((c for c in ('snap_ts', 'ts') if c in snaps.columns), None)
    if stamp is not None:
        taken = pd.to_datetime(pd.to_numeric(snaps[stamp], errors='coerce'),
                               unit='ms', utc=True)
        out['quote_age_seconds'] = (
            out['event_time'].values - taken.values) / pd.Timedelta(seconds=1)
    return out


def summarise_window(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per minute 0..15: the last snapshot AT OR BEFORE each mark.

    Pure-pandas, used by the tests and for a single window. The production path
    does the same thing in DuckDB across a whole partition, because a partition
    can hold 44 million snapshots and this cannot.
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
    # Last at or before the mark. A minute with nothing before it is OMITTED
    # rather than carried: emitting one would date a later book to an earlier
    # minute, which is a leak dressed as coverage.
    last = frame.groupby(['market_id', 'window_open', 'offset_minutes'],
                         as_index=False).last()
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

    `arg_max(col, ts)` is the last value at or before the minute mark by
    construction, because the GROUP BY is the minute. DuckDB streams and spills;
    pandas would have to hold 44 million rows to do the same thing.
    """
    import duckdb
    con = duckdb.connect()
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

    `market_ticker`, not `market_id` — the collection and the store disagree,
    and a wrong name does not raise. It lands as an absent column, and every
    reader downstream sees nulls it will read as missing data rather than as a
    loader bug.

    `expiration_value` is deliberately DROPPED: `venue_settlements` has no
    price column. The numeric proxy-bias check reads the JSONL directly, so
    inventing a field here would put the number somewhere nothing looks while
    implying it had been stored.
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
    log(f'  venue_settlements  {n:>8,} rows')
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
    store = ResearchStore(os.getenv('RESEARCH_STORE', 'data/research'))

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
