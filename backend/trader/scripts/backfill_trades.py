"""Backfill the Kalshi public TAPE for the markets we have a live book for.

**One question motivates this: would a RESTING order have filled?** Measured
2026-10-17 on 54,100 live-recorded rows, the taker-to-maker swing is ~1.5-2pp
per contract (about 0.72pp of fee plus a full cent of spread, the spread being
exactly 1.000c median) while the model's edge when it disagrees is ~1.1-1.5pp.
The execution cost is larger than the entire signal, so whether a resting bid
fills is the question the strategy now turns on.

**The book cannot answer it, and the failure is silent.** A resting bid fills
when an aggressive SELLER crosses into it; the quoted ask never moves. A first
attempt simulated fills from `venue_depth` by requiring `best_ask <= our_bid`,
which in a 1c market can only happen after a >=1c ADVERSE move -- so it defined
every fill to be adversely selected and returned -9.5pp regardless of the edge
gate. `venue_book_events` cannot answer it either: a delta cannot distinguish a
trade from a cancellation.

The tape can. Each print carries `taker_book_side`, naming which side the
AGGRESSOR crossed on, so a print at price P with a selling aggressor is evidence
a resting bid at P would have filled.

**Scope is deliberately narrow.** Only markets that already have a LIVE-recorded
book in `venue_depth` -- 2026-08-25 onward, ~15,300 markets. Earlier markets
have only the Predexon backfill, whose prices are not executable
(unbiased but 54x noisier and stamped a minute late), so a fill simulation over
them would measure the reconstruction rather than the venue.

Read-only: it places nothing and touches no account state. Resumable -- markets
already in `venue_trades` are skipped, so an interrupted run continues.

    python -m scripts.backfill_trades --dry-run     # count the work
    python -m scripts.backfill_trades               # do it
    python -m scripts.backfill_trades --limit 200   # a taste first
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
from typing import Optional

import pandas as pd

logger = logging.getLogger('backfill_trades')
DATASET = 'venue_trades'
#: How deep to page one market's tape. The client's default of 40 pages is
#: 40,000 prints, and a busy BTC window exceeds it -- measured 2026-10-21, one
#: 15-minute market printed 26,138 times and several tripped the cap. A
#: truncated tape is not a short one: the rows below the cut are MISSING, and
#: for a fill simulation the missing rows are prints that would have hit a
#: resting order. Set from the environment so a deep run does not need an edit.
MAX_PAGES = int(os.getenv('TAPE_MAX_PAGES', '150'))
# `live_ws_replay` belongs here: it is the venue's OWN websocket frames,
# archived in `venue_book_events` and folded back into a book afterwards --
# the same true deci-cent tick as `live_ws`, just reconstructed later. The
# docstring's reason for excluding older markets is that they carry only the
# Predexon backfill, whose prices are not executable; that reason does not
# reach a replay of the venue's own frames. Omitting it kept the 2026-09-17 ->
# 09-30 window out of scope, which is the exact stretch the replay recovered.
LIVE_SOURCES = ("'live'", "'live_ws'", "'live_touch'", "'live_ws_replay'")


def _args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    p.add_argument('--limit', type=int, default=None,
                   help='Stop after this many markets, OLDEST first. Time-biased '
                        'by construction — use --sample for an answer.')
    p.add_argument('--sample', type=int, default=None,
                   help='Fetch a RANDOM sample of this many markets instead. '
                        'A market with 25,000 prints needs 25 pages, so the '
                        'full 15,342 is 7-13 hours; a few thousand markets '
                        'answers the fill question with room to spare, and the '
                        'sample is random so it is not a slice of one regime.')
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--concurrency', type=int, default=4,
                   help='Parallel market fetches. Kept low on purpose: this '
                        'is another party API and the job is not urgent.')
    p.add_argument('--batch', type=int, default=400,
                   help='Markets per store write.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    return p.parse_args()


def _markets(store_root: Optional[str]) -> pd.DataFrame:
    """Markets with a LIVE-recorded book, and the window each belongs to."""
    import duckdb
    root = store_root or 'data/research'
    q = f"""
        select market_ticker, min(window_open) as window_open
        from read_parquet('{root}/venue_depth/**/*.parquet', union_by_name=true)
        where venue = 'kalshi' and source in ({','.join(LIVE_SOURCES)})
          and market_ticker is not null
        group by 1 order by 2
    """
    return duckdb.connect().execute(q).fetchdf()


def _already(store) -> set:
    try:
        have = store.read(DATASET)
    except Exception:                                          # noqa: BLE001
        return set()
    if have is None or not len(have) or 'market_ticker' not in have:
        return set()
    return set(have.market_ticker.dropna().unique())


def _symbol(ticker: str) -> Optional[str]:
    from core.config import SERIES_BY_SYMBOL
    for sym, series in SERIES_BY_SYMBOL.items():
        if series and ticker.upper().startswith(f'{series.upper()}-'):
            return sym
    return None


def _rows(trades, ticker: str, window_open, fetched_at) -> list[dict]:
    """Collapse a market's prints to one row per (minute, aggressor side).

    **The extremes are the point, not the average.** A resting bid at P is hit
    when a SELLING aggressor trades at or below P, so `min_price` on the selling
    side is the fill test; `max_price` on the buying side is its mirror for a
    resting offer. A mean would answer neither.

    Prints before the window opens or after it settles are dropped rather than
    clamped: a print at minute 16 belongs to the next market, and folding it in
    here would put a later market's tape inside this one's window.
    """
    import numpy as np
    sym = _symbol(ticker)
    w = pd.Timestamp(window_open)
    if w.tzinfo is None:
        w = w.tz_localize('UTC')
    buckets: dict = {}
    for t in trades:
        created = getattr(t, 'created_time', None)
        price = getattr(t, 'yes_price', None)
        if created is None or price is None:
            continue
        ts = pd.Timestamp(created)
        ts = ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')
        off = int((ts - w).total_seconds() // 60)
        if off < 0 or off > 15:
            continue
        side = getattr(t, 'taker_book_side', None) or 'unknown'
        n = float(getattr(t, 'contracts', 0) or 0)
        b = buckets.setdefault((off, side),
                               {'n': 0, 'c': 0.0, 'lo': np.inf, 'hi': -np.inf,
                                'pv': 0.0})
        b['n'] += 1
        b['c'] += n
        b['lo'] = min(b['lo'], float(price))
        b['hi'] = max(b['hi'], float(price))
        b['pv'] += float(price) * n
    out = []
    for (off, side), b in buckets.items():
        out.append({
            'venue': 'kalshi', 'symbol': sym, 'market_ticker': ticker,
            'event_time': w + pd.Timedelta(minutes=off),
            # Read after the fact, so what we could first have SEEN is now --
            # not the print. A backfill that stamps itself as live reintroduces
            # the lookahead this store's `available_time` exists to prevent.
            'available_time': fetched_at,
            # **'valid', not 'backfill'.** Quality is a GRADE, and the
            # vocabulary is ('invalid','unvalidated','suspicious','valid');
            # anything else is not a weaker grade but invisibility, because
            # `read` keeps QUALITY_LEVELS[index(min_quality):] and the
            # default is 'valid'. Measured 2026-10-21: all 31,462 rows of
            # `venue_trades` were unreadable, which also broke `_already()`
            # below -- it saw an empty set, so every run refetched from
            # scratch. A print the venue reported is a sound measurement.
            #
            # **This does NOT record that the row was backfilled**, and
            # `venue_trades` has no `source` column to record it in --
            # unlike `venue_depth`, where `source` separates a book
            # somebody recorded from the same book reconstructed later.
            # Now that a live recorder writes this dataset too, backfilled
            # and live prints are indistinguishable in it. That matters
            # here more than most places: this project has measured the
            # backfilled BOOK as 54x noisier than the live one and
            # sign-reversing in the tails.
            'quality': 'valid',
            'window_open': w,
            'offset_minutes': off,
            'taker_book_side': side,
            'n_prints': b['n'],
            'contracts': b['c'],
            'min_price': b['lo'],
            'max_price': b['hi'],
            'vwap': (b['pv'] / b['c']) if b['c'] > 0 else None,
        })
    return out


async def _fetch(client, ticker, sem):
    """The live route first, the historical host if it comes back empty.

    `/markets/trades` refuses to look back past a moving cutoff (~3 months) and
    everything older answers on `/historical/trades`, on a different host. The
    live era starts 2026-08-25, inside the cutoff today but not forever, so the
    fallback is not dead code.
    """
    from data_collection.kalshi_client import KalshiError
    async with sem:
        err = 'never attempted'
        # **429 is not an empty tape.** Measured 2026-10-18: at concurrency 6,
        # 886 of 2,000 markets came back `too_many_requests`, and without this
        # retry they were recorded as markets that simply never traded — the
        # same silent-failure shape as counting an error as an empty result.
        # Exponential backoff, and the whole semaphore slot is held while we
        # wait so a backoff actually reduces pressure instead of letting four
        # other coroutines keep hammering.
        for historical in (False, True):
            for attempt in range(5):
                try:
                    got = await client.market_trades(
                        ticker=ticker, limit=1000, historical=historical,
                        max_pages=MAX_PAGES)
                except KalshiError as exc:
                    err = str(exc)[:160]
                    if '429' in err or 'too_many_requests' in err:
                        await asyncio.sleep(2.0 * (2 ** attempt))
                        continue
                    logger.debug('%s historical=%s: %s', ticker, historical, exc)
                    break
                except Exception as exc:                       # noqa: BLE001
                    err = str(exc)[:160]
                    logger.debug('%s historical=%s: %s', ticker, historical, err)
                    break
                if got:
                    return got
                err = None
                break
        # **An ERROR is not an empty tape.** Returning [] for both made 20
        # failed requests print as "20 markets had no prints at all (normal)",
        # which is the silent-failure shape this repo exists to avoid.
        return err if err is not None else []


async def run(a) -> int:
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore(a.research_store) if a.research_store else ResearchStore()
    markets = _markets(a.research_store)
    done = _already(store)
    todo = markets[~markets.market_ticker.isin(done)]
    if a.sample and len(todo) > a.sample:
        todo = todo.sample(a.sample, random_state=a.seed).sort_values('window_open')
    elif a.limit:
        todo = todo.head(a.limit)
    print(f'{len(markets):,} markets with a live book '
          f'({markets.window_open.min()} -> {markets.window_open.max()})')
    print(f'{len(done):,} already backfilled, {len(todo):,} to fetch')
    if a.dry_run or not len(todo):
        if a.dry_run:
            print('\nDRY RUN — nothing fetched or written.')
        return 0

    sem = asyncio.Semaphore(max(1, a.concurrency))
    fetched_at = pd.Timestamp.now(tz='UTC')
    buf, written, empty, failed = [], 0, 0, 0
    first_error = None
    # The client holds a session; outside `async with` every request raises
    # "client is not open" and the old code counted those as empty tapes.
    async with KalshiClient(live=False) as client:   # read-only, cannot order
        for start in range(0, len(todo), a.batch):
            chunk = todo.iloc[start:start + a.batch]
            got = await asyncio.gather(*[_fetch(client, t, sem)
                                         for t in chunk.market_ticker])
            for (_, m), trades in zip(chunk.iterrows(), got):
                if isinstance(trades, str):
                    failed += 1
                    first_error = first_error or trades
                    continue
                if not trades:
                    empty += 1
                    continue
                buf.extend(_rows(trades, m.market_ticker, m.window_open,
                                 fetched_at))
            if buf:
                frame = pd.DataFrame(buf).dropna(subset=['event_time'])
                written += store.write(DATASET, frame)
                buf = []
            print(f'  {min(start + a.batch, len(todo)):,}/{len(todo):,} markets, '
                  f'{written:,} minute-rows, {empty:,} empty, {failed:,} FAILED')
            if failed and failed == min(start + a.batch, len(todo)):
                print(f'\n  every request failed — stopping rather than '
                      f'grinding through {len(todo):,} markets.\n  first error: '
                      f'{first_error}')
                return 1
    print(f'\ndone: {written:,} prints written, {empty:,} markets with an empty '
          f'tape (a 15-minute market that never traded is normal), '
          f'{failed:,} FAILED')
    if failed:
        print(f'  first error: {first_error}')
    return 0


def main() -> int:
    a = _args()
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s')
    return asyncio.run(run(a))


if __name__ == '__main__':
    raise SystemExit(main())
