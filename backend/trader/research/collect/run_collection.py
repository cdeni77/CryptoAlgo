"""The collection loop: claim work from the ledger, fetch it, record what came back.

Run each phase separately, in order:

    python -m research.collect.run_collection --phase catalog
    python -m research.collect.run_collection --phase seed
    python -m research.collect.run_collection --phase collect   # resumable
    python -m research.collect.run_collection --phase report    # coverage
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import gzip
import json
import os
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.collect.catalog import (                        # noqa: E402
    COLLECT_FROM, Predexon, kalshi_catalog, pm_catalog_by_grid, seed_from_catalogs,
)
from research.collect.fetchers import FIELDS, pack_kalshi, pack_pm  # noqa: E402
from research.collect.ledger import Ledger                    # noqa: E402
from research.collect.orchestrator import (                   # noqa: E402
    Breaker, RateLimiter, SingleWriterLock,
)

DATA = Path(os.getenv('COLLECT_DATA', 'data/collection'))
LEDGER_PATH = str(DATA / 'ledger.db')
LOCK_PATH = str(DATA / 'collect.lock')
KALSHI_CATALOG = str(DATA / 'kalshi_catalog.jsonl')
PM_CATALOG = str(DATA / 'pm_catalog.jsonl')
ARCHIVE = DATA / 'archive'
TAIL_SECONDS = 90


def log(*parts):
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp}]', *parts, flush=True)


# -- fetching ----------------------------------------------------------------

def fetch_kalshi(api: Predexon, item, _tokens):
    """Every snapshot in the window, following pagination to exhaustion.

    `start_time` and `end_time` are required. Without them the endpoint
    returns nothing, even for a window that has a book.
    """
    lo = int(item.window_open.timestamp() * 1000)
    hi = int((item.window_open + dt.timedelta(minutes=15, seconds=TAIL_SECONDS))
             .timestamp() * 1000)
    out, cursor, pages = [], None, 0
    while pages < 30:
        params = {'ticker': item.market_id, 'start_time': lo, 'end_time': hi,
                  'limit': 2000}
        if cursor:
            params['pagination_key'] = cursor
        payload, ok = api.get('/kalshi/orderbooks', params)
        if not ok:
            return None, 'request failed'
        rows = (payload or {}).get('snapshots') or (payload or {}).get('data') or []
        out += rows
        pages += 1
        page = (payload or {}).get('pagination') or {}
        cursor = page.get('pagination_key')
        if not page.get('has_more') or not cursor or not rows:
            break
    return out, None


def fetch_pm(api: Predexon, item, tokens):
    """Every snapshot in the window, following pagination to exhaustion.

    Busy windows exceed one 2,000-snapshot page, so this must paginate.
    """
    token = tokens.get(item.market_id)
    if not token:
        return None, 'no token_id in catalog'
    lo = int(item.window_open.timestamp() * 1000)
    hi = int((item.window_open + dt.timedelta(minutes=15, seconds=TAIL_SECONDS))
             .timestamp() * 1000)
    out, cursor, pages = [], None, 0
    while pages < 30:
        params = {'token_id': token, 'start_time': lo, 'end_time': hi,
                  'limit': 2000}
        if cursor:
            params['pagination_key'] = cursor
        payload, ok = api.get('/polymarket/orderbooks', params)
        if not ok:
            return None, 'request failed'
        rows = (payload or {}).get('snapshots') or (payload or {}).get('data') or []
        out += rows
        pages += 1
        page = (payload or {}).get('pagination') or {}
        cursor = page.get('pagination_key')
        # `not rows` stops a venue that claims has_more and returns nothing.
        if not page.get('has_more') or not cursor or not rows:
            break
    return out, None


FETCHERS = {'kalshi': (fetch_kalshi, pack_kalshi),
            'polymarket': (fetch_pm, pack_pm)}


def _archive_path(item) -> Path:
    return (ARCHIVE / f'venue={item.venue}' / f'symbol={item.symbol}'
            / f'month={item.window_open:%Y-%m}' / 'windows.jsonl.gz')


def write_window(item, snapshots, packed) -> int:
    """Archive the raw ladders and the packed summary for one window.

    Appends gzipped JSONL per (venue, symbol, month). Appending is crash-safe:
    a kill costs one partial line. Each append is its own gzip member, which
    `gzip.open` reads back. `consolidate` converts a closed month to Parquet.
    """
    path = _archive_path(item)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({
        'venue': item.venue, 'symbol': item.symbol,
        'market_id': item.market_id,
        'window_open': item.window_open.isoformat(),
        'fields': list(FIELDS), 'n': len(packed),
        'series': packed, 'ladders': snapshots,
    })
    # Level 6: near level 9's ratio at about a third of its time.
    with gzip.open(path, 'at', compresslevel=6) as handle:
        handle.write(record + '\n')
    return len(record)


# -- phases ------------------------------------------------------------------

def phase_catalog(api: Predexon, venue: str = 'both') -> None:
    """Rebuild one or both catalogs.

    Selectable per venue, because the two are independent and each is slow.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    if venue in ('kalshi', 'both'):
        log('KALSHI catalog')
        n = kalshi_catalog(api, KALSHI_CATALOG, log=log)
        log(f'  {n:,} Kalshi markets')
    if venue in ('polymarket', 'both'):
        # Targeted lookup, which is much cheaper than paging every asset.
        log('POLYMARKET catalog')
        n = pm_catalog_by_grid(api, PM_CATALOG, log=log)
        log(f'  {n:,} Polymarket markets')


def phase_seed() -> None:
    ledger = Ledger(LEDGER_PATH)
    seed_from_catalogs(ledger, [KALSHI_CATALOG, PM_CATALOG], log=log)
    log('  ledger:', ledger.counts())


def _load_tokens() -> dict:
    tokens = {}
    if os.path.exists(PM_CATALOG):
        with open(PM_CATALOG) as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    tokens[row['market_id']] = row['token_id']
                except Exception:                             # noqa: BLE001
                    continue
    return tokens


def phase_collect(api: Predexon, *, batch: int = 200, month=None,
                  max_windows: int = 0, since=None, workers: int = 4) -> int:
    ledger = Ledger(LEDGER_PATH)
    breaker = Breaker(threshold=0.25, window=40)
    tokens = _load_tokens()
    start = time.monotonic()
    done = errors = 0

    total_open = sum(v for k, v in ledger.counts().items()
                     if k in ('pending', 'error'))
    if max_windows:
        total_open = min(total_open, max_windows)
    log(f'starting: {total_open:,} windows outstanding')

    # Fetches are transfer-bound, so several run at once behind the shared
    # rate limiter.
    # One write lock per partition file, not one global lock. `locks_guard`
    # stops two threads from making two locks for the same path.
    write_locks: dict = {}
    locks_guard = threading.Lock()

    def lock_for(path) -> threading.Lock:
        key = str(path)
        with locks_guard:
            lk = write_locks.get(key)
            if lk is None:
                lk = write_locks[key] = threading.Lock()
        return lk

    def do(item):
        fetch, pack = FETCHERS[item.venue]
        snapshots, failure = fetch(api, item, tokens)
        if failure:
            return item, 'error', 0, 0, failure
        if not snapshots:
            # The venue answered with no book. Recorded, never retried.
            return item, 'empty', 0, 0, None
        packed = [pack(s) for s in snapshots]
        # Workers serialise only at the write to one partition file.
        with lock_for(_archive_path(item)):
            size = write_window(item, snapshots, packed)
        return item, 'ok', len(packed), size, None

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
      while True:
        items = ledger.claim(batch, month=month, since=since)
        if not items:
            log('nothing left to claim')
            return 0
        for item, status, snaps, size, failure in pool.map(do, items):
            # One thread writes the ledger: SQLite connections are not
            # shareable, and status transitions stay ordered.
            if status == 'error':
                ledger.record(item, 'error', error=failure)
                breaker.record(ok=False)
                errors += 1
            elif status == 'empty':
                ledger.record(item, 'empty')
                breaker.record(ok=True)
            else:
                ledger.record(item, 'ok', snapshots=snaps, bytes_=size)
                breaker.record(ok=True)
            done += 1

            if max_windows and done >= max_windows:
                rate = done / max(time.monotonic() - start, 1e-9)
                counts = ledger.counts()
                log(f'trial limit reached: {done:,} windows at {rate:.2f}/s '
                    f'({1 / rate:.2f}s each)')
                log(f'  ok={counts.get("ok", 0):,} empty={counts.get("empty", 0):,} '
                    f'err={counts.get("error", 0):,} | '
                    f'{api.throttled:,} throttled of {api.calls:,} calls')
                return 0

            if breaker.tripped:
                log(f'CIRCUIT BREAKER: {breaker.failure_rate:.0%} of the last '
                    f'{breaker.window} attempts failed. Pausing so an outage '
                    f'cannot poison the ledger. Re-run to resume.')
                return 2

            if done % 100 == 0:
                elapsed = max(time.monotonic() - start, 1e-9)
                rate = done / elapsed
                left = max(total_open - done, 0)
                eta = left / rate / 3600 if rate else 0
                counts = ledger.counts()
                # req/s against the venue limit shows whether the run is at
                # the ceiling or wasting slots.
                log(f'{done:,}/{total_open:,}  {rate:.2f} win/s  '
                    f'{api.calls / elapsed:.2f} req/s  '
                    f'({api.throttled:,} throttled of {api.calls:,})  '
                    f'ETA {eta:.1f}h  ok={counts.get("ok", 0):,} '
                    f'empty={counts.get("empty", 0):,} err={counts.get("error", 0):,}')


def phase_report() -> None:
    ledger = Ledger(LEDGER_PATH)
    print(f'{"month":9s} {"venue":11s} {"ok":>7s} {"empty":>7s} {"error":>7s} '
          f'{"pending":>8s} {"yield":>7s}')
    print('-' * 60)
    for row in ledger.coverage():
        y = f'{row["yield_pct"]:.0f}%' if row['yield_pct'] is not None else '-'
        print(f'{row["month"]:9s} {row["venue"]:11s} {row["ok"] or 0:7,} '
              f'{row["empty"] or 0:7,} {row["error"] or 0:7,} '
              f'{row["pending"] or 0:8,} {y:>7s}')
    print()
    print('totals:', ledger.counts())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--phase', required=True,
                        choices=('catalog', 'seed', 'collect', 'report'))
    parser.add_argument('--batch', type=int, default=200)
    parser.add_argument('--month', default=None,
                        help='limit to one YYYY-MM, for a trial slice')
    parser.add_argument('--workers', type=int, default=4,
                        help='concurrent fetches sharing the one rate limiter')
    parser.add_argument('--since', default=None,
                        help='only windows at or after this ISO date')
    parser.add_argument('--max-windows', type=int, default=0,
                        help='stop after N windows; for a bounded trial that '
                             'measures real per-window cost before committing '
                             'to the full run')
    parser.add_argument('--venue', default='both',
                        choices=('kalshi', 'polymarket', 'both'),
                        help='catalog phase only: which venue to rebuild')
    args = parser.parse_args()

    if args.phase == 'report':
        phase_report()
        return 0
    if args.phase == 'seed':
        phase_seed()
        return 0

    key = os.getenv('PREDEXON_API_KEY', '').strip()
    if not key:
        print('PREDEXON_API_KEY is not set.')
        return 1
    # An env var, because the limit depends on the plan: 1 req/s on the free
    # tier, 20 on Dev.
    rps = float(os.getenv('PREDEXON_RPS', '1'))
    api = Predexon(key, RateLimiter(rps))
    log(f'rate limit {rps:g} req/s, {args.workers} workers')
    # One collector at a time: the Predexon bucket is org-wide.
    with SingleWriterLock(LOCK_PATH):
        if args.phase == 'catalog':
            phase_catalog(api, args.venue)
            return 0
        since = (dt.datetime.fromisoformat(args.since).replace(tzinfo=dt.timezone.utc)
                 if args.since else None)
        return phase_collect(api, batch=args.batch, month=args.month,
                             max_windows=args.max_windows, since=since,
                             workers=args.workers)


if __name__ == '__main__':
    raise SystemExit(main())
