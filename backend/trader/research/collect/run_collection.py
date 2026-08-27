"""The collection loop: claim work from the ledger, fetch it, record what came back.

Run phases separately so each can be inspected before the next:

    python -m research.collect.run_collection --phase catalog   # ~1.3h
    python -m research.collect.run_collection --phase seed      # instant
    python -m research.collect.run_collection --phase collect   # ~47h, resumable
    python -m research.collect.run_collection --phase report    # coverage

`--phase collect` is interruptible at any moment: resume is a ledger query, so
restarting re-reads the same table and picks up whatever is not finished. A
`kill -9` costs at most the batch in flight.
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

    start_time/end_time are REQUIRED. Omitting them returns nothing even for a
    window that certainly has a book, which is how BOOK_COVERAGE_START came to
    sit five months late.
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

    This was ONE request, on the assumption that Polymarket returns a whole
    window at once. That held for the August trial, whose windows averaged 358
    snapshots, and is false on busy months: 3,097 collected windows came back
    with EXACTLY 2,000 snapshots and not one above it, while Kalshi — which
    paginates — had a single window at exactly 2,000 and thousands beyond.
    That distribution is the shape of a silent truncation, and it cost 44% of
    the Polymarket windows collected before it was found.

    Confirmed against the endpoint: a January BTC window returns 2,000
    snapshots with `has_more: true`, and the next page returns 2,000 more with
    `has_more` still true.
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
        # `not rows` guards a venue that claims has_more and returns nothing,
        # which would otherwise spin until the page cap.
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

    Gzipped JSONL per (venue, symbol, month) while collecting. Appending is
    crash-safe, which a Parquet rewrite is not: a partial line from a kill -9
    costs one window rather than a partition. Converting to Parquet/zstd is
    Phase 4's job, once the month is closed.

    Compressed because the raw ladders are 97% of the bytes and measure 19.5x
    smaller gzipped (21.4MB -> 1.1MB on a real partition). Uncompressed, the
    corpus projects to ~202GB against 477GB free; compressed it is ~10GB. Each
    append is its own gzip member, which `gzip.open` reads back transparently,
    so the crash-safety of appending survives the compression.
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
    # Level 6, measured on a real 3.58MB window: 25ms for 33.1x, against
    # level 9's 86ms for 36.5x and level 1's 9ms for 18.8x. The extra 61ms a
    # window that level 9 costs is ~2.2 hours across the corpus, to save ~10%
    # of an already-13GB archive.
    with gzip.open(path, 'at', compresslevel=6) as handle:
        handle.write(record + '\n')
    return len(record)


# -- phases ------------------------------------------------------------------

def phase_catalog(api: Predexon, venue: str = 'both') -> None:
    """Rebuild one or both catalogs.

    Selectable per venue because they are independent and each takes tens of
    minutes: re-running the pair to redo one of them wastes a walk that
    already exhausted the venue's pagination.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    if venue in ('kalshi', 'both'):
        log('KALSHI catalog')
        n = kalshi_catalog(api, KALSHI_CATALOG, log=log)
        log(f'  {n:,} Kalshi markets')
    if venue in ('polymarket', 'both'):
        # Targeted lookup rather than paging the whole venue: pagination walks
        # eight-plus assets to extract three and degrades with depth, from 1.8
        # days of history a minute down to 0.55 four hundred pages in.
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

    # Fetches are transfer-bound, not rate-bound: a window with a large book
    # spends 1.5-5.4s mostly downloading, while issuing ONE request. Run
    # several at once behind the shared limiter and the same 1 req/s budget
    # buys several times the throughput. Serial, the corpus measured ~77h;
    # the rate-limit floor is ~36h.
    # One lock PER PARTITION FILE, not one globally. Appends to different
    # (venue, symbol, month) files cannot corrupt each other, and a single
    # global lock serialises every write — measured 0.04-0.09s each, which at
    # 5 windows/s is ~45% duty and becomes the binding constraint once the
    # rate limit stops being one. `defaultdict` under its own lock so two
    # threads cannot mint two locks for the same path.
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
            # The venue answered and there is no book. An ANSWER, recorded as
            # such and never retried.
            return item, 'empty', 0, 0, None
        packed = [pack(s) for s in snapshots]
        # One append per (venue, symbol, month) file, so the workers serialise
        # only at the write, which measured under 0.1s.
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
            # The ledger is written from ONE thread: SQLite connections are not
            # shareable, and this keeps every status transition ordered.
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
                # req/s is the number that matters: the venue's limit is 1 and
                # a window costs ~2.6 requests, so the window rate is capped
                # near 0.39/s however many workers run. Printing calls and
                # throttles makes "are we at the ceiling or wasting slots?"
                # answerable instead of inferred.
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
    # Measured after the Dev upgrade: header reports limit 20, burst 40, and a
    # sweep to 20 req/s drew zero throttling. Kept as an env var rather than a
    # constant because the free tier is 1 and the difference is 20x — a wrong
    # default either wastes 19/20ths of the budget or gets the key throttled.
    rps = float(os.getenv('PREDEXON_RPS', '1'))
    api = Predexon(key, RateLimiter(rps))
    log(f'rate limit {rps:g} req/s, {args.workers} workers')
    # One collector at a time: the Predexon bucket is org-wide, and two
    # runners throttle each other into 429s that look exactly like empty books.
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
