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
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.collect.catalog import (                        # noqa: E402
    COLLECT_FROM, Predexon, kalshi_catalog, pm_catalog, seed_from_catalogs,
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
    token = tokens.get(item.market_id)
    if not token:
        return None, 'no token_id in catalog'
    lo = int(item.window_open.timestamp() * 1000)
    hi = int((item.window_open + dt.timedelta(minutes=15, seconds=TAIL_SECONDS))
             .timestamp() * 1000)
    payload, ok = api.get('/polymarket/orderbooks', {
        'token_id': token, 'start_time': lo, 'end_time': hi, 'limit': 2000})
    if not ok:
        return None, 'request failed'
    return ((payload or {}).get('snapshots')
            or (payload or {}).get('data') or []), None


FETCHERS = {'kalshi': (fetch_kalshi, pack_kalshi),
            'polymarket': (fetch_pm, pack_pm)}


def _archive_path(item) -> Path:
    return (ARCHIVE / f'venue={item.venue}' / f'symbol={item.symbol}'
            / f'month={item.window_open:%Y-%m}' / 'windows.jsonl')


def write_window(item, snapshots, packed) -> int:
    """Archive the raw ladders and the packed summary for one window.

    JSONL per (venue, symbol, month) while collecting: appending is
    crash-safe, which a Parquet rewrite is not, and a partial line from a
    kill -9 costs one window rather than a partition. Converting to
    Parquet/zstd is Phase 4's job, once the month is closed.
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
    with open(path, 'a') as handle:
        handle.write(record + '\n')
    return len(record)


# -- phases ------------------------------------------------------------------

def phase_catalog(api: Predexon) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    log('KALSHI catalog')
    n = kalshi_catalog(api, KALSHI_CATALOG, log=log)
    log(f'  {n:,} Kalshi markets')
    log('POLYMARKET catalog')
    n = pm_catalog(api, PM_CATALOG, log=log)
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


def phase_collect(api: Predexon, *, batch: int = 200, month=None) -> int:
    ledger = Ledger(LEDGER_PATH)
    breaker = Breaker(threshold=0.25, window=40)
    tokens = _load_tokens()
    start = time.monotonic()
    done = errors = 0

    total_open = sum(v for k, v in ledger.counts().items()
                     if k in ('pending', 'error'))
    log(f'starting: {total_open:,} windows outstanding')

    while True:
        items = ledger.claim(batch, month=month)
        if not items:
            log('nothing left to claim')
            return 0
        for item in items:
            fetch, pack = FETCHERS[item.venue]
            snapshots, failure = fetch(api, item, tokens)
            if failure:
                ledger.record(item, 'error', error=failure)
                breaker.record(ok=False)
                errors += 1
            elif not snapshots:
                # The venue answered and there is no book. An ANSWER, recorded
                # as such and never retried.
                ledger.record(item, 'empty')
                breaker.record(ok=True)
            else:
                packed = [pack(s) for s in snapshots]
                size = write_window(item, snapshots, packed)
                ledger.record(item, 'ok', snapshots=len(packed), bytes_=size)
                breaker.record(ok=True)
            done += 1

            if breaker.tripped:
                log(f'CIRCUIT BREAKER: {breaker.failure_rate:.0%} of the last '
                    f'{breaker.window} attempts failed. Pausing so an outage '
                    f'cannot poison the ledger. Re-run to resume.')
                return 2

            if done % 100 == 0:
                rate = done / max(time.monotonic() - start, 1e-9)
                left = max(total_open - done, 0)
                eta = left / rate / 3600 if rate else 0
                counts = ledger.counts()
                log(f'{done:,}/{total_open:,}  {rate:.2f}/s  ETA {eta:.1f}h  '
                    f'ok={counts.get("ok", 0):,} empty={counts.get("empty", 0):,} '
                    f'err={counts.get("error", 0):,}')


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
    api = Predexon(key, RateLimiter(1.0))
    # One collector at a time: the Predexon bucket is org-wide, and two
    # runners throttle each other into 429s that look exactly like empty books.
    with SingleWriterLock(LOCK_PATH):
        if args.phase == 'catalog':
            phase_catalog(api)
            return 0
        return phase_collect(api, batch=args.batch, month=args.month)


if __name__ == '__main__':
    raise SystemExit(main())
