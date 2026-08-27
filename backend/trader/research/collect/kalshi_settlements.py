"""The venue's own settlement for every Kalshi 15-minute window, with its PRICE.

Run:
    python -m research.collect.kalshi_settlements
    python -m research.collect.kalshi_settlements --report

**Why this exists, and why against Kalshi rather than Predexon.**

Predexon is missing `result` on 6,828 Kalshi markets that TRADED. A market
carrying $2.9M of volume unambiguously settled, so a blank result there is a
provider gap, not a void market — and Kalshi has them: the ticker
KXBTC15M-26AUG062245-45 reads `status: finalized, result: yes` on Kalshi's own
API and blank in our Predexon catalog.

It also carries `expiration_value`, the price the window actually settled at
(80383.64 on a real BTC window). Nothing else in this project holds that.
`_validate_label.py` can currently only score our Coinbase-derived UP/DOWN
against the venue's UP/DOWN, giving 97% agreement. With settlement PRICES the
question becomes numeric — by how many basis points does a one-minute Coinbase
mean differ from CF Benchmarks' BRTI? — which is a measurement of the proxy's
bias rather than a count of how often it flips the answer.

**Cost.** The list endpoint returns 200 markets a page, each already carrying
`result` and `expiration_value`, so the whole corpus is ~343 requests. Kalshi's
public API served six back-to-back requests with zero spacing and no
throttling, and it is a DIFFERENT rate bucket from Predexon's org-wide 1 req/s
— so this runs alongside the collection rather than competing with it. The
pacing below is politeness, not necessity.

**Its limit is retention, not rate.** Kalshi purges markets past roughly two
months; a ticker held in our own store 404s there. So this recovers the recent
months and cannot reach January — measured, the floor is about 2026-06-20.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.collect.fetchers import kalshi_window_open, verify_window  # noqa: E402

BASE = 'https://api.elections.kalshi.com/trade-api/v2'
SERIES = {'KXBTC15M': 'BTC-USD', 'KXETH15M': 'ETH-USD', 'KXSOL15M': 'SOL-USD'}
OUT = Path(os.getenv('COLLECT_DATA', 'data/collection')) / 'kalshi_settlements.jsonl'
# Politeness, not a limit we were given. Kalshi took 15 req/s without
# complaint; this finishes the corpus in about two minutes either way, and
# being a good citizen on a free unmetered API costs nothing here.
REQUESTS_PER_SECOND = float(os.getenv('KALSHI_RPS', '5'))
PAGE = 200                                    # the endpoint's own page size
SETTLED = ('finalized', 'settled', 'determined')


def wanted_series() -> list:
    return list(SERIES)


def _time(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def parse_market(m: dict):
    """One settled market as a record, or None if it should not be stored.

    Rejects rather than guesses in three cases, each mirroring a failure this
    project has already had:

      * a ticker that does not decode, or is not one of our three series;
      * a decoded window the venue's own open/close times contradict — the
        fifteen-minute-shift bug, which went unnoticed for weeks because every
        window was a valid window;
      * a market that has not settled. Storing a blank result would rebuild
        the exact ambiguity this file exists to remove.
    """
    ticker = str(m.get('ticker') or '')
    series = ticker.split('-')[0]
    if series not in SERIES:
        return None
    try:
        opened = kalshi_window_open(ticker)
    except ValueError:
        return None
    if verify_window(opened, venue_open=_time(m.get('open_time')),
                     venue_close=_time(m.get('close_time'))):
        return None
    result = str(m.get('result') or '').strip().lower()
    status = str(m.get('status') or '').strip().lower()
    if not result or status not in SETTLED:
        return None
    value = m.get('expiration_value')
    # Zero is kept. A settlement value is an observation, and a real zero
    # dropped as "missing" is the same error as a zero quote read as a price,
    # in the opposite direction.
    try:
        value = float(value) if value is not None else None
    except (TypeError, ValueError):
        value = None
    return {
        'venue': 'kalshi',
        'symbol': SERIES[series],
        'market_id': ticker,
        'window_open': opened.isoformat(),
        'status': status,
        'result': result,
        'expiration_value': value,
        'close_time': str(m.get('close_time') or ''),
    }


class Kalshi:
    def __init__(self, rps: float = REQUESTS_PER_SECOND):
        self.interval = 1.0 / max(rps, 0.1)
        self._next = 0.0
        self.calls = 0
        self.throttled = 0
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'User-Agent': 'Mozilla/5.0 (quarter research collector)'})

    def get(self, path: str, params: dict, *, tries: int = 5):
        for attempt in range(tries):
            now = time.monotonic()
            if now < self._next:
                time.sleep(self._next - now)
            self._next = time.monotonic() + self.interval
            self.calls += 1
            try:
                r = self.session.get(f'{BASE}{path}', params=params, timeout=45)
                if r.status_code == 429 or r.status_code >= 500:
                    self.throttled += 1
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if r.status_code >= 400:
                    return None, True          # answered; nothing there
                return r.json(), True
            except Exception:                                 # noqa: BLE001
                self.throttled += 1
                time.sleep(1.5 * (attempt + 1))
        return None, False


def collect(api: Kalshi, *, log=print) -> int:
    """Walk each series' settled markets, newest first, to the retention floor."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if OUT.exists():                          # idempotent: re-running tops up
        with open(OUT) as handle:
            for line in handle:
                try:
                    seen.add(json.loads(line)['market_id'])
                except Exception:             # noqa: BLE001
                    continue
        log(f'  {len(seen):,} settlements already held')

    written = 0
    with open(OUT, 'a') as handle:
        for series, symbol in SERIES.items():
            cursor, pages, kept = None, 0, 0
            while pages < 400:
                params = {'series_ticker': series, 'status': 'settled',
                          'limit': PAGE}
                if cursor:
                    params['cursor'] = cursor
                payload, ok = api.get('/markets', params)
                if not ok:
                    log(f'  {series}: request failed, stopping this series')
                    break
                markets = (payload or {}).get('markets') or []
                if not markets:
                    break
                pages += 1
                for m in markets:
                    row = parse_market(m)
                    if row is None or row['market_id'] in seen:
                        continue
                    seen.add(row['market_id'])
                    handle.write(json.dumps(row) + '\n')
                    kept += 1
                    written += 1
                handle.flush()
                cursor = (payload or {}).get('cursor')
                if not cursor:
                    break
            log(f'  {series}: {kept:,} new settlements over {pages} pages')
    return written


def report() -> None:
    """What this adds on top of the Predexon catalog."""
    if not OUT.exists():
        print('nothing collected yet'); return
    rows = []
    with open(OUT) as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except Exception:                                 # noqa: BLE001
                continue
    have_value = [r for r in rows if r.get('expiration_value') is not None]
    print(f'{len(rows):,} settlements, {len(have_value):,} with a settlement price')
    months = {}
    for r in rows:
        months.setdefault(r['window_open'][:7], 0)
        months[r['window_open'][:7]] += 1
    for m in sorted(months):
        print(f'  {m}: {months[m]:,}')

    # How many Predexon gaps does this actually close?
    cat = Path(os.getenv('COLLECT_DATA', 'data/collection')) / 'kalshi_catalog.jsonl'
    if not cat.exists():
        return
    missing = set()
    with open(cat) as handle:
        for line in handle:
            try:
                c = json.loads(line)
            except Exception:                                 # noqa: BLE001
                continue
            if not c.get('result'):
                missing.add(c['market_id'])
    filled = missing & {r['market_id'] for r in rows}
    print(f'\nPredexon rows with no result: {len(missing):,}')
    print(f'  now filled from Kalshi     : {len(filled):,}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--rps', type=float, default=REQUESTS_PER_SECOND)
    args = parser.parse_args()
    if args.report:
        report()
        return 0
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%H:%M:%S')
    print(f'[{stamp}] Kalshi settlements (own API, separate rate bucket)', flush=True)
    api = Kalshi(args.rps)
    n = collect(api)
    print(f'  {n:,} new settlements written to {OUT}')
    print(f'  {api.calls:,} requests, {api.throttled:,} throttled')
    report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
