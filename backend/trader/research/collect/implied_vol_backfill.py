"""Rebuild forward-looking sigma from the historical strike ladders.

Fits `record_implied_vol.implied_sigma` to each hourly KXBTCD/ETHD/SOLD ladder's
cross-section near its close, from Predexon tick series.
Run: `python -m research.collect.implied_vol_backfill [--rps 5] [--report]`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.collect.catalog import Predexon                 # noqa: E402
from research.collect.orchestrator import RateLimiter, SingleWriterLock  # noqa: E402
from scripts.record_implied_vol import (                      # noqa: E402
    MIN_STRIKES, implied_sigma, strike_of,
)

LADDER_SERIES = {'KXBTCD': 'BTC-USD', 'KXETHD': 'ETH-USD', 'KXSOLD': 'SOL-USD'}
DATA = Path(os.getenv('COLLECT_DATA', 'data/collection'))
OUT = DATA / 'implied_vol.jsonl'
STATE = DATA / 'implied_vol.state.json'
LOCK = DATA / 'implied_vol.lock'
# The same floor as the 15-minute books.
COLLECT_FROM = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)
MAX_PAGES = 30
# Ladders are hourly with ~80 strikes each, so the fetch is limited two ways.
#
# The sampled band, in minutes before the ladder's close: only the run-up is
# fetched. It gives the short-horizon sigma the 15-minute markets need, and it
# is where two-sided rungs exist. An hourly ladder has no quotes an hour out.
TAIL_MINUTES = int(os.getenv('IV_TAIL_MINUTES', '50'))
NEAREST_MINUTES = int(os.getenv('IV_NEAREST_MINUTES', '5'))
# History fetched before the first instant. `cross_section` takes the last tick
# at or before an instant, so the first instant needs ticks behind it.
LEAD_MINUTES = int(os.getenv('IV_LEAD_MINUTES', '5'))
# Take every Nth hourly ladder. Consecutive ladders overlap, and implied vol
# moves slowly, so a stride trades density for API cost.
LADDER_STRIDE = int(os.getenv('IV_LADDER_STRIDE', '1'))
# Only strikes near spot. A far rung sits outside implied_sigma's 0.01-0.99
# band and is discarded. Spot comes from local Coinbase bars.
NEAR_STRIKES = int(os.getenv('IV_NEAR_STRIKES', '24'))
# Latency, not rate, binds: see fetch_rungs.
WORKERS = int(os.getenv('IV_WORKERS', '12'))
# How often to invert inside the fetched window. Extra instants cost no
# request, and sigma is needed at several minutes-to-close.
SAMPLE_MINUTES = int(os.getenv('IV_SAMPLE_MINUTES', '5'))
# The markets endpoint's ceiling. A larger limit returns zero rows.
MARKETS_PAGE = 100


def log(*parts):
    print(f'[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}]', *parts, flush=True)


def sample_instants(opened: dt.datetime, closes: dt.datetime, minutes: int = 15):
    """The quarter-hour grid inside the ladder's life, excluding the close.

    At the close no time remains, so sigma is undefined.
    """
    step = dt.timedelta(minutes=minutes)
    out, when = [], opened
    while when < closes:
        out.append(when)
        when += step
    return out


def cross_section(series: dict, at_ms: int):
    """[(strike, P_above)] as the ladder stood at `at_ms`, sorted by strike.

    Point-in-time: uses the last tick at or before the instant, never a later
    one. Drops a one-sided quote, which gives no P(above).
    """
    rungs = []
    for strike, snaps in series.items():
        stamps = snaps['ts']
        idx = bisect_right(stamps, at_ms) - 1
        if idx < 0:
            continue
        bid, ask = snaps['bid'][idx], snaps['ask'][idx]
        if bid is None or ask is None:
            continue
        rungs.append((float(strike), (bid + ask) / 200.0))
    rungs.sort(key=lambda r: r[0])
    return rungs


def pack_series(snapshots):
    """A strike's tick series as parallel sorted lists, for bisect lookups."""
    rows = []
    for s in snapshots:
        ts = s.get('timestamp')
        if ts is None:
            continue
        bids = s.get('yes_bids') or []
        asks = s.get('yes_asks') or []
        bid = max((e['price'] for e in bids), default=None)
        ask = min((e['price'] for e in asks), default=None)
        rows.append((int(ts), bid, ask))
    # by timestamp only: a None bid or ask makes tuple comparison raise
    rows.sort(key=lambda r: r[0])
    return {'ts': [r[0] for r in rows], 'bid': [r[1] for r in rows],
            'ask': [r[2] for r in rows]}


def near_the_money(strikes: list, spot) -> list:
    """The NEAR_STRIKES rungs closest to spot, or all of them if spot is unknown.

    With spot unknown, keeps every strike rather than an arbitrary subset.
    """
    if spot is None or len(strikes) <= NEAR_STRIKES:
        return strikes
    return sorted(strikes, key=lambda sk: abs(sk[0] - spot))[:NEAR_STRIKES]


def load_bars():
    """Coinbase minute bars, for the spot used to centre the strike selection."""
    try:
        from core.datastore import ResearchStore
        frame = ResearchStore(os.getenv('RESEARCH_STORE')).read('minute_bars')
    except Exception as exc:                                  # noqa: BLE001
        log(f'  minute_bars unavailable ({str(exc)[:60]}); fetching all strikes')
        return None
    import pandas as pd
    frame = frame[['symbol', 'event_time', 'close']].copy()
    frame['event_time'] = pd.to_datetime(frame['event_time'], utc=True)
    return {sym: grp.sort_values('event_time')
            for sym, grp in frame.groupby('symbol')}


def spot_at(bars, symbol: str, when):
    if not bars or symbol not in bars:
        return None
    import pandas as pd
    grp = bars[symbol]
    idx = grp['event_time'].searchsorted(pd.Timestamp(when), side='right') - 1
    if idx < 0:
        return None
    try:
        return float(grp['close'].iloc[idx])
    except Exception:                                         # noqa: BLE001
        return None


def _time(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None


def _when(value):
    """An ISO date or instant from the command line, as an aware UTC datetime."""
    if not value:
        return None
    parsed = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.timezone.utc)


def ladders(api: Predexon, series: str) -> dict:
    """{event_ticker: (open, close)} for every ladder at or after the floor."""
    found, cursor, pages = {}, None, 0
    while pages < 400:
        params = {'series_ticker': series, 'limit': MARKETS_PAGE}
        if cursor:
            params['pagination_key'] = cursor
        payload, ok = api.get('/kalshi/markets', params)
        if not ok:
            break
        markets = (payload or {}).get('markets') or []
        if not markets:
            break
        pages += 1
        for m in markets:
            ev = m.get('event_ticker') or str(m.get('ticker') or '').rsplit('-', 1)[0]
            o, c = _time(m.get('open_time')), _time(m.get('close_time'))
            if not ev or not o or not c or o < COLLECT_FROM:
                continue
            found.setdefault(ev, (o, c))
        cursor = ((payload or {}).get('pagination') or {}).get('pagination_key')
        if not cursor:
            break
    return found


def strikes_of(api: Predexon, event_ticker: str) -> list:
    """Every strike market in one ladder, asked for by event.

    Queried by event, because a series walk spreads one ladder across pages.
    """
    payload, ok = api.get('/kalshi/markets',
                          {'event_ticker': event_ticker, 'limit': MARKETS_PAGE})
    if not ok:
        return []
    out = []
    for m in (payload or {}).get('markets') or []:
        strike = strike_of(m)
        if strike is not None:
            out.append((strike, str(m.get('ticker'))))
    return out


def book_of(api: Predexon, ticker: str, lo_ms: int, hi_ms: int):
    """Every snapshot for one strike across the ladder's life."""
    out, cursor, pages = [], None, 0
    while pages < MAX_PAGES:
        params = {'ticker': ticker, 'start_time': lo_ms, 'end_time': hi_ms,
                  'limit': 2000}
        if cursor:
            params['pagination_key'] = cursor
        payload, ok = api.get('/kalshi/orderbooks', params)
        if not ok:
            return out, False
        rows = (payload or {}).get('snapshots') or []
        out += rows
        pages += 1
        page = (payload or {}).get('pagination') or {}
        cursor = page.get('pagination_key')
        if not page.get('has_more') or not cursor or not rows:
            break
    return out, True


def select_ladders(every, done, stride: int, max_ladders: int = 0, *,
                   since=None, until=None) -> list:
    """Which ladders this run should walk, from the full sorted population.

    Order matters. The stride applies to every ladder before finished ones are
    removed, so a resume keeps the same grid. `since`/`until` filter by close
    time after the stride. The cap applies last.

    `every` as {ticker: (open, close)} orders by close time; ticker text sorts
    by month name, which is not time order.
    """
    if isinstance(every, dict):
        target = sorted(every, key=lambda e: (every[e][1], e))
    else:
        target = sorted(every)
    if stride > 1:
        target = target[::stride]
    if isinstance(every, dict) and (since is not None or until is not None):
        target = [e for e in target
                  if (since is None or every[e][1] >= since)
                  and (until is None or every[e][1] < until)]
    if max_ladders:
        target = target[:max_ladders]
    return [e for e in target if e not in done]


def windows_for(opened, closes):
    """(fetch_from, instants) for one ladder.

    The fetch starts LEAD_MINUTES before the first instant, so that instant
    has a tick behind it. Both are clamped to the ladder's open.
    """
    first = closes - dt.timedelta(minutes=TAIL_MINUTES)
    last = closes - dt.timedelta(minutes=NEAREST_MINUTES)
    first = max(first, opened)
    instants, at = [], first
    step = dt.timedelta(minutes=SAMPLE_MINUTES)
    while at <= last:
        if at >= opened:
            instants.append(at)
        at += step
    fetch_from = max(opened, first - dt.timedelta(minutes=LEAD_MINUTES))
    return fetch_from, instants


def fetch_rungs(api: Predexon, strikes, lo_ms: int, hi_ms: int, *,
                workers: int = 1) -> dict:
    """One ladder's strikes to packed tick series, fetched concurrently.

    Concurrent behind the shared limiter, because round-trip latency binds,
    not the rate. A strike with no book is omitted, so the result is the same
    at any worker count.
    """
    out: dict = {}
    if workers <= 1:
        for strike, ticker in strikes:
            snaps, _ = book_of(api, ticker, lo_ms, hi_ms)
            if snaps:
                out[strike] = pack_series(snaps)
        return out
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(book_of, api, ticker, lo_ms, hi_ms): strike
                   for strike, ticker in strikes}
        for fut in as_completed(futures):
            strike = futures[fut]
            try:
                snaps, _ = fut.result()
            except Exception:                                 # noqa: BLE001
                continue
            if snaps:
                out[strike] = pack_series(snaps)
    return out


def run(api: Predexon, *, only_series=None, max_ladders: int = 0,
        since=None, until=None) -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if STATE.exists():
        try:
            done = set(json.loads(STATE.read_text()).get('ladders', []))
            log(f'  resuming, {len(done):,} ladders already done')
        except (OSError, ValueError):
            done = set()

    bars = load_bars()
    written = 0
    with open(OUT, 'a') as handle:
        for series, symbol in LADDER_SERIES.items():
            if only_series and series != only_series:
                continue
            found = ladders(api, series)
            todo = select_ladders(found, done, LADDER_STRIDE, max_ladders,
                                  since=since, until=until)
            log(f'  {series}: {len(found):,} ladders in range, {len(todo):,} to do')
            for n, ev in enumerate(todo, 1):
                opened, closes = found[ev]
                # See windows_for.
                window_from, instants = windows_for(opened, closes)
                spot = spot_at(bars, symbol, closes)
                rungs_by_strike = fetch_rungs(
                    api, near_the_money(strikes_of(api, ev), spot),
                    int(window_from.timestamp() * 1000),
                    int(closes.timestamp() * 1000), workers=WORKERS)
                fits = 0
                for at in instants:
                    rungs = cross_section(rungs_by_strike, int(at.timestamp() * 1000))
                    if len(rungs) < MIN_STRIKES:
                        continue
                    minutes = (closes - at).total_seconds() / 60.0
                    fit = implied_sigma(rungs, minutes)
                    if fit is None:
                        continue
                    handle.write(json.dumps({
                        'venue': 'kalshi', 'symbol': symbol,
                        'event_ticker': ev,
                        'event_time': at.isoformat(),
                        'close_time': closes.isoformat(),
                        'minutes_to_close': round(minutes, 3),
                        'implied_sigma_per_min': fit.sigma_per_min,
                        'implied_spot': fit.implied_spot,
                        'atm_strike': fit.atm_strike,
                        'n_strikes': fit.n_strikes,
                        'r2': fit.r2,
                    }) + '\n')
                    fits += 1
                    written += 1
                handle.flush()
                done.add(ev)
                STATE.write_text(json.dumps({'ladders': sorted(done)}))
                if n % 10 == 0 or fits == 0:
                    log(f'    {series} {n}/{len(todo)}  {ev}  {fits} fits  '
                        f'({api.throttled:,} throttled of {api.calls:,})')
    return written


def report() -> None:
    if not OUT.exists():
        print('nothing collected yet'); return
    import statistics as st
    from collections import Counter
    per = {}
    months = Counter()
    for line in open(OUT):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        per.setdefault(r['symbol'], []).append(r)
        months[r['event_time'][:7]] += 1
    print(f'{"symbol":9s} {"fits":>8s} {"median bp/min":>14s} {"median R2":>10s} '
          f'{"median rungs":>13s}')
    for sym, rows in sorted(per.items()):
        sig = [1e4 * r['implied_sigma_per_min'] for r in rows]
        print(f'{sym:9s} {len(rows):8,} {st.median(sig):14.2f} '
              f'{st.median([r["r2"] for r in rows]):10.4f} '
              f'{st.median([r["n_strikes"] for r in rows]):13.0f}')
    print('\nby month:', dict(sorted(months.items())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--rps', type=float,
                        default=float(os.getenv('PREDEXON_RPS', '5')))
    parser.add_argument('--series', default=None, choices=sorted(LADDER_SERIES))
    parser.add_argument('--max-ladders', type=int, default=0,
                        help='stop after N ladders per series, for a trial')
    parser.add_argument('--since', default=None,
                        help='only ladders closing at or after this ISO date')
    parser.add_argument('--until', default=None,
                        help='only ladders closing before this ISO date')
    args = parser.parse_args()
    if args.report:
        report()
        return 0
    key = os.getenv('PREDEXON_API_KEY', '').strip()
    if not key:
        print('PREDEXON_API_KEY is not set.')
        return 1
    # Its own lock: a separate job from the book collection. The two share
    # the Predexon bucket, so their rates must sum under the plan's limit.
    with SingleWriterLock(str(LOCK)):
        log(f'implied vol backfill at {args.rps:g} req/s')
        api = Predexon(key, RateLimiter(args.rps))
        n = run(api, only_series=args.series, max_ladders=args.max_ladders,
                since=_when(args.since), until=_when(args.until))
        log(f'  {n:,} fits written to {OUT}')
        log(f'  {api.calls:,} requests, {api.throttled:,} throttled')
    report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
