"""Rebuild forward-looking sigma from the historical strike ladders.

    python -m research.collect.implied_vol_backfill --rps 5
    python -m research.collect.implied_vol_backfill --report

**Why this is worth its own collector.** The barrier framing says the
displacement is known exactly and `sigma_remaining` is the ONE quantity that
must be forecast. Every volatility feature in `core/features.py` is
backward-looking realised vol. A threshold ladder inverts to the market's own
FORWARD-looking sigma at R² > 0.95, which is a different kind of input
entirely.

The collection plan listed this as a feature family and marked it "already
recorded live". That was wrong, and it is the one real error the plan review
found: `venue_implied_vol` held 1,256 rows across 3 days, BTC only. Three days
trains nothing. The ladders reach January, so the history exists — it just was
never collected.

**Measured shape of the data.** A ladder (`KXBTCD-26JUL3017`) is one event
holding ~80 strike markets, opens 20:00 and closes 21:00 the following day —
about 25 hours. Each strike's historical book is a tick series carrying a
two-sided quote, e.g. bid 38 / ask 39 at a $63,999.99 strike, which is
P(above) = 0.385. Predexon and Kalshi's own API agree exactly on the ladder's
size (80 markets each).

**The inversion is imported, not rewritten.** `record_implied_vol.implied_sigma`
is the single definition and the live path calls it; a second copy would be two
definitions that agree until one is changed. What is genuinely new here is
assembling the cross-section, because the live recorder sees every strike's
current quote at once while history arrives as one series per strike that must
be aligned onto a common instant.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from bisect import bisect_right
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
# The ladders reach January, the same floor the 15-minute books have.
COLLECT_FROM = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)
MAX_PAGES = 30
# Ladders are HOURLY, not daily: 4,960 for BTC alone in range. At ~80 strikes
# each that is ~400k requests per asset, so two economies are needed and both
# are principled rather than arbitrary.
#
# TAIL_MINUTES — only the run-up to the close is fetched. A ladder lives ~25
# hours, but sigma from a ladder closing a day out is a 25-hour-horizon vol;
# the 15-minute markets need a SHORT horizon. The final stretch is the part
# that answers the question, and it is also where the ladder is liquid.
# Measured: a near-the-money strike generates ~14,000 ticks in 120 minutes, so
# a 120-minute window costs ~7 pages PER STRIKE and ~179 requests per ladder.
# The sample instants are 15 minutes apart, so almost all of those ticks are
# fetched and discarded. A short window costs ~1 page per strike and yields the
# shortest-horizon sigma, which is the one the 15-minute markets need.
TAIL_MINUTES = int(os.getenv('IV_TAIL_MINUTES', '20'))
# Ladders are HOURLY and overlap, so consecutive ones say almost the same
# thing. A stride trades sigma's sampling density against API hours: 1 gives
# hourly sigma, 4 gives it every four hours for a quarter of the cost. Implied
# vol moves slowly, so a stride is a cheap way to test whether the feature
# earns its place before paying for full density.
LADDER_STRIDE = int(os.getenv('IV_LADDER_STRIDE', '1'))
# NEAR_STRIKES — only strikes bracketing spot. Far from spot a rung sits at
# P ~ 0 or 1, outside implied_sigma's 0.01-0.99 band, so it is discarded after
# being paid for. Spot comes from the Coinbase bars we already hold, so the
# selection costs no requests.
NEAR_STRIKES = int(os.getenv('IV_NEAR_STRIKES', '24'))
# `limit=100` is the markets endpoint's ceiling — 200 comes back empty, the
# same behaviour the Polymarket markets endpoint has.
MARKETS_PAGE = 100


def log(*parts):
    print(f'[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}]', *parts, flush=True)


def sample_instants(opened: dt.datetime, closes: dt.datetime, minutes: int = 15):
    """Where to evaluate sigma: the quarter-hour grid inside the ladder's life.

    The same grid the 15-minute up/down windows decide on, so the feature lines
    up with the rows that consume it. The close itself is excluded: with no
    time remaining sigma is a division by sqrt(0).
    """
    step = dt.timedelta(minutes=minutes)
    out, when = [], opened
    while when < closes:
        out.append(when)
        when += step
    return out


def cross_section(series: dict, at_ms: int):
    """[(strike, P_above)] as the ladder stood at `at_ms`, sorted by strike.

    The last tick AT OR BEFORE the instant, never the nearest: a book is a step
    function, and letting a later tick inform an earlier instant is lookahead —
    exactly the leak this project treats as its worst failure mode.

    A one-sided quote is dropped rather than half-invented. The inversion needs
    P(above), and a single side does not give one.
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

    Rungs far from spot price at ~0 or ~1 and are dropped by the inversion's
    probability band, so fetching them is a request spent on a rung that is
    discarded. When spot is unknown the selection is skipped rather than
    guessed — paying for extra strikes is better than silently fitting sigma to
    an arbitrary subset.
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

    Asked directly rather than filtered out of a series walk: paging the series
    returns the ladders spread across pages, which made a full 80-strike ladder
    look like it held one strike.
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


def run(api: Predexon, *, only_series=None, max_ladders: int = 0) -> int:
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
            todo = [e for e in sorted(found) if e not in done]
            if LADDER_STRIDE > 1:
                todo = todo[::LADDER_STRIDE]
            if max_ladders:
                todo = todo[:max_ladders]
            log(f'  {series}: {len(found):,} ladders in range, {len(todo):,} to do')
            for n, ev in enumerate(todo, 1):
                opened, closes = found[ev]
                # Only the run-up to the close, for the horizon reason above.
                window_from = max(opened, closes - dt.timedelta(minutes=TAIL_MINUTES))
                spot = spot_at(bars, symbol, closes)
                rungs_by_strike = {}
                for strike, ticker in near_the_money(strikes_of(api, ev), spot):
                    snaps, ok = book_of(api, ticker,
                                        int(window_from.timestamp() * 1000),
                                        int(closes.timestamp() * 1000))
                    if snaps:
                        rungs_by_strike[strike] = pack_series(snaps)
                fits = 0
                for at in sample_instants(window_from, closes):
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
    args = parser.parse_args()
    if args.report:
        report()
        return 0
    key = os.getenv('PREDEXON_API_KEY', '').strip()
    if not key:
        print('PREDEXON_API_KEY is not set.')
        return 1
    # Its OWN lock, not the collector's: this shares the Predexon rate bucket
    # with the book collection, so the two rates must be chosen to sum under
    # the plan's limit — but they are separate jobs and either may run alone.
    with SingleWriterLock(str(LOCK)):
        log(f'implied vol backfill at {args.rps:g} req/s')
        api = Predexon(key, RateLimiter(args.rps))
        n = run(api, only_series=args.series, max_ladders=args.max_ladders)
        log(f'  {n:,} fits written to {OUT}')
        log(f'  {api.calls:,} requests, {api.throttled:,} throttled')
    report()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
