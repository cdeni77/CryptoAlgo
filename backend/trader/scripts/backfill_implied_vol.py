"""Rebuild `venue_implied_vol` from the venue's own historical strike ladders.

**The sparse coverage is an artifact of the RECORDER, not of the venue.**
`scripts/record_implied_vol.py` fetches `/markets?series_ticker=...&status=open`
— so it only ever saw ladders that happened to be open while it was running.
Measured 2026-10-21, that left 13-28% of windows with a fit before August
against 61-68% after, and under `--complete-cases` the group then caps the
whole training set at 20,839 of 84,795 quoted symbol-windows. It is the single
largest constraint on the sample, and unlike the book-shape features it is
entirely recoverable.

**How.** A KXBTCD event ticker is `series + close in New York local`, fully
determined by the hour, so the ladders can be CONSTRUCTED rather than listed —
the same unlock as `backfill_markets`. Verified against the live venue:
`/historical/markets?event_ticker=KXBTCD-26MAR0211` returns 75 strikes, each
carrying `floor_strike`.

**The listing's quotes are the FINAL state and must not be used.** A settled
market reads `result: yes, last_price 0.99`; inverting that ladder would
measure the outcome, not the market's forward view — lookahead of the worst
kind. The point-in-time quote comes from each strike's own candlesticks.

**Only near-the-money strikes are fetched.** `implied_sigma` needs four usable
rungs and the information is concentrated where the ladder is not already at 0
or 1, so the strikes nearest spot are the ones that identify sigma. Fetching
all 75 would cost 5x for rungs that are pinned. Spot comes from the stored
Coinbase bars, which is point-in-time safe: it is the bar at or before the
fit minute.

Read-only against the venue. Writes `venue_implied_vol` through the same
`fits_for` the live recorder uses, so a backfilled fit and a recorded one are
the same computation.

    python -m scripts.backfill_implied_vol --start 2026-01-07 --dry-run
    python -m scripts.backfill_implied_vol --start 2026-01-07 --concurrency 6
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import datetime as dt
import logging
import os
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger('backfill_implied_vol')

NY = ZoneInfo('America/New_York')
MONTHS = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']

#: Minutes before the hourly close at which to fit. The live recorder fits
#: whenever it runs; these give an hourly grid dense enough that
#: `iv_staleness_minutes` stays small for any 15-minute window.
FIT_OFFSETS = (50, 35, 20, 10)

#: How many strikes either side of spot to price. `implied_sigma` needs four
#: usable rungs; strikes far from spot are pinned at 0 or 1 and identify
#: nothing, so fetching all ~75 would cost 5x for no information.
STRIKES_EACH_SIDE = 11


def event_ticker(series: str, close_utc: dt.datetime) -> str:
    """`KXBTCD-26MAR0211` — series plus the CLOSE in New York local time."""
    local = close_utc.astimezone(NY)
    return '%s-%02d%s%02d%02d' % (series, local.year % 100,
                                  MONTHS[local.month - 1], local.day, local.hour)


def _money(raw):
    if raw in (None, ''):
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value / 100.0 if value > 1.0 else value


def candle_quote(candle) -> tuple:
    """`(bid, ask)` at a candle's close, under either endpoint's spelling."""
    def side(name):
        block = candle.get(name)
        if not isinstance(block, dict):
            return None
        for key in ('close_dollars', 'close'):
            got = _money(block.get(key))
            if got is not None:
                return got
        return None
    return side('yes_bid'), side('yes_ask')


async def _listing(client, ev):
    """The event's strikes, from whichever tier serves it.

    **429 is not a missing event, and this swallowed the difference.** The
    first version caught every exception and moved on, so under rate pressure
    a listing failed silently and the hour was dropped with no retry and no
    log — while `_candles` right below it backed off correctly. Verified
    2026-10-21: `KXBTCD-26JAN3119` reported "no listing" during a saturated
    run and returned 75 markets on a quiet retry. That is the same
    silent-failure shape as counting an error as an empty tape, and it
    suppressed the coverage this whole backfill exists to raise.
    """
    for base, path in ((client.historical_base_url, '/historical/markets'),
                       (None, '/markets')):
        for attempt in range(5):
            try:
                payload = await client._request(           # noqa: SLF001
                    'GET', path, params={'event_ticker': ev, 'limit': 200},
                    base=base)
            except Exception as exc:                       # noqa: BLE001
                if '429' in str(exc) or 'too_many_requests' in str(exc):
                    await asyncio.sleep(1.5 * (2 ** attempt))
                    continue
                break
            markets = payload.get('markets') or []
            if markets:
                return markets
            break
    return []


#: Requests lost to rate limiting. Non-zero means fits are thinner than the
#: venue's data, and the run should be repeated rather than believed.
THROTTLED: dict = collections.defaultdict(int)


async def _candles(client, ticker, start, end):
    """Historical per-ticker, then the plural batch path. Two endpoints, not three.

    **`/markets/{ticker}/candlesticks` is not tried, because it 404s for every
    1-hour strike market at every date.** It used to sit in the middle of this
    chain, costing one guaranteed-failed request per strike -- measured, 38 of
    88 requests in an old-era sample, so roughly 30% of the whole backfill
    spent proving a known 404. It also muddied the diagnosis: a strike that
    was merely empty and a strike that 404'd arrived here as the same `[]`.

    A 429 is retried with backoff. **Exhausting those retries is LOGGED**, and
    that is the point of the counter: a throttled request returns `[]`, which
    is indistinguishable from "this strike never traded", so silent exhaustion
    would quietly thin every fit and we would explain it away as a quiet
    market. That is the same silent-failure shape already fixed in `_listing`
    today, where a 429 was reported as "no listing".
    """
    for attempt in range(5):
        try:
            payload = await client._request(               # noqa: SLF001
                'GET', '/historical/markets/%s/candlesticks' % ticker,
                base=client.historical_base_url,
                params={'start_ts': int(start.timestamp()),
                        'end_ts': int(end.timestamp()), 'period_interval': 1})
        except Exception as exc:                           # noqa: BLE001
            if '429' in str(exc):
                if attempt == 4:
                    THROTTLED['candles'] += 1
                    logger.warning(
                        'candles for %s gave up after 5 rate-limited attempts; '
                        'this strike will look EMPTY (%d so far)',
                        ticker, THROTTLED['candles'])
                    break
                await asyncio.sleep(1.5 * (2 ** attempt))
                continue
            break
        got = payload.get('candlesticks') or []
        if got:
            return got
        break
    return await _batch_candles(client, ticker, start, end)


async def _batch_candles(client, ticker, start, end):
    """The BATCH endpoint, which is the only one serving the recent era.

    The two per-ticker endpoints TILE rather than overlap: historical serves
    2025-12 to 2026-08-21 and then returns nothing, and `/markets/{t}/
    candlesticks` returns nothing for a 1-hour strike market at any date we
    tested. `/markets/candlesticks` -- plural, a different path, keyed on
    `market_tickers` -- covers roughly the last 69 days. Without it the
    rebuild stops five days short of where the live recorder starts, which is
    exactly the seam that made this look unvalidatable: there was no era
    holding both a candle-derived fit and a live-recorded one to compare.

    Measured against 10 live fits in that overlap, the candle fit reproduces
    the live one to a median 0.01 bp/min -- so this is not an approximation of
    what we record, it is the same number by another route.

    The response nests one level deeper (`markets[].candlesticks[]`) and names
    the ticker `ticker` or `market_ticker` depending on tier, so it cannot
    share the parse above.
    """
    for attempt in range(5):
        try:
            payload = await client._request(               # noqa: SLF001
                'GET', '/markets/candlesticks',
                params={'market_tickers': ticker, 'period_interval': 1,
                        'start_ts': int(start.timestamp()),
                        'end_ts': int(end.timestamp())})
        except Exception as exc:                           # noqa: BLE001
            if '429' in str(exc):
                await asyncio.sleep(1.5 * (2 ** attempt))
                continue
            return []
        for market in (payload.get('markets') or []):
            if (market.get('ticker') or market.get('market_ticker')) == ticker:
                return market.get('candlesticks') or []
        return []
    return []


async def hour_rows(client, sem, series, symbol, close_utc, spot_at, fetched_at,
                    min_r2):
    """Every fit for one hourly ladder."""
    from scripts.record_implied_vol import fits_for

    ev = event_ticker(series, close_utc)
    open_utc = close_utc - dt.timedelta(hours=1)
    async with sem:
        markets = await _listing(client, ev)
    if not markets:
        return []

    spot = spot_at(symbol, close_utc - dt.timedelta(minutes=max(FIT_OFFSETS)))
    if spot is None:
        return []
    rungs = []
    for m in markets:
        try:
            rungs.append((float(m['floor_strike']), m['ticker']))
        except (KeyError, TypeError, ValueError):
            continue
    if len(rungs) < 4:
        return []
    rungs.sort(key=lambda r: abs(r[0] - spot))
    near = rungs[:STRIKES_EACH_SIDE * 2]

    async def one(strike, ticker):
        async with sem:
            cs = await _candles(client, ticker, open_utc, close_utc)
        by_minute = {}
        for c in cs:
            end = c.get('end_period_ts')
            if end is None:
                continue
            bid, ask = candle_quote(c)
            by_minute[int(end)] = (bid, ask)
        return strike, ticker, by_minute

    got = await asyncio.gather(*[one(s, t) for s, t in near])

    rows = []
    for mins in FIT_OFFSETS:
        at = close_utc - dt.timedelta(minutes=mins)
        synthetic = []
        for strike, ticker, by_minute in got:
            quote = by_minute.get(int(at.timestamp()))
            if not quote or quote[0] is None or quote[1] is None:
                continue
            synthetic.append({
                'event_ticker': ev, 'ticker': ticker,
                'floor_strike': strike, 'close_time': close_utc.isoformat(),
                'yes_bid_dollars': quote[0], 'yes_ask_dollars': quote[1],
            })
        if len(synthetic) < 4:
            continue
        fits, _ = fits_for(synthetic, now=at, symbol=symbol,
                           min_minutes=1.0, max_minutes=600.0, min_r2=min_r2)
        for row in fits:
            # The fit was computed from the ladder as of `at`; it could not
            # have been known before then.
            row['event_time'] = pd.Timestamp(at)
            row['available_time'] = fetched_at
            # **Stamped, because three pipes write this table and they are not
            # the same measurement.** Until 2026-10-21 there was no `source`
            # column, so a backfill that left `available_time` unset was
            # indistinguishable from a live recording -- `_prepare` defaults it
            # to `event_time`, and 100% of the pre-August rows read as zero-lag
            # and looked live. Candles carry no sizes, so unlike the live
            # recorder this cannot refuse a rung with nothing resting behind
            # it; that difference has to be visible in the data.
            row['source'] = 'ladder_candles'
            rows.append(row)
    return rows


async def run(args) -> int:
    from core.datastore import ResearchStore
    from data_collection.kalshi_client import KalshiClient
    from scripts.record_implied_vol import LADDERS

    store = ResearchStore(args.research_store)
    start = pd.Timestamp(args.start, tz='UTC').ceil('h')
    end = (pd.Timestamp(args.end, tz='UTC') if args.end
           else pd.Timestamp.now(tz='UTC')).floor('h')
    wanted = [(s, sym) for s, sym in LADDERS
              if not args.symbols or sym in args.symbols]

    bars = {}
    for _, sym in wanted:
        try:
            b = store.read('minute_bars', venue='coinbase_spot', symbols=[sym],
                           start=start - pd.Timedelta(hours=2), end=end,
                           min_quality=None)
            b['event_time'] = pd.to_datetime(b['event_time'], utc=True)
            bars[sym] = b.set_index('event_time').sort_index()['close'].astype(float)
        except Exception as exc:                           # noqa: BLE001
            logger.warning('%s: no bars (%s)', sym, str(exc)[:90])

    def spot_at(symbol, when):
        s = bars.get(symbol)
        if s is None or not len(s):
            return None
        seg = s.loc[:pd.Timestamp(when)]
        return float(seg.iloc[-1]) if len(seg) else None

    done = set()
    if not args.force:
        try:
            held = store.read('venue_implied_vol', min_quality=None)
            if len(held):
                held['event_time'] = pd.to_datetime(held['event_time'], utc=True)
                done = set(zip(held['symbol'].astype(str),
                               held['event_time'].dt.floor('h')))
        except Exception as exc:                           # noqa: BLE001
            logger.info('nothing stored yet (%s)', str(exc)[:80])

    hours = pd.date_range(start, end, freq='h', inclusive='left')
    jobs = [(s, sym, h) for s, sym in wanted for h in hours
            if (sym, h.floor('h')) not in done]
    logger.info('%d hour(s) x %d ladder(s) -> %d to fit (%d already held)',
                len(hours), len(wanted), len(jobs), len(done))
    if args.dry_run or not jobs:
        return 0

    fetched_at = dt.datetime.now(dt.timezone.utc)
    sem = asyncio.Semaphore(max(1, args.concurrency))
    written = asked = 0
    async with KalshiClient(live=False) as client:
        for i in range(0, len(jobs), args.batch):
            chunk = jobs[i:i + args.batch]
            got = await asyncio.gather(*[
                hour_rows(client, sem, s, sym, h.to_pydatetime(), spot_at,
                          fetched_at, args.min_r2) for s, sym, h in chunk])
            rows = [r for g in got for r in g]
            if rows:
                written += store.write('venue_implied_vol', pd.DataFrame(rows))
            asked += len(chunk)
            logger.info('  %d/%d ladders, %d fit(s) written', asked, len(jobs), written)
    logger.info('done: %d fit(s)', written)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--start', default='2026-01-07')
    p.add_argument('--end', default=None)
    p.add_argument('--symbols', nargs='*', default=None)
    p.add_argument('--concurrency', type=int, default=6)
    p.add_argument('--batch', type=int, default=120)
    p.add_argument('--min-r2', type=float, default=0.90)
    p.add_argument('--force', action='store_true')
    p.add_argument('--research-store', default=os.getenv('RESEARCH_STORE'))
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('-v', '--verbose', action='store_true')
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
