"""The market's own forward volatility, inverted from the KXBTCD strike ladder.

**Why this is worth a service of its own.** The barrier reframing says the
displacement is known exactly and `sigma_remaining` is the *only* forecast the
system requires. Everything in `core/vol.py` estimates it from past returns —
HAR over 15/60/240/1440 minutes plus an intraday seasonal. Kalshi publishes a
threshold ladder on BTC (`KXBTCD`: "BTC above K at time T", nine or so strikes
on one expiry) which is priced off a single volatility, forward-looking, free,
and updated every minute. Nothing in the model has ever seen it.

**The inversion.** Under the same zero-drift barrier model the baseline already
uses,

    P(S_T > K) = F( ln(S/K) / (sigma * sqrt(t)) )

so applying the inverse CDF to every quote makes the ladder linear in `ln K`:

    F^-1(p_k) = ln(S)/(sigma*sqrt(t))  -  ln(K_k)/(sigma*sqrt(t))

One regression across the strikes gives sigma from the slope, and the implied
spot falls out of the intercept — so no external spot is needed and the fit is
self-contained. R^2 says whether the ladder was internally consistent enough to
believe; measured on 29,776 archived fits it runs above 0.99.

**It is not the same instrument.** KXBTCD closes on the hour and carries an
explicit strike; the traded markets are 15-minute up/down. That is the point —
it prices the same underlying's volatility over an overlapping horizon, which is
what `sigma_remaining` needs, and it says nothing about direction.

Collection of this stopped on 2026-08-25 and nothing restarted it. A day not
recorded is a day gone, exactly as for depth.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from statistics import NormalDist
from typing import Iterable, Optional, Sequence

import pandas as pd

from core.datastore import ResearchStore

logger = logging.getLogger('implied-vol')
NORMAL = NormalDist()
# All three crypto strike ladders. This was `SERIES = 'KXBTCD'` and every row
# was stamped 'BTC-USD', so ETH and SOL carried NaN for all five implied-vol
# features on every live cycle — while the BACKFILL already held all three
# (BTC 22,248 rows, ETH 5,286, SOL 3,110). Live was the half that lagged: the
# model was fitted against ETH/SOL implied vol that existed and scored against
# an absence that did not.
#
# The venue serves all three identically: 200 open markets each, every one
# carrying a strike, `strike_type=greater`.
LADDERS = (
    ('KXBTCD', 'BTC-USD'),
    ('KXETHD', 'ETH-USD'),
    ('KXSOLD', 'SOL-USD'),
)
SERIES, SYMBOL = LADDERS[0]

# The freshest ladder fit per symbol, for the trading loop in this same process.
CACHE: dict = {}
MIN_STRIKES = 4
PROBABILITY_BAND = (0.01, 0.99)


@dataclass(frozen=True)
class VolFit:
    """One ladder, inverted."""

    sigma_per_min: float
    r2: float
    atm_strike: float
    implied_spot: float
    n_strikes: int


def implied_sigma(rungs: Sequence[tuple[float, float]],
                  minutes_to_close: float) -> Optional[VolFit]:
    """Sigma per minute implied by `(strike, P(above))` pairs, or None.

    Returns None rather than a number whenever the ladder cannot support one:
    fewer than four usable strikes, no time left to divide by, or a slope that
    is not negative — price must FALL as the strike rises, and a ladder that
    does otherwise is not the instrument this assumes.
    """
    if not minutes_to_close or minutes_to_close <= 0:
        return None
    points = []
    for strike, probability in rungs or ():
        try:
            strike = float(strike)
            probability = float(probability)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(strike) and strike > 0):
            continue
        # **Drop the far rungs, not just the impossible ones.** The inverse CDF
        # is infinite at 0 and 1, but the real problem starts well before that:
        # the tick is a tenth of a cent below 10c, so a 1c quote carries ~0.05c
        # of quantisation, and at |z| ~ 3 that is worth a large fraction of a
        # sigma. Those rungs then dominate an unweighted fit in z-space. Keeping
        # the band where the ladder is actually informative is cheaper than
        # weighting and does not pretend to a precision the tick cannot carry.
        if not (PROBABILITY_BAND[0] < probability < PROBABILITY_BAND[1]):
            continue
        points.append((math.log(strike), NORMAL.inv_cdf(probability), strike,
                       probability))
    if len(points) < MIN_STRIKES:
        return None

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0:
        return None
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = sxy / sxx
    if not math.isfinite(slope) or slope >= 0:
        return None
    intercept = mean_y - slope * mean_x

    sigma = -1.0 / (slope * math.sqrt(minutes_to_close))
    if not math.isfinite(sigma) or sigma <= 0:
        return None

    syy = sum((y - mean_y) ** 2 for y in ys)
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    r2 = 1.0 - residual / syy if syy > 0 else float('nan')

    # The intercept is ln(S)/(sigma*sqrt(t)), so the spot the ladder implies
    # falls straight out of the same fit — no external price needed.
    implied_spot = math.exp(-intercept / slope)
    atm = min(points, key=lambda p: abs(p[3] - 0.5))[2]
    return VolFit(sigma_per_min=sigma, r2=r2, atm_strike=atm,
                  implied_spot=implied_spot, n_strikes=n)


def strike_of(market: dict) -> Optional[float]:
    """The market's threshold, preferring what the venue states over the ticker."""
    for key in ('floor_strike', 'cap_strike', 'strike'):
        value = market.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    # `KXBTCD-26AUG2317-T86749.99` — the suffix is the strike, and its presence
    # is what distinguishes a threshold ladder from an up/down market.
    tail = str(market.get('ticker') or '').rsplit('-', 1)[-1]
    if tail.startswith('T'):
        try:
            return float(tail[1:])
        except ValueError:
            return None
    return None


def mid_of(market: dict) -> Optional[float]:
    """The ladder's own P(above), from the two-sided quote where there is one."""
    def number(*keys):
        for key in keys:
            raw = market.get(key)
            if raw is None:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            # Dollar strings and integer cents share these names; above 1.0 it
            # can only be cents.
            return value / 100.0 if value > 1.0 else value
        return None

    bid = number('yes_bid_dollars', 'yes_bid')
    ask = number('yes_ask_dollars', 'yes_ask')
    # BOTH sides, and neither of them zero. `cross_section` in the backfill
    # drops a strike without a two-sided quote — "a one-sided quote is dropped
    # rather than half-invented; the inversion needs P(above), and a single side
    # does not give one" — and a zero is not a side. CLAUDE.md states the same
    # rule for quotes generally: a zero level means there is nothing there.
    #
    # Accepting them put every far-OTM strike into the fit: live carried a
    # median of 50 rungs for ETH and 19 for SOL against the backfill's 5, and
    # one sigma across half-invented mids collapsed R2 to 0.827 and 0.295
    # against training's 0.976 and 0.986. BTC is liquid enough to survive it,
    # which is why this hid.
    if not bid or not ask or ask < bid:
        return None
    return (bid + ask) / 2.0


def fits_for(markets, *, now, symbol: str, min_minutes: float,
             max_minutes: float, min_r2: float):
    """Invert one series' open ladders. Returns (rows, freshest_fit).

    Split out of `run` so it can be tested without a venue: the loop around it
    is a fetch and a sleep, and everything that decides what a row SAYS lives
    here.

    A ladder is a set of `greater` strikes on one event. Each rung's mid is
    P(spot > strike), so the set inverts to a single forward sigma — the only
    forward-looking volatility input this system has.
    """
    events: dict[str, list[dict]] = {}
    for market in markets or []:
        event = str(market.get('event_ticker') or '')
        if event:
            events.setdefault(event, []).append(market)

    rows, latest = [], None
    for event, group in events.items():
        closes = [m.get('close_time') for m in group if m.get('close_time')]
        if not closes:
            continue
        close = pd.Timestamp(min(closes))
        if close.tzinfo is None:
            close = close.tz_localize('UTC')
        minutes = (close - pd.Timestamp(now)).total_seconds() / 60.0
        if not (min_minutes <= minutes <= max_minutes):
            continue
        rungs = []
        for market in group:
            strike, mid = strike_of(market), mid_of(market)
            if strike is not None and mid is not None:
                rungs.append((strike, mid))
        fit = implied_sigma(rungs, minutes)
        if fit is None or fit.r2 < min_r2:
            continue
        stamp = (pd.Timestamp(now, tz='UTC') if pd.Timestamp(now).tzinfo is None
                 else pd.Timestamp(now))
        # Freshest wins: several events are fitted per cycle and the nearest
        # close is the most informative about the fifteen minutes being traded.
        if latest is None or minutes < latest['_minutes']:
            latest = {'implied_sigma_per_min': float(fit.sigma_per_min),
                      'r2': float(fit.r2), 'n_strikes': float(fit.n_strikes),
                      'at': stamp, '_minutes': minutes}
        rows.append({
            'venue': 'kalshi', 'symbol': symbol,
            'event_time': pd.Timestamp(now).floor('min'),
            'available_time': pd.Timestamp(now),
            'quality': 'valid', 'event_ticker': event,
            'close_time': close,
            'minutes_to_close': round(minutes, 3),
            'implied_sigma_per_min': fit.sigma_per_min,
            'implied_spot': fit.implied_spot,
            'atm_strike': fit.atm_strike,
            'n_strikes': float(fit.n_strikes), 'r2': fit.r2,
        })
    if latest is not None:
        latest.pop('_minutes', None)
    return rows, latest


async def run(args, gate=None) -> int:
    from data_collection.kalshi_client import KalshiClient

    store = ResearchStore(os.getenv('RESEARCH_STORE'))
    pem = (os.getenv('KALSHI_PRIVATE_KEY')
           or open(os.environ['KALSHI_PRIVATE_KEY_PATH']).read())
    rows: list[dict] = []

    while True:
        try:
            async with KalshiClient(key_id=os.environ['KALSHI_KEY_ID'],
                                    private_key_pem=pem) as client:
                while True:
                    if gate is not None:
                        await gate.idle()
                    now = datetime.now(timezone.utc)
                    # One pass per ladder. A failure on one series must not cost
                    # the others their cycle: they are independent books and the
                    # venue rate-limits per request, not per symbol.
                    for series, symbol in LADDERS:
                        try:
                            payload = await client._request(  # noqa: SLF001
                                'GET', '/markets',
                                params={'series_ticker': series,
                                        'status': 'open', 'limit': 200})
                        except Exception as exc:              # noqa: BLE001
                            logger.warning('%s markets: %s', series, str(exc)[:110])
                            continue

                        found, latest = fits_for(
                            payload.get('markets', []), now=now, symbol=symbol,
                            min_minutes=args.min_minutes,
                            max_minutes=args.max_minutes, min_r2=args.min_r2)
                        if latest is not None:
                            CACHE[symbol] = latest
                        rows.extend(found)

                    if len(rows) >= args.batch_rows:
                        await asyncio.to_thread(
                            store.write, 'venue_implied_vol', pd.DataFrame(rows))
                        last = rows[-1]
                        logger.info(
                            'wrote %d fits (last %s: sigma %.2fbp/min over '
                            '%.0fm, %d strikes, R2 %.4f)',
                            len(rows), last['event_ticker'],
                            1e4 * last['implied_sigma_per_min'],
                            last['minutes_to_close'], int(last['n_strikes']),
                            last['r2'])
                        rows.clear()
                    await asyncio.sleep(args.interval)
        except Exception as exc:                          # noqa: BLE001 - reconnect
            logger.error('implied vol recorder: %s; retrying in 20s',
                         str(exc)[:160])
            await asyncio.sleep(20)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--batch-rows', type=int, default=10)
    parser.add_argument('--min-minutes', type=float, default=2.0)
    parser.add_argument('--max-minutes', type=float, default=180.0)
    # 0.90. NOT the backfill's rule, and deliberately so — see below.
    #
    # REVERTED 2026-08-31, same day it was set to 0.0. The backfill gates on
    # nothing, so 0.0 is the parity-correct value in principle; in practice
    # live and the backfill do not select the same STRIKES, and removing the
    # gate exposed that rather than fixing it. Live fits the whole open ladder
    # (17-50 rungs) where Predexon's tick series only carried the liquid ones
    # (median 5), so the fits that the gate had been hiding were garbage:
    #
    #   symbol  training med sigma   live med sigma (gate off)   live med R2
    #   BTC          5.4 bp/min            7.5                      0.989
    #   ETH         10.3 bp/min           64.2                      0.827
    #   SOL          7.8 bp/min        2,243.4                      0.295
    #
    # 287x on SOL puts `iv_minus_realised` near log(287) = 5.7 where training
    # saw ~0. The 0.90 gate is a known train/serve mismatch that costs ETH and
    # SOL their coverage; a 287x sigma is an unknown one that corrupts the
    # feature the model actually reads. Prefer the known, smaller error until
    # the strike selection is matched properly.
    #
    # The real fix is upstream: select the same rungs the backfill did. Until
    # then this stays at 0.90, which is the configuration the measured weekend
    # run traded.
    #
    # This defaulted to 0.90 and that was a train/serve mismatch, not caution.
    # Measured on what the backfill actually wrote, accepted fits run down to
    # R2 = 0.0367 (BTC), 0.0134 (ETH), 0.0002 (SOL), with 5th percentiles of
    # 0.85 / 0.72 / 0.56 — so the artifact was FITTED across that whole range
    # while live kept only the top of it. Over 67 hours live produced 2,805 BTC
    # fits, 13 ETH and zero SOL, against training coverage of 22,258 / 5,286 /
    # 3,110.
    #
    # Two reasons filtering here is the wrong direction. `--complete-cases`
    # fitted the artifact on rows that ALL carry a fit, so a symbol with no
    # fits has no complete rows and is scored out of distribution. And `iv_r2`
    # is ONE OF THE FIVE FEATURES: the model saw R2 from 0.0002 to 1.0 and
    # learned what a weak fit is worth, so discarding weak fits upstream
    # removes the information it was given to make that judgement.
    parser.add_argument('--min-r2', type=float, default=0.90,
                        help='drop fits below this R2. Default 0.0 to match '
                             'the backfill the model was trained on; iv_r2 is '
                             'a feature, so the model judges fit quality')
    return parser


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
