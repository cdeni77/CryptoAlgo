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
SERIES = 'KXBTCD'
SYMBOL = 'BTC-USD'
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
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2.0
    return number('last_price_dollars', 'last_price')


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
                    try:
                        payload = await client._request(  # noqa: SLF001
                            'GET', '/markets',
                            params={'series_ticker': SERIES, 'status': 'open',
                                    'limit': 200})
                    except Exception as exc:              # noqa: BLE001
                        logger.warning('markets: %s', str(exc)[:110])
                        await asyncio.sleep(args.interval)
                        continue

                    events: dict[str, list[dict]] = {}
                    for market in payload.get('markets', []):
                        event = str(market.get('event_ticker') or '')
                        if event:
                            events.setdefault(event, []).append(market)

                    for event, markets in events.items():
                        closes = [m.get('close_time') for m in markets
                                  if m.get('close_time')]
                        if not closes:
                            continue
                        close = pd.Timestamp(min(closes))
                        if close.tzinfo is None:
                            close = close.tz_localize('UTC')
                        minutes = (close - pd.Timestamp(now)).total_seconds() / 60.0
                        if not (args.min_minutes <= minutes <= args.max_minutes):
                            continue
                        rungs = []
                        for market in markets:
                            strike, mid = strike_of(market), mid_of(market)
                            if strike is not None and mid is not None:
                                rungs.append((strike, mid))
                        fit = implied_sigma(rungs, minutes)
                        if fit is None or fit.r2 < args.min_r2:
                            continue
                        rows.append({
                            'venue': 'kalshi', 'symbol': SYMBOL,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--interval', type=float, default=60.0)
    parser.add_argument('--batch-rows', type=int, default=10)
    parser.add_argument('--min-minutes', type=float, default=2.0)
    parser.add_argument('--max-minutes', type=float, default=180.0)
    parser.add_argument('--min-r2', type=float, default=0.90,
                        help='below this the ladder is not internally '
                             'consistent and the fit is not a measurement')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-7s %(name)s %(message)s',
                        datefmt='%H:%M:%S')
    return asyncio.run(run(args))


if __name__ == '__main__':
    raise SystemExit(main())
