"""Venue drivers. Each one knows its API's shape and nothing else.

A fetcher takes a market id and a window and returns that window's tick series
and a status. The orchestrator owns rate limits, retries, resume and storage.

Both venues pack into the same thirteen fields, with prices in integer cents.
Polymarket serves dollars, so it is converted here.
"""
from __future__ import annotations

import datetime as dt
import zoneinfo
from typing import Optional

ET = zoneinfo.ZoneInfo('America/New_York')
MONTHS = ('JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
          'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC')
WINDOW = dt.timedelta(minutes=15)

FIELDS = ('ts', 'best_bid', 'best_ask', 'bid_at_touch', 'ask_at_touch',
          'bid_1c', 'ask_1c', 'bid_5c', 'ask_5c', 'bid_levels', 'ask_levels',
          'bid_vol', 'ask_vol')


# -- identifiers -> the window they name -------------------------------------

def kalshi_window_open(ticker: str) -> dt.datetime:
    """The window a Kalshi ticker names, as UTC.

    `KXBTC15M-26JAN061730-30` encodes the window's close in US Eastern time
    (EST or EDT), so the open is fifteen minutes earlier.
    """
    try:
        stamp = ticker.split('-')[1]
        year = 2000 + int(stamp[0:2])
        month = MONTHS.index(stamp[2:5]) + 1
        day, hour, minute = int(stamp[5:7]), int(stamp[7:9]), int(stamp[9:11])
    except (IndexError, ValueError) as exc:
        raise ValueError(f'not a 15-minute Kalshi ticker: {ticker!r}') from exc
    close = dt.datetime(year, month, day, hour, minute, tzinfo=ET)
    return (close - WINDOW).astimezone(dt.timezone.utc)


def pm_window_open(slug: str) -> dt.datetime:
    """The window a Polymarket slug names, as UTC.

    The trailing unix stamp is the window's open, not its close.
    """
    stamp = (slug or '').rsplit('-', 1)[-1]
    if not stamp.isdigit():
        raise ValueError(f'no unix stamp in slug: {slug!r}')
    return dt.datetime.fromtimestamp(int(stamp), dt.timezone.utc)


def verify_window(derived: dt.datetime, *, venue_open: Optional[dt.datetime],
                  venue_close: Optional[dt.datetime],
                  tolerance: dt.timedelta = dt.timedelta(seconds=90)) -> Optional[str]:
    """Cross-check a derived window against the venue's own stated times.

    Returns None when they agree, or a description of the disagreement.
    Absent venue times are a failure, not a pass.
    """
    if venue_open is None and venue_close is None:
        return 'venue stated neither open nor close; cannot corroborate'
    if venue_open is not None and abs(venue_open - derived) > tolerance:
        return (f'derived open {derived.isoformat()} disagrees with venue open '
                f'{venue_open.isoformat()}')
    if venue_close is not None:
        expected = derived + WINDOW
        if abs(venue_close - expected) > tolerance:
            return (f'venue close {venue_close.isoformat()} is not 15 minutes '
                    f'after the derived open {derived.isoformat()}')
    return None


# -- packing -----------------------------------------------------------------

def _pack(bids, asks, ts):
    """Thirteen numbers, shared by both venues. Prices are integer cents."""
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)

    def within(side, best, sign, cents):
        if best is None:
            return 0
        return sum(s for p, s in side if 0 <= sign * (best - p) <= cents)

    return [
        ts, best_bid, best_ask,
        sum(s for p, s in bids if p == best_bid) if best_bid is not None else 0,
        sum(s for p, s in asks if p == best_ask) if best_ask is not None else 0,
        within(bids, best_bid, 1, 1), within(asks, best_ask, -1, 1),
        within(bids, best_bid, 1, 5), within(asks, best_ask, -1, 5),
        len(bids), len(asks),
        sum(s for _, s in bids), sum(s for _, s in asks),
    ]


def pack_kalshi(snapshot: dict) -> list:
    bids = [(b['price'], b['size']) for b in snapshot.get('yes_bids') or []]
    asks = [(a['price'], a['size']) for a in snapshot.get('yes_asks') or []]
    return _pack(bids, asks, snapshot.get('timestamp'))


def pack_pm(snapshot: dict) -> list:
    """Polymarket serves dollar strings; convert to cents to match Kalshi."""
    def side(raw):
        out = []
        for entry in raw or []:
            try:
                out.append((int(round(float(entry['price']) * 100)),
                            float(entry['size'])))
            except (TypeError, ValueError, KeyError):
                continue
        return out
    return _pack(side(snapshot.get('bids')), side(snapshot.get('asks')),
                 snapshot.get('timestamp'))
