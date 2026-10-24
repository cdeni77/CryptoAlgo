"""Market resolution must not depend on the venue's `status=open` filter.

Kalshi has two status vocabularies. The LIST filter takes
`open|unopened|closed|settled`; a market's own `status` reads
`initialized|active|finalized`. On 2026-10-24 `status=open` began returning
nothing while a market sat `active` with a full orderbook, so every window
resolved to None and the loop abstained on tradeable markets -- scoring NaN,
because a market-init model has no init score without a price.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from data_collection.kalshi_client import TRADEABLE_STATUS, KalshiClient

SETTLE = dt.datetime(2026, 10, 24, 16, 45, tzinfo=dt.timezone.utc)


def _market(ticker, status, close):
    return {'ticker': ticker, 'status': status, 'close_time': close}


class _Fake(KalshiClient):
    """Only `markets()` is exercised; nothing here touches the network."""

    def __init__(self, filtered, unfiltered):
        self._filtered, self._unfiltered = filtered, unfiltered
        self.calls = []

    async def markets(self, **params):
        self.calls.append(params.get('status'))
        return list(self._filtered if params.get('status') else self._unfiltered)


LIVE = _market('KXBTC15M-26OCT241245-45', 'active', '2026-10-24T16:45:00Z')
OLD = _market('KXBTC15M-26OCT241230-30', 'finalized', '2026-10-24T16:30:00Z')
NEXT = _market('KXBTC15M-26OCT241300-00', 'initialized', '2026-10-24T17:00:00Z')


def test_it_falls_back_when_the_filter_returns_nothing():
    c = _Fake(filtered=[], unfiltered=[OLD, LIVE, NEXT])
    out = asyncio.run(c.resolve_window_market('KXBTC15M', SETTLE))
    assert out is not None and out['ticker'] == LIVE['ticker']
    # Filtered first, unfiltered only as a fallback.
    assert c.calls == ['open', None]


def test_the_fallback_refuses_a_market_that_is_not_tradeable():
    """A finalized or initialized market must never be traded."""
    c = _Fake(filtered=[], unfiltered=[OLD, NEXT])
    assert asyncio.run(c.resolve_window_market('KXBTC15M', SETTLE)) is None


def test_the_filter_is_still_preferred_when_it_works():
    """If the venue restores the mapping, the cheap path is used and the
    unfiltered page is never fetched."""
    c = _Fake(filtered=[LIVE], unfiltered=[])
    out = asyncio.run(c.resolve_window_market('KXBTC15M', SETTLE))
    assert out['ticker'] == LIVE['ticker']
    assert c.calls == ['open']


def test_a_neighbouring_window_is_still_refused():
    """The tolerance still applies: abstain rather than trade a neighbour."""
    c = _Fake(filtered=[], unfiltered=[_market('X', 'active', '2026-10-24T17:00:00Z')])
    assert asyncio.run(c.resolve_window_market('KXBTC15M', SETTLE)) is None


def test_tradeable_status_covers_both_vocabularies():
    assert 'active' in TRADEABLE_STATUS and 'open' in TRADEABLE_STATUS
