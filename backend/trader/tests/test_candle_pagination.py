"""Tests that the Coinbase candle fetcher loses no minute and stores no partial bar.

Both failures are silent: a dropped minute per request, and a bar whose values are
incomplete. Only a test that counts can catch them.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data_collection.coinbase_connector import (
    MAX_CANDLES_PER_REQUEST, CoinbaseRESTClient,
)


class FakeVenue:
    """Fake Coinbase candles endpoint with the venue's boundary rules.

    `start` and `end` are both inclusive. When more candles fall in the range than
    `limit`, the venue returns the newest `limit` and drops the oldest.
    """

    def __init__(self, *, first: datetime, last: datetime, step_seconds: int = 60):
        self.first, self.last, self.step = first, last, step_seconds
        self.requests: list[tuple[int, int, int]] = []

    async def __call__(self, method, path, params=None, **kwargs):
        params = params or {}
        start = int(params['start'])
        end = int(params['end'])
        limit = int(params['limit'])
        self.requests.append((start, end, limit))
        first = int(self.first.replace(tzinfo=timezone.utc).timestamp())
        last = int(self.last.replace(tzinfo=timezone.utc).timestamp())
        stamps = [t for t in range(first, last + 1, self.step) if start <= t <= end]
        stamps = stamps[-limit:]  # newest `limit`, as the venue does
        return 200, {'candles': [
            {'start': str(t), 'open': '100', 'high': '101', 'low': '99',
             'close': '100.5', 'volume': '1'} for t in reversed(stamps)
        ]}


def client_with(venue: FakeVenue) -> CoinbaseRESTClient:
    client = CoinbaseRESTClient(api_key=None, api_secret=None)
    client._request = venue                            # noqa: SLF001 - that is the seam
    return client


@pytest.mark.asyncio
async def test_a_request_never_spans_more_candles_than_its_limit_allows():
    """No request spans more candle starts than its limit, counting both ends."""
    first = datetime(2025, 1, 1, 0, 0)
    venue = FakeVenue(first=first, last=first + timedelta(minutes=5000))
    await client_with(venue).get_candles_range(
        'BTC-USD', '1m', first, first + timedelta(minutes=5000))

    assert venue.requests, 'the fetcher made no request at all'
    for start, end, limit in venue.requests:
        spanned = (end - start) // 60 + 1  # inclusive on both ends
        assert spanned <= limit, (
            f'asked for {spanned} candle starts with limit {limit}; the venue '
            f'returns the newest {limit} and the rest are lost'
        )


@pytest.mark.asyncio
async def test_a_multi_batch_range_comes_back_with_no_holes():
    """A multi-batch range returns every minute, with no holes at batch boundaries."""
    first = datetime(2025, 1, 1, 0, 0)
    minutes = 5000
    last = first + timedelta(minutes=minutes)
    venue = FakeVenue(first=first, last=last)
    bars = await client_with(venue).get_candles_range('BTC-USD', '1m', first, last)

    times = [b.event_time for b in bars]
    assert times == sorted(times), 'bars came back out of order'
    assert len(times) == len(set(times)), 'a candle was returned twice'
    expected = [first + timedelta(minutes=i) for i in range(minutes + 1)]
    missing = sorted(set(expected) - set(times))
    assert not missing, f'{len(missing)} minutes lost, first at {missing[:3]}'


@pytest.mark.asyncio
async def test_the_span_is_exactly_one_short_of_the_limit():
    """Each request uses exactly the full limit, not just at most the limit."""
    first = datetime(2025, 1, 1, 0, 0)
    venue = FakeVenue(first=first, last=first + timedelta(minutes=5000))
    await client_with(venue).get_candles_range(
        'BTC-USD', '1m', first, first + timedelta(minutes=5000))
    start, end, limit = venue.requests[0]
    assert limit == MAX_CANDLES_PER_REQUEST
    assert (end - start) // 60 + 1 == MAX_CANDLES_PER_REQUEST


@pytest.mark.asyncio
async def test_the_minute_still_in_progress_is_never_returned():
    """The bar for the minute in progress is never returned, since its values are
    partial.

    The pipeline only accepts bars newer than the last stored one, so a stored partial
    bar is never corrected.
    """
    now = datetime.now(timezone.utc).replace(tzinfo=None, second=30, microsecond=0)
    in_progress = now.replace(second=0)
    venue = FakeVenue(first=in_progress - timedelta(minutes=30), last=in_progress)
    bars = await client_with(venue).get_candles('BTC-USD', '1m',
                                               in_progress - timedelta(minutes=30), now)

    assert bars, 'the closed bars were dropped too'
    newest = max(b.event_time for b in bars)
    assert newest == in_progress - timedelta(minutes=1), (
        f'newest bar is {newest}, but the minute at {in_progress} has not closed'
    )
    for bar in bars:
        assert bar.available_time <= now, (
            f'bar at {bar.event_time} closes at {bar.available_time}, after now'
        )
