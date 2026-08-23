"""The two ways the candle fetcher used to lose data, pinned.

Both were invisible for five years because they fail *quietly*: one dropped a
single minute per request and the other stored a bar whose numbers were merely
wrong rather than absent. Neither has a natural symptom, so they need a test that
counts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from data_collection.coinbase_connector import (
    MAX_CANDLES_PER_REQUEST, CoinbaseRESTClient,
)


class FakeVenue:
    """Coinbase's candles endpoint, with its actual boundary semantics.

    `start` and `end` are both **inclusive**, and when more candles fall in the
    range than `limit` allows the venue returns the **newest** `limit` of them.
    That combination is what the off-by-one fell into: asking for a `limit`-wide
    span names `limit + 1` candles, so the oldest is silently dropped.
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
        stamps = stamps[-limit:]                      # newest `limit`, as the venue does
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
    """The root cause, stated directly.

    `batch_duration = tf_seconds * 300` over an inclusive range names 301 candle
    starts for a limit of 300. This asserts the arithmetic, so the bug cannot
    come back by way of someone "simplifying" the `- 1`.
    """
    first = datetime(2025, 1, 1, 0, 0)
    venue = FakeVenue(first=first, last=first + timedelta(minutes=5000))
    await client_with(venue).get_candles_range(
        'BTC-USD', '1m', first, first + timedelta(minutes=5000))

    assert venue.requests, 'the fetcher made no request at all'
    for start, end, limit in venue.requests:
        spanned = (end - start) // 60 + 1               # inclusive on both ends
        assert spanned <= limit, (
            f'asked for {spanned} candle starts with limit {limit}; the venue '
            f'returns the newest {limit} and the rest are lost'
        )


@pytest.mark.asyncio
async def test_a_multi_batch_range_comes_back_with_no_holes():
    """The symptom: one minute in 301, forever.

    5,000 minutes spans several batches, so a boundary error shows up as missing
    minutes rather than as an exception. Before the fix this returned 4,984 of
    5,000 with the holes exactly 301 apart.
    """
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
    """Pin the constant, not just the inequality.

    `spanned <= limit` would also pass for a fetcher that requested one candle
    at a time. The intent is to use the whole allowance and no more.
    """
    first = datetime(2025, 1, 1, 0, 0)
    venue = FakeVenue(first=first, last=first + timedelta(minutes=5000))
    await client_with(venue).get_candles_range(
        'BTC-USD', '1m', first, first + timedelta(minutes=5000))
    start, end, limit = venue.requests[0]
    assert limit == MAX_CANDLES_PER_REQUEST
    assert (end - start) // 60 + 1 == MAX_CANDLES_PER_REQUEST


@pytest.mark.asyncio
async def test_the_minute_still_in_progress_is_never_returned():
    """A bar is data only once it has closed.

    Every call site passes `utc_now()` as `end`, so the newest candle the venue
    offers is the one for the minute in progress — partial high, low, close and
    volume. It was stored, and `DataPipeline` only accepts `event_time >
    last_known`, so it was never corrected. Observed in the real store: a scrape
    started at 04:17:29Z stored `04:17:00` holding 51.9% of a minute's volume.
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
