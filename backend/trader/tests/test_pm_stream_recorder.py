"""Tests for the Polymarket socket recorder `scripts.record_pm_stream`.

Silence returns on a timeout, the recorder resubscribes as the window rolls, and it
publishes the shape `cross_venue_row` reads.
"""
from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from scripts import record_pm_stream as rec


def test_silence_is_a_condition_not_a_wait():
    """A silent socket returns a reason instead of blocking forever."""
    async def _run():
        async def never():
            await asyncio.sleep(3600)

        reason = await rec.consume(_Stream(never), until=None,
                                   silence_seconds=0.2)
        return reason

    reason = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    assert 'silen' in reason.lower(), reason


def test_the_refresh_deadline_fires_even_while_frames_pour_in():
    """A busy socket still hits its refresh deadline at the window rollover."""
    async def _run():
        async def flood():
            return {'event_type': 'price_change', 'price_changes': []}
        return await rec.consume(_Stream(flood), until=0.2, silence_seconds=5.0)

    reason = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(_run())
    assert 'refresh' in reason.lower() or 'window' in reason.lower(), reason


def test_publish_matches_the_shape_cross_venue_row_reads():
    """The published entry matches `record_pm_ladder.CACHE`: cents, plus a stamp for the
    staleness check.
    """
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}],
                'asks': [{'price': '0.46', 'size': '150'}]})
    rec.publish(book, {'T': 'BTC-USD'})
    entry = rec.CACHE['BTC-USD']
    assert entry['best_bid'] == pytest.approx(44.0), 'cents, not dollars'
    assert entry['best_ask'] == pytest.approx(46.0)
    assert isinstance(entry['at'], pd.Timestamp)


def test_a_one_sided_book_is_not_published_as_a_touch():
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'T', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}], 'asks': []})
    rec.publish(book, {'T': 'BTC-USD'})
    assert 'BTC-USD' not in rec.CACHE or rec.CACHE['BTC-USD'].get('best_ask') is None


class _Stream:
    """A stand-in socket whose next frame comes from `factory`."""

    def __init__(self, factory):
        self._factory = factory

    def events(self):
        factory = self._factory

        async def _gen():
            while True:
                out = await factory()
                if out is not None:
                    yield out
        return _gen()


def test_the_streamed_window_is_the_one_kalshi_settles():
    """The slug names the window open that Kalshi's ticker settles 15 minutes later.

    Kalshi's ticker is keyed on the settle time and Polymarket's slug on the window
    open. Reading the slug as a close would shift every window by 15 minutes.
    """
    import datetime as dt
    from scripts.record_pm_ladder import slug_for, window_of

    # A decision inside the 22:30-22:45 window, which Kalshi's -2245 ticker names.
    now = dt.datetime(2026, 9, 3, 22, 37, tzinfo=dt.timezone.utc)
    opened = window_of(slug_for('btc', now))
    assert opened == pd.Timestamp('2026-09-03 22:30', tz='UTC'), (
        'the slug must name the window OPEN that Kalshi settles at 22:45')
    assert opened + pd.Timedelta(minutes=15) == pd.Timestamp(
        '2026-09-03 22:45', tz='UTC')


def test_a_decision_at_the_boundary_takes_the_window_it_is_inside():
    """At the boundary the new window is used (floor, never ceil)."""
    import datetime as dt
    from scripts.record_pm_ladder import slug_for, window_of

    at_open = dt.datetime(2026, 9, 3, 22, 45, tzinfo=dt.timezone.utc)
    assert window_of(slug_for('btc', at_open)) == pd.Timestamp(
        '2026-09-03 22:45', tz='UTC')
    one_before = dt.datetime(2026, 9, 3, 22, 44, 59, tzinfo=dt.timezone.utc)
    assert window_of(slug_for('btc', one_before)) == pd.Timestamp(
        '2026-09-03 22:30', tz='UTC')


def test_a_rolled_window_does_not_leave_the_old_book_readable():
    """The cache records which window its reading belongs to.

    At a rollover the cache can still hold the previous window's book with a recent
    stamp, which passes the staleness check. Only the window tells them apart.
    """
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'OLD', 'timestamp': '1788487941413',
                'bids': [{'price': '0.44', 'size': '250'}],
                'asks': [{'price': '0.46', 'size': '150'}]})
    rec.publish(book, {'OLD': 'BTC-USD'}, window=pd.Timestamp('2026-09-03 22:30', tz='UTC'))
    assert rec.CACHE['BTC-USD']['window'] == pd.Timestamp('2026-09-03 22:30', tz='UTC')

    # The window rolls; the recorder has not caught up yet.
    entry = rec.CACHE['BTC-USD']
    assert entry['window'] != pd.Timestamp('2026-09-03 22:45', tz='UTC'), (
        'the cached reading belongs to the previous window and a consumer must '
        'be able to tell')


def test_publishing_a_new_window_replaces_the_old_reading():
    rec.CACHE.clear()
    book = rec.PmBookCache()
    book.apply({'event_type': 'book', 'asset_id': 'NEW', 'timestamp': '1788487941413',
                'bids': [{'price': '0.30', 'size': '10'}],
                'asks': [{'price': '0.32', 'size': '10'}]})
    rec.publish(book, {'NEW': 'BTC-USD'}, window=pd.Timestamp('2026-09-03 22:45', tz='UTC'))
    assert rec.CACHE['BTC-USD']['window'] == pd.Timestamp('2026-09-03 22:45', tz='UTC')
    assert rec.CACHE['BTC-USD']['best_bid'] == pytest.approx(30.0)
